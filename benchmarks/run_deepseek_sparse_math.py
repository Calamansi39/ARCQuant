"""Native DeepSeek MATH/GPQA generation with ARCQuant HiF4 + TEAL/WINA.

ARC is the default. Use --quantization hif4 only for explicitly labelled
ablations. The external tokenizer, generation scheduler and scorer stay pinned.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ('benchmarks/run_deepseek_sparse_math.py', 'model/deepseek_generation_arc.py',
           'model/deepseek_generation_runtime.py', 'model/deepseek_experiment.py',
           'model/deepseek_adapter.py', 'model/sparse_utils.py')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def runtime_fingerprint():
    return {path: digest(ROOT / path) for path in SOURCES}


def calibration_metadata(path, source, converted, world):
    from deepseek_v4_support import source_signature, runtime_signature
    manifest = json.loads((path / 'manifest.json').read_text())
    conversion = json.loads((Path(converted) / 'conversion.json').read_text())
    signature = source_signature(source)
    if (manifest['source_signature'] != signature or manifest['world_size'] != world
            or manifest['runtime_signature'] != runtime_signature()):
        raise ValueError('Calibration model/TP/runtime signature mismatch')
    if (conversion['source_signature'] != signature or conversion['world_size'] != world
            or not conversion['completed']):
        raise ValueError('Converted checkpoint model/TP signature mismatch')
    return dict(manifest_sha256=digest(path / 'manifest.json'),
                rank_sha256={str(rank): digest(path / f'rank{rank}.pt') for rank in range(world)},
                world_size=world, dataset=manifest['dataset'], samples=manifest['samples'],
                length=manifest['length'], arc_policy=manifest['arc_policy'])


def patched_prepare(original, metadata):
    """Insert identity BEFORE native protocol comparison, hashing and saving."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    body = tree.body[0].body
    position = next((i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                     and ast.unparse(node) == 'out = Path(a.out_dir)'), None)
    if position is None:
        raise RuntimeError('Unsupported native protocol preparation; refusing ARC resume')
    body[position:position] = ast.parse("""
protocol['arcquant'] = arcquant_metadata
protocol['quantization']['arc'] = arcquant_metadata['arc']
if arcquant_metadata['arc']:
    protocol['quantization']['fp4'] = 'HiF4 W4A4; sparse mask, calibrated channel reorder, quantized activation residual channels; dense GEMM'
""").body
    ast.fix_missing_locations(tree)
    namespace = {**original.__globals__, 'arcquant_metadata': metadata}
    exec(compile(tree, '<arcquant-generation-protocol>', 'exec'), namespace)
    return namespace[original.__name__]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--sparse-method', choices=('none', 'teal', 'wina'), required=True)
    parser.add_argument('--sparsity', type=float, required=True)
    parser.add_argument('--quantization', choices=('arc', 'hif4'), default='arc')
    parser.add_argument('--benchmark', choices=('math500', 'gpqa'), default='math500')
    parser.add_argument('--base-eval-root', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    sparse, remaining = parser.parse_known_args(argv)
    if not 0 <= sparse.sparsity < 1 or (sparse.sparse_method == 'none' and sparse.sparsity != 0):
        parser.error('sparsity must be in [0, 1); method none requires zero')
    arc = sparse.quantization == 'arc'
    if (arc or sparse.sparse_method == 'teal') and sparse.calibration is None:
        parser.error('--calibration is required for ARC and TEAL')
    sys.path.insert(0, str(sparse.base_eval_root.expanduser().resolve()))

    import deepseek_v4_quant as quant
    import deepseek_v4_lcb_batch_quant as request_quant
    import deepseek_generative_batch as batch
    import run_generative_eval as reference
    from model.deepseek_generation_arc import DeepSeekGenerationArcMixin, install_arc_grouped_attention
    from model.deepseek_generation_runtime import DeepSeekGenerationTealMixin, DeepSeekGenerationWinaMixin

    args = reference.parse_args(['--model', 'deepseek', '--benchmark', sparse.benchmark, *remaining])
    if args.quant != 'hifp4' or args.mode != 'w4a4':
        parser.error('This launcher requires --quant hifp4 --mode w4a4')
    if args.reuse_completed:
        parser.error('--reuse-completed is disabled for sparse/ARC evaluations')
    calibration = None
    if sparse.calibration is not None:
        sparse.calibration = sparse.calibration.expanduser().resolve()
        world = int(os.environ.get('WORLD_SIZE', '0'))
        if not world:
            world = json.loads((Path(args.converted_model) / 'conversion.json').read_text())['world_size']
        calibration = calibration_metadata(sparse.calibration, args.source_model, args.converted_model, world)
    sources = runtime_fingerprint()
    metadata = dict(schema_version=2, method=sparse.sparse_method, sparsity=sparse.sparsity,
                    allocation='uniform', arc=arc, quantization=sparse.quantization,
                    kernel='HiF4-value-reconstruction+dense-GEMM', calibration=calibration,
                    implementation_sha256=sources)
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata['runtime_sha256'] = identity
    args.out_dir = str(Path(args.out_dir) /
                       f'arcquant-{sparse.quantization}-{sparse.sparse_method}-{sparse.sparsity:g}-{identity[:12]}')
    out = Path(args.out_dir)

    base_runtime = request_quant.RequestQuantRuntime
    mixin = DeepSeekGenerationTealMixin if sparse.sparse_method == 'teal' else DeepSeekGenerationWinaMixin
    bases = (DeepSeekGenerationArcMixin, mixin, base_runtime) if arc else (mixin, base_runtime)
    SparseRuntime = type('SparseRuntime', bases, {})
    original_install = quant.install_reference_model_patches
    original_ragged = batch.install_ragged_attention
    original_prepare = reference.prepare
    native_prepare = patched_prepare(original_prepare, metadata)
    runtimes = []

    def install(ds, runtime):
        original_install(ds, runtime)
        if arc:
            install_arc_grouped_attention(ds, runtime, 'standard')
        original_transformer = ds.Transformer

        def transformer(*ctor_args, **ctor_kwargs):
            import torch
            model = original_transformer(*ctor_args, **ctor_kwargs)
            values = None
            if arc or sparse.sparse_method == 'teal':
                rank = os.environ.get('RANK', '0')
                values = torch.load(sparse.calibration / f'rank{rank}.pt', map_location='cpu', weights_only=True)
            options = {'calibration': values} if sparse.sparse_method == 'teal' else {}
            runtime.configure_sparse(ds, model, sparse.sparsity, **options)
            if arc:
                runtime.configure_arc(ds, model, values, sparse.sparse_method)
            runtimes.append(runtime)
            return model

        ds.Transformer = transformer

    def ragged(ds, runtime):
        original_ragged(ds, runtime)
        if arc:
            install_arc_grouped_attention(ds, runtime, 'ragged')

    def prepare(a, *prep_args, **prep_kwargs):
        result = native_prepare(a, *prep_args, **prep_kwargs)
        if int(os.environ.get('RANK', '0')) == 0:
            reference.atomic_json(out / 'arcquant_sparse_runtime.json', metadata)
            for name in sources:
                target = out / 'arcquant_sources' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
        return result

    request_quant.RequestQuantRuntime = SparseRuntime
    quant.install_reference_model_patches = install
    batch.install_ragged_attention = ragged
    reference.prepare = prepare
    succeeded = False
    try:
        result = int(reference.run(args) or 0)
        succeeded = result == 0
        return result
    finally:
        if runtimes:
            rank = int(os.environ.get('RANK', '0'))
            reference.atomic_json(out / f'arcquant_runtime_rank{rank}.json',
                                  dict(runtime_sha256=identity, run_succeeded=succeeded,
                                       rank=rank, **runtimes[-1].stats()))
        request_quant.RequestQuantRuntime = base_runtime
        quant.install_reference_model_patches = original_install
        batch.install_ragged_attention = original_ragged
        reference.prepare = original_prepare


if __name__ == '__main__':
    raise SystemExit(main())
