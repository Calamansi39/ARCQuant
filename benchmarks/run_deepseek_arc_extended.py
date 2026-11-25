"""Dense HiF4 W4A4 + ARC with the pinned LongBench / batched LCB protocols.

The external evaluator and calibration are required assets. Continuation uses
only this ARC run's 64K answers and freezes both external and ARC source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'run.json').is_file():
    ROOT = ROOT / 'runtime' / 'arcquant_extension'
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def release_previous():
    previous = os.environ.get('ARC_PREVIOUS_HOLD')
    if not previous:
        return
    folder = Path(previous)
    (folder/'release').touch()
    for path in folder.glob('rank*.json'):
        pid = json.loads(path.read_text())['pid']
        deadline = time.monotonic()+120
        while process_alive(pid):
            if time.monotonic() > deadline:
                raise RuntimeError('Previous ARC worker did not release its CUDA context')
            time.sleep(.1)


def hold_for_successor():
    """Keep this worker's reusable cache alive during CPU scoring/preparation."""
    hold = os.environ.get('ARC_HOLD_DIR')
    if not hold:
        return
    from lcb_protocol import atomic_json
    folder = Path(hold)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/f'rank{os.environ.get("RANK", "0")}.json', dict(pid=os.getpid(), completed_at=time.time()))
    owner = int(os.environ['ARC_QUEUE_OWNER_PID'])
    while not (folder/'release').exists() and process_alive(owner):
        time.sleep(.2)


def sources():
    from benchmarks.run_deepseek_sparse_math import SOURCES
    return dict.fromkeys((*SOURCES, 'model/__init__.py', 'model/quantize.py',
        'model/kv_cache.py', 'model/optional_agemm.py', 'model/bridge.py', 'model/deepseek_arc_long.py',
        'benchmarks/run_deepseek_arc_extended.py'))


def metadata(calibration, source, converted):
    from benchmarks.run_deepseek_sparse_math import calibration_metadata
    value = dict(schema='arcquant-dense-extended-v1', arc=True, method='none', sparsity=0.,
        quantization='HiF4 W4A4 + ARC', arc_chunk_rows=4096,
        kernel='HiF4-value-reconstruction+dense-GEMM',
        calibration=calibration_metadata(calibration, source, converted, 2),
        implementation_sha256={name: sha(ROOT/name) for name in sources()})
    value['runtime_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


class Injection:
    def __init__(self, calibration, identity, output, guard_root=None):
        self.calibration, self.identity, self.output = Path(calibration), identity, Path(output)
        self.runtimes, self.patches, self.guards = [], [], []
        self.guard_root = guard_root

    def patch(self, module, name, replacement):
        self.patches.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    def runtime_type(self, base):
        from model.deepseek_arc_long import LongArcMixin
        from model.deepseek_generation_runtime import DeepSeekGenerationWinaMixin
        return type('DenseLongArcRuntime', (LongArcMixin, DeepSeekGenerationWinaMixin, base), {})

    def install(self, original):
        def wrapped(ds, runtime):
            from model.deepseek_generation_arc import install_arc_grouped_attention
            original(ds, runtime)
            install_arc_grouped_attention(ds, runtime, 'standard')
            constructor = ds.Transformer

            def build(*args, **kwargs):
                import torch
                model = constructor(*args, **kwargs)
                rank = os.environ.get('RANK', '0')
                values = torch.load(self.calibration/f'rank{rank}.pt', map_location='cpu', weights_only=True)
                runtime.configure_sparse(ds, model, 0.)
                runtime.configure_arc(ds, model, values, 'none')
                self.runtimes.append(runtime)
                if self.guard_root:
                    from scripts.magi_memory_guard import CacheGuard
                    class DenseGuard(CacheGuard):
                        def after_forward(self, module, inputs, output):
                            if module is model:
                                self.armed = True
                                self.refill('successful DS forward')
                    guard = DenseGuard(torch, headroom_mib=4096, status_dir=self.guard_root).install()
                    self.guards.append(guard)
                    # Native schedulers already broadcast their worklist before
                    # construction, initializing CUDA/NCCL. Keep reusable cache
                    # while safetensors loads; four GiB stay free for libraries.
                    guard.armed = True
                    guard.refill('model allocated; checkpoint loading', force=True)
                return model
            ds.Transformer = build
        return wrapped

    def ragged(self, original):
        def wrapped(ds, runtime):
            from model.deepseek_generation_arc import install_arc_grouped_attention
            original(ds, runtime)
            install_arc_grouped_attention(ds, runtime, 'ragged')
        return wrapped

    def finish(self, succeeded):
        from lcb_protocol import atomic_json
        if self.runtimes:
            rank = int(os.environ.get('RANK', '0'))
            stats = self.runtimes[-1].stats()
            atomic_json(self.output/f'arcquant_runtime_rank{rank}.json',
                dict(runtime_sha256=self.identity['runtime_sha256'], run_succeeded=succeeded, rank=rank, **stats))
        if succeeded:
            hold_for_successor()
        for guard in self.guards:
            guard.close()
        for module, name, old in reversed(self.patches):
            setattr(module, name, old)


def extended(options, remaining):
    import deepseek_v4_quant as quant
    import eval_deepseek_v4_extended as reference
    identity = metadata(options.calibration, options.source_model, options.converted_model)
    quant_args = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    quant_args.add_argument('--quant', default='bf16')
    quant_args.add_argument('--mode', default='w4a4')
    selected, _ = quant_args.parse_known_args(remaining)
    if selected.quant != 'hifp4' or selected.mode != 'w4a4':
        raise ValueError('Require --quant hifp4 --mode w4a4')
    injection = Injection(options.calibration, identity, options.out_dir, options.memory_guard_dir)
    original_signature = reference.runtime_signature
    injection.patch(reference, 'runtime_signature', lambda: {**original_signature(), 'arcquant': identity})
    prepare = reference.prepare_output

    def prepare_bound(a, items):
        path = prepare(a, items)  # Original compares ARC identity before any resume.
        reference.atomic_json(Path(a.out_dir)/'arcquant_sparse_runtime.json', identity)
        for name in sources():
            target = Path(a.out_dir)/'arcquant_sources'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        return path
    injection.patch(reference, 'prepare_output', prepare_bound)
    cli = ['--benchmark', options.stage, '--source-model', str(options.source_model),
        '--converted-model', str(options.converted_model), '--out-dir', str(options.out_dir), *remaining]
    succeeded = False
    try:
        if options.stage == 'longbench-v2':
            import run_deepseek_longbench as entry
            injection.patch(entry, 'LongPrefillRuntime', injection.runtime_type(entry.LongPrefillRuntime))
            injection.patch(quant, 'install_reference_model_patches', injection.install(quant.install_reference_model_patches))
        else:
            import run_deepseek_lcb_batch_all as entry
            injection.patch(entry, 'RequestQuantRuntime', injection.runtime_type(entry.RequestQuantRuntime))
            scheduler = entry.scheduler
            injection.patch(scheduler, 'install_reference_model_patches', injection.install(scheduler.install_reference_model_patches))
            injection.patch(scheduler, 'install_ragged_attention', injection.ragged(scheduler.install_ragged_attention))
        if not any(flag in remaining for flag in ('--dry-run', '--evaluate-only')):
            release_previous()
        entry.main(cli)
        succeeded = True
    finally:
        injection.finish(succeeded)


def continuation(options):
    run_dir = options.run_dir.resolve()
    manifest = run_dir/'run.json'
    if manifest.exists():
        # Verify native snapshot before executing any frozen implementation.
        sys.path.insert(0, str(options.base_eval_root/'continue_lcb_code'))
        import deepseek_workflow as bootstrapper
        config = bootstrapper.bootstrap(run_dir)
        frozen = Path(config['assets']['code_dir'])/Path(__file__).name
        if Path(__file__).resolve() != frozen.resolve():
            os.execv(sys.executable, [sys.executable, str(frozen), *sys.argv[1:]])
        sys.modules.pop('deepseek_workflow', None)
        sys.path.insert(0, config['assets']['code_dir'])
        sys.path.insert(0, config['assets']['runtime_dir'])
    else:
        sys.path.insert(0, str(options.base_eval_root/'continue_lcb_code'))
    import deepseek_workflow as workflow
    if not manifest.exists():
        workflow.PROJECT = options.base_eval_root
    from lcb_queue import SharedQueue
    if options.stage == 'continue-prepare':
        source_cfg = json.loads((options.source_root/'hifp4-w4a4.jsonl.config.json').read_text())
    else:
        config = workflow.bootstrap(run_dir)
        source_cfg = json.loads((Path(config['assets']['source_dir'])/'hifp4-w4a4.jsonl.config.json').read_text())
    expected = metadata(options.calibration, source_cfg['source_model'], source_cfg['converted_model'])
    if source_cfg['runtime_signature'].get('arcquant') != expected:
        raise ValueError('Continuation requires identical ARC source, calibration and implementation')
    if options.stage == 'continue-prepare':
        original_freeze = workflow.freeze
        def freeze(*args):
            stage, hashes = original_freeze(*args)
            for name in sources():
                relative = 'arcquant_extension/'+name
                dest = stage/'runtime'/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((ROOT/name).read_bytes())
                hashes['runtime'][relative] = sha(dest)
            for name in ('scripts/__init__.py', 'scripts/magi_memory_guard.py'):
                source = options.base_eval_root/name
                if source.exists():
                    dest = stage/'runtime'/name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(source.read_bytes())
                    hashes['runtime'][name] = sha(dest)
            dest = stage/'code'/Path(__file__).name
            dest.write_bytes(Path(__file__).read_bytes())
            hashes['code'][dest.name] = sha(dest)
            return stage, hashes
        workflow.freeze = freeze
        from types import SimpleNamespace
        workflow.prepare(SimpleNamespace(run_dir=run_dir, source_root=options.source_root,
            quant='hifp4', batch_size=8, cache_gib=48, prefill_cache_gib=8))
        return
    queue = SharedQueue(run_dir)
    workflow.verify(queue)
    if options.stage == 'continue-collect':
        workflow.collect(queue)
        return
    import deepseek_replay as replay
    import deepseek_v4_quant as quant
    import deepseek_v4_lcb_batch as batch
    import deepseek_worker as worker
    injection = Injection(options.calibration, expected, run_dir, options.memory_guard_dir)
    injection.patch(replay, 'ContinuationRuntime', injection.runtime_type(replay.ContinuationRuntime))
    injection.patch(quant, 'install_reference_model_patches', injection.install(quant.install_reference_model_patches))
    injection.patch(batch, 'install_ragged_attention', injection.ragged(batch.install_ragged_attention))
    original_clear = replay.clear_weight_cache
    def clear(runtime):
        original_clear(runtime)
        runtime.arc_cache.clear()
        runtime.arc_cache_bytes = 0
    injection.patch(replay, 'clear_weight_cache', clear)
    succeeded = False
    try:
        release_previous()
        sys.argv = [worker.__file__, '--run-dir', str(run_dir)]
        worker.main()
        succeeded = True
    finally:
        injection.finish(succeeded)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--stage', required=True, choices=('longbench-v2', 'livecodebench',
        'continue-prepare', 'continue-worker', 'continue-collect'))
    parser.add_argument('--base-eval-root', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--source-model', type=Path)
    parser.add_argument('--converted-model', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--memory-guard-dir', type=Path)
    options, remaining = parser.parse_known_args(argv)
    options.base_eval_root = options.base_eval_root.resolve()
    os.environ['LCB_PROJECT_ROOT'] = str(options.base_eval_root)
    sys.path.insert(0, str(options.base_eval_root))
    if options.stage.startswith('continue-'):
        if remaining or options.run_dir is None:
            parser.error('Continuation requires --run-dir and no unknown options')
        continuation(options)
    else:
        if not all((options.source_model, options.converted_model, options.out_dir)):
            parser.error('Require --source-model, --converted-model, --out-dir')
        extended(options, remaining)


if __name__ == '__main__':
    main()
