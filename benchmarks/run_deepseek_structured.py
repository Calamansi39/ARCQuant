"""Run native DeepSeek evaluation with strict magnitude 2:4 + HiF4 W4A4.

All rows are masked in selected projections. No ARC residual or TEAL/WINA
transform is added. Output fingerprints keep this experiment separate from
other methods and from runs using different projection selections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configuration(args):
    import math
    if not math.isfinite(args.layer_coverage) or not 0 <= args.layer_coverage <= 1:
        raise ValueError("--layer-coverage must be finite and in [0, 1]")
    if args.mask_chunk_rows < 1:
        raise ValueError("--mask-chunk-rows must be positive")
    selected = None
    if args.selection:
        selected = json.loads(args.selection.read_text())
        if not isinstance(selected, list) or any(not isinstance(k, str) for k in selected):
            raise ValueError("--selection must contain a JSON list of projection names")
        if len(selected) != len(set(selected)):
            raise ValueError("--selection contains duplicate names")
    return {
        "schema_version": 1, "method": "magnitude-structured24", "quant": "hifp4",
        "mode": "w4a4", "pattern": "2:4", "share_index": False,
        "layer_coverage": args.layer_coverage, "scope": args.scope,
        "selection": sorted(selected) if selected is not None else None,
        "selection_policy": "explicit" if selected is not None else "sha256-name-order per area (not sensitivity optimized)",
        "coverage_policy": "meet requested fraction separately in each in-scope area",
        "mask_chunk_rows": args.mask_chunk_rows,
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__), ROOT / "model/structured_sparsity.py",
                     ROOT / "model/deepseek_structured_runtime.py")},
    }


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True,
                   choices=("math500", "gpqa-diamond", "longbench-v2", "livecodebench"))
    p.add_argument("--base-eval-root", required=True, type=Path)
    p.add_argument("--layer-coverage", type=float, default=1.)
    p.add_argument("--scope", choices=("all", "attention", "ffn"), default="all")
    p.add_argument("--selection", type=Path)
    p.add_argument("--mask-chunk-rows", type=int, default=4096)
    p.add_argument("--smoke-steps", type=int, default=0)
    own, remaining = p.parse_known_args(argv)
    config = configuration(own)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    sys.path.insert(0, str(own.base_eval_root.expanduser().resolve()))
    import deepseek_v4_quant as quant
    from model.deepseek_structured_runtime import DeepSeekStructured24Mixin
    generative = own.benchmark in ("math500", "gpqa-diamond")
    if own.smoke_steps < 0 or (own.smoke_steps and not generative):
        raise ValueError("Bounded smoke steps require the MATH/GPQA generation evaluator")
    if generative:
        import deepseek_v4_lcb_batch_quant as runtime_module
        import run_generative_eval as reference
        attribute = "RequestQuantRuntime"
        native_benchmark = "gpqa" if own.benchmark == "gpqa-diamond" else own.benchmark
        args = reference.parse_args(["--model", "deepseek", "--benchmark", native_benchmark, *remaining])
        if args.model != "deepseek" or args.benchmark != native_benchmark:
            raise ValueError("Structured evaluator model/benchmark cannot be overridden")
    else:
        runtime_module = quant
        import eval_deepseek_v4_extended as reference
        attribute = "DeepSeekV4QuantRuntime"
        args = reference.parse_args(["--benchmark", own.benchmark, *remaining])
    if args.quant != "hifp4" or args.mode != "w4a4":
        raise ValueError("Specify --quant hifp4 --mode w4a4 for this evaluator")
    out = Path(args.out_dir) / f"structured24-{fingerprint}"
    if own.smoke_steps:
        # Native guard permits only base_eval/validation. Also permit this
        # repository's validation tree, retaining the unscored smoke protocol.
        allowed = (ROOT / "validation", own.base_eval_root.resolve() / "validation")
        if not any(out.resolve().is_relative_to(path) for path in allowed):
            raise ValueError("Bounded smoke must write under ARCQuant/validation or base_eval/validation")
        args.smoke_steps = own.smoke_steps
    args.out_dir = str(out)
    if not generative:
        args.tag = f"structured24-{fingerprint}" + (f"-{args.tag}" if args.tag else "")
    instances = []
    base = getattr(runtime_module, attribute)

    class Runtime(DeepSeekStructured24Mixin, base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            instances.append(self)

    original_install = quant.install_reference_model_patches

    def install(ds, runtime):
        original_install(ds, runtime)
        original_transformer = ds.Transformer

        def transformer(*a, **kw):
            model = original_transformer(*a, **kw)
            # This only reads logical layout; weights are still uninitialized.
            # Selection never depends on uninitialized or stale column norms.
            runtime.configure_structured(ds, model, coverage=own.layer_coverage,
                scope=own.scope, selected_names=config["selection"], chunk_rows=own.mask_chunk_rows)
            return model

        ds.Transformer = transformer

    manifest = out / "structured_config.json"
    if int(os.environ.get("RANK", "0")) == 0:
        if manifest.exists():
            if json.loads(manifest.read_text()) != config:
                raise ValueError("Output configuration mismatch; choose a new output directory")
        elif out.exists() and any(out.iterdir()):
            raise ValueError("Refusing to resume a directory without structured provenance")
        atomic_json(manifest, config)
    original_signature = None
    if not generative:
        original_signature = reference.runtime_signature
        reference.runtime_signature = lambda: {**original_signature(), "arcquant_structured": config}
    setattr(runtime_module, attribute, Runtime)
    quant.install_reference_model_patches = install
    succeeded = False
    try:
        result = reference.run(args)
        succeeded = True
        return int(result or 0)
    finally:
        setattr(runtime_module, attribute, base)
        quant.install_reference_model_patches = original_install
        if original_signature is not None:
            reference.runtime_signature = original_signature
        for runtime in instances:
            if hasattr(runtime, "structured"):
                try:
                    atomic_json(out / f"structured_rank{os.environ.get('RANK', '0')}.json", {
                        "evaluation_returned_successfully": succeeded,
                        "rank": int(os.environ.get("RANK", "0")),
                        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                        "configuration": config, "runtime": runtime.stats(),
                        "note": "Configured coverage is not observed coverage or quality acceptance.",
                    })
                finally:
                    runtime.remove_structured_hooks()


if __name__ == "__main__":
    raise SystemExit(main())
