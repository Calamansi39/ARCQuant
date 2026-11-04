"""Run the native DeepSeek MATH-500 evaluator with ARCQuant TEAL/WINA.

The evaluator and Math-Verify scorer remain in the pinned ``base_eval`` tree.
This launcher injects the same request runtime used by the evaluator and adds
the sparse method at the projection boundary.  Partial smoke runs must use a
``validation/`` output directory and are never treated as formal scores.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sparse-method", choices=("teal", "wina"), required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--base-eval-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        help="TEAL rank*.pt calibration directory (required for teal)")
    sparse, remaining = parser.parse_known_args(argv)
    if not 0 <= sparse.sparsity < 1:
        parser.error("--sparsity must be in [0, 1)")
    if sparse.sparse_method == "teal" and sparse.calibration is None:
        parser.error("--calibration is required for TEAL")
    sys.path.insert(0, str(sparse.base_eval_root.expanduser().resolve()))

    import deepseek_v4_quant as quant
    import deepseek_v4_lcb_batch_quant as request_quant
    import run_generative_eval as reference
    from model.deepseek_generation_runtime import (
        DeepSeekGenerationTealMixin, DeepSeekGenerationWinaMixin)

    base_runtime = request_quant.RequestQuantRuntime
    mixin = (DeepSeekGenerationTealMixin if sparse.sparse_method == "teal"
             else DeepSeekGenerationWinaMixin)

    class SparseRuntime(mixin, base_runtime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sparse_sparsity = float(sparse.sparsity)

    original_install = quant.install_reference_model_patches

    def install(ds, runtime):
        original_install(ds, runtime)
        original_transformer = ds.Transformer

        def transformer(*args, **kwargs):
            model = original_transformer(*args, **kwargs)
            if sparse.sparse_method == "teal":
                rank = os.environ.get("RANK", "0")
                import torch
                calibration = torch.load(
                    str(sparse.calibration / f"rank{rank}.pt"),
                    map_location="cpu", weights_only=True)
                runtime.configure_sparse(ds, model, sparse.sparsity,
                                         calibration=calibration)
            else:
                runtime.configure_sparse(ds, model, sparse.sparsity)
            return model

        ds.Transformer = transformer

    request_quant.RequestQuantRuntime = SparseRuntime
    original_prepare = reference.prepare

    def prepare(args, *prep_args, **prep_kwargs):
        result = original_prepare(args, *prep_args, **prep_kwargs)
        if int(os.environ.get("RANK", "0")) == 0:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "arcquant_sparse_runtime.json").write_text(
                __import__("json").dumps({
                    "method": sparse.sparse_method,
                    "sparsity": sparse.sparsity,
                    "runtime_sha256": hashlib.sha256(
                        (ROOT / "model/deepseek_generation_runtime.py").read_bytes()
                    ).hexdigest(),
                    "calibration": str(sparse.calibration) if sparse.calibration else None,
                }, indent=2) + "\n")
        return result

    quant.install_reference_model_patches = install
    reference.prepare = prepare
    try:
        args = reference.parse_args(["--model", "deepseek", "--benchmark", "math500", *remaining])
        runtime_hash = hashlib.sha256((ROOT / "model/deepseek_generation_runtime.py").read_bytes()).hexdigest()
        tag = f"arcquant-{sparse.sparse_method}-{sparse.sparsity:g}-{runtime_hash[:12]}"
        args.out_dir = str(Path(args.out_dir) / tag)
        return int(reference.run(args) or 0)
    finally:
        request_quant.RequestQuantRuntime = base_runtime
        quant.install_reference_model_patches = original_install
        reference.prepare = original_prepare


if __name__ == "__main__":
    raise SystemExit(main())
