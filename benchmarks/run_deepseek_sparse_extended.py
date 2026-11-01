"""Run the pinned DeepSeek LongBench/LiveCodeBench evaluator with WINA.

The external evaluator remains untouched.  This launcher injects a small
runtime subclass before the evaluator constructs its Transformer, so its
generation and scoring protocol remains the same while projection inputs are
masked by native WINA top-k.  TEAL is rejected here until a signed histogram
calibration pass is added for the generation scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _hash_source() -> str:
    return hashlib.sha256((ROOT / "model/deepseek_generation_runtime.py").read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sparse-method", choices=("teal", "wina"), required=True)
    p.add_argument("--sparsity", type=float, required=True)
    p.add_argument("--base-eval-root", type=Path, required=True)
    p.add_argument("--calibration", type=Path,
                   help="TEAL rank*.pt calibration directory (required for teal)")
    sparse, remaining = p.parse_known_args(argv)
    if not 0 <= sparse.sparsity < 1:
        p.error("--sparsity must be in [0, 1)")
    sys.path.insert(0, str(sparse.base_eval_root.expanduser().resolve()))
    import deepseek_v4_quant as quant
    import eval_deepseek_v4_extended as reference
    from model.deepseek_generation_runtime import (
        DeepSeekGenerationTealMixin, DeepSeekGenerationWinaMixin)

    base_runtime = quant.DeepSeekV4QuantRuntime

    mixin = DeepSeekGenerationTealMixin if sparse.sparse_method == "teal" else DeepSeekGenerationWinaMixin

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
                if sparse.calibration is None:
                    raise ValueError("--calibration is required for TEAL")
                import os
                import torch
                rank = os.environ.get("RANK", "0")
                calibration = torch.load(
                    str(sparse.calibration / f"rank{rank}.pt"),
                    map_location="cpu", weights_only=True)
                runtime.configure_sparse(ds, model, sparse.sparsity,
                                         calibration=calibration)
            else:
                runtime.configure_sparse(ds, model, sparse.sparsity)
            return model

        ds.Transformer = transformer

    quant.DeepSeekV4QuantRuntime = SparseRuntime
    quant.install_reference_model_patches = install
    # The external runner writes its runtime signature into every config. Add
    # the ARCQuant source hash and method parameters so a dense output cannot
    # accidentally be resumed as a sparse run.
    original_signature = reference.runtime_signature

    def signature():
        return {**original_signature(), "arcquant_sparse_runtime_sha256": _hash_source(),
                "arcquant_sparse_method": sparse.sparse_method,
                "arcquant_sparse_sparsity": sparse.sparsity}

    reference.runtime_signature = signature
    try:
        args = reference.parse_args(remaining)
        if args.benchmark not in ("longbench-v2", "livecodebench"):
            raise ValueError("This launcher supports LongBench-v2 and LiveCodeBench only")
        args.sparse_method = sparse.sparse_method
        args.sparsity = sparse.sparsity
        sparse_tag = f"arcquant-{sparse.sparse_method}-{sparse.sparsity:g}"
        args.tag = f"{sparse_tag}-{args.tag}" if args.tag else sparse_tag
        return int(reference.run(args) or 0)
    finally:
        quant.DeepSeekV4QuantRuntime = base_runtime
        quant.install_reference_model_patches = original_install
        reference.runtime_signature = original_signature


if __name__ == "__main__":
    raise SystemExit(main())
