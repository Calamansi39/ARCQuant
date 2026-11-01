"""Measure dense, masked-dense and actual sparse GEMM on one projection.

This is a kernel diagnostic, not an end-to-end model speed claim.  The
masked-dense path is what ARCQuant currently executes in its PyTorch HiF4
accuracy runtime.  The CSR path performs a real sparse matrix multiplication
when the installed PyTorch backend supports it, making the difference between
"many zeros" and "less work" explicit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, *, warmup: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iterations


def benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {device}")
    if args.hidden < 1 or args.output < 1 or args.tokens < 1 or args.batch < 1:
        raise ValueError("batch, tokens, hidden and output must be positive")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    x = torch.randn(args.batch * args.tokens, args.hidden, device=device, dtype=dtype)
    weight = torch.randn(args.output, args.hidden, device=device, dtype=dtype)
    dense_ms = 1e3 * _time(lambda: F.linear(x, weight), warmup=args.warmup,
                            iterations=args.iterations, device=device)
    rows = []
    for sparsity in args.sparsities:
        if not 0 <= sparsity < 1:
            raise ValueError("sparsities must be in [0, 1)")
        keep = max(1, int(args.hidden * (1 - sparsity)))
        indices = x.abs().topk(keep, dim=-1).indices
        mask = torch.zeros_like(x, dtype=torch.bool).scatter_(1, indices, True)
        masked = x * mask

        def masked_dense():
            return F.linear(masked, weight)

        masked_ms = 1e3 * _time(masked_dense, warmup=args.warmup,
                                 iterations=args.iterations, device=device)
        sparse_ms = None
        sparse_error = None
        try:
            sparse_x = masked.to_sparse_csr()
            def csr_mm():
                return torch.sparse.mm(sparse_x, weight.transpose(0, 1))
            sparse_ms = 1e3 * _time(csr_mm, warmup=args.warmup,
                                    iterations=args.iterations, device=device)
            sparse_error = float((csr_mm() - masked_dense()).abs().max().item())
        except (RuntimeError, NotImplementedError) as exc:
            sparse_error = str(exc).splitlines()[0]
        rows.append({
            "target_sparsity": sparsity,
            "actual_zero_fraction": float((~mask).float().mean().item()),
            "kept_channels": keep,
            "masked_dense_ms": masked_ms,
            "masked_dense_speedup_vs_dense": dense_ms / masked_ms if masked_ms else None,
            "torch_csr_ms": sparse_ms,
            "torch_csr_speedup_vs_dense": dense_ms / sparse_ms if sparse_ms else None,
            "torch_csr_max_abs_error": sparse_error,
        })
    return {
        "schema_version": 1,
        "benchmark": "single_projection_sparse_latency",
        "device": str(device),
        "dtype": args.dtype,
        "shape": {"batch": args.batch, "tokens": args.tokens,
                  "hidden": args.hidden, "output": args.output},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "dense_ms": dense_ms,
        "rows": rows,
        "interpretation": "masked_dense is zeroing overhead; torch_csr is an actual sparse backend check, not a TEAL/WINA kernel result",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--output", type=int, default=4096)
    p.add_argument("--sparsities", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        p.error("--warmup must be non-negative and --iterations must be positive")
    result = benchmark(args)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
