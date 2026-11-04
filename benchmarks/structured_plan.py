"""Portable command builder for strict 2:4 experiments (no GPU imports)."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--benchmark", required=True,
                   choices=("math500", "gpqa-diamond", "longbench-v2", "livecodebench"))
    p.add_argument("--base-eval-root", required=True)
    p.add_argument("--source-model", required=True)
    p.add_argument("--converted-model", required=True)
    p.add_argument("--layer-coverage", type=float, default=1.)
    p.add_argument("--scope", choices=("all", "attention", "ffn"), default="all")
    p.add_argument("--selection", type=Path)
    p.add_argument("--gpu-ids", default="0,1")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="default: GPQA 8192 (native thinking budget), other tasks 128 for diagnostics")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--out-dir", default="validation/deepseek_structured24")
    p.add_argument("--dry-run", action="store_true", help="native evaluator prerequisite check without loading GPU weights")
    p.add_argument("--wait-for-gpus", action="store_true", help="auto-select two idle GPUs at launch, ignoring fixed --gpu-ids")
    p.add_argument("--min-free-mib", type=int, default=138000)
    p.add_argument("--poll-seconds", type=int, default=60)
    return p


def build_plan(args):
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 8192 if args.benchmark == "gpqa-diamond" else 128
    if not math.isfinite(args.layer_coverage) or not 0 <= args.layer_coverage <= 1:
        raise ValueError("--layer-coverage must be finite and in [0, 1]")
    ids = args.gpu_ids.split(",")
    if len(ids) != 2 or len(set(ids)) != 2 or any(not s.isdecimal() for s in ids):
        raise ValueError("Two distinct numeric GPU IDs are required")
    if args.limit != -1 and args.limit < 1:
        raise ValueError("--limit must be -1 or positive")
    if max_new_tokens < 1 or not 1 <= args.batch_size <= 20 or args.smoke_steps < 0:
        raise ValueError("Invalid generation budget/batch size")
    if args.benchmark == "gpqa-diamond" and max_new_tokens < 1026:
        raise ValueError("Native GPQA needs >=1026 tokens for thinking + close + 1024 final reserve; use --smoke-steps for short diagnostics")
    if getattr(args, "min_free_mib", 138000) < 1 or getattr(args, "poll_seconds", 60) < 1:
        raise ValueError("GPU wait thresholds must be positive")
    generative = args.benchmark in ("math500", "gpqa-diamond")
    if not generative and (args.batch_size != 1 or args.smoke_steps):
        raise ValueError("Extended evaluator supports batch-size=1 and no --smoke-steps here")
    output = Path(args.out_dir).expanduser().resolve()
    if args.smoke_steps and "validation" not in output.parts:
        raise ValueError("Bounded smoke output must be under a validation directory")
    env_path = os.environ.get("ARCQUANT_BASE_EVAL_ENV")
    python = str(Path(env_path) / "bin/python") if env_path else sys.executable
    command = [python]
    if not args.dry_run:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
    command += [str(ROOT / "benchmarks/run_deepseek_structured.py"),
                "--benchmark", args.benchmark, "--layer-coverage", str(args.layer_coverage),
                "--scope", args.scope, "--quant", "hifp4", "--mode", "w4a4",
                "--limit", str(args.limit), "--max-new-tokens", str(max_new_tokens),
                "--out-dir", str(output)]
    for key in ("base_eval_root", "source_model", "converted_model"):
        command += ["--" + key.replace("_", "-"), str(Path(getattr(args, key)).expanduser().resolve())]
    if args.selection:
        command += ["--selection", str(args.selection.expanduser().resolve())]
    if generative:
        command += ["--batch-size", str(args.batch_size), "--smoke-steps", str(args.smoke_steps)]
    if args.dry_run:
        command.append("--dry-run")
    env = {"CUDA_VISIBLE_DEVICES": args.gpu_ids,
           "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
           "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", "1")}
    return {"model": "deepseek-v4-flash", "benchmark": args.benchmark,
            "method": "magnitude-structured24", "quantization": "HiF4 W4A4",
            "layer_coverage": args.layer_coverage, "element_mask_fraction": 0.5,
            "scope": args.scope, "share_index": False, "status": "planned",
            "gpu_selection": "auto-idle-pair" if getattr(args, "wait_for_gpus", False) and not args.dry_run else "explicit",
            "coverage_note": "Counts named linear projections, not blocks; no quality acceptance claim.",
            "command": command, "env": env,
            "shell_command": " ".join(k + "=" + shlex.quote(v) for k, v in env.items()) + " " + shlex.join(command),
            "wait_note": "shell_command invokes the evaluator directly; use structured-run --wait-for-gpus for scheduling"}
