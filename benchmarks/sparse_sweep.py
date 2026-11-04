"""Build reproducible DeepSeek TEAL/WINA sparsity sweep commands.

This module only writes a JSON plan.  It does not reserve GPUs or start an
evaluator; use the emitted commands after the 1-question and 8-question gates
have passed.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILE = ROOT / "configs" / "sparsity_profiles.json"


def load_profiles(path: Path = PROFILE_FILE) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("profiles"), dict):
        raise ValueError(f"Invalid sparsity profile file: {path}")
    return data


def validate_rate(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("sparsity rates must be finite numbers in [0, 1)")
    return value


def rates_for(profile: str, method: str, *, path: Path = PROFILE_FILE) -> list[float]:
    data = load_profiles(path)
    try:
        values = data["profiles"][profile]["rates"][method]
    except KeyError as exc:
        raise ValueError(f"No sparsity rates for profile={profile!r}, method={method!r}") from exc
    rates = [validate_rate(value) for value in values]
    if not rates:
        raise ValueError(f"Empty sparsity profile for profile={profile!r}, method={method!r}")
    return rates


def build_deepseek_plan(args: argparse.Namespace) -> dict:
    if args.benchmark not in ("gpqa-diamond", "math500", "longbench-v2", "livecodebench"):
        raise ValueError("DeepSeek sparse sweep supports gpqa-diamond, math500, longbench-v2, and livecodebench")
    methods = tuple(dict.fromkeys(item.strip() for item in args.methods.split(",") if item.strip()))
    if not methods or any(method not in ("teal", "wina") for method in methods):
        raise ValueError("--methods must contain teal and/or wina")
    if not args.base_eval_root or not args.source_model or not args.converted_model:
        raise ValueError("--base-eval-root, --source-model, and --converted-model are required")
    if "teal" in methods and not args.calibration:
        raise ValueError("TEAL requires --calibration")
    if args.benchmark == "gpqa-diamond" and not args.data:
        raise ValueError("GPQA sparse plan requires --data")
    runs = []
    for method in methods:
        for sparsity in rates_for(args.profile, method):
            tag = f"{sparsity:g}"
            if args.benchmark == "gpqa-diamond":
                output = Path(args.out_root).expanduser().resolve() / args.benchmark / f"{method}-{tag}.json"
                command = [
                    "python", str(ROOT / "arc.py"), "experiment", "--stage", "evaluate",
                    "--gpu-ids", args.gpu_ids, "--base-eval-root",
                    str(Path(args.base_eval_root).expanduser().resolve()),
                    "--source-model", str(Path(args.source_model).expanduser().resolve()),
                    "--converted-model", str(Path(args.converted_model).expanduser().resolve()),
                    "--calibration", str(Path(args.calibration).expanduser().resolve()),
                    "--benchmark", args.benchmark, "--data",
                    str(Path(args.data).expanduser().resolve()), "--variant", method,
                    "--sparsity", tag, "--limit", str(args.limit), "--max-length", "4096",
                    "--output", str(output),
                ]
            else:
                runner = (ROOT / "benchmarks" /
                          ("run_deepseek_sparse_math.py" if args.benchmark == "math500"
                           else "run_deepseek_sparse_extended.py"))
                output = Path(args.out_root).expanduser().resolve() / args.benchmark / f"{method}-{tag}"
                command = [
                    "torchrun", "--standalone", "--nproc-per-node=2", str(runner),
                    "--base-eval-root", str(Path(args.base_eval_root).expanduser().resolve()),
                    "--sparse-method", method, "--sparsity", tag,
                    "--model", "deepseek", "--benchmark", args.benchmark,
                    "--source-model", str(Path(args.source_model).expanduser().resolve()),
                    "--converted-model", str(Path(args.converted_model).expanduser().resolve()),
                    "--quant", "hifp4", "--mode", "w4a4",
                    "--limit", str(args.limit), "--max-new-tokens", str(args.max_new_tokens),
                    "--out-dir", str(output),
                ]
                if method == "teal" and args.calibration:
                    command += ["--calibration", str(Path(args.calibration).expanduser().resolve())]
            runs.append({
                "method": method,
                "sparsity": sparsity,
                "sparsity_percent": int(round(sparsity * 100)),
                "keep_fraction": 1.0 - sparsity,
                "command": command,
                "shell_command": "CUDA_VISIBLE_DEVICES=" + args.gpu_ids + " " + shlex.join(command),
                "output_dir": str(output),
                "status": "planned",
            })
    return {
        "schema_version": 1,
        "definition": load_profiles()["definition"],
        "model": "deepseek-v4-flash",
        "benchmark": args.benchmark,
        "profile": args.profile,
        "methods": list(methods),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "runs": runs,
        "gate": "Run 1 question, then 8 questions, before changing limit to the full dataset.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--profile", default="baseline_sweep")
    p.add_argument("--methods", default="wina,teal", help="comma-separated methods")
    p.add_argument("--benchmark", required=True,
                   choices=("gpqa-diamond", "math500", "longbench-v2", "livecodebench"))
    p.add_argument("--base-eval-root", required=True)
    p.add_argument("--source-model", required=True)
    p.add_argument("--converted-model", required=True)
    p.add_argument("--calibration")
    p.add_argument("--data", help="GPQA CSV when --benchmark=gpqa-diamond")
    p.add_argument("--gpu-ids", default="0,1")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out-root", default=str(ROOT / "runs" / "deepseek_sparse_sweep"))
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(build_deepseek_plan(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
