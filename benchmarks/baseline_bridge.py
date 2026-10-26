"""Plan calls to the existing, pinned server-local baseline evaluators.

This backend keeps the benchmark protocol and model loaders in base_eval. Its
FP4 numbers are baseline W4A4 fake quantization, not ARCQuant's residual-channel
method. Activation sparsity is therefore intentionally unavailable here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from benchmarks.paths import REPOSITORY_ROOT, asset_path, asset_root, resolve_path, variant_path


class BaselineError(ValueError):
    pass


PRECISION = {"BF16": "bf16", "HiF4": "hifp4", "NVFP4": "nvfp4", "MXFP4": "mxfp4"}
SUPPORTED = {
    "deepseek-v4-flash": {"gpqa-diamond", "math500", "longbench-v2", "livecodebench"},
    "minimax-h3": {"vbench"},
    "magi-2-preview": {"vbench"},
}


def _model_path(args, entry: dict) -> Path:
    return asset_path(entry, args.model, "models", override=args.model_path)


def _backend_root(args, catalog: dict) -> Path:
    spec = catalog["backends"]["base-eval"]
    raw = args.base_eval_root or os.environ.get(spec["path_env"]) or spec.get("default_path")
    return resolve_path(raw) if raw else REPOSITORY_ROOT / "external" / "base_eval"


def _revision(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    completed = subprocess.run(["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"], cwd=root,
                               capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_baseline_plan(args, catalog: dict, repository_root: Path) -> dict:
    if args.method != "none":
        raise BaselineError("base-eval supports baseline quantization only; TEAL/WINA/xmask require ARCQuant adapters.")
    if args.quant_type not in PRECISION:
        raise BaselineError("base-eval supports BF16, HiF4, NVFP4 and MXFP4.")
    if args.model not in SUPPORTED or args.benchmark not in SUPPORTED[args.model]:
        raise BaselineError(f"No base-eval adapter for {args.model} with {args.benchmark}.")
    benchmark = catalog["benchmarks"][args.benchmark]
    if args.profile not in benchmark["profiles"]:
        raise BaselineError(f"Unknown profile {args.profile!r}; choose {', '.join(benchmark['profiles'])}.")
    model_path = _model_path(args, catalog["models"][args.model])
    backend_root = _backend_root(args, catalog)
    data_root = asset_path(benchmark, args.benchmark, "datasets")
    precision = PRECISION[args.quant_type]
    tag = "bf16" if precision == "bf16" else f"{precision}-w4a4"
    gpu_ids = args.gpu_ids or ("auto" if args.model == "minimax-h3" else "0,1")
    if args.model != "minimax-h3" and len(gpu_ids.split(",")) != 2:
        raise BaselineError(f"{args.model} baseline needs two GPU IDs, for example --gpu-ids 0,1.")
    if gpu_ids != "auto" and (not all(part.isdigit() for part in gpu_ids.split(",")) or
                               len(set(gpu_ids.split(","))) != len(gpu_ids.split(","))):
        raise BaselineError("--gpu-ids must be 'auto' or distinct comma-separated GPU numbers.")

    env = {"HF_HOME": os.environ.get("HF_HOME", str(asset_root() / "hf_home")),
           "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
           "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", "0")}
    for variable in ("H3_PYTHON", "VBENCH_PYTHON", "MAGI2_PYTHON", "CPU_PYTHON"):
        configured = os.environ.get(f"ARCQUANT_{variable}")
        if configured:
            env[variable] = str(resolve_path(configured))
    if os.environ.get("ARCQUANT_BASE_EVAL_ENV"):
        env["ARCQUANT_BASE_EVAL_ENV"] = str(resolve_path(os.environ["ARCQUANT_BASE_EVAL_ENV"]))
    results_root = repository_root / "runs" / "baseline_results"
    limit = benchmark["profiles"][args.profile]["limit"]
    if args.model == "deepseek-v4-flash":
        converted = variant_path(catalog["models"][args.model], "tp2",
                                 override=args.converted_model)
        env.update(QUANTS=precision, SOURCE_MODEL=str(model_path), TP_MODEL=str(converted),
                   RESULTS_ROOT=str(results_root))
        if args.benchmark in ("gpqa-diamond", "math500"):
            task = "gpqa" if args.benchmark == "gpqa-diamond" else "math500"
            command = ["bash", str(backend_root / "scripts/run_all_generative.sh"),
                       "deepseek", gpu_ids, task, str(limit)]
            output_dir = results_root / model_path.name / f"{task}_generative" / precision
        else:
            task = "longbench-v2" if args.benchmark == "longbench-v2" else "livecodebench"
            output_dir = results_root / model_path.name / task / tag
            command = ["bash", str(backend_root / "scripts/run_all_deepseek_v4_extended.sh"),
                       task, gpu_ids, "--limit", str(limit), "--out-dir", str(output_dir)]
    else:
        if args.benchmark != "vbench":
            raise BaselineError("Video models currently use the VBench evaluator only.")
        # A transferred VBench checkout can belong to another server user.
        # Trust only that checkout for the launched evaluator process.
        git_slot = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
        env.update({"GIT_CONFIG_COUNT": str(git_slot + 1),
                    f"GIT_CONFIG_KEY_{git_slot}": "safe.directory",
                    f"GIT_CONFIG_VALUE_{git_slot}": str(data_root)})
        env.update(MODEL=str(model_path), TAGS=tag,
                   VBENCH_REPO=str(data_root),
                   VBENCH_CACHE_DIR=os.environ.get("VBENCH_CACHE_DIR", str(asset_root() / "vbench_cache")))
        sampling = "--tiny" if args.profile == "tiny" else "--official"
        if args.model == "minimax-h3":
            command = ["bash", str(backend_root / "scripts/run_all_vbench_dynamic.sh"),
                       "all", gpu_ids, sampling]
        else:
            env["MAGI2_CKPT_ROOT"] = str(model_path)
            env["MAGI2_REPO"] = os.environ.get("MAGI2_REPO", str(REPOSITORY_ROOT / "external" / "MAGI-2-preview"))
            command = ["bash", str(backend_root / "scripts/run_all_magi2_vbench.sh"),
                       "all", gpu_ids, sampling]
        output_dir = backend_root / "results" / model_path.name / "vbench_standard" / tag
        if args.profile == "tiny":
            output_dir /= "tiny"

    return {"backend": "base-eval", "backend_revision": _revision(backend_root),
            "backend_expected_revision": catalog["backends"]["base-eval"]["revision"],
            "protocol": "external BF16/W4A4 baseline; no ARCQuant residual channels or activation sparsity",
            "model": args.model, "model_path": str(model_path), "benchmark": args.benchmark,
            "dataset_path": str(data_root), "profile": args.profile, "method": args.method,
            "quant_type": args.quant_type, "gpu_ids": gpu_ids,
            "command": command, "cwd": str(backend_root), "env": env,
            "result_path": str(output_dir)}
