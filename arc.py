"""Plan and run ARCQuant experiments from the repository root.

This control script intentionally uses only Python's standard library. The
legacy model code and its GPU dependencies are imported only by ``run``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmarks.asset_inventory import audit_benchmark, audit_catalog, audit_model
from benchmarks.baseline_bridge import BaselineError, build_baseline_plan


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "configs" / "experiments.json"


class PlanError(ValueError):
    pass


def check_runtime_compatibility() -> None:
    """Avoid loading large weights with the wrong Transformers model API."""
    try:
        installed = version("transformers")
    except PackageNotFoundError:
        installed = "not installed"
    required = "4.46.3"  # requirements.txt
    if installed != required:
        raise PlanError(
            f"ARCQuant requires transformers=={required}; this Python has {installed}. "
            "Install requirements.txt in a compatible environment before prepare/run."
        )


def read_catalog(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model(args: argparse.Namespace, catalog: dict) -> tuple[dict, Path]:
    model = catalog["models"].get(args.model)
    if model is None:
        raise PlanError(f"Unknown model {args.model!r}; use 'python arc.py list'.")
    raw_path = args.model_path or os.environ.get(model.get("path_env", "")) or model.get("default_path")
    if not raw_path:
        raise PlanError(f"Set {model['path_env']} or pass --model-path for {args.model}.")
    return model, Path(raw_path).expanduser().resolve()


def cache_paths(model_path: Path, dataset: str, metric: str) -> list[Path]:
    prefix = f"{model_path.name.lower()}_"
    return [ROOT / "saved" / f"{prefix}{suffix}_{dataset}_{metric}.pt" for suffix in
            ("reorder_index", "select_num", "act_scales")]


def missing_teal_histograms(root: Path, model_path: Path) -> list[Path]:
    """Find required per-layer files before loading large model weights."""
    config_path = model_path / "config.json"
    layer_count = 1
    if config_path.is_file():
        layer_count = int(json.loads(config_path.read_text(encoding="utf-8")).get("num_hidden_layers", 1))
    missing = []
    for index in range(layer_count):
        for part in ("self_attn", "mlp"):
            path = root / f"layer-{index}" / part / "histograms.pt"
            if not path.is_file():
                missing.append(path)
    return missing


def build_plan(args: argparse.Namespace, catalog: dict | None = None) -> dict:
    catalog = catalog or read_catalog()
    if getattr(args, "backend", "arcquant") == "base-eval":
        return build_baseline_plan(args, catalog, ROOT)
    if args.quant_type == "BF16":
        raise PlanError("BF16 is available through --backend base-eval; choose an ARCQuant quantization type here.")
    model, model_path = resolve_model(args, catalog)
    benchmark = catalog["benchmarks"].get(args.benchmark)
    if benchmark is None:
        raise PlanError(f"Unknown benchmark {args.benchmark!r}; use 'python arc.py list'.")
    if args.method not in catalog["methods"]:
        raise PlanError(f"Unknown method {args.method!r}; use 'python arc.py list'.")
    if model["status"] != "legacy-ready":
        raise PlanError(f"{args.model} is registered at {model_path}, but has no ARCQuant model adapter yet.")
    if benchmark["status"] not in ("legacy-ready", "lm-eval-ready"):
        raise PlanError(f"{args.benchmark} is registered, but has no evaluator adapter yet.")
    if args.method in ("teal", "wina") and model["family"] != "llama":
        raise PlanError("The integrated TEAL and WINA paths currently support Llama only.")
    if args.method in ("teal", "wina"):
        if not 0 <= args.sparsity < 1:
            raise PlanError("--sparsity must be in [0, 1).")
    if args.method == "teal" and not args.histogram_path:
        raise PlanError("TEAL requires --histogram-path pointing to its calibration histograms.")
    if args.method in ("none", "xmask") and args.sparsity != 0:
        raise PlanError("Choose --method teal or wina before setting --sparsity.")
    if args.method == "xmask":
        if model["family"] != "llama":
            raise PlanError("The unified xmask CLI currently supports Llama only.")
        if not args.x_mask_ckpt:
            raise PlanError("xmask requires --x-mask-ckpt from the calibration script.")
    if args.profile not in benchmark["profiles"]:
        choices = ", ".join(benchmark["profiles"])
        raise PlanError(f"Unknown profile {args.profile!r} for {args.benchmark}; choose {choices}.")

    command = [
        sys.executable, str(ROOT / "model" / "main.py"), str(model_path),
        "--model_family", model["family"],
        "--quant_type", args.quant_type,
        "--sparse_method", "none" if args.method == "xmask" else args.method,
        "--dataset", args.calibration_dataset,
        "--act_sort_metric", args.act_sort_metric,
        "--seed", str(args.seed),
    ]
    if args.method in ("teal", "wina"):
        command += ["--sparsity", str(args.sparsity)]
        if args.method == "teal":
            command += ["--teal_histogram_path", str(Path(args.histogram_path).expanduser().resolve())]
        if args.disable_prefill_sparsity:
            command.append("--disable_prefill_sparsity")
    if args.method == "xmask":
        command += ["--use_x_mask", "--x_mask_ckpt", str(Path(args.x_mask_ckpt).expanduser().resolve()),
                    "--x_mask_tau", str(args.x_mask_tau), "--x_mask_alpha", str(args.x_mask_alpha),
                    "--x_mask_r_thr", str(args.x_mask_r_thr)]
        if args.rec:
            command.append("--rec")
        if args.softmax_alpha_ckpt:
            command += ["--softmax_alpha_ckpt", str(Path(args.softmax_alpha_ckpt).expanduser().resolve())]
    if args.kv_cache:
        command.append("--kv_cache")
    if benchmark["runner"] == "ppl":
        command += ["--eval_ppl", "--tasks"]
    else:
        command += [
            "--tasks", benchmark["tasks"],
            "--num_fewshot", str(benchmark["fewshot"]),
            "--lm_eval_limit", str(benchmark["profiles"][args.profile]["limit"]),
            "--fewshot_overrides", "{}",
        ]
        if args.lm_eval_include_path:
            command += ["--lm_eval_include_path", str(Path(args.lm_eval_include_path).expanduser().resolve())]
    return {
        "model": args.model,
        "model_path": str(model_path),
        "model_family": model["family"],
        "benchmark": args.benchmark,
        "profile": args.profile,
        "method": args.method,
        "quant_type": args.quant_type,
        "command": command,
        "cache_paths": [str(path) for path in cache_paths(model_path, args.calibration_dataset, args.act_sort_metric)],
    }


def prepare_plan(args: argparse.Namespace, catalog: dict | None = None) -> dict:
    catalog = catalog or read_catalog()
    model, model_path = resolve_model(args, catalog)
    if model["status"] != "legacy-ready":
        raise PlanError(f"{args.model} has no ARCQuant model adapter for calibration yet.")
    command = [
        sys.executable, str(ROOT / "reorder_indices.py"),
        "--model", str(model_path), "--dataset", args.dataset,
        "--act_sort_metric", args.act_sort_metric,
        "--samples", str(args.samples), "--seqlen", str(args.seqlen),
    ]
    return {"model": args.model, "model_path": str(model_path), "command": command,
            "cache_paths": [str(path) for path in cache_paths(model_path, args.dataset, args.act_sort_metric)]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARCQuant experiment CLI")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list methods, models and benchmarks")
    sub.add_parser("assets", help="audit local model weights and benchmark data without GPU")
    prep = sub.add_parser("prepare", help="produce ARCQuant reorder and calibration caches")
    prep.add_argument("--model", required=True)
    prep.add_argument("--model-path")
    prep.add_argument("--dataset", choices=["wikitext2", "c4", "pile", "humaneval", "arc_mix"], default="wikitext2")
    prep.add_argument("--act-sort-metric", choices=["max", "mean", "frobenius", "hessian"], default="max")
    prep.add_argument("--samples", type=int, default=128)
    prep.add_argument("--seqlen", type=int, default=2048)
    prep.add_argument("--dry-run", action="store_true")

    for action in ("plan", "run"):
        p = sub.add_parser(action, help="print a plan" if action == "plan" else "execute and record a run")
        p.add_argument("--backend", choices=["arcquant", "base-eval"], default="arcquant")
        p.add_argument("--base-eval-root", help="source directory for the pinned external baseline evaluator")
        p.add_argument("--gpu-ids", help="physical GPU IDs for base-eval, e.g. 0,1; MiniMax also accepts auto")
        p.add_argument("--converted-model", help="two-way converted DeepSeek weights for base-eval")
        p.add_argument("--method", choices=["none", "teal", "wina", "xmask"], required=True)
        p.add_argument("--quant-type", choices=["BF16", "HiF4", "NVFP4", "MXFP4", "INT4"], default="HiF4")
        p.add_argument("--model", required=True)
        p.add_argument("--model-path")
        p.add_argument("--benchmark", required=True)
        p.add_argument("--profile", default="full")
        p.add_argument("--calibration-dataset", choices=["wikitext2", "c4", "pile", "humaneval", "arc_mix"], default="wikitext2")
        p.add_argument("--act-sort-metric", choices=["max", "mean", "frobenius", "hessian"], default="max")
        p.add_argument("--histogram-path")
        p.add_argument("--x-mask-ckpt")
        p.add_argument("--x-mask-tau", type=float, default=1.0)
        p.add_argument("--x-mask-alpha", type=float, default=1.0)
        p.add_argument("--x-mask-r-thr", type=float, default=-1.0)
        p.add_argument("--softmax-alpha-ckpt")
        p.add_argument("--rec", action="store_true")
        p.add_argument("--sparsity", type=float, default=0.0)
        p.add_argument("--disable-prefill-sparsity", action="store_true")
        p.add_argument("--kv-cache", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--lm-eval-include-path")
        if action == "run":
            p.add_argument("--run-dir", type=Path, default=ROOT / "runs")
    return parser


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalog = read_catalog()
        if args.action == "list":
            for key in ("methods", "models", "benchmarks"):
                print(f"{key}:")
                for name, entry in catalog[key].items():
                    baseline = entry.get("baseline_benchmarks") or entry.get("baseline_models")
                    extra = "  base-eval: " + ", ".join(baseline) if baseline else ""
                    print(f"  {name:<22} {entry['status']}{extra}")
            return 0
        if args.action == "assets":
            report = audit_catalog(catalog)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if all(value["status"] == "ready" for group in report.values()
                            for value in group.values()) else 2
        if args.action == "prepare":
            plan = prepare_plan(args, catalog)
            if args.dry_run:
                print(json.dumps(plan, indent=2, ensure_ascii=False))
                return 0
            check_runtime_compatibility()
            if not Path(plan["model_path"]).is_dir():
                raise PlanError(f"Model directory does not exist: {plan['model_path']}")
            return subprocess.run(plan["command"], cwd=ROOT, check=False).returncode
        plan = build_plan(args, catalog)
        if args.action == "plan":
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        if plan.get("backend") == "base-eval":
            model_entry = {**catalog["models"][args.model], "path_env": "",
                           "default_path": plan["model_path"]}
            if args.model == "deepseek-v4-flash":
                model_entry["variants"] = {**model_entry["variants"],
                                           "tp2": plan["env"]["TP_MODEL"]}
            model_check = audit_model(args.model, model_entry)
            benchmark_entry = {**catalog["benchmarks"][args.benchmark],
                               "data_env": "", "default_path": plan["dataset_path"]}
            data_check = audit_benchmark(args.benchmark, benchmark_entry)
            if model_check["status"] != "ready" or data_check["status"] != "ready" or (args.model == "deepseek-v4-flash" and model_check["tp2"]["status"] != "ready"):
                raise PlanError("Baseline assets are incomplete; run 'python arc.py assets' for details.")
            args.run_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            manifest_path = args.run_dir / f"{stamp}_{args.model}_base-eval_{args.benchmark}.json"
            manifest = {**plan, "source_revision": _git_revision(), "started_at_utc": stamp,
                        "status": "running"}
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            try:
                code = subprocess.run(plan["command"], cwd=plan["cwd"],
                                      env={**os.environ, **plan["env"]}, check=False).returncode
            except BaseException:
                manifest["status"] = "interrupted"
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
                raise
            manifest["returncode"] = code
            manifest["status"] = "completed" if code == 0 else "failed"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            print(f"Run manifest: {manifest_path}")
            return code
        check_runtime_compatibility()
        if not Path(plan["model_path"]).is_dir():
            raise PlanError(f"Model directory does not exist: {plan['model_path']}")
        missing = [path for path in plan["cache_paths"] if not Path(path).is_file()]
        if missing:
            raise PlanError("Missing calibration caches: " + ", ".join(missing) + ". Run 'python arc.py prepare' first.")
        if args.method == "teal":
            histogram_root = Path(args.histogram_path).expanduser()
            if not histogram_root.is_dir():
                raise PlanError(f"TEAL histogram directory does not exist: {args.histogram_path}")
            missing = missing_teal_histograms(histogram_root, Path(plan["model_path"]))
            if missing:
                raise PlanError(f"Missing {len(missing)} TEAL histogram files; first: {missing[0]}")
        if args.method == "xmask":
            if not Path(args.x_mask_ckpt).expanduser().is_file():
                raise PlanError(f"xmask checkpoint does not exist: {args.x_mask_ckpt}")
            if args.softmax_alpha_ckpt and not Path(args.softmax_alpha_ckpt).expanduser().is_file():
                raise PlanError(f"softmax alpha checkpoint does not exist: {args.softmax_alpha_ckpt}")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        manifest_path = args.run_dir / f"{stamp}_{args.model}_{args.method}_{args.benchmark}.json"
        if catalog["benchmarks"][args.benchmark]["runner"] == "lm-eval":
            result_path = args.run_dir / f"{stamp}_{args.model}_{args.method}_{args.benchmark}_results.json"
            plan["command"] += ["--output_file", str(result_path)]
            plan["result_path"] = str(result_path)
        manifest = {**plan, "source_revision": _git_revision(), "started_at_utc": stamp, "status": "running"}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        try:
            code = subprocess.run(plan["command"], cwd=ROOT, check=False).returncode
        except BaseException:
            manifest["status"] = "interrupted"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            raise
        manifest["returncode"] = code
        manifest["status"] = "completed" if code == 0 else "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"Run manifest: {manifest_path}")
        return code
    except (PlanError, BaselineError) as exc:
        print(f"arc: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
