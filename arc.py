"""Plan and run ARCQuant experiments from the repository root.

This control script intentionally uses only Python's standard library. The
legacy model code and its GPU dependencies are imported only by ``run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmarks.asset_inventory import MODEL_FILES, audit_benchmark, audit_catalog, audit_model
from benchmarks.baseline_bridge import BaselineError, build_baseline_plan
from benchmarks.fetch_assets import FetchError, backend_link_status, fetch_entry
from benchmarks.paths import asset_path, asset_root
from benchmarks.preflight import preflight
from benchmarks.provision import execute_provision, provision_plan
from benchmarks.scorer_assets import (audit_scorer, auxiliary_entries, entries as scorer_entries,
                                       fetch_scorer)
from benchmarks.sparse_strategy import StrategyError, load_strategy


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
    return model, asset_path(model, args.model, "models", override=args.model_path)


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
    allocation = getattr(args, "allocation", "uniform")
    if args.method not in ("teal", "wina") and allocation != "uniform":
        raise PlanError("--allocation greedy requires --method teal or wina.")
    if args.method not in ("teal", "wina") and getattr(args, "strategy", None):
        raise PlanError("--strategy requires --method teal or wina.")
    if args.method in ("teal", "wina") and allocation == "uniform" and getattr(args, "strategy", None):
        raise PlanError("--strategy requires --allocation greedy.")
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
            histogram_path = Path(args.histogram_path).expanduser().resolve() if args.histogram_path else (
                asset_root() / "calibration" / args.model / "teal" / "histograms"
            ).resolve()
            command += ["--teal_histogram_path", str(histogram_path)]
        if allocation == "greedy":
            strategy_value = getattr(args, "strategy", None)
            strategy_path = Path(strategy_value).expanduser().resolve() if strategy_value else (
                asset_root() / "calibration" / args.model / args.method / "strategy.json"
            ).resolve()
            command += ["--sparse_strategy_path", str(strategy_path)]
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
        "allocation": allocation if args.method in ("teal", "wina") else None,
        "histogram_path": str(histogram_path) if args.method == "teal" else None,
        "strategy_path": str(strategy_path) if args.method in ("teal", "wina") and allocation == "greedy" else None,
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


def calibration_plan(args: argparse.Namespace, catalog: dict | None = None) -> dict:
    catalog = catalog or read_catalog()
    model, model_path = resolve_model(args, catalog)
    if model["family"] != "llama" or model["status"] != "legacy-ready":
        raise PlanError("Native TEAL/WINA calibration currently supports the legacy Llama adapter only.")
    if not 0 < args.sparsity < 1:
        raise PlanError("--sparsity must be in (0, 1).")
    if args.method == "wina" and args.allocation == "uniform":
        raise PlanError("Uniform WINA needs no offline calibration; use 'arc.py prepare' for ARCQuant caches.")
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        asset_root() / "calibration" / args.model / args.method
    ).resolve()
    command = [sys.executable, str(ROOT / "model" / "calibrate_sparse.py"),
               "--model-path", str(model_path), "--method", args.method,
               "--allocation", args.allocation, "--output-dir", str(output),
               "--dataset", args.dataset, "--samples", str(args.samples),
               "--seqlen", str(args.seqlen), "--sparsity", str(args.sparsity),
               "--step", str(args.step), "--last-fraction", str(args.last_fraction),
               "--device", args.device, "--seed", str(args.seed)]
    return {"model": args.model, "model_path": str(model_path), "method": args.method,
            "allocation": args.allocation, "output_dir": str(output), "command": command}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARCQuant experiment CLI")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list methods, models and benchmarks")
    assets = sub.add_parser("assets", help="audit local model weights and benchmark data without GPU")
    assets.add_argument("--model", action="append", help="check only this model; may be repeated")
    assets.add_argument("--benchmark", action="append", help="check only this benchmark; may be repeated")
    doctor = sub.add_parser("doctor", help="check concrete evaluation prerequisites without running a model")
    doctor.add_argument("--model", required=True)
    doctor.add_argument("--benchmark", required=True)
    doctor.add_argument("--backend", choices=["base-eval", "arcquant"], default="base-eval")
    doctor.add_argument("--method", choices=["none", "teal", "wina", "xmask"], default="none")
    doctor.add_argument("--model-path")
    doctor.add_argument("--base-eval-root")
    doctor.add_argument("--converted-model")
    doctor.add_argument("--gpu-ids")
    doctor.add_argument("--skip-gpu", action="store_true", help="report GPU as unchecked")
    doctor.add_argument("--hash-scorer", action="store_true", help="hash all VBench/BERT weights")
    doctor.add_argument("--check-remote", action="store_true", help="test evaluator Git URL access for migration")
    doctor.add_argument("--skip-runtime-probe", action="store_true", help="only check runtime executable paths")
    doctor.add_argument("--native-dry-run", action="store_true",
                        help="run the pinned evaluator's CPU-only model and prompt plan")
    fetch = sub.add_parser("fetch", help="download a recorded model, benchmark or evaluator source")
    fetch_target = fetch.add_mutually_exclusive_group(required=True)
    fetch_target.add_argument("--model", help="model key from 'arc.py list'")
    fetch_target.add_argument("--benchmark", help="benchmark key from 'arc.py list'")
    fetch_target.add_argument("--backend", choices=["base-eval"])
    fetch_target.add_argument("--runtime", choices=["magi-2-preview"])
    fetch_target.add_argument("--scorer", choices=["vbench"], help="download hash-pinned VBench/BERT scoring weights")
    fetch.add_argument("--dry-run", action="store_true")
    fetch.add_argument("--source", help="local Git checkout or alternate URL for a Git backend/runtime")
    provision = sub.add_parser("provision", help="prepare a target pair's missing files in dependency order")
    provision.add_argument("--model", required=True)
    provision.add_argument("--benchmark", required=True)
    provision.add_argument("--backend-source", help="authorized base-eval Git URL or local checkout")
    provision.add_argument("--verify-scorer-hashes", action="store_true")
    provision.add_argument("--execute", action="store_true", help="download/convert missing files")
    prep = sub.add_parser("prepare", help="produce ARCQuant reorder and calibration caches")
    prep.add_argument("--model", required=True)
    prep.add_argument("--model-path")
    prep.add_argument("--dataset", choices=["wikitext2", "c4", "pile", "humaneval", "arc_mix"], default="wikitext2")
    prep.add_argument("--act-sort-metric", choices=["max", "mean", "frobenius", "hessian"], default="max")
    prep.add_argument("--samples", type=int, default=128)
    prep.add_argument("--seqlen", type=int, default=2048)
    prep.add_argument("--dry-run", action="store_true")
    cal = sub.add_parser("calibrate", help="generate native TEAL histograms and optional TEAL/WINA greedy allocations")
    cal.add_argument("--model", required=True)
    cal.add_argument("--model-path")
    cal.add_argument("--method", choices=["teal", "wina"], required=True)
    cal.add_argument("--allocation", choices=["uniform", "greedy"], default="uniform")
    cal.add_argument("--sparsity", type=float, required=True)
    cal.add_argument("--dataset", choices=["wikitext2", "c4", "arc_mix"], default="wikitext2")
    cal.add_argument("--samples", type=int, default=8)
    cal.add_argument("--seqlen", type=int, default=256)
    cal.add_argument("--step", type=float, default=0.025,
                     help="greedy rate increment for a Q-sized weight matrix")
    cal.add_argument("--last-fraction", type=float, default=0.25)
    cal.add_argument("--device", default="cuda:0")
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument("--output-dir")
    cal.add_argument("--dry-run", action="store_true")

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
        p.add_argument("--allocation", choices=["uniform", "greedy"], default="uniform")
        p.add_argument("--strategy", help="per-layer greedy strategy.json; defaults to assets/calibration/<model>/<method>/strategy.json")
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


def runtime_env() -> dict[str, str]:
    """Keep CLI downloads and dataset loaders on the same portable HF cache."""
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(asset_root() / "hf_home"))
    return env


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
            if args.model or args.benchmark:
                unknown_models = set(args.model or ()) - set(catalog["models"])
                unknown_benchmarks = set(args.benchmark or ()) - set(catalog["benchmarks"])
                if unknown_models or unknown_benchmarks:
                    raise PlanError(f"Unknown assets: models={sorted(unknown_models)}, "
                                    f"benchmarks={sorted(unknown_benchmarks)}")
                unchecked_models = set(args.model or ()) - set(MODEL_FILES)
                unchecked_benchmarks = {name for name in args.benchmark or ()
                                        if "data_file" not in catalog["benchmarks"][name]}
                if unchecked_models or unchecked_benchmarks:
                    raise PlanError(f"No file audit rules: models={sorted(unchecked_models)}, "
                                    f"benchmarks={sorted(unchecked_benchmarks)}")
                report = {
                    "models": {name: audit_model(name, catalog["models"][name])
                               for name in dict.fromkeys(args.model or ())},
                    "benchmarks": {name: audit_benchmark(name, catalog["benchmarks"][name])
                                   for name in dict.fromkeys(args.benchmark or ())},
                }
            else:
                report = audit_catalog(catalog)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if all(value["status"] == "ready" for group in report.values()
                            for value in group.values()) else 2
        if args.action == "doctor":
            report = preflight(catalog, model=args.model, benchmark=args.benchmark,
                               backend=args.backend, method=args.method,
                               model_override=args.model_path,
                               backend_override=args.base_eval_root,
                               converted_override=args.converted_model,
                               gpu_ids=args.gpu_ids, check_gpu=not args.skip_gpu,
                               hash_scorer=args.hash_scorer,
                               probe_runtime=not args.skip_runtime_probe,
                               check_remote=args.check_remote,
                               native_dry_run=args.native_dry_run)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["status"] == "ready" else 2
        if args.action == "provision":
            steps = provision_plan(catalog, model=args.model, benchmark=args.benchmark,
                                   backend_source=args.backend_source,
                                   verify_scorer_hashes=args.verify_scorer_hashes)
            completed = (execute_provision(catalog, steps, model=args.model,
                                           benchmark=args.benchmark,
                                           backend_source=args.backend_source)
                         if args.execute else [])
            if args.execute:
                steps = provision_plan(catalog, model=args.model, benchmark=args.benchmark,
                                       backend_source=args.backend_source,
                                       verify_scorer_hashes=True)
            print(json.dumps({"model": args.model, "benchmark": args.benchmark,
                              "executed": args.execute, "completed": completed, "steps": steps,
                              "next": f"python arc.py doctor --model {args.model} --benchmark {args.benchmark}"
                              }, indent=2, ensure_ascii=False))
            return 0 if not args.execute or all(step["status"] == "ready" for step in steps) else 2
        if args.action == "fetch":
            if args.scorer:
                if args.source:
                    raise PlanError("--source is not supported for scorer weights")
                if args.dry_run:
                    files = scorer_entries()
                    report = {"scorer": "vbench", "files": len(files) + len(auxiliary_entries()),
                              "bytes": sum(item["size"] for _, item, _ in files) +
                              sum(item["size"] for item in auxiliary_entries()),
                              "sources": sorted({m["repo_id"] for m, _, _ in files} |
                                                {item["url"] for item in auxiliary_entries()}),
                              "status": audit_scorer(hash_contents=False)["status"]}
                else:
                    report = fetch_scorer()
                print(json.dumps(report, indent=2, ensure_ascii=False))
                return 0 if args.dry_run or report["status"] == "ready" else 2
            kind, name = next((kind, getattr(args, kind)) for kind in
                              ("model", "benchmark", "backend", "runtime") if getattr(args, kind))
            print(json.dumps(fetch_entry(catalog, kind, name, dry_run=args.dry_run,
                                         source_override=args.source),
                             indent=2, ensure_ascii=False))
            return 0
        if args.action == "prepare":
            plan = prepare_plan(args, catalog)
            if args.dry_run:
                print(json.dumps(plan, indent=2, ensure_ascii=False))
                return 0
            check_runtime_compatibility()
            if not Path(plan["model_path"]).is_dir():
                raise PlanError(f"Model directory does not exist: {plan['model_path']}")
            return subprocess.run(plan["command"], cwd=ROOT, env=runtime_env(), check=False).returncode
        if args.action == "calibrate":
            plan = calibration_plan(args, catalog)
            if args.dry_run:
                print(json.dumps(plan, indent=2, ensure_ascii=False))
                return 0
            check_runtime_compatibility()
            if not Path(plan["model_path"]).is_dir():
                raise PlanError(f"Model directory does not exist: {plan['model_path']}")
            return subprocess.run(plan["command"], cwd=ROOT, env=runtime_env(), check=False).returncode
        plan = build_plan(args, catalog)
        if args.action == "plan":
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        if plan.get("backend") == "base-eval":
            backend_root = Path(plan["cwd"])
            if backend_root == ROOT / "external" / "base_eval":
                links = backend_link_status(backend_root)
                if not links["data_link"]:
                    raise PlanError(
                        "Portable base-eval data link does not match ARCQUANT_ASSET_ROOT; "
                        "rerun 'python arc.py fetch --backend base-eval' with the same asset root."
                    )
                if not links["activation_link"]:
                    raise PlanError(
                        "Portable base-eval activation link is missing or points elsewhere; "
                        "rerun 'python arc.py fetch --backend base-eval'."
                    )
            if plan["backend_revision"] != plan["backend_expected_revision"]:
                raise PlanError(
                    f"base-eval checkout at {backend_root} must be revision "
                    f"{plan['backend_expected_revision']}; found {plan['backend_revision'] or 'no Git checkout'}. "
                    "Run 'python arc.py fetch --backend base-eval' or pass a matching --base-eval-root."
                )
            if not Path(plan["evaluator_launcher"]).is_file():
                raise PlanError(f"base-eval launcher is missing: {plan['evaluator_launcher']}")
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
                                      env={**runtime_env(), **plan["env"]}, check=False).returncode
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
            histogram_root = Path(plan["histogram_path"])
            if not histogram_root.is_dir():
                raise PlanError(f"TEAL histogram directory does not exist: {histogram_root}; run 'arc.py calibrate --method teal' first.")
            missing = missing_teal_histograms(histogram_root, Path(plan["model_path"]))
            if missing:
                raise PlanError(f"Missing {len(missing)} TEAL histogram files; first: {missing[0]}")
            calibration_file = histogram_root.parent / "calibration.json"
            if calibration_file.is_file():
                calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
                config_hash = hashlib.sha256((Path(plan["model_path"]) / "config.json").read_bytes()).hexdigest()
                if calibration.get("model_config_sha256") != config_hash:
                    raise PlanError(f"TEAL histograms at {histogram_root} were made for another model config")
        if plan.get("strategy_path"):
            config_file = Path(plan["model_path"]) / "config.json"
            layer_count = int(json.loads(config_file.read_text(encoding="utf-8"))["num_hidden_layers"])
            load_strategy(plan["strategy_path"], method=args.method, layer_count=layer_count,
                          target_sparsity=args.sparsity, model_config=config_file)
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
        if args.method in ("teal", "wina"):
            stats_path = args.run_dir / f"{stamp}_{args.model}_{args.method}_{args.benchmark}_sparsity.json"
            plan["command"] += ["--sparse_stats_file", str(stats_path)]
            plan["sparse_stats_path"] = str(stats_path)
        manifest = {**plan, "source_revision": _git_revision(), "started_at_utc": stamp, "status": "running"}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        try:
            code = subprocess.run(plan["command"], cwd=ROOT, env=runtime_env(), check=False).returncode
        except BaseException:
            manifest["status"] = "interrupted"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            raise
        manifest["returncode"] = code
        manifest["status"] = "completed" if code == 0 else "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"Run manifest: {manifest_path}")
        return code
    except (PlanError, BaselineError, FetchError, StrategyError) as exc:
        print(f"arc: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
