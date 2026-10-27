"""Ordered, resumable preparation of catalogued evaluation files."""

from __future__ import annotations

import subprocess
import sys
import hashlib
from pathlib import Path

from benchmarks.asset_inventory import audit_benchmark, audit_model
from benchmarks.fetch_assets import FetchError, backend_link_status, ensure_backend_links, fetch_entry
from benchmarks.launch_deepseek_extended import PINNED_SHA256
from benchmarks.paths import REPOSITORY_ROOT, asset_path, variant_path
from benchmarks.preflight import _converted_deepseek, _revision
from benchmarks.scorer_assets import audit_scorer, fetch_scorer


def provision_plan(catalog: dict, *, model: str, benchmark: str,
                   backend_source: str | None = None, verify_scorer_hashes: bool = False) -> list[dict]:
    if model not in catalog["models"] or benchmark not in catalog["benchmarks"]:
        raise ValueError("Unknown model or benchmark")
    if benchmark not in catalog["models"][model].get("baseline_benchmarks", ()):
        raise ValueError(f"No baseline protocol for {model} on {benchmark}")
    entry = catalog["models"][model]
    data = catalog["benchmarks"][benchmark]
    steps = []
    model_check = audit_model(model, entry)
    steps.append({"step": "model", "status": "ready" if model_check["status"] == "ready" else "needed",
                  "path": model_check["path"], "command": f"python arc.py fetch --model {model}"})
    data_check = audit_benchmark(benchmark, data)
    steps.append({"step": "benchmark", "status": "ready" if data_check["status"] == "ready" else "needed",
                  "path": data_check["path"], "command": f"python arc.py fetch --benchmark {benchmark}"})
    evaluator = REPOSITORY_ROOT / "external/base_eval"
    expected = catalog["backends"]["base-eval"]["revision"]
    valid = _revision(evaluator) == expected
    steps.append({"step": "evaluator", "status": "ready" if valid else "needed",
                  "path": str(evaluator), "command": "python arc.py fetch --backend base-eval" +
                  (f" --source {backend_source}" if backend_source else "")})
    links = backend_link_status(evaluator)
    sibling = evaluator.parent / "data"
    steps.append({"step": "data_link", "status": "ready" if links["data_link"] else "needed", "path": str(sibling),
                  "command": "python arc.py fetch --backend base-eval" +
                  (f" --source {backend_source}" if backend_source else "")})
    activation = evaluator.parent / "activate_arcquant.sh"
    steps.append({"step": "activation_link", "status": "ready" if links["activation_link"] else "needed",
                  "path": str(activation), "command": "python arc.py fetch --backend base-eval" +
                  (f" --source {backend_source}" if backend_source else "")})
    launcher = ("scripts/run_all_generative.sh" if model == "deepseek-v4-flash" and
                benchmark in ("gpqa-diamond", "math500") else
                "scripts/run_all_deepseek_v4_extended.sh" if model == "deepseek-v4-flash" else
                "scripts/run_all_vbench_dynamic.sh" if model == "minimax-h3" else
                "scripts/run_all_magi2_vbench.sh")
    script = evaluator / launcher
    launcher_ready = script.is_file() and (model != "deepseek-v4-flash" or benchmark in (
        "gpqa-diamond", "math500") or hashlib.sha256(script.read_bytes()).hexdigest() == PINNED_SHA256)
    steps.append({"step": "evaluator_launcher", "status": "ready" if launcher_ready else "needed",
                  "path": str(script), "command": "python arc.py fetch --backend base-eval" +
                  (f" --source {backend_source}" if backend_source else "")})
    if model == "deepseek-v4-flash":
        source = asset_path(entry, model, "models")
        converted = variant_path(entry, "tp2")
        ok, detail = _converted_deepseek(source, converted) if model_check["status"] == "ready" else (
            False, "source model not available yet")
        steps.append({"step": "deepseek_tp2", "status": "ready" if ok else "needed",
                      "path": str(converted), "detail": detail,
                      "command": "python external/base_eval/convert_deepseek_v4.py --source-model " +
                      f"{source} --world-size 2 --output {converted}"})
    if model == "magi-2-preview":
        runtime = REPOSITORY_ROOT / "external/MAGI-2-preview"
        ok = _revision(runtime) == entry["runtime_source"]["revision"]
        steps.append({"step": "magi_runtime", "status": "ready" if ok else "needed",
                      "path": str(runtime), "command": "python arc.py fetch --runtime magi-2-preview"})
    if benchmark == "vbench":
        scorer = audit_scorer(hash_contents=verify_scorer_hashes)
        steps.append({"step": "vbench_scorer", "status": "ready" if scorer["status"] == "ready"
                      else "unchecked" if scorer["status"] == "size-only" else "needed",
                      "path": scorer["path"],
                      "verification": scorer["status"],
                      "command": "python arc.py fetch --scorer vbench"})
    return steps


def execute_provision(catalog: dict, steps: list[dict], *, model: str, benchmark: str,
                      backend_source: str | None = None) -> list[dict]:
    """Execute only missing steps in dependency order; stop at the first error."""
    completed = []
    for action in steps:
        step = action["step"]
        if action["status"] == "ready":
            completed.append({"step": step, "status": "already-ready"})
            continue
        if step == "model":
            fetch_entry(catalog, "model", model)
        elif step == "benchmark":
            fetch_entry(catalog, "benchmark", benchmark)
        elif step in ("evaluator", "data_link", "activation_link", "evaluator_launcher"):
            evaluator = REPOSITORY_ROOT / "external/base_eval"
            # A pinned checkout needs no network access to repair sibling links.
            if step in ("data_link", "activation_link") and _revision(evaluator) == (
                    catalog["backends"]["base-eval"]["revision"]):
                ensure_backend_links(evaluator)
            else:
                fetch_entry(catalog, "backend", "base-eval", source_override=backend_source)
            if step == "evaluator_launcher":
                script = Path(action["path"])
                if not script.is_file() or (model == "deepseek-v4-flash" and benchmark not in (
                        "gpqa-diamond", "math500") and
                        hashlib.sha256(script.read_bytes()).hexdigest() != PINNED_SHA256):
                    raise FetchError(f"Pinned evaluator launcher is missing or changed: {script}; "
                                     "restore the authorized checkout")
        elif step == "deepseek_tp2":
            source = asset_path(catalog["models"][model], model, "models")
            output = variant_path(catalog["models"][model], "tp2")
            script = REPOSITORY_ROOT / "external/base_eval/convert_deepseek_v4.py"
            if not script.is_file():
                raise FileNotFoundError(f"Pinned converter is missing: {script}")
            env_prefix = os.environ.get("ARCQUANT_BASE_EVAL_ENV")
            python = str(Path(env_prefix) / "bin/python") if env_prefix else sys.executable
            subprocess.run([python, str(script), "--source-model", str(source),
                            "--world-size", "2", "--output", str(output)],
                           check=True, cwd=script.parent)
        elif step == "magi_runtime":
            fetch_entry(catalog, "runtime", "magi-2-preview")
        elif step == "vbench_scorer":
            report = fetch_scorer()
            if report["status"] != "ready":
                raise RuntimeError("VBench scorer assets remain incomplete")
        else:
            raise ValueError(f"Unknown provision step: {step}")
        completed.append({"step": step, "status": "completed"})
    return completed
