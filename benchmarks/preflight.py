"""Read-only evaluation preflight, with explicit baseline/method boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmarks.asset_inventory import audit_benchmark, audit_model
from benchmarks.baseline_bridge import SUPPORTED
from benchmarks.fetch_assets import backend_link_status
from benchmarks.launch_deepseek_extended import PINNED_SHA256
from benchmarks.paths import REPOSITORY_ROOT, asset_path, asset_root, executable_path, resolve_path, variant_path
from benchmarks.scorer_assets import audit_scorer, scorer_root


GPU_MIN_FREE_MIB = {"deepseek-v4-flash": 132 * 1024, "minimax-h3": 79000,
                    "magi-2-preview": 142000}
TARGET_MODELS = frozenset(GPU_MIN_FREE_MIB)


def _check(items: list, name: str, status: str, detail: str, fix: str | None = None) -> None:
    entry = {"name": name, "status": status, "detail": detail}
    if fix and status == "blocked":
        entry["fix"] = fix
    items.append(entry)


def _revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(["git", "-c", f"safe.directory={path.resolve()}",
                             "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _source_signature(source: Path) -> dict:
    names = ("config.json", "model.safetensors.index.json", "inference/model.py",
             "inference/convert.py")
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}


def _converted_deepseek(source: Path, converted: Path) -> tuple[bool, str]:
    manifest = converted / "conversion.json"
    if not manifest.is_file():
        return False, f"missing conversion.json at {converted}"
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("completed") is not True or meta.get("world_size") != 2:
            return False, "conversion manifest is not a completed TP=2 conversion"
        if meta.get("source_signature") != _source_signature(source):
            return False, "converted weights do not match source model files"
        for rank in range(2):
            name = f"model{rank}-mp2.safetensors"
            file = converted / name
            if not file.is_file() or file.stat().st_size != meta["shard_bytes"][name]:
                return False, f"converted shard missing or wrong size: {name}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"invalid TP=2 conversion: {exc}"
    return True, "TP=2 shards, sizes and source signature match conversion manifest"


def _gpu_free() -> dict[int, int] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        return None
    try:
        return {int(parts[0]): int(parts[1]) for line in result.stdout.splitlines()
                if (parts := [part.strip() for part in line.split(",")]) and len(parts) == 2}
    except ValueError:
        return None


def _runtime_python(env_name: str, candidates: list[Path]) -> Path | None:
    explicit = os.environ.get(env_name)
    if explicit:
        return executable_path(explicit)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _python_probe(python: Path, code: str, label: str, *, pythonpath: Path | None = None,
                  extra_env: dict[str, str] | None = None) -> tuple[bool, str]:
    command = [str(python), "-c", code]
    env = os.environ.copy()
    env.update(extra_env or {})
    if pythonpath:
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{env.get('PYTHONPATH', '')}"
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45,
                                check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode:
        return False, (result.stderr or result.stdout).strip()[-600:]
    return True, f"{python}: {label}"


def _native_runtime_code(script: Path, stage: str) -> str:
    """Run the pinned evaluator's import-only CPU check in its own Python."""
    return ("import runpy, sys; sys.argv = ['check_minimax_runtime.py', " +
            repr(stage) + ", '--allow-cpu']; runpy.run_path(" + repr(str(script)) +
            ", run_name='__main__')")


def _native_model_plan(model: str, entry: dict, model_path: Path, data_path: Path,
                       evaluator: Path, converted_override: str | None) -> tuple[bool, str]:
    """Run the pinned model's CPU-only dry run, including its real prompt plan."""
    env = os.environ.copy()
    env.update({"VBENCH_REPO": str(data_path),
                "VBENCH_FULL_INFO": str(data_path / "vbench/VBench_full_info.json"),
                "VBENCH_CACHE_DIR": str(scorer_root()),
                "HF_HOME": str(asset_root() / "hf_home"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    if model == "deepseek-v4-flash":
        prefix = os.environ.get("ARCQUANT_BASE_EVAL_ENV")
        python = Path(prefix) / "bin/python" if prefix else None
        script = evaluator / "run_deepseek_v4_baseline.py"
        converted = variant_path(entry, "tp2", override=converted_override)
        env["WORLD_SIZE"] = "2"
        command = ([str(python), str(script), "--dry-run", "--source-model", str(model_path),
                    "--converted-model", str(converted), "--quant", "hifp4", "--mode", "w4a4"]
                   if python else [])
    elif model == "minimax-h3":
        python = _runtime_python("ARCQUANT_H3_PYTHON", [evaluator / ".venvs/minimax-h3/bin/python"])
        script = evaluator / "gen_vbench_minimax.py"
        command = ([str(python), str(script), "--model", str(model_path), "--quant", "hifp4",
                    "--mode", "w4a4", "--protocol", "tiny16", "--dry-run"] if python else [])
    else:
        python = _runtime_python("ARCQUANT_MAGI2_PYTHON", [evaluator / ".venvs/magi2-h20/bin/python"])
        script = evaluator / "gen_vbench_magi2.py"
        runtime = (resolve_path(os.environ["MAGI2_REPO"]) if os.environ.get("MAGI2_REPO")
                   else REPOSITORY_ROOT / "external/MAGI-2-preview")
        command = ([str(python), str(script), "--model", str(model_path), "--repo", str(runtime),
                    "--num-gpus", "2", "--quant", "hifp4", "--mode", "w4a4",
                    "--protocol", "tiny16", "--dry-run"] if python else [])
    if not python or not python.is_file() or not script.is_file():
        return False, f"Native dry-run needs a configured Python and {script}"
    try:
        result = subprocess.run(command, cwd=evaluator, env=env, capture_output=True,
                                text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode:
        return False, (result.stderr or result.stdout).strip()[-900:]
    scope = ("source and TP2 model layout" if model == "deepseek-v4-flash" else
             "model layout and VBench tiny prompt plan")
    return True, f"{model} HiF4/W4A4 CPU dry-run passed; {scope} checked"


def _remote_revision(url: str, revision: str) -> tuple[bool, str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    try:
        result = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True,
                                text=True, timeout=15, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode:
        return False, (result.stderr or result.stdout).strip()[-500:]
    head = result.stdout.split()[0] if result.stdout.split() else "missing"
    return True, f"remote access works; HEAD={head}; pinned {revision} is checked by provision"


def preflight(catalog: dict, *, model: str, benchmark: str, backend: str = "base-eval",
              method: str = "none", model_override: str | None = None,
              backend_override: str | None = None, converted_override: str | None = None,
              gpu_ids: str | None = None, check_gpu: bool = True,
              hash_scorer: bool = False, probe_runtime: bool = True,
              check_remote: bool = False, native_dry_run: bool = False) -> dict:
    """Return actionable checks. ``ready`` describes the selected protocol only."""
    if model not in catalog["models"] or benchmark not in catalog["benchmarks"]:
        raise ValueError("Unknown model or benchmark; run 'python arc.py list'")
    if backend not in ("base-eval", "arcquant"):
        raise ValueError(f"Unknown backend: {backend}")
    checks = []
    entry = catalog["models"][model]
    data = catalog["benchmarks"][benchmark]
    model_path = asset_path(entry, model, "models", override=model_override)
    data_path = asset_path(data, benchmark, "datasets")
    model_audit = audit_model(model, {**entry, "path_env": "", "default_path": str(model_path)})
    data_audit = audit_benchmark(benchmark, {**data, "data_env": "", "default_path": str(data_path)})
    _check(checks, "model_files", "ready" if model_audit["status"] == "ready" else "blocked",
           f"{model_path}: {model_audit['status']}; {len(model_audit.get('issues', []))} issues",
           f"python arc.py fetch --model {model}" if model_audit["status"] != "ready" else None)
    _check(checks, "benchmark_files", "ready" if data_audit["status"] == "ready" else "blocked",
           f"{data_path}: {data_audit['status']}; {len(data_audit.get('issues', []))} issues",
           f"python arc.py fetch --benchmark {benchmark}" if data_audit["status"] != "ready" else None)

    if backend == "arcquant":
        if model in TARGET_MODELS:
            _check(checks, "model_adapter", "blocked",
                   "Target model lacks end-to-end ARCQuant calibration and forward/evaluator wiring",
                   "See docs/TARGET_MODELS.md; projection-level tests are not a model evaluation")
        elif method in ("teal", "wina") and entry["family"] != "llama":
            _check(checks, "model_adapter", "blocked", "TEAL/WINA are wired for the Llama model family only")
        elif entry["status"] != "legacy-ready" or data["status"] not in ("legacy-ready", "lm-eval-ready"):
            _check(checks, "model_adapter", "blocked", "No ARCQuant model/evaluator adapter for this pair")
        else:
            _check(checks, "model_adapter", "ready", "Legacy ARCQuant model adapter is available")
    else:
        if method != "none":
            _check(checks, "method", "blocked", "External baseline evaluator has no TEAL/WINA or ARC residual channels")
        if benchmark not in SUPPORTED.get(model, ()):
            _check(checks, "protocol", "blocked", "No registered baseline protocol for this model and task")
        else:
            _check(checks, "protocol", "ready", "External BF16/W4A4 baseline protocol; no ARC residual channels")
        spec = catalog["backends"]["base-eval"]
        raw = backend_override or os.environ.get(spec["path_env"])
        evaluator = resolve_path(raw) if raw else REPOSITORY_ROOT / "external/base_eval"
        actual = _revision(evaluator)
        expected = spec["revision"]
        _check(checks, "evaluator_revision", "ready" if actual == expected else "blocked",
               f"{evaluator}: {actual or 'missing'}; expected {expected}",
               "python arc.py fetch --backend base-eval --source /authorized/base_eval"
               if actual != expected else None)
        launcher = ("scripts/run_all_generative.sh" if model == "deepseek-v4-flash" and
                    benchmark in ("gpqa-diamond", "math500") else
                    "scripts/run_all_deepseek_v4_extended.sh" if model == "deepseek-v4-flash" else
                    "scripts/run_all_vbench_dynamic.sh" if model == "minimax-h3" else
                    "scripts/run_all_magi2_vbench.sh")
        script = evaluator / launcher
        launcher_ready = script.is_file() and (model != "deepseek-v4-flash" or benchmark in (
            "gpqa-diamond", "math500") or hashlib.sha256(script.read_bytes()).hexdigest() == PINNED_SHA256)
        _check(checks, "evaluator_launcher", "ready" if launcher_ready else "blocked",
               str(script), "Restore the pinned evaluator checkout" if not launcher_ready else None)
        if check_remote:
            ok, detail = _remote_revision(spec["source_url"], expected)
            _check(checks, "evaluator_remote", "ready" if ok else "blocked", detail,
                   "Provide an authorized remote or copy an authorized Git checkout" if not ok else None)
        else:
            _check(checks, "evaluator_remote", "info",
                   "Remote access not needed for this local run; use --check-remote before moving machines")
        links = backend_link_status(evaluator)
        sibling = evaluator.parent / "data"
        _check(checks, "evaluator_data_link", "ready" if links["data_link"] else "blocked",
               f"{sibling} -> {asset_root()}",
               "python arc.py fetch --backend base-eval --source /authorized/base_eval"
               if not links["data_link"] else None)
        activation = evaluator.parent / "activate_arcquant.sh"
        _check(checks, "evaluator_activation_link", "ready" if links["activation_link"] else "blocked",
               f"{activation} -> {REPOSITORY_ROOT / 'benchmarks/activate_base_eval.sh'}",
               "python arc.py fetch --backend base-eval --source /authorized/base_eval"
               if not links["activation_link"] else None)
        if model == "deepseek-v4-flash":
            converted = variant_path(entry, "tp2", override=converted_override)
            ok, detail = _converted_deepseek(model_path, converted)
            _check(checks, "deepseek_tp2", "ready" if ok else "blocked", detail,
                   "Run the pinned DeepSeek converter; see docs/ASSETS_AND_BASELINES.md" if not ok else None)
            executable = _runtime_python("ARCQUANT_BASE_EVAL_ENV", [])
            python = executable / "bin/python" if executable else None
            _check(checks, "deepseek_python", "ready" if python and python.is_file() else "blocked",
                   str(python) if python else "ARCQUANT_BASE_EVAL_ENV is unset",
                   "export ARCQUANT_BASE_EVAL_ENV=/path/to/deepseek-eval-env")
            if python and python.is_file() and probe_runtime:
                code = ("import torch, safetensors, transformers, triton; "
                        "from importlib.metadata import version; "
                        "assert version('transformers') == '5.9.0'; "
                        "assert hasattr(torch, 'float4_e2m1fn_x2')")
                ok, detail = _python_probe(python, code, "DeepSeek FP4 API and transformers 5.9.0")
                _check(checks, "deepseek_imports", "ready" if ok else "blocked", detail)
            elif python and python.is_file():
                _check(checks, "deepseek_imports", "unchecked", "Runtime imports not tested")
        elif model in ("minimax-h3", "magi-2-preview"):
            candidates = [evaluator / ".venvs/vbench/bin/python"]
            scorer_python = _runtime_python("ARCQUANT_VBENCH_PYTHON", candidates)
            _check(checks, "vbench_python", "ready" if scorer_python and scorer_python.is_file()
                   else "blocked", str(scorer_python) if scorer_python else "VBench Python not configured",
                   "export ARCQUANT_VBENCH_PYTHON=/path/to/vbench-env/bin/python")
            scorer = audit_scorer(hash_contents=hash_scorer)
            _check(checks, "vbench_weights", "blocked" if scorer["status"] == "incomplete" else
                   "ready" if scorer["status"] == "ready" else "unchecked",
                   f"{scorer['path']}: {scorer['file_count']} files, {len(scorer['issues'])} issues; "
                   f"{'hash verified' if hash_scorer else 'sizes only'}",
                   "python arc.py fetch --scorer vbench" if scorer["issues"] else None)
            data_rev = _revision(data_path)
            expected_data_rev = data["source"]["revision"]
            _check(checks, "vbench_source", "ready" if data_rev == expected_data_rev else "blocked",
                   f"{data_path}: {data_rev or 'missing'}; expected {expected_data_rev}",
                   "python arc.py fetch --benchmark vbench" if data_rev != expected_data_rev else None)
            if model == "minimax-h3":
                generator = _runtime_python("ARCQUANT_H3_PYTHON", [evaluator / ".venvs/minimax-h3/bin/python"])
                _check(checks, "minimax_python", "ready" if generator and generator.is_file()
                       else "blocked", str(generator) if generator else "MiniMax Python not configured",
                       "export ARCQUANT_H3_PYTHON=/path/to/h3-env/bin/python")
                if generator and generator.is_file() and probe_runtime:
                    code = ("import torch, diffusers, transformers; "
                            "from importlib.metadata import version; "
                            "from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep; "
                            "assert version('diffusers') == '0.40.0'; "
                            "assert version('transformers') == '5.9.0'")
                    ok, detail = _python_probe(generator, code, "MiniMax H3 Diffusers and Transformers APIs")
                    _check(checks, "minimax_imports", "ready" if ok else "blocked", detail)
                    script = evaluator / "scripts/check_minimax_runtime.py"
                    if script.is_file():
                        ok, detail = _python_probe(generator, _native_runtime_code(script, "gen"),
                                                   "pinned MiniMax generator imports",
                                                   extra_env={"PYTHONDONTWRITEBYTECODE": "1"})
                        _check(checks, "minimax_native_imports", "ready" if ok else "blocked", detail)
                    else:
                        _check(checks, "minimax_native_imports", "blocked", f"missing {script}")
                elif generator and generator.is_file():
                    _check(checks, "minimax_imports", "unchecked", "Runtime imports not tested")
                    _check(checks, "minimax_native_imports", "unchecked", "Pinned evaluator imports not tested")
            else:
                runtime = resolve_path(os.environ["MAGI2_REPO"]) if os.environ.get("MAGI2_REPO") else (
                    REPOSITORY_ROOT / "external/MAGI-2-preview")
                expected_runtime = entry["runtime_source"]["revision"]
                actual_runtime = _revision(runtime)
                _check(checks, "magi_runtime_source", "ready" if actual_runtime == expected_runtime
                       else "blocked", f"{runtime}: {actual_runtime or 'missing'}",
                       "python arc.py fetch --runtime magi-2-preview" if actual_runtime != expected_runtime else None)
                generator = _runtime_python("ARCQUANT_MAGI2_PYTHON", [evaluator / ".venvs/magi2-h20/bin/python"])
                _check(checks, "magi_python", "ready" if generator and generator.is_file()
                       else "blocked", str(generator) if generator else "MAGI-2 Python not configured",
                       "export ARCQUANT_MAGI2_PYTHON=/path/to/magi2-env/bin/python")
                ffmpeg_bin = os.environ.get("ARCQUANT_FFMPEG_BIN")
                media_path = (f"{resolve_path(ffmpeg_bin)}{os.pathsep}{os.environ.get('PATH', '')}"
                              if ffmpeg_bin else None)
                ffmpeg = shutil.which("ffmpeg", path=media_path)
                ffprobe = shutil.which("ffprobe", path=media_path)
                _check(checks, "ffmpeg", "ready" if ffmpeg else "blocked",
                       ffmpeg or "ffmpeg missing", "Install FFmpeg or set ARCQUANT_FFMPEG_BIN")
                _check(checks, "ffprobe", "ready" if ffprobe else "blocked",
                       ffprobe or "ffprobe missing", "Install FFmpeg or set ARCQUANT_FFMPEG_BIN")
                if generator and generator.is_file() and probe_runtime:
                    code = ("import sys, torch, flash_attn_interface, magi_attention, magi_compiler; "
                            "from transformers import Qwen3_5TextModel; "
                            "assert sys.version_info[:2] == (3, 12)")
                    ok, detail = _python_probe(generator, code, "MAGI-2 Python 3.12 and Hopper extension APIs")
                    _check(checks, "magi_imports", "ready" if ok else "blocked", detail)
                elif generator and generator.is_file():
                    _check(checks, "magi_imports", "unchecked", "Runtime imports not tested")
            if scorer_python and scorer_python.is_file() and probe_runtime:
                code = ("import torch, vbench, detectron2._C; "
                        "from importlib.metadata import version; "
                        "assert version('transformers') == '4.33.2'")
                ok, detail = _python_probe(scorer_python, code, "VBench and Detectron2 scoring APIs",
                                           pythonpath=data_path)
                _check(checks, "vbench_imports", "ready" if ok else "blocked", detail)
                script = evaluator / "scripts/check_minimax_runtime.py"
                if scorer["status"] != "incomplete" and script.is_file():
                    cache = scorer_root()
                    extra = {"VBENCH_REPO": str(data_path), "VBENCH_CACHE_DIR": str(cache),
                             "HF_HOME": str(asset_root() / "hf_home"),
                             "TORCH_HOME": str(cache / "torch_hub"),
                             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                             "PYTHONDONTWRITEBYTECODE": "1"}
                    ok, detail = _python_probe(scorer_python, _native_runtime_code(script, "eval"),
                                               "pinned VBench scorer imports", pythonpath=data_path,
                                               extra_env=extra)
                    _check(checks, "vbench_native_imports", "ready" if ok else "blocked", detail)
                else:
                    _check(checks, "vbench_native_imports", "blocked",
                           "Pinned scorer check or scoring cache is incomplete")
            elif scorer_python and scorer_python.is_file():
                _check(checks, "vbench_imports", "unchecked", "Runtime imports not tested")
                _check(checks, "vbench_native_imports", "unchecked", "Pinned evaluator imports not tested")

        if native_dry_run:
            if model not in TARGET_MODELS or benchmark not in SUPPORTED.get(model, ()):
                _check(checks, "native_model_plan", "blocked",
                       "No pinned CPU dry-run for this model and benchmark")
            elif (model_audit["status"] != "ready" or data_audit["status"] != "ready" or
                    actual != expected or not launcher_ready):
                _check(checks, "native_model_plan", "blocked",
                       "Model, data or pinned evaluator is incomplete; prepare them first")
            else:
                ok, detail = _native_model_plan(model, entry, model_path, data_path,
                                                evaluator, converted_override)
                _check(checks, "native_model_plan", "ready" if ok else "blocked", detail)

    if check_gpu and model in GPU_MIN_FREE_MIB:
        free = _gpu_free()
        needed = 2 if model != "minimax-h3" else 1
        if gpu_ids and gpu_ids != "auto":
            raw_ids = gpu_ids.split(",")
            selected = [int(item) for item in raw_ids] if all(item.isdigit() for item in raw_ids) else []
        else:
            selected = sorted(free, key=free.get, reverse=True)[:needed] if free else []
        allowed_count = ((2, 4, 8) if model == "magi-2-preview" else
                         tuple(range(1, 9)) if model == "minimax-h3" else (2,))
        if (gpu_ids == "auto" and model != "minimax-h3") or len(selected) not in allowed_count or len(set(selected)) != len(selected) or not free or any(
                free.get(index, 0) < GPU_MIN_FREE_MIB[model] for index in selected):
            _check(checks, "gpu_capacity", "blocked",
                   f"Need {needed} GPU(s) with at least {GPU_MIN_FREE_MIB[model]} MiB free each; "
                   f"selected={selected}; free={free or 'unavailable'}",
                   "Wait for suitable GPUs or select another machine; do not interrupt existing jobs")
        else:
            _check(checks, "gpu_capacity", "ready", f"selected={selected}; free MiB={free}")
    elif model in GPU_MIN_FREE_MIB:
        _check(checks, "gpu_capacity", "unchecked", "GPU availability skipped; rerun without --skip-gpu before evaluation")

    blocked = sum(item["status"] == "blocked" for item in checks)
    unchecked = sum(item["status"] == "unchecked" for item in checks)
    return {"model": model, "benchmark": benchmark, "backend": backend, "method": method,
            "status": "blocked" if blocked else "unchecked" if unchecked else "ready",
            "blocked": blocked, "unchecked": unchecked, "checks": checks,
            "scope": "Baseline readiness only" if backend == "base-eval" else "ARCQuant method readiness"}
