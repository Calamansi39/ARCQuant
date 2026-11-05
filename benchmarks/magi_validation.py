"""Fail-closed MAGI sparse coverage and zero-sparsity validation gates."""

import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("benchmarks/run_magi2_sparse.py", "model/magi2_sparse_runtime.py",
           "model/magi2_sparse_moe.py", "model/magi2_calibration.py", "model/sparse_utils.py")
TINY_MANIFEST = "5823e6399ba7e31ba5303dc6d63526c30656e2c45a6ae1c7ade2482a1d23e958"


def validate_sparse_runtime(folder, method, sparsity, *, check_sources=True):
    folder = Path(folder)
    reports = []
    for rank in range(2):
        report = json.loads((folder / f"sparse_runtime_rank{rank}.json").read_text())
        if not report.get("complete") or report.get("error") or report["method"] != method or report["sparsity"] != sparsity:
            raise ValueError(f"Incomplete/mismatched MAGI sparse runtime evidence: rank {rank}")
        if check_sources and report["sources"] != {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}:
            raise ValueError("MAGI sparse code changed since validation")
        if set(report["stages"]) != {"preview", "refiner"}:
            raise ValueError("Missing preview/refiner runtime evidence")
        for stage, details in report["stages"].items():
            counts = details["stats"]
            if counts.get("nonfinite", 0) or details.get("output_nonfinite", 0):
                raise ValueError(f"Nonfinite MAGI activations: {stage}")
            if not counts.get("total_elements") or not details.get("coverage"):
                raise ValueError(f"Empty MAGI runtime coverage: {stage}")
            for name, kind in details["coverage"].items():
                seen = any(key == name or key.startswith(name + ".") for key in details["modules"])
                if kind == "routed" and sparsity == 0:
                    seen = seen or details.get("zero_sparsity_native_bypass", {}).get(name, 0) > 0
                if not seen:
                    raise ValueError(f"Unexecuted MAGI projection: {stage}/{name}")
            rate = counts["masked_elements"] / counts["total_elements"]
            if sparsity == 0 and rate != 0:
                raise ValueError("Zero sparsity unexpectedly masked activations")
            if method == "wina" and sparsity and abs(rate - sparsity) > .02:
                raise ValueError(f"WINA observed mask rate {rate} differs from target {sparsity}")
        if not report["videos"] or any(not video["peak_allocated_bytes"] for video in report["videos"]):
            raise ValueError("No real-video memory/timing evidence")
        reports.append(report)
    return reports


def decoded_video_hash(path):
    result = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                      "-f", "hash", "-hash", "sha256", "-"], text=True)
    return result.strip()


def validate_zero_equivalence(baseline, sparse):
    baseline, sparse = Path(baseline), Path(sparse)
    validate_sparse_runtime(sparse, "wina", 0.)
    configs = [json.loads((folder / "generation_config.json").read_text()) for folder in (baseline, sparse)]
    ignored = {"arcquant_sparse", "cache_dir"}
    if {key: value for key, value in configs[0].items() if key not in ignored} != {
            key: value for key, value in configs[1].items() if key not in ignored}:
        raise ValueError("Zero equivalence requires the same model, prompt, runtime and protocol")
    baseline_jobs = (baseline / "generation_jobs.jsonl").read_bytes()
    if baseline_jobs != (sparse / "generation_jobs.jsonl").read_bytes():
        raise ValueError("Zero equivalence prompts/seeds differ")
    videos = [sorted((folder / "_cache").glob("*.mp4")) for folder in (baseline, sparse)]
    if not videos[0] or len(videos[0]) != len(videos[1]):
        raise ValueError("Zero equivalence videos missing")
    results = []
    for original, masked in zip(*videos):
        original_hash, masked_hash = decoded_video_hash(original), decoded_video_hash(masked)
        if original.name != masked.name or original_hash != masked_hash:
            raise ValueError("Zero sparsity differs from baseline decoded video")
        results.append(dict(video=original.name, decoded_sha256=original_hash))
    return dict(complete=True, baseline=str(baseline), sparse=str(sparse), videos=results)


def validate_tiny_scores(folder):
    folder = Path(folder)
    generation = json.loads((folder / "gen_summary.json").read_text())
    scores = json.loads((folder / "vbench_summary.json").read_text())
    config = generation["generation_config"]
    if generation.get("protocol") != "tiny16" or generation.get("num_videos") != 740 or generation.get("num_prompts") != 108:
        raise ValueError("Not a complete MAGI Tiny generation suite")
    if len(set(generation.get("dims", []))) != 16 or set(scores.get("scores", {})) != set(generation["dims"]):
        raise ValueError("MAGI Tiny requires all sixteen scored dimensions")
    if generation.get("subset_manifest_sha256") != TINY_MANIFEST or scores.get("subset_manifest_sha256") != TINY_MANIFEST:
        raise ValueError("MAGI Tiny manifest mismatch")
    if config.get("preview_steps") != 50 or config.get("refiner_steps") != 5 or config.get("resolution") != "1080p" or config.get("prompt_enhancement") is not False:
        raise ValueError("MAGI Tiny protocol differs from the 50+5 baseline")
    coverage = scores.get("temporal_flickering_coverage", {})
    if coverage.get("input_prompts") != 10 or coverage.get("input_candidates") != 250 or coverage.get("missing_candidate_videos"):
        raise ValueError("Incomplete flickering candidate coverage")
    if coverage.get("passed_prompts") != 10 or coverage.get("passed_videos") != 50 or coverage.get("missing_static_prompts"):
        raise ValueError("Incomplete flickering static selection coverage")
    totals = scores.get("tiny_scores", {})
    if any(not isinstance(totals.get(key), (int, float)) or not math.isfinite(totals[key]) for key in ("total", "quality", "semantic")):
        raise ValueError("Missing/invalid Tiny aggregate scores")
    return {key: totals[key] for key in ("total", "quality", "semantic")}
