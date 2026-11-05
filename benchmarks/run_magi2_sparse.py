"""Run pinned MAGI generation with auditable WINA/TEAL video adaptations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = ("benchmarks/run_magi2_sparse.py", "model/magi2_sparse_runtime.py",
           "model/magi2_sparse_moe.py", "model/magi2_calibration.py", "model/sparse_utils.py")


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def calibration_identity(config, prompts_hash):
    keys = ("checkpoint_revision", "checkpoint_manifest_sha256", "checkpoint_files",
            "source_revision", "source_sha256", "runtime_versions", "resolution", "seconds",
            "preview_steps", "refiner_steps", "cp_size", "ep_size", "quant", "mode", "seed",
            "prompt_enhancement", "expert_backend", "moe_execution")
    return {**{key: config[key] for key in keys}, "sparse_sources": source_hashes(),
            "calibration_prompts_sha256": prompts_hash,
            "forward_bucket_width": 10, "disable_torch_compile": True}


def validate_prompts(path, reference):
    data = json.loads(path.read_text())
    prompts = data["prompts"]
    if not prompts or len(prompts) != len(set(prompts)) or any(not isinstance(item, str) or not item.strip() for item in prompts):
        raise ValueError("Calibration prompts must be distinct nonempty strings")
    evaluation = reference.load_prompts(reference.STANDARD_DIMS, -1, "standard16")
    overlaps = set(prompts) & {prompt for values in evaluation.values() for prompt in values}
    if overlaps:
        raise ValueError("Calibration prompts overlap the VBench evaluation set")
    return prompts


def main(argv=None):
    initial_sources = source_hashes()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sparse-method", choices=("wina", "teal", "calibrate"), required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--base-eval-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-prompts", type=Path, default=ROOT / "configs/magi_calibration_prompts.json")
    parser.add_argument("--norm-cache", type=Path)
    sparse, remaining = parser.parse_known_args(argv)
    if not 0 <= sparse.sparsity < 1:
        parser.error("--sparsity must be in [0, 1)")
    if "--out-dir" not in remaining:
        parser.error("Sparse/calibration runs require an independent --out-dir")
    if sparse.sparse_method == "calibrate" and sparse.sparsity:
        parser.error("Calibration must use zero sparsity")
    if sparse.sparse_method == "teal" and sparse.sparsity and sparse.calibration is None:
        parser.error("TEAL requires --calibration; no fallback is permitted")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("NCCL_ALGO", "^NVLS")
    sys.path.insert(0, str(sparse.base_eval_root.expanduser().resolve()))
    import torch
    import magi2_quant
    import magi2_runtime
    import gen_vbench_magi2 as reference
    from model.magi2_calibration import ActivationRecorder, VideoCalibration, digest
    from model.magi2_sparse_runtime import Magi2SparseController, install_sparse, prepare_norms

    native_args = reference.parse_args(remaining)
    if native_args.quant not in ("hifp4", "bf16"):
        parser.error("MAGI sparse experiments only support HiF4 or BF16")
    if sparse.sparse_method == "calibrate" and native_args.protocol != "custom10":
        parser.error("Calibration is a disjoint custom run, never tiny16/standard16")
    slot = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    os.environ[f"GIT_CONFIG_KEY_{slot}"] = "safe.directory"
    os.environ[f"GIT_CONFIG_VALUE_{slot}"] = str(native_args.repo.expanduser().resolve())
    os.environ["GIT_CONFIG_COUNT"] = str(slot + 1)
    out = native_args.out_dir.expanduser().resolve()
    rank = int(os.environ.get("RANK", "0"))
    controllers, context, timings = [], {}, []
    recorder = ActivationRecorder() if sparse.sparse_method == "calibrate" else None
    prompts_hash = hashlib.sha256(sparse.calibration_prompts.read_bytes()).hexdigest()
    calibration_hash = hashlib.sha256(sparse.calibration.read_bytes()).hexdigest() if sparse.calibration else None
    original_apply, original_prepare = magi2_quant.apply_magi2_quant, reference.prepare
    original_initialize, original_prompts = magi2_runtime.initialize_pipeline, reference.load_prompts

    def prepare(args):
        result = original_prepare(args)
        if recorder is not None and not args.dry_run and any(result[1].glob("*.mp4")):
            raise ValueError("Calibration needs a fresh video cache; cached videos cannot recreate missing activations")
        config = result[3]
        identity = calibration_identity(config, prompts_hash)
        context["calibration_identity"] = identity
        if sparse.calibration:
            context["calibration"] = VideoCalibration.load(sparse.calibration, expected=identity)
        config["arcquant_sparse"] = dict(method=sparse.sparse_method, sparsity=sparse.sparsity,
            variant="magi-projection-shared-mask-v1", sources=source_hashes(),
            calibration_sha256=calibration_hash, norm_source="original-pre-quantization",
            calibration_prompts_sha256=prompts_hash if recorder else None,
            disable_torch_compile=True, role="calibration" if recorder else "evaluation")
        return result

    def flush(complete=False, error=None):
        if native_args.dry_run or not controllers:
            return
        out.mkdir(parents=True, exist_ok=True)
        reports = {controller.stage: controller.snapshot() for controller in controllers}
        report = dict(schema="magi-sparse-runtime-v1", complete=complete, error=error,
                      rank=rank, hostname=os.uname().nodename, method=sparse.sparse_method,
                      sparsity=sparse.sparsity, sources=initial_sources, stages=reports,
                      videos=timings, calibration_sha256=calibration_hash)
        reference.atomic_write_json(out / f"sparse_runtime_rank{rank}.json", report)
        if recorder is not None:
            temporary = out / f"calibration-rank{rank}.tmp"
            torch.save(dict(**recorder.state(), complete=complete,
                            identity=context["calibration_identity"]), temporary)
            temporary.replace(out / f"calibration-rank{rank}.pt")

    def apply(model, quant, act_quant=True, device="cpu"):
        stage = "preview" if not controllers else "refiner"
        if len(controllers) >= 2:
            raise ValueError("Unexpected extra MAGI model; explicit stage mapping required")
        norms = {}
        if sparse.sparse_method == "wina":
            identity = dict(context["calibration_identity"], stage=stage, rank=rank)
            cache = (sparse.norm_cache or out.parent / "original_norms") / f"{digest(identity)}.pt"
            if cache.exists():
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                if saved["identity"] != identity:
                    raise ValueError("Original norm cache identity mismatch")
                norms = saved["norms"]
            else:
                norms = prepare_norms(model, device=device)
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(f".{os.getpid()}.tmp")
                torch.save(dict(identity=identity, norms=norms), temporary)
                temporary.replace(cache)
        controller = Magi2SparseController(sparse.sparsity, method=sparse.sparse_method,
            norms=norms, calibration=context.get("calibration"), recorder=recorder, stage=stage)
        report = original_apply(model, quant, act_quant, device)
        install_sparse(model, controller)
        expected = set(report["dense"] + report["grouped"] + report["routed"])
        if set(controller.coverage) != expected:
            raise ValueError("Sparse projection coverage differs from pinned quantization coverage")
        controllers.append(controller)
        report["arcquant_sparse_installation"] = controller.snapshot()
        return report

    def initialize(args, timer):
        pipeline, reports, text_stats = original_initialize(args, timer)
        original_run = pipeline.run_offline

        def run_offline(*args, **kwargs):
            for controller in controllers:
                controller.begin_video()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            result = original_run(*args, **kwargs)
            torch.cuda.synchronize()
            timings.append(dict(seconds=time.monotonic() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                prompt_sha256=hashlib.sha256(kwargs["prompt"].encode()).hexdigest()))
            flush()
            if any(controller.stats["nonfinite"] or controller.snapshot()["output_nonfinite"] for controller in controllers):
                raise ValueError("Nonfinite activations detected; refusing completed sparse video")
            return result

        pipeline.run_offline = run_offline
        return pipeline, reports, text_stats

    try:
        if recorder is not None:
            prompts = validate_prompts(sparse.calibration_prompts, reference)
            reference.load_prompts = lambda dims, limit, protocol: {"subject_consistency": prompts}
        reference.prepare = prepare
        magi2_quant.apply_magi2_quant = apply
        magi2_runtime.initialize_pipeline = initialize
        result = reference.main(remaining)
        if source_hashes() != initial_sources:
            raise ValueError("MAGI sparse implementation changed during execution")
        if recorder is not None and not native_args.dry_run and not controllers:
            raise ValueError("Calibration cannot reuse videos without runtime activation shards")
        if not native_args.dry_run and controllers and not (out / "gen_summary.json").is_file():
            raise ValueError("Generation remains partial; sparse/calibration runtime is not complete")
        flush(complete=True)
        return int(result or 0)
    except BaseException as exc:
        flush(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        magi2_quant.apply_magi2_quant = original_apply
        magi2_runtime.initialize_pipeline = original_initialize
        reference.prepare, reference.load_prompts = original_prepare, original_prompts


if __name__ == "__main__":
    raise SystemExit(main())
