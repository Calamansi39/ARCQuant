"""Diagnostic MAGI HiF4/uniform validation; not the ARC/greedy campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import queue_video
from benchmarks.gpu_wait import wait_for_idle_gpus
from benchmarks.magi_validation import SOURCES, validate_sparse_runtime, validate_zero_equivalence

GRID = (.5, .6, .7, .8, .9)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--baseline", type=Path, required=True, help="Independent completed/pending baseline queue directory")
    result.add_argument("--gpu-ids", default="5,6")
    result.add_argument("--poll-seconds", type=float, default=30)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--tiny-after-validation", action="store_true",
                        help="Rejected: formal Tiny requires ARC compensation and learned greedy strategies")
    return result


def queue_args(args, method, sparsity, profile="smoke", calibration=None, zero=None):
    values = ["--model", "magi-2-preview", "--method", method, "--sparsity", str(sparsity),
              "--quant", "hifp4", "--steps", "50", "--refiner-steps", "5",
              "--profile", profile, "--gpu-ids", args.gpu_ids,
              "--out-dir", str(args.out_dir / f"{method}-s{sparsity:g}-{profile}")]
    if sparsity == 0:
        values += ["--stage", "generate"]
    if calibration:
        values += ["--calibration", str(calibration)]
    if zero:
        values += ["--magi-zero-evidence", str(zero)]
    return queue_video.parser().parse_args(values)


def run(args):
    if args.tiny_after_validation:
        raise ValueError("Uniform/no-ARC diagnostics cannot launch formal Tiny suites; "
                         "implement ARC compensation and learned greedy strategies first")
    args.out_dir = args.out_dir.expanduser().resolve()
    args.baseline = args.baseline.expanduser().resolve()
    if args.poll_seconds <= 0:
        raise ValueError("Polling interval must be positive")
    catalog = json.loads((ROOT / "configs/experiments.json").read_text())
    initial = queue_video.build_plan(queue_args(args, "wina", 0), catalog)
    if not args.execute:
        print(json.dumps(dict(baseline=str(args.baseline), output=str(args.out_dir),
            gpu_ids=args.gpu_ids, phases=["baseline", "wina-zero-equivalence", "wina-50-smoke",
            "disjoint-4-video-calibration", "teal-50-smoke", "remaining-8-smokes",
            "stop-after-validation"], allocation="uniform", arc_compensation=False,
            planned_tiny_videos=0), indent=2))
        return 0
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = (*SOURCES, "benchmarks/magi_campaign.py", "benchmarks/magi_validation.py",
             "benchmarks/queue_video.py", "configs/magi_calibration_prompts.json")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    identity = dict(sources=hashes, baseline=str(args.baseline), gpu_ids=args.gpu_ids,
                    tiny_after_validation=args.tiny_after_validation, initial=initial["identity"])
    with (out / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = out / "campaign_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Campaign code/config changed; use a new output directory")
        queue_video.atomic_json(identity_path, identity)
        status = dict(pid=os.getpid(), hostname=os.uname().nodename, completed=[],
                      baseline=str(args.baseline), gpu_ids=args.gpu_ids)

        def save(state, **extra):
            status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
            queue_video.atomic_json(out / "campaign_status.json", status)

        def unchanged():
            if hashes != {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}:
                raise ValueError("Campaign implementation changed; refusing mixed-code results")

        def execute_point(method, sparsity, profile="smoke", calibration=None, zero=None):
            unchanged()
            options = queue_args(args, method, sparsity, profile, calibration, zero)
            name = f"{method}-s{sparsity:g}-{profile}"
            save("running", phase=name)
            plan = queue_video.build_plan(options, catalog)
            queue_video.execute(options, plan, catalog)
            status["completed"].append(name)
            save("between_stages")
            return Path(plan["output"]) / profile

        try:
            save("waiting_for_baseline", phase="baseline")
            while True:
                unchanged()
                baseline_status = args.baseline / "status.json"
                state = json.loads(baseline_status.read_text()) if baseline_status.exists() else {}
                if state.get("state") == "failed":
                    raise ValueError(f"Baseline failed: {state.get('error')}")
                if state.get("state") == "completed":
                    break
                time.sleep(args.poll_seconds)
            zero_folder = execute_point("wina", 0.)
            evidence = validate_zero_equivalence(args.baseline / "smoke", zero_folder)
            zero_file = out / "zero_equivalence.json"
            queue_video.atomic_json(zero_file, evidence)
            execute_point("wina", .5)
            save("calibrating", phase="disjoint-4-video-calibration")
            options = queue_args(args, "wina", 0.)
            plan = queue_video.build_plan(options, catalog)
            phase = plan["phases"][0]
            command = list(phase["command"])
            command[command.index("--sparse-method") + 1] = "calibrate"
            calibration_out = out / "calibration"
            command[command.index("--out-dir") + 1] = str(calibration_out)
            command[command.index("--tag") + 1] = "magi-disjoint-calibration-50p5-v1"
            artifact = out / "teal_calibration.pt"
            env = {**os.environ, **plan["env"]}
            calibration_complete = all((calibration_out / f"calibration-rank{rank}.pt").exists() for rank in range(2))
            if not artifact.exists():
                if not calibration_complete:
                    with wait_for_idle_gpus(2, 142000, args.poll_seconds, candidates=args.gpu_ids.split(",")) as devices:
                        unchanged()
                        queue_video.run_child(command, {**env, "CUDA_VISIBLE_DEVICES": devices}, plan["cwd"], out / "calibration.log")
                queue_video.run_child([plan["identity"]["generator_python"], str(ROOT / "benchmarks/calibrate_magi2.py"),
                    "--input", str(calibration_out), "--output", str(artifact)],
                    {**env, "CUDA_VISIBLE_DEVICES": ""}, str(ROOT), out / "calibration-merge.log")
            execute_point("teal", .5, calibration=artifact)
            for sparsity in GRID[1:]:
                for method in ("wina", "teal"):
                    execute_point(method, sparsity, calibration=artifact if method == "teal" else None)
            validation = {}
            for method in ("wina", "teal"):
                for sparsity in GRID:
                    folder = out / f"{method}-s{sparsity:g}-smoke" / "smoke"
                    validate_sparse_runtime(folder, method, sparsity)
                    validation[f"{method}:{sparsity:g}"] = dict(folder=str(folder),
                        generation_sha256=hashlib.sha256((folder / "gen_summary.json").read_bytes()).hexdigest())
            queue_video.atomic_json(out / "validated_smokes.json", validation)
            save("completed", phase=None)
        except BaseException as exc:
            save("failed", error=f"{type(exc).__name__}: {exc}")
            raise
    return 0


def main(argv=None):
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
