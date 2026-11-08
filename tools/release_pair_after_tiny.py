#!/usr/bin/env python3
"""Reserve a TP2 pair for release after its current tiny evaluator finishes.

This deliberately terminates only the scheduler. The evaluator was launched in
its own session, so it continues to finish the current task. No next task can
be claimed on the pair after the scheduler is frozen.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from benchmarks.queue_deepseek_tiny_sweep import process_identity, summary_complete


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                    suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def proc(pid: int):
    return process_identity(pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-pid", type=int, required=True)
    parser.add_argument("--evaluator-pid", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", nargs=2, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    out = ROOT / "runs/deepseek_tiny_sweep_20261106"
    task_dir = out / "claims" / args.task
    state_path = task_dir / "state.json"
    state = json.loads(state_path.read_text())
    scheduler = proc(args.scheduler_pid)
    evaluator = proc(args.evaluator_pid)
    if not scheduler or not evaluator:
        raise SystemExit("scheduler or evaluator is not alive")
    scheduler_cmd = " ".join(scheduler["command"])
    evaluator_cmd = " ".join(evaluator["command"])
    if "queue_deepseek_tiny_sweep.py" not in scheduler_cmd or "--gpu-candidates" not in scheduler_cmd:
        raise SystemExit("scheduler identity mismatch")
    if f"--gpu-candidates {','.join(args.gpu)}" not in scheduler_cmd:
        raise SystemExit("scheduler is not the requested GPU pair")
    if "run_deepseek_greedy_eval.py" not in evaluator_cmd or args.task not in evaluator_cmd:
        raise SystemExit("evaluator identity mismatch")

    record = dict(schema="tiny-pair-release-v1", task=args.task, host=args.host,
                  gpu_ids=args.gpu, scheduler_pid=args.scheduler_pid,
                  evaluator_pid=args.evaluator_pid, phase="freezing_scheduler",
                  started_at=time.time(), scheduler_command=scheduler["command"],
                  evaluator_command=evaluator["command"])
    atomic_json(out / "release-teal-0.6-gpqa.json", record)
    os.kill(args.scheduler_pid, signal.SIGTERM)
    time.sleep(2)
    if proc(args.scheduler_pid):
        os.kill(args.scheduler_pid, signal.SIGKILL)
    record["phase"] = "waiting_for_current_evaluator"
    atomic_json(out / "release-teal-0.6-gpqa.json", record)

    while proc(args.evaluator_pid):
        time.sleep(args.poll_seconds)

    summaries = sorted((out / args.task).glob("arcquant-*/summary.json"))
    complete = False
    if summaries:
        try:
            task = dict(id=args.task, method="teal", sparsity=.6, benchmark="gpqa", limit=32)
            complete = summary_complete(out / args.task, task, max_new_tokens=8192)
        except (OSError, ValueError, KeyError, TypeError):
            complete = False
    state = json.loads(state_path.read_text()) if state_path.is_file() else state
    state.update(phase="completed" if complete else "failed",
                 return_code=0 if complete else None, summary_complete=complete,
                 finished_at=time.time(), release_reserved=True)
    atomic_json(state_path, state)
    record.update(phase="released" if complete else "evaluator_exited_unverified",
                  evaluator_exited_at=time.time(), summary_complete=complete)
    atomic_json(out / "release-teal-0.6-gpqa.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
