#!/usr/bin/env python3
"""Run one job while holding ARCQuant's cooperating GPU advisory locks.

This prevents another ARCQuant queue on the same host from selecting the same
GPU. It cannot block unrelated users or an administrator from launching work.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    p.add_argument("--reservation", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    gpus = [x.strip() for x in args.gpus.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(x.isdecimal() for x in gpus):
        p.error("--gpus must contain two distinct numeric indices")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        p.error("a command is required after --")
    lock_root = Path("/tmp") / f"arcquant-gpu-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    handles = []
    try:
        for index in sorted(gpus):
            handle = (lock_root / f"gpu{index}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise SystemExit(f"GPU {index} already has a cooperating ARCQuant job")
            handles.append(handle)
        args.reservation.parent.mkdir(parents=True, exist_ok=True)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        record = dict(schema="arcquant-gpu-reservation-v1", pid=os.getpid(), gpus=gpus,
                      hostname=__import__("socket").gethostname(), command=args.command,
                      started_at=time.time(), phase="running")
        args.reservation.write_text(json.dumps(record, indent=2) + "\n")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus),
                   HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        with args.log.open("w") as stream:
            process = subprocess.Popen(args.command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            record["child_pid"] = process.pid
            args.reservation.write_text(json.dumps(record, indent=2) + "\n")
            return_code = process.wait()
        record.update(phase="completed" if return_code == 0 else "failed",
                      return_code=return_code, finished_at=time.time())
        args.reservation.write_text(json.dumps(record, indent=2) + "\n")
        return return_code
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
