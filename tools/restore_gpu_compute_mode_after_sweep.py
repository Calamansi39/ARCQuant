#!/usr/bin/env python3
"""Restore GPU compute mode after a bounded ARCQuant sweep is terminal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def terminal(sweep: Path) -> bool:
    claims = sorted((sweep / "claims").glob("*/state.json"))
    if len(claims) != 4:
        return False
    states = [json.loads(path.read_text()) for path in claims]
    return all(state.get("phase") in ("completed", "failed") for state in states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    gpus = [item.strip() for item in args.gpus.split(",")]
    if not gpus or not all(item.isdecimal() for item in gpus):
        parser.error("--gpus must be numeric indices")
    log = args.sweep / "compute-mode-restore.json"
    args.sweep.mkdir(parents=True, exist_ok=True)
    while not terminal(args.sweep):
        log.write_text(json.dumps(dict(schema="arcquant-compute-mode-restore-v1", phase="waiting",
                                       gpus=gpus, checked_at=time.time()), indent=2) + "\n")
        time.sleep(args.interval)
    result = subprocess.run(["nvidia-smi", "-i", ",".join(gpus), "-c", "DEFAULT"],
                            text=True, capture_output=True)
    log.write_text(json.dumps(dict(schema="arcquant-compute-mode-restore-v1",
                                   phase="restored" if result.returncode == 0 else "restore_failed",
                                   gpus=gpus, return_code=result.returncode,
                                   stdout=result.stdout, stderr=result.stderr,
                                   restored_at=time.time()), indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
