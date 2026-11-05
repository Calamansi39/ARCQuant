"""Wait for idle GPUs; cooperating launchers use local advisory locks.

The locks are not cluster reservations and cannot block unrelated jobs.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import time


def idle_gpus(min_free_mib, candidates=None):
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    ids = []
    for line in query.stdout.splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) != 3 or not all(v.isdecimal() for v in fields):
            continue
        index, free, util = map(int, fields)
        if free < min_free_mib or util > 5 or (candidates is not None and str(index) not in candidates):
            continue
        apps = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
                               "--format=csv,noheader"], text=True, capture_output=True, check=True)
        if not apps.stdout.strip():
            ids.append(str(index))
    return ids


@contextmanager
def wait_for_idle_gpus(count=2, min_free_mib=138000, poll_seconds=60, settle_seconds=3,
                       candidates=None):
    """Lease a local GPU group; release every advisory lock on all exits.

    ``candidates`` contains physical nvidia-smi indices, not CUDA logical IDs.
    Existing DeepSeek queues use the same lock directory through the wrapper below.
    """
    if count < 1 or min_free_mib <= 0 or poll_seconds <= 0 or settle_seconds < 0:
        raise ValueError("Invalid GPU waiting thresholds")
    if candidates is not None:
        candidates = [str(i) for i in candidates]
        if len(set(candidates)) != len(candidates) or len(candidates) < count or not all(
                i.isdecimal() for i in candidates):
            raise ValueError("GPU candidates must be distinct indices, with at least count entries")
    locks = []
    # Keep these locks on local storage, not the shared network filesystem.
    root = Path("/tmp") / f"arcquant-gpu-locks-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    try:
        while True:
            ids = []
            for index in idle_gpus(min_free_mib, candidates):
                handle = (root / f"gpu{index}.lock").open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                locks.append(handle)
                ids.append(index)
                if len(ids) == count:
                    break
            if len(ids) == count:
                time.sleep(settle_seconds)
                if set(idle_gpus(min_free_mib, ids)) == set(ids):
                    print(f"[gpu-wait] starting on GPUs {','.join(ids)}", flush=True)
                    yield ",".join(ids)
                    return
            for handle in locks:
                handle.close()
            locks.clear()
            print(f"[gpu-wait] waiting for {count} idle GPUs with >= {min_free_mib} MiB free", flush=True)
            time.sleep(poll_seconds)
    finally:
        for handle in locks:
            handle.close()


def wait_for_idle_pair(min_free_mib=138000, poll_seconds=60, settle_seconds=3):
    """Backward compatible entry for the DeepSeek queues."""
    return wait_for_idle_gpus(2, min_free_mib, poll_seconds, settle_seconds)
