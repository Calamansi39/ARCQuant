#!/usr/bin/env python3
"""Run resumable dense evaluations from a shared plan on the first idle TP2 pair.

Each job has a shared file lock. Finished answers are resumed by the native
evaluator; a failed attempt retains all outputs and pauses for diagnosis.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json
from benchmarks.gpu_wait import wait_for_idle_gpus
from benchmarks.queue_sparse_baselines import arc_health
from benchmarks.verified_results import validate_answers


def result_directory(job):
    paths = list(Path(job['output']).glob('arcquant-*/protocol.json'))
    if len(paths) != 1:
        raise ValueError('Missing/ambiguous prepared protocol: ' + job['id'])
    return paths[0].parent


def complete(job):
    try:
        root = result_directory(job)
        summary, protocol, _, _ = validate_answers(root, job['expected'])
        return (summary['is_full_dataset'] and not summary['is_smoke_test']
                and protocol['arcquant']['method'] == 'none'
                and protocol['arcquant']['sparsity'] == 0 and arc_health(root)[0])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--worker', required=True)
    p.add_argument('--gpu-candidates', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    folder = a.plan.resolve().parent
    status_path = folder / f'worker-{a.worker}.json'
    state = dict(hostname=socket.gethostname(), pid=os.getpid(), phase='starting')
    mutex, done = threading.Lock(), threading.Event()
    child = None

    def save(phase=None, **extra):
        with mutex:
            if phase:
                state['phase'] = phase
            state.update(extra, updated_at=time.time())
            atomic_json(status_path, state)

    def heartbeat():
        while not done.wait(15):
            save()

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    threading.Thread(target=heartbeat, daemon=True).start()
    with (folder / f'worker-{a.worker}.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not all(complete(j) for j in plan['jobs']):
                save('waiting_for_idle_tp2', gpu_ids=[], job=None)
                with wait_for_idle_gpus(count=2, candidates=a.gpu_candidates.split(','),
                                       poll_seconds=10, settle_seconds=2) as ids:
                    # Select only after GPUs are available: an occupied host must
                    # not pin a job and block the other host from making progress.
                    with ExitStack() as resources:
                        selected = None
                        for job in plan['jobs']:
                            if complete(job):
                                continue
                            handle = (folder / f"job-{job['id']}.lock").open('a')
                            try:
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                handle.close()
                                continue
                            resources.callback(handle.close)
                            selected = job
                            break
                        if selected is None:
                            save('other_workers_running', gpu_ids=[])
                            continue
                        command = selected['command']
                        command = [command[0], '-m', 'torch.distributed.run', '--standalone',
                                   '--nproc-per-node=2', *command[1:]]
                        log = folder / f"{selected['id']}-{a.worker}.log"
                        save('running', job=selected['id'], gpu_ids=ids.split(','), log=str(log), command=command)
                        with log.open('a') as stream:
                            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=ids, HF_HUB_OFFLINE='1',
                                         HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4'))
                            save(child_pid=child.pid)
                            rc = child.wait()
                            child = None
                        if rc or not complete(selected):
                            raise RuntimeError(f"{selected['id']}: exit {rc}; outputs retained, see {log}")
                        save('job_completed', child_pid=None)
                time.sleep(2)
            save('completed', job=None, child_pid=None, gpu_ids=[])
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            save('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
            raise
        finally:
            done.set()


if __name__ == '__main__':
    main()
