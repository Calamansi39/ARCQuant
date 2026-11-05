"""Run the MATH-500 sparse sweep under one continuous, advisory GPU lease.

The lease coordinates ARCQuant launchers on this host. It does not reserve GPUs
against unrelated users or keep CUDA memory allocated between model processes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.gpu_wait import idle_gpus
from benchmarks.queue_sparse_baselines import main as run_queue


@contextmanager
def hold_pair(ids, min_free_mib, poll_seconds, report, lock_root=None):
    """Keep both locks through all methods and stages, releasing on every exit."""
    root = lock_root or Path('/tmp') / f'arcquant-gpu-locks-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    handles = []
    try:
        while True:
            for index in sorted(ids, key=int):
                handle = (root / f'gpu{index}.lock').open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    break
                handles.append(handle)
            if len(handles) == 2:
                break
            for handle in handles:
                handle.close()
            handles.clear()
            report('waiting_for_gpu_locks')
            time.sleep(poll_seconds)
        while True:
            if set(idle_gpus(min_free_mib, ids)) == set(ids):
                # Recheck after the previous worker has fully released CUDA.
                time.sleep(2)
                if set(idle_gpus(min_free_mib, ids)) == set(ids):
                    break
            report('waiting_for_gpu_memory')
            time.sleep(poll_seconds)
        report('lease_acquired')
        yield ','.join(ids)
    finally:
        for handle in handles:
            handle.close()


def source_hash():
    paths = ['benchmarks/queue_sparse_sweep.py', 'benchmarks/queue_sparse_baselines.py',
             'benchmarks/run_deepseek_sparse_math.py',
             'model/deepseek_generation_runtime.py', 'model/sparse_utils.py']
    return hashlib.sha256(b''.join((ROOT / p).read_bytes() for p in paths)).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-eval-root', 'source-model', 'converted-model', 'calibration', 'out-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--gpu-ids', required=True, help='exactly two physical indices, e.g. 0,1')
    p.add_argument('--sparsities', default='0.5,0.6,0.7,0.8,0.9')
    p.add_argument('--methods', default='teal,wina')
    p.add_argument('--skip', action='append', default=[], metavar='METHOD:SPARSITY',
                   help='omit an already running/completed combination; repeatable')
    p.add_argument('--min-free-mib', type=int, default=138000)
    p.add_argument('--poll-seconds', type=float, default=2)
    p.add_argument('--dry-run', action='store_true', help='native CPU preflight, without GPU locks')
    a = p.parse_args(argv)
    ids = [s.strip() for s in a.gpu_ids.split(',')]
    if len(ids) != 2 or len(set(ids)) != 2 or not all(s.isdecimal() for s in ids):
        p.error('--gpu-ids must contain two distinct physical indices')
    if a.min_free_mib <= 0 or a.poll_seconds <= 0:
        p.error('GPU waiting thresholds must be positive')
    methods = list(dict.fromkeys(s.strip() for s in a.methods.split(',') if s.strip()))
    if not methods or any(s not in ('teal', 'wina') for s in methods):
        p.error('--methods must contain teal and/or wina')
    try:
        sparsities = list(dict.fromkeys(float(s.strip()) for s in a.sparsities.split(',')))
        skip = {(m, float(s)) for m, s in (item.split(':') for item in a.skip)}
    except ValueError:
        p.error('invalid --sparsities or --skip (expected METHOD:SPARSITY)')
    if not sparsities or any(not 0 <= s < 1 for s in sparsities):
        p.error('sparsities must be in [0, 1)')
    if any(m not in methods or s not in sparsities for m, s in skip):
        p.error('--skip must refer to a requested method and sparsity')
    for key in ('base_eval_root', 'source_model', 'converted_model', 'calibration', 'out_dir'):
        setattr(a, key, getattr(a, key).expanduser().resolve())
    for path in (a.base_eval_root, a.source_model, a.converted_model,
                 a.calibration / 'rank0.pt', a.calibration / 'rank1.pt'):
        if not path.exists():
            p.error(f'missing asset: {path}')
    jobs = [{'method': m, 'sparsity': s, 'status': 'pending'}
            for s in sparsities for m in methods if (m, s) not in skip]
    if not jobs:
        p.error('no combinations remain after --skip')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / 'sweep.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (a.out_dir / 'status.json').exists():
            p.error('use a fresh --out-dir; existing sweep results are retained')
        code = source_hash()
        state = dict(pid=os.getpid(), hostname=os.uname().nodename,
                     gpu_ids=','.join(ids), lease_scope='local advisory locks only',
                     code_sha256=code, jobs=jobs, skipped=a.skip, dry_run=a.dry_run)

        def save(status=None):
            if status:
                state['state'] = status
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            tmp = a.out_dir / 'status.tmp'
            tmp.write_text(json.dumps(state, indent=2) + '\n')
            tmp.replace(a.out_dir / 'status.json')

        def stop(signum, frame):
            raise KeyboardInterrupt(f'received signal {signum}')

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        save('preflight' if a.dry_run else 'waiting_for_gpu_locks')
        print(f'[sweep] {len(jobs)} combinations on GPUs {state["gpu_ids"]}', flush=True)
        try:
            lease = nullcontext(','.join(ids)) if a.dry_run else hold_pair(
                ids, a.min_free_mib, a.poll_seconds, save)
            with lease as gpu_ids:
                for job in jobs:
                    if source_hash() != code:
                        raise RuntimeError('source changed; restart into a fresh output directory')
                    name = f'{job["method"]}-{job["sparsity"]:g}'
                    destination = a.out_dir / name
                    job.update(status='running', output_dir=str(destination))
                    save('preflight' if a.dry_run else 'running')
                    args = ['--out-dir', str(destination), '--python', a.python,
                            '--sparsity', str(job['sparsity']), '--methods', job['method'], '--full']
                    for key in ('base_eval_root', 'source_model', 'converted_model', 'calibration'):
                        args += ['--' + key.replace('_', '-'), str(getattr(a, key))]
                    if a.dry_run:
                        args.append('--dry-run')
                    # Inner stages borrow the outer lease; they never release/reacquire it.
                    rc = run_queue(args, gpu_lease=lambda **_: nullcontext(gpu_ids))
                    job.update(status=('preflight_passed' if a.dry_run else 'completed')
                               if rc == 0 else 'needs_debug', exit_code=rc)
                    save()
                    print(f'[sweep] {name}: {job["status"]}', flush=True)
            failed = any(j['status'] == 'needs_debug' for j in jobs)
            save('finished_with_failures' if failed else 'completed')
            return int(failed)
        except KeyboardInterrupt as exc:
            state['error'] = str(exc)
            save('interrupted')
            return 130
        except Exception as exc:
            state['error'] = str(exc)
            save('blocked')
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
