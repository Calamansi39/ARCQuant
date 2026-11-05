"""Claim and run DeepSeek ARCQuant tiny cross-evaluations on idle local TP2 pairs.

The worker is intentionally a scheduler, not a new evaluator. It waits for a
complete greedy strategy and calibration manifest, verifies strategy identity,
then runs the existing ``run_deepseek_greedy_eval.py`` under torchrun. Multiple
hosts may run this file against the shared output directory; atomic claim dirs
ensure each task is owned once. A task is never marked complete unless the
external summary says the selected cases are complete and the protocol records
ARC HiF4 W4A4 plus the requested method/rate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import SCHEMA, atomic_json
from benchmarks.gpu_wait import idle_gpus

DEFAULT_RATES = (0.5, 0.6, 0.7, 0.8, 0.9)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ready(root, method):
    root = Path(root)
    strategy = root / method / 'strategy.json'
    calibration = root / 'calibration' / 'manifest.json'
    manifest = root / 'search_manifest.json'
    if not all(p.is_file() for p in (strategy, calibration, manifest)):
        return None
    s, c, m = read(strategy), read(calibration), read(manifest)
    if (s.get('schema') != SCHEMA or s.get('method') != method or not s.get('complete')
        or s.get('quantization') != 'arc' or s.get('transform') is not False):
        return None
    if (not c.get('complete') or c.get('schema') != SCHEMA or c.get('quantization') != 'arc'
        or c.get('world_size') != 2):
        return None
    for key in ('source_signature', 'world_size', 'quantization', 'samples', 'length',
                'data_sha256', 'implementation_sha256', 'calibration_rank_sha256'):
        if s.get(key) != c.get(key):
            return None
    if m.get('methods') != [method] or m.get('quantization') != 'arc':
        return None
    if any(abs(float(p.get('target', -1)) - rate) > 1e-8 for p in s.get('profiles', []) for rate in []):
        return None
    return dict(root=str(root.resolve()), method=method,
                strategy=strategy.resolve(), calibration=str((root/'calibration').resolve()),
                strategy_sha256=sha(strategy), calibration_manifest_sha256=sha(calibration),
                search_manifest_sha256=sha(manifest), targets=s.get('profiles', []),
                ready_at=now())


def choose(candidates, method):
    # A race directory is allowed, but select the first complete artifact in
    # the explicit order. The scheduler never merges strategies from directories.
    for candidate in candidates:
        result = ready(candidate, method)
        if result:
            return result
    return None


def make_tasks(out, methods, benchmarks, limits, rates):
    tasks = []
    for method in methods:
        for rate in rates:
            for benchmark in benchmarks:
                tasks.append(dict(id=f'{method}-{rate:g}-{benchmark}', method=method,
                                  sparsity=rate, benchmark=benchmark, limit=limits[benchmark],
                                  output=str((out / f'{method}-{rate:g}-{benchmark}').resolve())))
    return tasks


def claim(task_dir, task, host):
    task_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        task_dir.mkdir()
    except FileExistsError:
        return False
    atomic_json(task_dir/'owner.json', dict(task=task, hostname=host, pid=os.getpid(), claimed_at=now()))
    return True


def summary_complete(path, task):
    p = Path(path) / 'summary.json'
    if not p.is_file():
        return False
    try:
        s = read(p)
    except Exception:
        return False
    return (s.get('complete') is True and s.get('completed_candidates') == task['limit']
            and s.get('questions') == task['limit'] and s.get('is_smoke_test') is True)


def load_task_state(task_dir):
    p = task_dir/'state.json'
    return read(p) if p.is_file() else {}


def save_state(task_dir, **data):
    previous = load_task_state(task_dir)
    previous.update(data, updated_at=now())
    atomic_json(task_dir/'state.json', previous)


def run_task(task, task_dir, strategy, args, gpu_ids):
    output = Path(task['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
        str(ROOT/'benchmarks/run_deepseek_greedy_eval.py'), '--strategy', strategy['strategy'],
        '--sparse-method', task['method'], '--sparsity', str(task['sparsity']),
        '--quantization', 'arc', '--base-eval-root', str(args.base_eval_root),
        '--source-model', str(args.source_model), '--converted-model', str(args.converted_model),
        '--calibration', strategy['calibration'], '--benchmark', task['benchmark'],
        '--model', 'deepseek', '--quant', 'hifp4', '--mode', 'w4a4', '--limit', str(task['limit']),
        '--max-new-tokens', str(args.max_new_tokens), '--out-dir', str(output)]
    log = task_dir / f'attempt-{load_task_state(task_dir).get("attempt", 0)}.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpu_ids), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    save_state(task_dir, phase='running', command=command, gpu_ids=list(gpu_ids),
               strategy=strategy, output=str(output), attempt=load_task_state(task_dir).get('attempt', 0),
               started_at=now())
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        save_state(task_dir, pid=process.pid)
        return_code = process.wait()
    complete = return_code == 0 and summary_complete(output, task)
    save_state(task_dir, phase='completed' if complete else 'failed', return_code=return_code,
               summary_complete=complete, finished_at=now())
    return complete


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--strategy-root', action='append', nargs=2, metavar=('METHOD', 'ROOT'), required=True,
                   help='candidate strategy root; repeat for wina/teal and race/primary, in preference order')
    p.add_argument('--base-eval-root', type=Path, required=True)
    p.add_argument('--source-model', type=Path, required=True)
    p.add_argument('--converted-model', type=Path, required=True)
    p.add_argument('--gpu-candidates', default='0,1,2,3,4,5,6,7')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--host-label', default=socket.gethostname() if 'socket' in globals() else os.uname().nodename)
    p.add_argument('--benchmarks', default='math500,gpqa')
    p.add_argument('--math-limit', type=int, default=100)
    p.add_argument('--gpqa-limit', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--retry', type=int, default=1)
    return p.parse_args()


def main():
    args = parse()
    import socket
    args.host_label = args.host_label or socket.gethostname()
    methods = ('wina', 'teal')
    benchmarks = tuple(v for v in args.benchmarks.split(',') if v in ('math500', 'gpqa'))
    if not benchmarks:
        raise SystemExit('no supported benchmark')
    limits = {'math500': args.math_limit, 'gpqa': args.gpqa_limit}
    if any(v < 1 for v in limits.values()):
        raise SystemExit('limits must be positive')
    candidates = {method: [] for method in methods}
    for method, root in args.strategy_root:
        if method not in candidates:
            raise SystemExit(f'unknown method {method}')
        candidates[method].append(Path(root))
    args.out_dir = args.out_dir.resolve(); args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(args.out_dir, methods, benchmarks, limits, DEFAULT_RATES)
    atomic_json(args.out_dir/'sweep_manifest.json', dict(schema='deepseek-tiny-sweep-v1', created_at=now(),
        host=args.host_label, benchmarks=benchmarks, limits=limits, rates=DEFAULT_RATES,
        tasks=tasks, protocol='ARC HiF4 W4A4 + native greedy strategy; generated-answer scoring'))
    gpu_candidates = [v.strip() for v in args.gpu_candidates.split(',') if v.strip().isdigit()]
    if len(set(gpu_candidates)) < 2:
        raise SystemExit('need at least two candidate GPUs')
    heartbeat = args.out_dir / f'heartbeat-{args.host_label}.json'
    while True:
        selected = {method: choose(candidates[method], method) for method in methods}
        atomic_json(heartbeat, dict(host=args.host_label, phase='waiting_for_complete_greedy',
            strategies=selected, updated_at=now()))
        if all(selected.values()):
            break
        time.sleep(args.poll_seconds)
    tasks_pending = tasks[:]
    while tasks_pending:
        atomic_json(heartbeat, dict(host=args.host_label, phase='scheduling',
            remaining=[t['id'] for t in tasks_pending], updated_at=now()))
        available = idle_gpus(138000, gpu_candidates)
        if len(available) < 2:
            time.sleep(args.poll_seconds); continue
        pair = available[:2]
        task = None
        for candidate in tasks_pending:
            task_dir = args.out_dir / 'claims' / candidate['id']
            if summary_complete(candidate['output'], candidate):
                tasks_pending.remove(candidate); continue
            state = load_task_state(task_dir)
            if state.get('phase') == 'completed':
                tasks_pending.remove(candidate); continue
            if state.get('phase') == 'running':
                continue
            if state.get('attempt', 0) > args.retry:
                tasks_pending.remove(candidate); continue
            if claim(task_dir, candidate, args.host_label):
                task = candidate; break
        if task is None:
            if not tasks_pending: break
            time.sleep(args.poll_seconds); continue
        task_dir = args.out_dir / 'claims' / task['id']
        state = load_task_state(task_dir)
        save_state(task_dir, attempt=state.get('attempt', 0) + 1)
        try:
            complete = run_task(task, task_dir, selected[task['method']], args, pair)
        except Exception as exc:
            save_state(task_dir, phase='failed', error=repr(exc), finished_at=now())
            complete = False
        if not complete:
            # Keep the claim directory and state for audit; a retry requires a
            # new claim marker and is intentionally bounded by --retry.
            failed = task_dir.with_name(task_dir.name + f'.retry{load_task_state(task_dir).get("attempt", 1)}')
            try:
                task_dir.rename(failed)
            except FileNotFoundError:
                pass
        else:
            tasks_pending.remove(task)
    atomic_json(heartbeat, dict(host=args.host_label, phase='completed', updated_at=now()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
