"""Run DeepSeek ARC tiny tasks on idle TP2 pairs, with resumable shared claims.

Run one detached worker per disjoint GPU pair. Shared task locks prevent duplicate
work across hosts; local GPU locks prevent overlapping cooperating workers.
Restarting a worker adopts its live local evaluator without restarting inference.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import SCHEMA
from benchmarks.gpu_wait import idle_gpus
from benchmarks.result_tables import verify_greedy_evidence

DEFAULT_RATES = (0.5, 0.6, 0.7, 0.8, 0.9)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    """Use unique temp files even when different hosts have identical PIDs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, indent=2)
            stream.write('\n')
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
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
    targets = [float(p.get('target', -1)) for p in s.get('profiles', [])]
    if any(not any(abs(target-rate) < 1e-8 for target in targets) for rate in DEFAULT_RATES):
        return None
    return dict(root=str(root.resolve()), method=method,
                strategy=str(strategy.resolve()), calibration=str((root/'calibration').resolve()),
                strategy_sha256=sha(strategy), calibration_manifest_sha256=sha(calibration),
                search_manifest_sha256=sha(manifest), targets=targets, ready_at=now())


def choose(candidates, method):
    for candidate in candidates:
        result = ready(candidate, method)
        if result:
            return result
    return None


def make_tasks(out, methods, benchmarks, limits, rates):
    # Complete both methods at each rate before advancing to the next rate.
    return [dict(id=f'{method}-{rate:g}-{benchmark}', method=method, sparsity=rate,
                 benchmark=benchmark, limit=limits[benchmark],
                 output=str((out / f'{method}-{rate:g}-{benchmark}').resolve()))
            for rate in rates for method in methods for benchmark in benchmarks]


def summary_complete(path, task, strategy=None, max_new_tokens=8192):
    """Verify nested evaluator output, coverage, quantization and greedy evidence."""
    root = Path(path)
    for p in [root/'summary.json', *sorted(root.glob('arcquant-*/summary.json'))]:
        try:
            s = read(p)
            if not (s.get('complete') is True and s.get('completed_candidates') == task['limit']
                    and s.get('questions') == task['limit'] and s.get('is_smoke_test') is True
                    and s.get('benchmark') == task['benchmark'] and s.get('quant') == 'hifp4'):
                continue
            directory = p.parent
            protocol = read(directory/'protocol.json')
            runtime = read(directory/'arcquant_sparse_runtime.json')
            if not (protocol.get('model') == 'deepseek' and protocol.get('quant') == 'hifp4'
                    and protocol.get('mode') == 'w4a4' and protocol.get('benchmark') == task['benchmark']
                    and protocol['generation']['max_new_tokens'] == max_new_tokens
                    and runtime.get('method') == task['method']
                    and abs(float(runtime.get('sparsity', -1))-task['sparsity']) < 1e-8
                    and runtime.get('arc') is True and runtime.get('quantization') == 'arc'
                    and runtime.get('allocation') == 'greedy'):
                continue
            identity = verify_greedy_evidence(directory, runtime)
            if strategy and any(identity[key] != strategy[key] for key in (
                    'strategy_sha256', 'calibration_manifest_sha256')):
                continue
            ids = protocol['selected_ids']
            rows = [json.loads(line) for line in (directory/'outputs.jsonl').read_text().splitlines() if line.strip()]
            if (len(ids) != task['limit'] or len(set(ids)) != task['limit']
                    or len(rows) != task['limit'] or {r['id'] for r in rows} != set(ids)
                    or any(r['sample_index'] != 0 or type(r['score']['correct']) is not bool
                           or r['score']['rules_version'] != s['scoring_rules_version'] for r in rows)):
                continue
            correct = sum(r['score']['correct'] for r in rows)
            if correct != s['correct_candidates'] or abs(correct/task['limit']-s['pass_at_1']) > 1e-8:
                continue
            return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def load_task_state(task_dir):
    p = task_dir/'state.json'
    return read(p) if p.is_file() else {}


def save_state(task_dir, **data):
    previous = load_task_state(task_dir)
    previous.update(data, updated_at=now())
    atomic_json(task_dir/'state.json', previous)


def try_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def process_identity(pid):
    try:
        root = Path('/proc')/str(pid)
        stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(start_ticks=stat[19], command=(root/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
    except (OSError, ValueError):
        return None


def evaluator_alive(state):
    ident = process_identity(state.get('pid'))
    return bool(ident and ident['command'] == state.get('command') and
                (not state.get('start_ticks') or ident['start_ticks'] == state['start_ticks']))


def owned_locally(task_dir, state, args):
    # Legacy labels are accepted only when explicitly reused by this worker.
    if state.get('hostname'):
        return state['hostname'] == socket.gethostname()
    owner = task_dir/'owner.json'
    return owner.is_file() and read(owner).get('hostname') == args.host_label


def attempts_used(task_dir):
    attempts = [load_task_state(task_dir).get('attempt', 0)]
    for archive in task_dir.parent.glob(task_dir.name+'.retry*'):
        suffix = archive.name.rsplit('.retry', 1)[-1]
        if suffix.isdigit():
            attempts.append(max(int(suffix), load_task_state(archive).get('attempt', 0)))
    return max(attempts)


def heartbeat(args, phase, **data):
    atomic_json(args.out_dir/f'heartbeat-{args.host_label}.json', dict(
        host=args.host_label, hostname=socket.gethostname(), pid=os.getpid(),
        phase=phase, updated_at=now(), **data))


def monitor(task, task_dir, strategy, args, process=None):
    state = load_task_state(task_dir)
    ident = process_identity(state['pid'])
    save_state(task_dir, hostname=socket.gethostname(), monitor_pid=os.getpid(),
               start_ticks=ident['start_ticks'] if ident else state.get('start_ticks'))
    while (process.poll() is None if process is not None else evaluator_alive(load_task_state(task_dir))):
        save_state(task_dir, monitor_pid=os.getpid())
        heartbeat(args, 'running', task=task['id'], evaluator_pid=state['pid'], gpu_ids=state['gpu_ids'])
        time.sleep(args.poll_seconds)
    return_code = process.returncode if process is not None else None
    complete = summary_complete(task['output'], task, strategy, args.max_new_tokens)
    save_state(task_dir, phase='completed' if complete else 'failed', return_code=return_code,
               summary_complete=complete, finished_at=now())
    return complete


def run_task(task, task_dir, strategy, args, gpu_ids):
    output = Path(task['output'])
    command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
        str(ROOT/'benchmarks/run_deepseek_greedy_eval.py'), '--strategy', strategy['strategy'],
        '--sparse-method', task['method'], '--sparsity', str(task['sparsity']),
        '--quantization', 'arc', '--base-eval-root', str(args.base_eval_root),
        '--source-model', str(args.source_model), '--converted-model', str(args.converted_model),
        '--calibration', strategy['calibration'], '--benchmark', task['benchmark'],
        '--model', 'deepseek', '--quant', 'hifp4', '--mode', 'w4a4', '--limit', str(task['limit']),
        '--max-new-tokens', str(args.max_new_tokens), '--out-dir', str(output)]
    attempt = attempts_used(task_dir) + 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpu_ids), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    atomic_json(task_dir/'owner.json', dict(task=task, hostname=args.host_label,
                physical_hostname=socket.gethostname(), pid=os.getpid(), claimed_at=now()))
    save_state(task_dir, phase='launching', command=command, gpu_ids=list(gpu_ids),
               strategy=strategy, output=str(output), attempt=attempt, pid=None, start_ticks=None,
               hostname=socket.gethostname(), started_at=now(), error=None)
    with (task_dir/f'attempt-{attempt}.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save_state(task_dir, phase='running', pid=process.pid)
        return monitor(task, task_dir, strategy, args, process)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--strategy-root', action='append', nargs=2, metavar=('METHOD', 'ROOT'), required=True)
    p.add_argument('--base-eval-root', type=Path, required=True)
    p.add_argument('--source-model', type=Path, required=True)
    p.add_argument('--converted-model', type=Path, required=True)
    p.add_argument('--gpu-candidates', default='0,1')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--host-label', default=socket.gethostname())
    p.add_argument('--benchmarks', default='math500,gpqa')
    p.add_argument('--math-limit', type=int, default=100)
    p.add_argument('--gpqa-limit', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--retry', type=int, default=1, help='additional attempts after the initial attempt')
    return p.parse_args()


def schedule(args, tasks, candidates, gpu_candidates):
    while True:
        selected = {method: choose(roots, method) for method, roots in candidates.items()}
        pending, exhausted = [], []
        for task in tasks:
            strategy = selected[task['method']]
            folder = args.out_dir/'claims'/task['id']
            lease = try_lock(folder/'.worker.lock')
            if lease is None:
                pending.append(task['id'])
                continue
            try:
                state = load_task_state(folder)
                local = owned_locally(folder, state, args)
                if state.get('phase') in ('running', 'launching'):
                    if local and evaluator_alive(state):
                        pending.append(task['id'])
                        if set(state['gpu_ids']).issubset(gpu_candidates):
                            monitor(task, folder, state.get('strategy', strategy), args)
                        continue
                    if not local:
                        # Never infer a remote PID is dead from the local /proc.
                        if summary_complete(task['output'], task, strategy, args.max_new_tokens):
                            save_state(folder, phase='completed', summary_complete=True)
                        else:
                            pending.append(task['id'])
                        continue
                if summary_complete(task['output'], task, strategy, args.max_new_tokens):
                    if state.get('phase') != 'completed':
                        save_state(folder, phase='completed', summary_complete=True)
                    continue
                if attempts_used(folder) >= 1 + args.retry:
                    exhausted.append(task['id'])
                    continue
                pending.append(task['id'])
                if strategy is None:
                    continue
                available = idle_gpus(138000, gpu_candidates)
                if len(available) < 2:
                    continue
                try:
                    run_task(task, folder, strategy, args, available[:2])
                except Exception as exc:
                    save_state(folder, phase='failed', error=repr(exc), finished_at=now())
            finally:
                lease.close()
        if not pending:
            heartbeat(args, 'failed' if exhausted else 'completed', exhausted=exhausted)
            return 1 if exhausted else 0
        heartbeat(args, 'waiting', remaining=pending, exhausted=exhausted)
        time.sleep(args.poll_seconds)


def main():
    args = parse()
    methods = ('wina', 'teal')
    benchmarks = tuple(args.benchmarks.split(','))
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or set(benchmarks)-{'math500', 'gpqa'}:
        raise SystemExit('unsupported or duplicate benchmark')
    limits = {'math500': args.math_limit, 'gpqa': args.gpqa_limit}
    if min(limits.values()) < 1 or args.poll_seconds <= 0 or args.retry < 0:
        raise SystemExit('invalid limits, polling interval or retry count')
    candidates = {method: [] for method in methods}
    for method, root in args.strategy_root:
        if method not in candidates:
            raise SystemExit(f'unknown method {method}')
        candidates[method].append(Path(root))
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(args.out_dir, methods, benchmarks, limits, DEFAULT_RATES)
    atomic_json(args.out_dir/f'sweep_manifest-{args.host_label}.json', dict(
        schema='deepseek-tiny-sweep-v2', created_at=now(), host=args.host_label,
        benchmarks=benchmarks, limits=limits, rates=DEFAULT_RATES, tasks=tasks,
        protocol='ARC HiF4 W4A4 + native greedy strategy; generated-answer scoring'))
    gpu_candidates = [v.strip() for v in args.gpu_candidates.split(',')]
    if len(gpu_candidates) != 2 or len(set(gpu_candidates)) != 2 or not all(v.isdecimal() for v in gpu_candidates):
        raise SystemExit('run one worker per distinct TP2 pair (--gpu-candidates 0,1)')
    locks = []
    try:
        for index in sorted(gpu_candidates):
            lease = try_lock(Path('/tmp')/f'arcquant-gpu-locks-{os.getuid()}'/f'gpu{index}.lock')
            if lease is None:
                raise SystemExit(f'GPU {index} already has a cooperating worker; not launching a duplicate')
            locks.append(lease)
        return schedule(args, tasks, candidates, gpu_candidates)
    finally:
        for lease in locks:
            lease.close()


if __name__ == '__main__':
    raise SystemExit(main())
