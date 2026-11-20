#!/usr/bin/env python3
"""Let current MAGI search rounds save, yield GPUs to DS dense, then resume MAGI.

The MAGI numerical code is never edited (its hash is part of its checkpoints).
Only the explicitly identified controller and its recorded search groups are
signalled. The controller is suspended first so it cannot dispatch new work.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json
from tools.queue_deepseek_dense import complete


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=stat[19],
                    command=(p / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
    except FileNotFoundError:
        return None


def alive(record):
    return identity(record['pid']) == record


def checkpoint(path):
    """Validate every persisted decision, including a partial current block."""
    from sparse_eval.magi.protocol import candidates
    data = json.loads(path.read_text())
    sizes = data['sizes']
    current = dict.fromkeys(sizes, 0.)
    for row in data['history']:
        options = candidates(sizes, current, data['target'], data['step'])
        errors = row['errors']
        if set(errors) != set(options) or not all(math.isfinite(e) and e >= 0 for e in errors.values()):
            raise ValueError(f'Invalid checkpoint errors: {path}')
        best = min(errors, key=lambda name: (errors[name], name))
        if row['selected'] != best or row['allocation'] != options[best]:
            raise ValueError(f'Invalid saved decision: {path}')
        current = options[best]
    return dict(path=str(path), rounds=len(data['history']), complete=bool(data['complete']),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mtime_ns=path.stat().st_mtime_ns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--controller-dir', type=Path, required=True)
    p.add_argument('--dense-plan', type=Path, required=True)
    a = p.parse_args()
    output = a.dense_plan.resolve().parent
    if not (output / 'ready.json').exists():
        p.error('Prepare and validate dense reuse before changing dispatch')
    folder = a.controller_dir.resolve()
    original = identity(a.controller_pid)
    if original is None or 'sparse_eval.magi.run' not in original['command']:
        p.error('The requested PID is not the live MAGI controller')
    if Path(sys.executable).resolve() != Path(original['command'][0]).resolve():
        # Python 3.12 changed float summation; exact greedy replay must use the
        # search interpreter, not the DS environment's Python 3.11.
        os.execv(original['command'][0], [original['command'][0], __file__, *sys.argv[1:]])
    base = Path(os.readlink(f'/proc/{a.controller_pid}/cwd'))
    sys.path.insert(0, str(base))
    environment = dict(item.split('=', 1) for item in
        Path(f'/proc/{a.controller_pid}/environ').read_bytes().decode().split('\0') if item)
    # Keep the original environment only in memory, never publish credentials.
    state = dict(schema='magi-checkpoint-handoff-v1', phase='preparing',
                 hostname=os.uname().nodename, pid=os.getpid(), controller=original,
                 dense_plan=str(a.dense_plan.resolve()), jobs={}, started_at=time.time())
    previous_path = output / 'magi-handoff.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous['controller'] != original:
            raise ValueError('Existing handoff belongs to another controller')
        state = previous
        state['pid'] = os.getpid()

    def save(phase=None):
        if phase:
            state['phase'] = phase
        state['updated_at'] = time.time()
        atomic_json(output / 'magi-handoff.json', state)

    with (output / 'magi-handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.kill(original['pid'], signal.SIGSTOP)
        try:
            current = json.loads((folder / 'status.json').read_text())
            planned = {r['key']: r for r in json.loads((folder / 'plan.json').read_text())['jobs']}
            for key, job in current['jobs'].items():
                if job['state'] != 'running':
                    continue
                record = identity(job['pid'])
                if record is None:
                    continue
                cmd = record['command']
                if planned[key]['stage'] != 'search' or cmd != planned[key]['command'] or os.getpgid(job['pid']) != job['pid']:
                    raise ValueError('Only recorded independent search process groups may yield')
                shard, total = map(int, cmd[cmd.index('--block-shard') + 1].split('/'))
                method, quant = cmd[cmd.index('--method') + 1], cmd[cmd.index('--quant') + 1]
                search = folder.parent / 'search' / f'{method}-{quant}'
                paths = [search / f'{stage}.block.layers.{i}.json'
                         for index, (stage, i) in enumerate([*[("preview", i) for i in range(40)],
                                                            *[("refiner", i) for i in range(30)]])
                         if index % total == shard]
                initial = {str(path): path.stat().st_mtime_ns for path in paths if path.exists()}
                state['jobs'].setdefault(key, dict(process=record, gpu_ids=job['gpus'], phase='awaiting_saved_round',
                                          paths=[str(path) for path in paths], initial_mtimes=initial))
            actual_children = {int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal()
                               and _parent(p) == original['pid'] and identity(int(p.name))}
            if actual_children != {job['process']['pid'] for job in state['jobs'].values() if alive(job['process'])}:
                raise ValueError('Controller children changed or are not fully accounted for')
            save('waiting_for_checkpoints')
        except BaseException:
            if alive(original):
                os.kill(original['pid'], signal.SIGCONT)
            raise
        while any(job['phase'] not in ('yielded', 'finished') for job in state['jobs'].values()):
            for job in state['jobs'].values():
                if job['phase'] in ('yielded', 'finished'):
                    continue
                if not alive(job['process']):
                    job['phase'] = 'yielded' if job.get('signal_sent_at') else 'finished'
                    job['saved_checkpoints'] = [checkpoint(Path(v)) for v in job['paths'] if Path(v).exists()]
                    continue
                if job.get('signal_sent_at'):
                    if time.time() - job['signal_sent_at'] > 60:
                        os.killpg(job['process']['pid'], signal.SIGKILL)
                    continue
                changed = [Path(v) for v in job['paths'] if Path(v).exists()
                           and Path(v).stat().st_mtime_ns != job['initial_mtimes'].get(v)]
                if changed:
                    proof = checkpoint(max(changed, key=lambda v: v.stat().st_mtime_ns))
                    if proof['rounds']:
                        job.update(checkpoint_before_yield=proof, phase='stopping_after_saved_round',
                                   signal_sent_at=time.time())
                        save()
                        os.killpg(job['process']['pid'], signal.SIGTERM)
            save()
            time.sleep(2)
        # TERM is handled only after CONT. All recorded GPU children have exited.
        if alive(original):
            os.kill(original['pid'], signal.SIGTERM)
            os.kill(original['pid'], signal.SIGCONT)
            deadline = time.time() + 30
            while alive(original) and time.time() < deadline:
                time.sleep(1)
            if alive(original):
                os.kill(original['pid'], signal.SIGKILL)
        save('dense_priority_active')
        jobs = json.loads(a.dense_plan.read_text())['jobs']
        while not all(complete(job) for job in jobs):
            save()
            time.sleep(20)
        # Original capture/search stages validate and reuse their own artifacts.
        with (output / 'magi-resumed.log').open('a') as log:
            resumed = subprocess.Popen(original['command'], cwd=base, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['resumed_controller'] = identity(resumed.pid)
        save('magi_resumed')


def _parent(path):
    try:
        return int((path / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None


if __name__ == '__main__':
    main()
