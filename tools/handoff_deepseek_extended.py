#!/usr/bin/env python3
"""Finish a verified live MAGI search, run a fixed DS queue, resume its controller.

Only the controller is STOPped: its existing search children continue naturally.
Controller-held MAGI reservations persist; additional ARC advisory locks exclude
our other queues. No signals are ever sent to the current search process group.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def identity(pid):
    try:
        path = Path('/proc')/str(pid)
        stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
        command = (path/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        return dict(pid=pid, start_ticks=stat[19], command=command)
    except FileNotFoundError:
        return None


def same(record):
    return record is not None and identity(record['pid']) == record


def exit_status(record):
    path = Path('/proc')/str(record['pid'])
    fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
    if fields[19] != record['start_ticks']:
        raise RuntimeError('Search PID was reused')
    return int(fields[49]) if fields[0] == 'Z' else None


def check_hashes(plan):
    for filename, expected in plan['pinned_files'].items():
        with Path(filename).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Queued source changed: '+filename)


def validate_controller(plan):
    controller, search = plan['controller'], plan['search']
    if not same(controller) or not same(search):
        raise ValueError('Recorded controller/search identity no longer matches')
    status = read(Path(plan['controller_dir'])/'status.json')
    running = {k:v for k,v in status['jobs'].items() if v['state'] == 'running'}
    row = running.get(plan['search_key'])
    if (len(running) != 1 or row is None or row['pid'] != search['pid']
            or row['gpus'] != plan['gpus'] or os.getpgid(search['pid']) != search['pid']):
        raise ValueError('Expected exactly the recorded live search on the reserved pair')
    planned = {r['key']:r for r in read(Path(plan['controller_dir'])/'plan.json')['jobs']}
    if planned[plan['search_key']]['command'] != search['command']:
        raise ValueError('Search command differs from controller plan')
    fields = Path(f'/proc/{search["pid"]}/stat').read_text().rsplit(')',1)[1].split()
    if int(fields[1]) != controller['pid']:
        raise ValueError('Search is not a child of the recorded controller')


def watch_owner(plan_path, owner, owner_ticks):
    """Fail-safe if the supervisor is killed: wait for its GPU workers, resume MAGI."""
    folder = plan_path.parent
    plan = read(plan_path)
    while True:
        current = identity(owner)
        if current is None or current['start_ticks'] != owner_ticks:
            break
        time.sleep(2)
    state_path = folder/'handoff.json'
    state = read(state_path) if state_path.exists() else {}
    # ARC workers release their tail holds when owner dies. Do not resume MAGI
    # while a useful orphaned DS job is still computing.
    for record in state.get('children', []):
        while same(record):
            time.sleep(2)
    if same(plan['controller']):
        os.kill(plan['controller']['pid'], signal.SIGCONT)
    atomic(folder/'watchdog.json', dict(phase='controller_resumed', at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--watch-owner', type=int)
    parser.add_argument('--owner-ticks')
    args = parser.parse_args()
    if args.watch_owner:
        return watch_owner(args.plan.resolve(), args.watch_owner, args.owner_ticks)
    plan_path = args.plan.resolve()
    plan, folder = read(plan_path), plan_path.parent
    if socket.gethostname() != plan['hostname']:
        raise ValueError('Handoff must run on the recorded host')
    if Path(sys.executable).resolve() != Path(plan['magi_python']).resolve():
        os.execv(plan['magi_python'], [plan['magi_python'], __file__, *sys.argv[1:]])
    lock = (folder/'handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_hashes(plan)
    validate_controller(plan)
    gpu_locks = []
    lock_dir = Path('/tmp')/f'arcquant-gpu-locks-{os.getuid()}'
    lock_dir.mkdir(exist_ok=True, mode=0o700)
    for gpu in sorted(plan['gpus']):
        handle = (lock_dir/f'gpu{gpu}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_locks.append(handle)
    state = dict(schema='ds-extended-handoff-v1', pid=os.getpid(), hostname=socket.gethostname(),
        gpus=plan['gpus'], controller=plan['controller'], search=plan['search'],
        jobs={}, children=[], started=time.time())
    processes = []

    def save(phase=None):
        if phase:
            state['phase'] = phase
        state['updated'] = time.time()
        atomic(folder/'handoff.json', state)

    def stopped(*_):
        raise KeyboardInterrupt('Handoff supervisor interrupted')
    signal.signal(signal.SIGTERM, stopped)
    owner = identity(os.getpid())
    with (folder/'watchdog.log').open('a') as log:
        subprocess.Popen([sys.executable, __file__, '--plan', str(plan_path),
            '--watch-owner', str(os.getpid()), '--owner-ticks', owner['start_ticks']],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save('arming')
    os.kill(plan['controller']['pid'], signal.SIGSTOP)
    previous_hold = None
    try:
        # Freeze dispatch only. The independent torchrun group continues.
        validate_controller(plan)
        save('waiting_for_current_search_natural_completion')
        while exit_status(plan['search']) is None:
            if not same(plan['controller']):
                raise RuntimeError('MAGI controller disappeared')
            paths = list(Path(plan['search_dir']).glob('*.json'))
            state['search_completed_blocks'] = sum(read(p).get('complete', False) for p in paths)
            save()
            time.sleep(2)
        if exit_status(plan['search']) != 0:
            raise RuntimeError('Current search failed; DS will not mask its failure')
        # Exact greedy replay uses the original Python 3.12 interpreter.
        sys.path.insert(0, plan['repo'])
        sys.path.insert(0, plan['base_eval'])
        from tools.yield_magi_at_checkpoint import checkpoint
        paths = [Path(plan['search_dir'])/f'{stage}.block.layers.{i}.json'
                 for stage, n in (('preview',40), ('refiner',30)) for i in range(n)]
        proof = [checkpoint(p) for p in paths]
        if not all(p['complete'] for p in proof):
            raise ValueError('Search exited without 70 complete validated checkpoints')
        for path in paths:
            data = read(path)
            if (data['method'] != 'wina' or data['quant'] != 'hifp4' or data['target'] != .6
                    or data['step'] != .01 or not data['history']
                    or abs(data['history'][-1]['budget']-.6) > 1e-8):
                raise ValueError('Search checkpoint did not reach the requested budget')
        atomic(folder/'magi-search-completed.json', proof)
        check_hashes(plan)
        save('running_ds_queue')
        env = dict(os.environ, **plan['environment'], CUDA_VISIBLE_DEVICES=','.join(plan['gpus']),
                   ARC_QUEUE_OWNER_PID=str(os.getpid()))
        for job in plan['jobs']:
            record = dict(state='starting', started=time.time(), command=job['command'])
            state['jobs'][job['id']] = record
            save()
            job_env = dict(env)
            hold = folder/'holds'/job['id']
            if job['gpu']:
                # Tail holds survive CPU preparation; the new worker releases
                # its predecessor only after imports and source checks pass.
                hold.mkdir(parents=True, exist_ok=False)
                job_env['ARC_HOLD_DIR'] = str(hold)
                if previous_hold:
                    job_env['ARC_PREVIOUS_HOLD'] = str(previous_hold)
            with (folder/(job['id']+'.log')).open('a') as log:
                process = subprocess.Popen(job['command'], cwd=plan['repo'], env=job_env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=tuple(handle.fileno() for handle in gpu_locks))
            processes.append(process)
            child = identity(process.pid)
            state['children'].append(child)
            record.update(state='running', process=child)
            save()
            while True:
                code = process.poll()
                ready = job['gpu'] and all((hold/f'rank{r}.json').exists() for r in range(2))
                if ready or code is not None:
                    if code is not None and code != 0:
                        raise RuntimeError(f'{job["id"]} failed ({code}); saved answers retained')
                    if job['gpu'] and not ready:
                        raise RuntimeError('GPU worker exited without completion handshake')
                    break
                save()
                time.sleep(2)
            record.update(state='completed', finished=time.time())
            if job['gpu']:
                previous_hold = hold
            save()
        save('ds_complete_resuming_magi')
    except BaseException as exc:
        state['error'] = str(exc)
        save('failed_resuming_magi')
        raise
    finally:
        # Only our newly launched DS groups are terminated on supervision
        # failure. The original search group is never signalled.
        for path in (folder/'holds').glob('*') if (folder/'holds').exists() else []:
            (path/'release').touch()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if same(plan['controller']):
            os.kill(plan['controller']['pid'], signal.SIGCONT)
            state['controller_resumed_at'] = time.time()
        save('failed_magi_resumed' if state.get('error') else 'completed_magi_resumed')


if __name__ == '__main__':
    main()
