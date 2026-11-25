#!/usr/bin/env python3
"""Supersede a verified waiting queue, checkpoint MAGI, and run DS40 first.

The superseded MAGI controller is retired, never implicitly resumed. Its saved
search files remain valid. Additional jobs may be appended atomically to plan;
started job definitions cannot change. GPU locks are inherited by GPU workers.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.handoff_deepseek_extended import identity, same, read, atomic, validate_controller


def live(record):
    return same(record) and Path(f'/proc/{record["pid"]}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'


def finish_check(job):
    if not job.get('expected'):
        return
    from benchmarks.verified_results import validate_answers
    from benchmarks.queue_sparse_baselines import arc_health
    from benchmarks.result_tables import verify_greedy_evidence
    root=Path(job['output'])
    summary, protocol, _, _=validate_answers(root,job['expected'])
    if not summary['is_full_dataset'] or summary['is_smoke_test'] or not arc_health(root)[0]:
        raise ValueError('Full ARC evidence failed: '+job['id'])
    verify_greedy_evidence(root,protocol['arcquant'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--previous-plan',type=Path,required=True)
    p.add_argument('--previous-owner',type=int,required=True)
    p.add_argument('--previous-watchdog',type=int,required=True)
    a=p.parse_args()
    old=read(a.previous_plan)
    if socket.gethostname()!=old['hostname']:
        raise ValueError('Wrong host')
    if Path(sys.executable).resolve()!=Path(old['magi_python']).resolve():
        os.execv(old['magi_python'],[old['magi_python'],__file__,*sys.argv[1:]])
    sys.path.insert(0,old['base_eval'])
    from tools.yield_magi_at_checkpoint import checkpoint
    folder=a.plan.resolve().parent
    if not read(folder/'ready.json')['ready']:
        raise ValueError('New queue has not passed preflight')
    singleton=(folder/'handoff.lock').open('a')
    fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    validate_controller(old)
    previous=read(a.previous_plan.parent/'handoff.json')
    if previous['jobs'] or previous['children'] or previous['phase']!='waiting_for_current_search_natural_completion':
        raise ValueError('Previous queue has already started work; manual reconciliation required')
    owner,watch=identity(a.previous_owner),identity(a.previous_watchdog)
    for record in (owner,watch):
        if not record or str(a.previous_plan) not in record['command'] or not any(v.endswith('handoff_deepseek_extended.py') for v in record['command']):
            raise ValueError('Previous supervisor identity mismatch')
    if '--watch-owner' not in watch['command'] or watch['command'][watch['command'].index('--watch-owner')+1]!=str(owner['pid']):
        raise ValueError('Wrong watchdog owner')
    state=dict(schema='ds40-priority-handoff-v1',pid=os.getpid(),host=socket.gethostname(),
        gpus=old['gpus'],controller=old['controller'],search=old['search'],jobs={},started=time.time())
    def save(phase=None):
        if phase: state['phase']=phase
        state['updated']=time.time()
        atomic(folder/'handoff.json',state)
    save('superseding_waiting_queue')
    # STOP the owner, retire its fail-safe, then remove only this idle CPU owner.
    # SIGTERM of the old owner would incorrectly restart the no-ARC controller.
    os.kill(owner['pid'],signal.SIGSTOP)
    os.kill(watch['pid'],signal.SIGTERM)
    while live(watch): time.sleep(.1)
    if not same(owner): raise ValueError('Owner changed during handoff')
    os.kill(owner['pid'],signal.SIGKILL)
    while live(owner): time.sleep(.1)
    atomic(a.previous_plan.parent/'superseded.json',dict(by=str(a.plan),at=time.time(),reason='DS40 full highest priority'))
    locks=[]
    directory=Path('/tmp')/f'arcquant-gpu-locks-{os.getuid()}'
    directory.mkdir(exist_ok=True)
    for gpu in old['gpus']:
        handle=(directory/f'gpu{gpu}.lock').open('a')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.append(handle)
    search_dir=Path(old['search_dir'])
    initial={str(path):path.stat().st_mtime_ns for path in search_dir.glob('*.json')}
    save('waiting_for_saved_magi_round')
    while live(old['search']):
        changed=[path for path in search_dir.glob('*.json') if path.stat().st_mtime_ns!=initial.get(str(path))]
        if changed:
            proof=checkpoint(max(changed,key=lambda path:path.stat().st_mtime_ns))
            if proof['rounds']:
                state['saved_checkpoint']=proof
                save('yielding_magi_after_checkpoint')
                os.killpg(old['search']['pid'],signal.SIGTERM)
                deadline=time.monotonic()+60
                while live(old['search']) and time.monotonic()<deadline: time.sleep(.2)
                if live(old['search']): os.killpg(old['search']['pid'],signal.SIGKILL)
                break
        save()
        time.sleep(2)
    # Proof includes partial blocks; no completed work is deleted or relabelled.
    atomic(folder/'preserved_magi_checkpoints.json',[checkpoint(path) for path in sorted(search_dir.glob('*.json'))])
    if live(old['controller']):
        # Its search has exited. TERM while stopped, then CONT only to process TERM.
        os.kill(old['controller']['pid'],signal.SIGTERM)
        os.kill(old['controller']['pid'],signal.SIGCONT)
        deadline=time.monotonic()+30
        while live(old['controller']) and time.monotonic()<deadline: time.sleep(.2)
        if live(old['controller']): os.kill(old['controller']['pid'],signal.SIGKILL)
    processes=[]
    previous_hold=None
    def stop(*_): raise KeyboardInterrupt('Priority queue interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:
        save('running_ds40_full')
        while True:
            plan=read(a.plan)
            for job in plan['jobs']:
                if job['id'] in state['jobs'] and state['jobs'][job['id']]['definition']!=job:
                    raise ValueError('Started job definition changed')
            pending=[job for job in plan['jobs'] if job['id'] not in state['jobs']]
            if not pending: break
            job=pending[0]
            env=dict(os.environ,**old['environment'],CUDA_VISIBLE_DEVICES=','.join(old['gpus']),ARC_QUEUE_OWNER_PID=str(os.getpid()))
            hold=folder/'holds'/job['id']
            if job['gpu']:
                hold.mkdir(parents=True,exist_ok=False)
                env['ARC_HOLD_DIR']=str(hold)
                if previous_hold: env['ARC_PREVIOUS_HOLD']=str(previous_hold)
            with (folder/(job['id']+'.log')).open('a') as log:
                child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    pass_fds=tuple(h.fileno() for h in locks))
            processes.append(child)
            row=dict(definition=job,state='running',process=identity(child.pid),started=time.time())
            state['jobs'][job['id']]=row
            save()
            while True:
                code=child.poll()
                ready=job['gpu'] and all((hold/f'rank{rank}.json').exists() for rank in range(2))
                if ready or code is not None:
                    if code not in (None,0): raise RuntimeError(f'{job["id"]}: exit {code}; outputs retained')
                    if job['gpu'] and not ready: raise RuntimeError('Missing successful worker handshake')
                    break
                save()
                time.sleep(2)
            finish_check(job)
            row.update(state='completed',finished=time.time())
            if job['gpu']: previous_hold=hold
            save()
        save('completed')
    except BaseException as exc:
        state['error']=repr(exc)
        save('failed_outputs_preserved')
        raise
    finally:
        for path in (folder/'holds').glob('*'):
            (path/'release').touch()
        for child in processes:
            if child.poll() is None:
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()


if __name__=='__main__':
    main()
