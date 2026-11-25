#!/usr/bin/env python3
"""Shared job claims for independent TP2 workers on multiple hosts."""
import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json
from benchmarks.gpu_wait import wait_for_idle_gpus
from tools.handoff_deepseek_priority import finish_check


def read(p):return json.loads(Path(p).read_text())


def verified(group,base):
    job=group['job']
    if job.get('validation')=='extended':
        from benchmarks.deepseek_arc_extended_audit import audit
        audit(Path(job['output']),job['stage'],Path(base))
    elif job.get('validation')=='magi-arc':
        output=Path(job['command'][job['command'].index('--output')+1])
        if not all(read(output/f'rank{r}.json')['complete'] for r in range(2)):
            raise ValueError('Incomplete MAGI ARC GPU validation')
    else:finish_check(job)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--worker')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--borrow-reservation',action='store_true')
    p.add_argument('--previous-hold',type=Path)
    p.add_argument('--finish-external')
    a=p.parse_args();plan=read(a.plan);folder=a.plan.parent/'dispatch'
    folder.mkdir(exist_ok=True)
    if a.finish_external:
        group=next(g for g in plan['groups'] if g['job']['id']==a.finish_external)
        verified(group,plan['base_eval'])
        atomic_json(folder/(a.finish_external+'.done.json'),dict(completed=time.time(),external=True))
        return
    if not a.worker:p.error('--worker required')
    state=dict(host=socket.gethostname(),pid=os.getpid(),phase='waiting_for_gpus',jobs={},gpus=[])
    status=folder/(a.worker+'.json')
    def save(phase=None):
        if phase:state['phase']=phase
        state['updated']=time.time();atomic_json(status,state)
    save()
    def stop(*_):raise KeyboardInterrupt('Queue interrupted')
    signal.signal(signal.SIGTERM,stop)
    processes=[];previous=a.previous_hold
    def done(group):return (folder/(group['job']['id']+'.done.json')).exists()
    with (folder/(a.worker+'.lock')).open('a') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reservation=nullcontext(a.gpus) if a.borrow_reservation else wait_for_idle_gpus(
            count=2,candidates=a.gpus.split(','),poll_seconds=10,settle_seconds=2)
        with reservation as ids:
            state['gpus']=ids.split(',')
            try:
                while True:
                    plan=read(a.plan)
                    selected=None;claim=None
                    for group in plan['groups']:
                        key=group['job']['id']
                        if done(group) or group.get('external') or (folder/(key+'.failed.json')).exists():continue
                        if group['job'].get('validation')=='magi-arc' and any(not done(g) for g in plan['groups'] if g['job'].get('validation')!='magi-arc'):continue
                        claim=(folder/(key+'.lock')).open('a')
                        try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:claim.close();claim=None;continue
                        if done(group):claim.close();claim=None;continue
                        selected=group;break
                    if selected is None:
                        save('no_unclaimed_ready_jobs_releasing_gpus');break
                    job=selected['job'];key=job['id']
                    try:
                        hold=folder/'holds'/a.worker/key
                        hold.mkdir(parents=True,exist_ok=False)
                        env=dict(os.environ,**plan['environment'],CUDA_VISIBLE_DEVICES=ids,ARC_QUEUE_OWNER_PID=str(os.getpid()),ARC_HOLD_DIR=str(hold))
                        if previous:env['ARC_PREVIOUS_HOLD']=str(previous)
                        with (folder/(key+'-'+a.worker+'.log')).open('a') as log:
                            child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(claim.fileno(),))
                        processes.append(child)
                        state['jobs'][key]=dict(state='running',pid=child.pid,started=time.time())
                        save('running')
                        while True:
                            code=child.poll()
                            ready=all((hold/f'rank{r}.json').exists() for r in range(2))
                            if code is not None or ready:
                                if code not in (0,None) or not ready:raise RuntimeError(f'{key}: exit {code}, handshake={ready}')
                                break
                            save();time.sleep(2)
                        for post in selected['post']:
                            with (folder/(key+'-publish.log')).open('a') as log:
                                subprocess.run(post['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                        verified(selected,plan['base_eval'])
                        atomic_json(folder/(key+'.done.json'),dict(completed=time.time(),worker=a.worker))
                        state['jobs'][key].update(state='completed',finished=time.time())
                        previous=hold;save()
                    except BaseException as exc:
                        atomic_json(folder/(key+'.failed.json'),dict(error=repr(exc),at=time.time(),worker=a.worker))
                        state['error']=repr(exc);save('failed_outputs_preserved');raise
                    finally:claim.close()
            finally:
                for path in (folder/'holds'/a.worker).glob('*'):
                    (path/'release').touch()
                for child in processes:
                    if child.poll() is None:
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=30)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
