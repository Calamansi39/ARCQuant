#!/usr/bin/env python3
"""Preflight DS40 LongBench/LCB; append them and missing dense controls to queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--dense-plan',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text());dense=json.loads(a.dense_plan.read_text())
    folder=a.plan.parent
    if any(j['id'].endswith('-longbench') for j in plan['jobs']): p.error('Extended jobs already present')
    initial=list(plan['jobs'])
    jobs=[]
    for stage in ('longbench','lcb64'):
        old=next(j for j in dense['jobs'] if j['id']==stage)
        for method in ('teal','wina'):
            seed=next(j for j in initial if j.get('method')==method)
            source=seed['command']
            cmd=list(old['command'])
            output=folder/(method+'-extended')/stage
            for flag,value in (('--calibration',source[source.index('--calibration')+1]),
                               ('--out-dir',str(output)),('--memory-guard-dir',str(folder/'memory_guard'))):
                cmd[cmd.index(flag)+1]=value
            cmd += ['--sparse-method',method,'--sparsity','0.4','--strategy',source[source.index('--strategy')+1]]
            script=next(i for i,c in enumerate(cmd) if c.endswith('run_deepseek_arc_extended.py'))
            if cmd[script-1]!='--':
                cmd.insert(script,'--');script+=1
            key=f'{method}-0.4-{stage}'
            with (folder/(key+'-preflight.log')).open('w') as log:
                subprocess.run([cmd[0],*cmd[script:],'--dry-run'],cwd=ROOT,
                    env=dict(os.environ,**dense['environment'],CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
            jobs.append(dict(id=key,gpu=True,validation='extended',method=method,sparsity=.4,
                question_count=old['expected'],stage=stage,output=str(output),command=cmd))
            jobs.append(dict(id=key+'-publish',gpu=False,command=[cmd[0],str(ROOT/'tools/finish_deepseek_priority.py'),
                '--plan',str(a.plan),'--base-eval',dense['base_eval'],'--stage',stage,'--directory',str(output)]))
    for stage in ('longbench','lcb64'):
        job=dict(next(j for j in dense['jobs'] if j['id']==stage))
        job.update(id='dense-'+stage,validation='extended',stage=stage,method='none',sparsity=0.)
        job['question_count']=job.pop('expected')
        job['command']=list(job['command'])
        directory=folder/'dense-extended'/stage
        job['command'][job['command'].index('--out-dir')+1]=str(directory)
        job['output']=str(directory)
        script=next(i for i,c in enumerate(job['command']) if c.endswith('run_deepseek_arc_extended.py'))
        if job['command'][script-1]!='--':
            job['command'].insert(script,'--');script+=1
        with (folder/('dense-'+stage+'-preflight.log')).open('w') as log:
            subprocess.run([job['command'][0],*job['command'][script:],'--dry-run'],cwd=ROOT,
                env=dict(os.environ,**dense['environment'],CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
        jobs.append(job)
        jobs.append(dict(id='dense-'+stage+'-publish',gpu=False,command=[job['command'][0],
            str(ROOT/'tools/finish_deepseek_priority.py'),'--plan',str(a.plan),'--base-eval',dense['base_eval'],
            '--stage',stage,'--directory',str(directory)]))
    # Read again after preflights, refusing an unexpected concurrent edit.
    if json.loads(a.plan.read_text())!=plan: raise ValueError('Queue changed while preparing')
    plan['jobs']=[]
    for job in initial:
        plan['jobs'].append(job)
        plan['jobs'].append(dict(id=job['id']+'-publish',gpu=False,command=[job['command'][0],
            str(ROOT/'tools/finish_deepseek_priority.py'),'--plan',str(a.plan),'--base-eval',dense['base_eval']]))
    plan['jobs']+=jobs
    atomic_json(a.plan,plan)
    print(json.dumps(dict(plan=str(a.plan),jobs=len(plan['jobs']),gpu_jobs=sum(j['gpu'] for j in plan['jobs']))))


if __name__=='__main__':main()
