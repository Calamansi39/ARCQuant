#!/usr/bin/env python3
"""Write an explicit MAGI ARC dependency plan. Never auto-launch 740 videos."""
import argparse
import json
import os
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-eval','model','repo','teacher-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--vbench-python')
    a=p.parse_args()
    sys.path.insert(0,str(a.base_eval));os.environ['ARC_BASE_EVAL']=str(a.base_eval.resolve())
    import gen_vbench_magi2 as native
    from integrations.magi_arc.protocol import sources,sha256
    args=native.parse_args(['--model',str(a.model),'--repo',str(a.repo),'--quant','hifp4','--mode','w4a4',
        '--protocol','tiny16','--num-inference-steps','50','--refiner-num-inference-steps','5'])
    count=int(os.environ.get('GIT_CONFIG_COUNT','0'))
    os.environ.update({f'GIT_CONFIG_KEY_{count}':'safe.directory',f'GIT_CONFIG_VALUE_{count}':str(a.repo.resolve()),'GIT_CONFIG_COUNT':str(count+1)})
    _,_,_,config,_,videos=native.prepare(args)
    teacher=json.loads((a.teacher_root/'dense/hifp4/identity.json').read_text())
    for name,expected in teacher['source_sha256'].items():
        if sha256(a.base_eval/name)!=expected:raise ValueError('Changed ordinary HiF4 teacher source: '+name)
    for key in ('model','checkpoint_manifest_sha256','checkpoint_files','runtime_versions','source_revision','resolution','preview_steps','refiner_steps'):
        if teacher['generation'][key]!=config[key]:raise ValueError('Incompatible teacher '+key)
    jobs=[]
    for method in ('none','teal','wina'):
        root=a.output/method
        channels=a.output/('none' if method=='teal' else method)/'channels'
        command=[a.python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            str(ROOT/'benchmarks/run_magi_arc.py'),'--base-eval-root',str(a.base_eval),'--method',method,
            '--arc-calibration',str(channels),'--root',str(root),'--model',str(a.model),'--repo',str(a.repo),
            '--asset-dir',str(a.teacher_root/'svd'),
            '--quant','hifp4','--mode','w4a4','--protocol','tiny16','--num-inference-steps','50',
            '--refiner-num-inference-steps','5']
        if method!='teal':
            jobs.append(dict(id=method+'-stats',depends=[],command=[*command,'--stage','stats','--teacher-root',str(a.teacher_root)],gpu=2))
        if method!='none':
            jobs.append(dict(id=method+'-capture',depends=[('none' if method=='teal' else method)+'-stats'],command=[*command,'--stage','capture'],gpu=2))
            jobs.append(dict(id=method+'-search',depends=[method+'-capture'],command=[*command,'--stage','search','--target','0.4'],gpu=2))
            jobs.append(dict(id=method+'-extract',depends=[method+'-search'],command=[command[0],*command[5:],'--stage','extract','--target','0.4','--extract-target','0.4'],gpu=0))
        extra=['--sparsity','0' if method=='none' else '0.4']
        if method!='none':extra+=['--allocation',str(root/'allocations'/f'{method}-hifp4-s0.4.json')]
        depends=[method+('-stats' if method=='none' else '-extract')]
        smoke=[*command,'--stage','generate',*extra,'--out-dir',str(root/'tiny'),'--max-new-videos','1','--defer-finalize']
        jobs.append(dict(id=method+'-first-video',depends=depends,command=smoke,gpu=2,validation_required=True))
        jobs.append(dict(id=method+'-tiny740',depends=[method+'-first-video'],gpu=2,
            command=[*command,'--stage','generate',*extra,'--out-dir',str(root/'tiny')],
            release_gate='first video finite/encoded/ARC evidence verified and disk capacity checked'))
        jobs.append(dict(id=method+'-score16',depends=[method+'-tiny740'],gpu=1,
            command=[a.vbench_python or str(a.base_eval/'.venvs/vbench/bin/python'),str(a.base_eval/'eval_vbench.py'),str(root/'tiny'),'--device','cuda']))
    a.output.mkdir(parents=True,exist_ok=True)
    destination=a.output/'plan.json'
    if destination.exists():raise ValueError('Plan exists; refusing overwrite')
    destination.write_text(json.dumps(dict(schema='magi-arc-evaluation-plan-v1',state='waiting_for_DS40_and_GPU_validation',
        sources=sources(),videos=len(videos),jobs=jobs,automatic_full_generation=False,
        storage=dict(free_gib=round(shutil.disk_usage(a.output).free/2**30,1),estimated_new_gib=250,
                     reason='two ARC teacher captures about 69 GiB each; three 740-video caches about 30 GiB each, plus statistics and margin'),
        reuse='none and TEAL share unrotated ARC channel statistics; WINA needs separate SVD-coordinate statistics'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(plan=str(destination),jobs=len(jobs),videos=len(videos),auto_launch=False)))


if __name__=='__main__':main()
