#!/usr/bin/env python3
"""CPU preflight and fixed queue preparation on the host running the MAGI search."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.handoff_deepseek_extended import atomic, identity, validate_controller


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-eval-root','source-model','converted-model','calibration','out-dir','controller-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--search-key', default='search-wina-hifp4-0')
    p.add_argument('--gpus', default='2,3')
    p.add_argument('--report-pythonpath', default='')
    a = p.parse_args()
    for name in ('base_eval_root','source_model','converted_model','calibration','out_dir','controller_dir'):
        setattr(a, name, getattr(a, name).resolve())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if (a.out_dir/'plan.json').exists():
        p.error('Plan already exists; inspect it rather than replacing an active queue')
    state = json.loads((a.controller_dir/'status.json').read_text())
    controller = identity(a.controller_pid)
    search = identity(state['jobs'][a.search_key]['pid'])
    plan = dict(schema='deepseek-arc-extended-queue-v1', created=time.time(), hostname=socket.gethostname(),
        repo=str(ROOT), base_eval=str(a.base_eval_root), controller=controller, search=search,
        controller_dir=str(a.controller_dir), magi_python=controller['command'][0],
        search_key=a.search_key, search_dir=str(a.controller_dir.parent/'search/wina-hifp4'),
        gpus=a.gpus.split(','), jobs=[])
    validate_controller(plan)
    env = dict(HF_HOME=str(a.base_eval_root.parent/'data/hf_home'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', DEEPSEEK_FAST_QUANT='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    if a.report_pythonpath:
        env['PYTHONPATH'] = a.report_pythonpath
    plan['environment'] = env
    script = str(ROOT/'benchmarks/run_deepseek_arc_extended.py')
    common = [script, '--base-eval-root', str(a.base_eval_root), '--calibration', str(a.calibration)]
    model = ['--source-model', str(a.source_model), '--converted-model', str(a.converted_model)]
    gpu = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2','--']
    continuation_dir = a.out_dir/'lcb128/runs/hifp4'
    for stage, name, extras in (
        ('longbench-v2','longbench',['--cache-gib','8','--max-input-tokens','131072',
         '--max-new-tokens','8192','--temperature','0.1','--top-p','0.95','--top-k','20','--disable-thinking']),
        ('livecodebench','lcb64',['--cache-gib','16','--max-new-tokens','65536',
         '--temperature','1','--top-p','0.95','--top-k','20','--enable-thinking','--reasoning-effort','max',
         '--seed','20261008','--batch-size','16','--effective-cache-gib','48','--cpu-threads','1'])):
        command = [*common, '--stage',stage,*model,'--out-dir',str(a.out_dir/name),
            '--quant','hifp4','--mode','w4a4','--limit','-1','--n','1',*extras,
            '--memory-guard-dir',str(a.out_dir/'memory_guard')]
        with (a.out_dir/(name+'-preflight.log')).open('w') as log:
            subprocess.run([sys.executable,*command,'--dry-run'], cwd=ROOT,
                env=dict(os.environ, **env, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=900)
        plan['jobs'].append(dict(id=name, gpu=True, expected=503 if name == 'longbench' else 175,
                                 command=[*gpu,*command]))
        plan['jobs'].append(dict(id=name+'-publish', gpu=False, command=[sys.executable,
            str(ROOT/'tools/finish_deepseek_extended.py'),'--base-eval',str(a.base_eval_root),
            '--run-dir',str(a.out_dir),'--stage',name]))
    for stage in ('continue-prepare','continue-worker','continue-collect'):
        command = [*common,'--stage',stage,'--source-root',str(a.out_dir/'lcb64'),
            '--run-dir',str(continuation_dir),'--memory-guard-dir',str(a.out_dir/'memory_guard')]
        plan['jobs'].append(dict(id=stage, gpu=stage == 'continue-worker',
            command=[*(gpu if stage == 'continue-worker' else [sys.executable]),*command]))
    plan['jobs'].append(dict(id='lcb128-publish',gpu=False,command=[sys.executable,
        str(ROOT/'tools/finish_deepseek_extended.py'),'--base-eval',str(a.base_eval_root),
        '--run-dir',str(a.out_dir),'--stage','lcb128']))
    from benchmarks.run_deepseek_arc_extended import sources
    names = list(ROOT/name for name in sources())
    names += [ROOT/name for name in ('tools/handoff_deepseek_extended.py',
        'tools/finish_deepseek_extended.py','benchmarks/deepseek_arc_extended_audit.py')]
    names += list((a.base_eval_root/'continue_lcb_code').glob('deepseek_*.py'))
    names += [a.base_eval_root/name for name in ('run_deepseek_longbench.py','deepseek_v4_longbench.py',
        'run_deepseek_lcb_batch_all.py','run_deepseek_lcb_batch.py','deepseek_v4_lcb_batch.py',
        'deepseek_v4_lcb_batch_quant.py','deepseek_v4_lcb_performance.py','scripts/magi_memory_guard.py',
        'rescore_longbench_v2.py','longbench_policy.py')]
    sys.path.insert(0, str(a.base_eval_root))
    from deepseek_v4_support import runtime_signature
    plan['pinned_files'] = {str(path):sha(path) for path in names}
    plan['pinned_files'].update({str(a.base_eval_root/name):value for name,value in runtime_signature().items()})
    # Import report dependencies now: completion must not discover a missing package.
    import openpyxl
    atomic(a.out_dir/'plan.json', plan)
    atomic(a.out_dir/'ready.json', dict(cpu_preflight='passed', gpu_validation='pending first real run',
        longbench_questions=503, lcb_questions=175, plan_sha256=sha(a.out_dir/'plan.json'), at=time.time()))
    print(json.dumps(dict(plan=str(a.out_dir/'plan.json'), jobs=len(plan['jobs']), status='ready'), ensure_ascii=False))


if __name__ == '__main__':
    main()
