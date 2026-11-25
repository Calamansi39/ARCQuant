#!/usr/bin/env python3
"""Prepare both 40% methods with complete GPQA/MATH protocols and strict reuse."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json
from benchmarks.reuse_dense_prefix import import_prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-eval-root', 'source-model', 'converted-model', 'search-root', 'tiny-root', 'out-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if (a.out_dir/'plan.json').exists():
        p.error('Plan exists; refusing to replace a queued experiment')
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    jobs = []
    for benchmark, count in [('gpqa',198), ('math500',500)]:
        for method in ('teal','wina'):
            key = f'{method}-0.4-{benchmark}'
            search = a.search_root/(method+'-search')
            dest = a.out_dir/key
            args = ['--strategy',str(search/method/'strategy.json'),'--sparse-method',method,
                '--sparsity','0.4','--quantization','arc','--base-eval-root',str(a.base_eval_root),
                '--source-model',str(a.source_model),'--converted-model',str(a.converted_model),
                '--calibration',str(search/'calibration'),'--benchmark',benchmark,'--quant','hifp4',
                '--mode','w4a4','--limit','-1','--batch-size','20','--max-new-tokens','8192','--out-dir',str(dest)]
            native = [sys.executable,str(ROOT/'benchmarks/run_deepseek_greedy_eval.py'),*args]
            with (a.out_dir/(key+'-prepare.log')).open('a') as log:
                subprocess.run([*native,'--dry-run'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                targets=list(dest.glob('arcquant-*/protocol.json'))
                sources=list((a.tiny_root/key).glob('arcquant-*/protocol.json'))
                if len(targets)!=1 or len(sources)!=1:
                    raise ValueError('Ambiguous/missing source or target protocol')
                reused=import_prefix(sources[0].parent,targets[0].parent,greedy=True)['reused']
                subprocess.run([*native,'--evaluate-only'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                str(ROOT/'tools/run_deepseek_priority_worker.py'),'--guard-dir',str(a.out_dir/'memory_guard'),*args]
            jobs.append(dict(id=key,gpu=True,method=method,sparsity=.4,expected=count,reused=reused,
                output=str(targets[0].parent),command=command))
    atomic_json(a.out_dir/'plan.json',dict(schema='ds-arc40-priority-v1',jobs=jobs))
    atomic_json(a.out_dir/'ready.json',dict(ready=True,reused=sum(j['reused'] for j in jobs),
        remaining=sum(j['expected']-j['reused'] for j in jobs)))
    print(json.dumps(jobs,ensure_ascii=False))


if __name__=='__main__':
    main()
