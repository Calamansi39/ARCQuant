#!/usr/bin/env python3
"""Prepare full ARC dense GPQA/MATH and optionally reuse identical Tiny answers."""
from __future__ import annotations

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
    for name in ('base-eval-root', 'source-model', 'converted-model', 'calibration', 'out-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--tiny-root', type=Path, help='Completed none-0-gpqa/none-0-math500 source folders')
    a = p.parse_args()
    folder = a.out_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    reused = 0
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for benchmark, expected in [('gpqa', 198), ('math500', 500)]:
        key = f'none-0-{benchmark}'
        destination = folder / key
        command = [sys.executable, str(ROOT / 'benchmarks/run_deepseek_sparse_math.py'),
            '--sparse-method', 'none', '--sparsity', '0', '--quantization', 'arc',
            '--benchmark', benchmark, '--model', 'deepseek', '--quant', 'hifp4', '--mode', 'w4a4',
            '--limit', '-1', '--batch-size', '20', '--max-new-tokens', '8192', '--out-dir', str(destination)]
        for name in ('base_eval_root', 'source_model', 'converted_model', 'calibration'):
            command += ['--' + name.replace('_', '-'), str(getattr(a, name).resolve())]
        with (folder / f'{key}-prepare.log').open('a') as log:
            subprocess.run([*command, '--dry-run'], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            paths = list(destination.glob('arcquant-*/protocol.json'))
            if len(paths) != 1:
                raise ValueError('Ambiguous protocol; use a separate run folder')
            job = dict(id=key, benchmark=benchmark, expected=expected, output=str(destination), command=command)
            if a.tiny_root:
                sources = list((a.tiny_root / key).glob('arcquant-*/protocol.json'))
                if len(sources) != 1:
                    raise ValueError('Missing/ambiguous Tiny source')
                job['source'] = str(sources[0].parent.resolve())
                reused += import_prefix(sources[0].parent, paths[0].parent)['reused']
            subprocess.run([*command, '--evaluate-only'], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append(job)
    plan = dict(schema='deepseek-arc-dense-full-v1', jobs=jobs,
                scope='Full ARC HiF4 W4A4 dense GPQA198/MATH500; preserved native protocol and answers')
    atomic_json(folder / 'plan.json', plan)
    atomic_json(folder / 'ready.json', dict(ready=True, reused=reused, remaining=698-reused))
    print(json.dumps(dict(plan=str(folder / 'plan.json'), reused=reused, remaining=698-reused)))


if __name__ == '__main__':
    main()
