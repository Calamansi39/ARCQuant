import json
import os

base = '/gemini/code/NMSparsity/ARC_lbc/results/rsparse_bench_20260318'
rest_path = os.path.join(base, 'rest.json')
mmlu_path = os.path.join(base, 'mmlu.json')
out_path = os.path.join(base, 'summary.json')


def extract_task_metrics(payload, task):
    task_groups = payload.get('metrics', {}).get(task, {})
    if task in task_groups:
        metrics = task_groups[task]
    elif len(task_groups) == 1:
        metrics = next(iter(task_groups.values()))
    else:
        metrics = {}
    return {k: v for k, v in metrics.items() if k.startswith('acc')}

with open(rest_path) as f:
    rest = json.load(f)
with open(mmlu_path) as f:
    mmlu = json.load(f)

summary = {
    'model': 'Llama-3.1-8B',
    'method': 'rsparse+arc',
    'quant_type': 'NVFP4',
    'results': {
        'arc_challenge': extract_task_metrics(rest, 'arc_challenge'),
        'arc_easy': extract_task_metrics(rest, 'arc_easy'),
        'boolq': extract_task_metrics(rest, 'boolq'),
        'openbookqa': extract_task_metrics(rest, 'openbookqa'),
        'piqa': extract_task_metrics(rest, 'piqa'),
        'rte': extract_task_metrics(rest, 'rte'),
        'winogrande': extract_task_metrics(rest, 'winogrande'),
        'mmlu': extract_task_metrics(mmlu, 'mmlu'),
    },
    'sources': {
        'rest': rest_path,
        'mmlu': mmlu_path,
    },
}

with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(out_path)
