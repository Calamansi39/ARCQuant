"""CPU-only identities, exact greedy prefixes and experiment configuration."""
import hashlib
import json
import math
import os
from pathlib import Path

BASE = Path(os.environ['ARC_BASE_EVAL']).resolve()
ROOT = Path(__file__).resolve().parent
SCHEMA = 'magi-arc-sparse-v1'
DEFAULT_ROOT = BASE / 'results/MAGI-2-preview/arc'
BASELINE_TAGS = {'bf16':'bf16-50p5','hifp4':'hifp4-w4a4-50p5'}


def bind_dense_baseline(config):
    """Bind a zero-sparsity ARC configuration; this is NOT a completed score."""
    from .calibration import arc_identity
    return dict(kind='required-arc-zero-control-configuration', completed_score=False,
        arcquant=arc_identity(),generation={k:config[k] for k in (
            'checkpoint_manifest_sha256','quant','mode','resolution','preview_steps','refiner_steps',
            'cp_size','ep_size','runtime_versions','source_revision')})


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sources():
    names = [p for p in ROOT.rglob('*') if p.suffix in ('.py', '.json', '.sh')]
    names += [BASE / p for p in ('magi2_runtime.py', 'magi2_quant.py', 'magi2_batched_moe.py',
              'magi2_fast_quant.py', 'arcquant_fakequant.py', 'gen_vbench_magi2.py',
              'magi2_vbench_worker.py', 'magi2_conditioning.py', 'vbench_generation_cache.py', 'vbench_protocol.py')]
    repo=ROOT.parents[1]
    names += [repo/'model/quantize.py']
    return {('repo/'+str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p.relative_to(BASE))): sha256(p) for p in sorted(names)}


def candidates(sizes, current, target, step):
    total = sum(sizes.values())
    budget = sum(sizes[n] * current[n] for n in sizes) / total
    # A small projection can saturate before consuming the full step. Finish
    # that budget interval next, so 40%/50% remain actual visited prefixes of
    # a 60% search instead of interpolated/scaled approximations.
    next_budget = min(target, (math.floor((budget + 1e-10) / step) + 1) * step)
    result = {}
    for name in sorted(sizes):
        delta = min((next_budget - budget) * total / sizes[name], 1 - current[name],
                    (target - budget) * total / sizes[name])
        if delta > 1e-10:
            result[name] = dict(current, **{name: current[name] + delta})
    return result


def replay(record, target):
    """Validate the complete path, then return an actual prefix at target."""
    sizes, step = record['sizes'], record['step']
    maximum = record['target']
    if not 0 <= target <= maximum or not math.isfinite(target):
        raise ValueError('Requested target is outside the calibrated path')
    current = dict.fromkeys(sizes, 0.)
    prefix = dict(current) if target == 0 else None
    total = sum(sizes.values())
    for i, row in enumerate(record['history']):
        options = candidates(sizes, current, maximum, step)
        errors = row['errors']
        if set(options) != set(errors) or not all(math.isfinite(v) and v >= 0 for v in errors.values()):
            raise ValueError(f'Invalid candidates at greedy round {i}')
        best = min(errors, key=lambda key: (errors[key], key))
        if row['selected'] != best or row['allocation'] != options[best]:
            raise ValueError(f'Invalid greedy decision at round {i}')
        current = options[best]
        budget = sum(sizes[n] * current[n] for n in sizes) / total
        if abs(budget - row['budget']) > 1e-9:
            raise ValueError('Greedy budget mismatch')
        if abs(budget - target) < 1e-9:
            prefix = dict(current)
    if abs(sum(sizes[n] * current[n] for n in sizes) / total - maximum) > 1e-9:
        raise ValueError('Calibration has not reached its target')
    if prefix is None:
        raise ValueError(f'{target} is not a visited greedy budget; choose a path boundary (step {step})')
    return prefix


def load_allocation(path, method, quant):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != SCHEMA or (data['method'], data['quant']) != (method, quant):
        raise ValueError('Wrong allocation schema/method/precision')
    if data['source_sha256'] != sources():
        raise ValueError('Allocation source differs; regenerate/revalidate calibration explicitly')
    if not data.get('layers') or any(not math.isfinite(v) or not 0 <= v <= 1 for v in data['layers'].values()):
        raise ValueError('Invalid allocation values')
    for block, entry in data['search_files'].items():
        search = Path(path).parent / entry['path']
        if sha256(search) != entry['sha256']:
            raise ValueError(f'Changed search record: {block}')
        selected = replay(json.loads(search.read_text()), data['target'])
        if any(data['layers'].get(k) != v for k,v in selected.items()):
            raise ValueError('Allocation is not the recorded greedy prefix')
    covered = {k for entry in data['search_files'].values()
               for k in json.loads((Path(path).parent/entry['path']).read_text())['sizes']}
    if covered != set(data['layers']):
        raise ValueError('Allocation contains missing or extra projections')
    return data
