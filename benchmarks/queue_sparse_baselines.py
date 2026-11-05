"""Wait for two idle GPUs; validate TEAL/WINA before full MATH-500 runs.

Each method progresses independently through 1 question, 8 questions, and
optionally the full dataset. A failed validation blocks only that method.
Status and logs survive disconnects; invoke with setsid or a job scheduler.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.gpu_wait import wait_for_idle_pair


def health(directory, expected):
    """Check coverage and generation integrity, without selecting on score.

    A length-truncated candidate is a valid completed record in the pinned
    evaluator and is counted as an incorrect answer by its scorer.  It should
    be reported in the smoke detail, but must not prevent the full run from
    starting; otherwise one hard question can silently suppress an entire
    method.
    """
    summaries = list(directory.glob('*/summary.json'))
    if len(summaries) != 1:
        return False, 'missing or ambiguous summary'
    summary = json.loads(summaries[0].read_text())
    outputs = summaries[0].with_name('outputs.jsonl')
    rows = [json.loads(line) for line in outputs.read_text().splitlines()] if outputs.exists() else []
    if (not summary.get('complete') or summary.get('questions') != expected
            or len(rows) != expected or len({r['id'] for r in rows}) != expected):
        return False, 'incomplete question coverage'
    invalid = []
    truncated = 0
    for row in rows:
        generation = row.get('generation', {})
        reason = generation.get('finish_reason')
        if reason == 'length':
            truncated += 1
            continue
        if (reason != 'eos' or not row.get('final_output', '').strip()
                or row.get('score', {}).get('extracted_answer') is None):
            invalid.append(row.get('id', '<unknown>'))
    if invalid:
        return False, f'invalid generation records: {invalid[:3]}'
    detail = f"{expected} answers; {summary.get('correct_candidates')} correct"
    if truncated:
        detail += f"; {truncated} length-truncated (scored incorrect)"
    return True, detail


def run_child(command, env, log, timeout):
    # Isolate torchrun so a timed-out smoke also terminates its GPU workers.
    with log.open('w') as handle:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main(argv=None, *, gpu_lease=wait_for_idle_pair):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base-eval-root', 'source-model', 'converted-model', 'calibration', 'out-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--sparsity', type=float, default=.5)
    p.add_argument('--methods', default='teal,wina',
                   help='comma-separated methods to run (teal,wina)')
    p.add_argument('--full', action='store_true', help='automatically run 500 questions after both validation stages')
    p.add_argument('--dry-run', action='store_true', help='check native evaluation prerequisites without requesting GPUs')
    p.add_argument('--poll-seconds', type=int, default=60)
    a = p.parse_args(argv)
    if not 0 <= a.sparsity < 1:
        p.error('sparsity must be in [0, 1)')
    for key in ('base_eval_root', 'source_model', 'converted_model', 'calibration', 'out_dir'):
        setattr(a, key, getattr(a, key).expanduser().resolve())
    for path in (a.base_eval_root, a.source_model, a.converted_model,
                 a.calibration / 'rank0.pt', a.calibration / 'rank1.pt'):
        if not path.exists():
            p.error(f'missing asset: {path}')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = lambda: hashlib.sha256(b''.join(path.read_bytes() for path in (
        ROOT / 'model/deepseek_generation_runtime.py', ROOT / 'model/sparse_utils.py',
        ROOT / 'benchmarks/run_deepseek_sparse_math.py', Path(__file__)))).hexdigest()
    code = fingerprint()
    methods = tuple(dict.fromkeys(item.strip() for item in a.methods.split(',') if item.strip()))
    if not methods or any(method not in ('teal', 'wina') for method in methods):
        p.error('--methods must contain teal and/or wina')
    state = {'benchmark': 'MATH-500', 'quantization': 'HiF4 W4A4', 'sparsity': a.sparsity,
             'code_sha256': code, 'full_requested': a.full, 'methods': {m: [] for m in methods}}

    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        tmp = a.out_dir / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(a.out_dir / 'status.json')

    def command(method, count, destination):
        cmd = [a.python]
        if not a.dry_run:
            cmd += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2']
        cmd += [str(ROOT / 'benchmarks/run_deepseek_sparse_math.py'), '--sparse-method', method,
                '--sparsity', str(a.sparsity), '--quant', 'hifp4', '--mode', 'w4a4',
                '--limit', str(count), '--batch-size', str(min(count, 20)),
                '--max-new-tokens', '512' if count == 1 else '8192', '--out-dir', str(destination)]
        for key in ('base_eval_root', 'source_model', 'converted_model', 'calibration'):
            cmd += ['--' + key.replace('_', '-'), str(getattr(a, key))]
        if a.dry_run:
            cmd.append('--dry-run')
        return cmd

    with (a.out_dir / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (a.out_dir / 'status.json').exists():
            p.error('use a fresh --out-dir; previous queue status is retained for audit')
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
        blocked = set()
        stages = (1,) if a.dry_run else ((1, 8, 500) if a.full else (1, 8))
        for count in stages:
            for method in methods:
                if method in blocked:
                    continue
                destination = a.out_dir / ('full' if count == 500 else 'validation') / method / str(count)
                entry = {'questions': count, 'status': 'preflight' if a.dry_run else 'waiting_for_gpus',
                         'output_dir': str(destination)}
                state['methods'][method].append(entry)
                save()
                log = a.out_dir / f'{method}-{count}.log'
                try:
                    if a.dry_run:
                        rc = run_child(command(method, count, destination), env, log, 600)
                    else:
                        with gpu_lease(poll_seconds=a.poll_seconds) as ids:
                            if fingerprint() != code:
                                raise RuntimeError('source changed while queued; restart with a new output directory')
                            entry.update(status='running', gpu_ids=ids)
                            save()
                            rc = run_child(command(method, count, destination),
                                {**env, 'CUDA_VISIBLE_DEVICES': ids}, log,
                                1800 if count == 1 else (14400 if count == 8 else None))
                    ok, detail = ((rc == 0, f'exit={rc}') if a.dry_run or rc != 0
                                  else health(destination, count))
                    entry.update(status=('preflight_passed' if a.dry_run else 'completed') if ok else 'needs_debug',
                                 detail=detail, exit_code=rc)
                except Exception as exc:
                    entry.update(status='needs_debug', detail=str(exc))
                if entry['status'] == 'needs_debug':
                    blocked.add(method)
                save()
                print(f"{method} {count}: {entry['status']} ({entry.get('detail')})", flush=True)
        return 1 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
