"""Resume a native DS sparsity probe: one-GPU layer search, then TP2 evaluation.

GPU leases check both memory and compute processes. Status gets a heartbeat
while waiting as well as running. No other user's job is stopped or shared.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json
from benchmarks.gpu_wait import wait_for_idle_gpus
from benchmarks.queue_deepseek_tiny_sweep import make_tasks, sha, summary_complete


def pending_evaluations(evaluation, search, sparsity, gpqa_limit, math_limit):
    """Skip only saved answers whose protocol, coverage and strategy still match."""
    tasks = make_tasks(evaluation, ['teal', 'wina'], ['gpqa', 'math500'],
                       dict(gpqa=gpqa_limit, math500=math_limit), [sparsity])
    calibration_sha = sha(search / 'calibration/manifest.json')
    identities = {method: dict(strategy_sha256=sha(search / method / 'strategy.json'),
                               calibration_manifest_sha256=calibration_sha)
                  for method in ('teal', 'wina')}
    pending = []
    for task in tasks:
        task['strategy_identity'] = identities[task['method']]
        if not summary_complete(task['output'], task, task['strategy_identity']):
            pending.append(task)
    return pending


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'base-eval-root', 'source-model', 'converted-model', 'calibration-data'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sparsity', type=float, default=.3)
    p.add_argument('--search-gpu-candidates', default='0,1,2,3,4,5,6,7')
    p.add_argument('--eval-gpu-candidates', default='0,1,2,3,4,5,6,7')
    p.add_argument('--gpqa-limit', type=int, default=8)
    p.add_argument('--math-limit', type=int, default=16)
    a = p.parse_args()
    if not 0 < a.sparsity < 1 or min(a.gpqa_limit, a.math_limit) < 1:
        p.error('Invalid sparsity or question count')
    run_dir = a.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    queue_lock = (run_dir / 'supervisor.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run_dir / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    search, evaluation = run_dir / 'search', run_dir / 'eval'
    state = dict(schema='deepseek-native-probe-queue-v2', phase='starting',
                 hostname=os.uname().nodename, pid=os.getpid(), sparsity=a.sparsity,
                 search_physical_gpus=1, evaluation_physical_gpus=2,
                 gpqa_limit=a.gpqa_limit, math_limit=a.math_limit,
                 updated_at=time.time())
    mutex, done = threading.Lock(), threading.Event()
    child = None

    def save(phase=None, **extra):
        with mutex:
            if phase:
                state['phase'] = phase
            state.update(extra, updated_at=time.time())
            atomic_json(run_dir / 'status.json', state)

    def heartbeat():
        while not done.wait(20):
            save()

    def stop(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    threading.Thread(target=heartbeat, daemon=True).start()
    assets = []
    for name in ('base_eval_root', 'source_model', 'converted_model', 'calibration_data'):
        assets += ['--' + name.replace('_', '-'), str(getattr(a, name).resolve())]
    atomic_json(run_dir / 'queue_plan.json', dict(vars(a), **{
        key: str(value) for key, value in vars(a).items() if isinstance(value, Path)
    }))

    def run(command, log_name, gpu_ids):
        nonlocal child
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids, HF_HUB_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', GLOO_SOCKET_IFNAME='lo')
        with (run_dir / log_name).open('a') as stream:
            stream.write('\n' + json.dumps(command) + '\n')
            stream.flush()
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True, stdin=subprocess.DEVNULL)
            save(child_pid=child.pid, gpu_ids=gpu_ids.split(','), log=log_name)
            rc = child.wait()
            child = None
        if rc:
            raise RuntimeError(f'{log_name} exited with code {rc}')

    def torchrun(script):
        return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc-per-node=2', str(ROOT / script)]

    def search_complete():
        # Validate complete coverage and the exact requested weighted budget.
        from benchmarks.deepseek_greedy_strategy import load_strategy
        manifest_path = search / 'search_manifest.json'
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text())
        for method in ('teal', 'wina'):
            path = search / method / 'strategy.json'
            if not path.exists() or not json.loads(path.read_text()).get('complete'):
                return False
            conversion = json.loads((a.converted_model / 'conversion.json').read_text())
            load_strategy(path, method=method, sparsity=a.sparsity,
                          source_signature=manifest['source_signature'], world_size=2,
                          quantization='arc', n_layers=conversion['config']['n_layers'])
        calibration = search / 'calibration/manifest.json'
        return calibration.exists() and json.loads(calibration.read_text()).get('complete')

    try:
        if not search_complete():
            save('waiting_for_single_search_gpu', gpu_ids=[])
            with wait_for_idle_gpus(count=1, min_free_mib=138000, poll_seconds=15,
                                   settle_seconds=3, candidates=a.search_gpu_candidates.split(',')) as ids:
                save('searching_30' if a.sparsity == .3 else 'searching')
                command = torchrun('benchmarks/colocated_deepseek_search.py') + assets + [
                    '--out-dir', str(search), '--quantization', 'arc', '--targets', str(a.sparsity),
                    '--samples', '128', '--search-samples', '20', '--search-batch-size', '20',
                    '--length', '512', '--step', '0.025', '--last-fraction', '0.25',
                    '--histogram-sample-limit', '16384', '--layer-streaming', '--methods', 'teal,wina',
                    '--gpu-memory-gib', '60']
                if (search / 'search_manifest.json').exists():
                    command.append('--resume')
                run(command, 'search.log', ids)
                if not search_complete():
                    raise RuntimeError('Search exited without complete native strategies/calibration')
        tasks = pending_evaluations(evaluation, search, a.sparsity, a.gpqa_limit, a.math_limit)
        pending_ids = [task['id'] for task in tasks]
        if not tasks:
            save('completed', child_pid=None, gpu_ids=[], pending_task_ids=[])
            return 0
        save('waiting_for_tp2_evaluation', gpu_ids=[], child_pid=None,
             pending_task_ids=pending_ids)
        with wait_for_idle_gpus(count=2, min_free_mib=138000, poll_seconds=15,
                               settle_seconds=3, candidates=a.eval_gpu_candidates.split(',')) as ids:
            for task in tasks:
                method, benchmark, limit = task['method'], task['benchmark'], task['limit']
                output = Path(task['output'])
                save('evaluating', method=method, benchmark=benchmark, limit=limit)
                command = torchrun('benchmarks/run_deepseek_greedy_eval.py') + [
                    '--strategy', str(search / method / 'strategy.json'), '--sparse-method', method,
                    '--sparsity', str(a.sparsity), '--quantization', 'arc',
                    '--base-eval-root', str(a.base_eval_root), '--source-model', str(a.source_model),
                    '--converted-model', str(a.converted_model), '--calibration', str(search / 'calibration'),
                    '--benchmark', benchmark, '--model', 'deepseek', '--quant', 'hifp4', '--mode', 'w4a4',
                    '--limit', str(limit), '--max-new-tokens', '8192', '--out-dir', str(output)]
                run(command, f'eval-{method}-{benchmark}.log', ids)
                if not summary_complete(output, task, task['strategy_identity']):
                    raise RuntimeError(f'Incomplete or incompatible evaluation: {output}')
                pending_ids.remove(task['id'])
                save(child_pid=None, pending_task_ids=pending_ids)
            save('completed', child_pid=None, gpu_ids=[])
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
             error=repr(exc), child_pid=None, gpu_ids=[])
        raise
    finally:
        done.set()
        queue_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
