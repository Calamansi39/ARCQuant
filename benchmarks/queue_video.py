"""Portable MiniMax/MAGI generation and scoring queue; planning is the default.

The pinned evaluator owns sampling, generation, cache validation and scoring.
This launcher owns only paths, GPU leases, stage ordering and resume identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.asset_inventory import audit_benchmark, audit_model
from benchmarks.baseline_bridge import _revision
from benchmarks.gpu_wait import wait_for_idle_gpus
from benchmarks.paths import asset_path, asset_root, executable_path
from benchmarks.scorer_assets import audit_scorer, scorer_root


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['minimax-h3', 'magi-2-preview'])
    p.add_argument('--model-path', type=Path)
    p.add_argument('--base-eval-root', type=Path, default=ROOT / 'external/base_eval')
    p.add_argument('--magi-repo', type=Path, default=ROOT / 'external/MAGI-2-preview')
    p.add_argument('--generator-python')
    p.add_argument('--scorer-python')
    p.add_argument('--method', choices=['none', 'wina', 'teal'], default='none')
    p.add_argument('--calibration', type=Path)
    p.add_argument('--magi-zero-evidence', type=Path)
    p.add_argument('--sparsity', type=float, default=0.)
    p.add_argument('--quant', choices=['hifp4', 'bf16'], default='hifp4')
    p.add_argument('--profile', choices=['smoke', 'tiny', 'official'], default='tiny')
    p.add_argument('--stage', choices=['all', 'generate', 'score'], default='all')
    p.add_argument('--steps', type=int, default=50,
                   help='MAGI defaults to the existing 50+5 baseline protocol; official default is 100+5')
    p.add_argument('--refiner-steps', type=int, default=5)
    p.add_argument('--gpu-ids', help='allowed physical indices, e.g. 0,1,2,3; default: all')
    p.add_argument('--min-free-mib', type=int)
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='preflight, wait for GPUs, then run')
    return p


def build_plan(a, catalog):
    if a.steps < 1 or a.refiner_steps < 1 or a.poll_seconds <= 0:
        raise ValueError('Steps and polling interval must be positive')
    if not 0 <= a.sparsity < 1 or (a.method == 'none' and a.sparsity != 0):
        raise ValueError('Use --method wina for nonzero sparsity; sparsity must be in [0,1)')
    if a.method != 'none' and a.quant != 'hifp4':
        raise ValueError('Sparse video queues require HiF4')
    if a.method == 'teal' and (a.model != 'magi-2-preview' or not a.calibration):
        raise ValueError('Video TEAL requires MAGI and a completed --calibration')
    if a.calibration and a.method != 'teal':
        raise ValueError('--calibration is only valid for TEAL')
    if a.magi_zero_evidence and (a.model != 'magi-2-preview' or a.method == 'none'):
        raise ValueError('--magi-zero-evidence requires a MAGI sparse method')
    if a.model == 'magi-2-preview' and a.method != 'none' and a.profile != 'smoke' and not a.magi_zero_evidence:
        raise ValueError('Full MAGI sparse suites require validated --magi-zero-evidence; run smoke first')
    count = 1 if a.model == 'minimax-h3' else 2
    candidates = a.gpu_ids.split(',') if a.gpu_ids else None
    if candidates is not None and (len(set(candidates)) != len(candidates) or
            len(candidates) < count or not all(i.isdecimal() for i in candidates)):
        raise ValueError(f'--gpu-ids requires at least {count} distinct physical indices')
    minimum = a.min_free_mib if a.min_free_mib is not None else (79000 if count == 1 else 142000)
    if minimum <= 0:
        raise ValueError('--min-free-mib must be positive')
    env_name = 'ARCQUANT_H3_PYTHON' if count == 1 else 'ARCQUANT_MAGI2_PYTHON'
    generator = str(executable_path(a.generator_python or os.environ.get(env_name, sys.executable)))
    scorer = str(executable_path(a.scorer_python or os.environ.get('ARCQUANT_VBENCH_PYTHON', sys.executable)))
    model = asset_path(catalog['models'][a.model], a.model, 'models', override=a.model_path)
    data = asset_path(catalog['benchmarks']['vbench'], 'vbench', 'datasets')
    backend, out = a.base_eval_root.expanduser().resolve(), a.out_dir.expanduser().resolve()
    files = ['benchmarks/queue_video.py', 'benchmarks/run_minimax_sparse.py',
             'model/video_sparse_runtime.py'] if a.method == 'wina' else ['benchmarks/queue_video.py']
    if a.model == 'magi-2-preview' and a.method != 'none':
        files = ['benchmarks/queue_video.py', 'benchmarks/magi_validation.py',
                 'benchmarks/run_magi2_sparse.py', 'model/magi2_sparse_runtime.py',
                 'model/magi2_sparse_moe.py', 'model/magi2_calibration.py', 'model/sparse_utils.py']
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    identity = dict(model=a.model, model_path=str(model), method=a.method, sparsity=a.sparsity,
                    quant=a.quant, mode='w4a4' if a.quant != 'bf16' else 'bf16',
                    profile=a.profile, steps=a.steps, refiner_steps=a.refiner_steps,
                    model_revision=catalog['models'][a.model]['source']['revision'],
                    backend_revision=catalog['backends']['base-eval']['revision'],
                    generator_python=generator, scorer_python=scorer, sources=hashes)
    if a.model == 'magi-2-preview':
        identity['allocator'] = 'expandable_segments:True'
        identity['nccl_algo'] = '^NVLS'
    if a.calibration:
        identity['calibration_sha256'] = hashlib.sha256(a.calibration.read_bytes()).hexdigest()
    if a.magi_zero_evidence:
        identity['zero_evidence_sha256'] = hashlib.sha256(a.magi_zero_evidence.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    tag = f'{a.quant}-{a.method}-s{a.sparsity:g}-{a.steps}p{a.refiner_steps}-{digest}'
    runtime_env = dict(HF_HOME=str(asset_root() / 'hf_home'),
                       HF_HUB_CACHE=str(asset_root() / 'hf_home/hub'),
                       HUGGINGFACE_HUB_CACHE=str(asset_root() / 'hf_home/hub'),
                       TRANSFORMERS_CACHE=str(asset_root() / 'hf_home/hub'),
                       HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       VBENCH_REPO=str(data), VBENCH_FULL_INFO=str(data / 'vbench/VBench_full_info.json'),
                       VBENCH_CACHE_DIR=str(scorer_root()), TORCH_HOME=str(scorer_root() / 'torch_hub'),
                       PYTHONPATH=str(data) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                       PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '8'),
                       NCCL_ALGO=os.environ.get('NCCL_ALGO', '^NVLS'))
    media = os.environ.get('ARCQUANT_FFMPEG_BIN')
    if media:
        runtime_env['PATH'] = str(Path(media).expanduser()) + os.pathsep + os.environ.get('PATH', '')
    # Scope shared-checkout trust to the child; never alter global Git config.
    slot = int(os.environ.get('GIT_CONFIG_COUNT', '0'))
    for trusted in (data, backend, a.magi_repo.expanduser().resolve()):
        runtime_env[f'GIT_CONFIG_KEY_{slot}'] = 'safe.directory'
        runtime_env[f'GIT_CONFIG_VALUE_{slot}'] = str(trusted)
        slot += 1
    runtime_env['GIT_CONFIG_COUNT'] = str(slot)
    if count == 2:
        runtime_env.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_ALGO='^NVLS',
                           OMP_NUM_THREADS='16')
        runtime_env['MAGI2_KERNEL_CACHE_ROOT'] = str(Path('/tmp') / f'arcquant-magi2-{os.getuid()}' / digest)

    def generation(profile, directory):
        script = backend / ('gen_vbench_minimax.py' if count == 1 else 'gen_vbench_magi2.py')
        command = [generator, str(script)]
        if a.method != 'none':
            wrapper = 'run_minimax_sparse.py' if count == 1 else 'run_magi2_sparse.py'
            command = [generator, str(ROOT / 'benchmarks' / wrapper),
                       '--base-eval-root', str(backend), '--sparse-method', a.method,
                       '--sparsity', str(a.sparsity)]
            if a.calibration:
                command += ['--calibration', str(a.calibration.expanduser().resolve())]
        command += ['--model', str(model), '--quant', a.quant, '--mode', 'w4a4',
                    '--protocol', {'smoke': 'custom10', 'tiny': 'tiny16', 'official': 'standard16'}[profile],
                    '--num-inference-steps', str(a.steps), '--seed', '0', '--tag', tag,
                    '--out-dir', str(directory)]
        if profile == 'smoke':
            command += ['--dims', 'subject_consistency', '--limit', '1']
        if count == 2:
            command += ['--repo', str(a.magi_repo.expanduser().resolve()), '--num-gpus', '2',
                        '--resolution', '1080p', '--refiner-num-inference-steps', str(a.refiner_steps),
                        '--conditioning-cache', str(out / 'conditioning')]
        else:
            command += ['--offload', 'model_cpu']
        return command

    preflight = []
    if a.stage != 'score':
        preflight.append(generation(a.profile, out / a.profile) + ['--dry-run'])
        if count == 1:
            preflight.append([generator, str(backend / 'scripts/check_minimax_runtime.py'), 'gen', '--allow-cpu'])
        else:
            preflight.append([generator, '-c',
                'import sys, torch, flash_attn_interface, magi_attention, magi_compiler; '
                'from transformers import Qwen3_5TextModel; '
                'from magi_attention.api import flex_flash_attn_func; '
                'from magi_attention import magi_attn_ext; '
                'assert sys.version_info[:2] == (3,12), "MAGI runtime requires Python 3.12"; '
                'print("MAGI Hopper runtime imports passed")'])
    if a.stage != 'generate':
        preflight += [[scorer, str(backend / 'scripts/check_minimax_runtime.py'), 'eval', '--allow-cpu']]
    phases = []
    profiles = ['smoke'] if a.profile == 'smoke' else ['smoke', a.profile]
    for profile in profiles:
        folder = out / profile
        if a.stage != 'score':
            gen = generation(profile, folder)
            if count == 2:
                gen = [generator, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=2'] + gen[1:]
            phases.append(dict(name=f'{profile}-generate', command=gen, gpus=count,
                               min_free_mib=minimum, result=str(folder / 'gen_summary.json'),
                               profile=profile))
        if a.stage != 'generate' and (profile != 'smoke' or a.profile == 'smoke'):
            phases.append(dict(name=f'{profile}-score',
                               command=[scorer, str(backend / 'eval_vbench.py'), str(folder), '--device', 'cuda'],
                               gpus=1, min_free_mib=16000,
                               result=str(folder / 'vbench_summary.json'), profile=profile))
    return dict(identity=identity, tag=tag, cwd=str(backend), output=str(out), env=runtime_env,
                candidates=candidates, preflight=preflight, phases=phases,
                note='Video WINA is an architecture adaptation of the selection rule; no 2:4, '
                     'no ARC residual channels, and no sparse-kernel speed claim.')


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def check_backend(plan):
    root = Path(plan['cwd'])
    if _revision(root) != plan['identity']['backend_revision']:
        raise ValueError('Evaluator revision mismatch; prepare the pinned base_eval first')
    changed = subprocess.check_output(['git', '-c', f'safe.directory={root}', '-C', str(root),
                                       'diff', 'HEAD', '--name-only'], text=True)
    if changed.strip():
        raise ValueError('Evaluator tracked files changed; use a clean pinned checkout')


def hardware_candidates(plan):
    """Do not wait forever for a pair that this server cannot ever provide."""
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in result.splitlines():
        index, name, memory, cap = [f.strip() for f in line.split(',')]
        if plan['candidates'] is None or index in plan['candidates']:
            rows.append(dict(index=index, name=name, memory_mib=int(memory), compute_cap=float(cap)))
    for phase in plan['phases']:
        valid = [r for r in rows if r['memory_mib'] >= phase['min_free_mib'] and (
            not (plan['identity']['model'] == 'magi-2-preview' and phase['name'].endswith('generate'))
            or r['compute_cap'] >= 9)]
        if len(valid) < phase['gpus']:
            raise ValueError(f"{phase['name']}: incompatible GPU count/memory/architecture: {rows}")
        phase['candidates'] = [r['index'] for r in valid]
    return rows


def checked_result(phase):
    path = Path(phase['result'])
    if not path.is_file():
        raise ValueError(f"{phase['name']} exited without a completed summary: {path}")
    data = json.loads(path.read_text())
    if phase['name'].endswith('generate'):
        expected_protocol = {'smoke': 'custom10', 'tiny': 'tiny16', 'official': 'standard16'}[phase['profile']]
        if data.get('protocol') != expected_protocol:
            raise ValueError('Generation summary protocol mismatch')
        expected = {'smoke': 1, 'tiny': 740, 'official': 6220}[phase['profile']]
        if data.get('num_videos') != expected:
            raise ValueError(f'Incomplete video suite: expected {expected} records')
    else:
        key = {'smoke': 'scores', 'tiny': 'tiny_scores', 'official': 'official_scores'}[phase['profile']]
        if not data.get(key):
            raise ValueError(f'VBench {key} missing; partial metrics are not a full result')


def terminate_child(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_child(command, env, cwd, log):
    print('[run] ' + shlex.join(command), flush=True)
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            terminate_child(process)
            raise
    if code:
        raise RuntimeError(f'Child exited {code}; see {log}')


def execute(a, plan, catalog):
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_file = out / 'identity.json'
        if identity_file.exists() and json.loads(identity_file.read_text()) != plan['identity']:
            raise ValueError('Model/method/protocol/code changed; use a new output directory')
        if not identity_file.exists() and any((out / profile).exists() for profile in ('smoke', 'tiny', 'official')):
            raise ValueError('Existing output has no queue identity; use a new directory')
        atomic_json(identity_file, plan['identity'])
        status = dict(state='preflight', model=a.model, method=a.method, sparsity=a.sparsity,
                      pid=os.getpid(), hostname=os.uname().nodename, stages={})

        def save(state, **updates):
            status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **updates)
            atomic_json(out / 'status.json', status)

        save('preflight')
        try:
            if a.magi_zero_evidence:
                from benchmarks.magi_validation import validate_zero_equivalence
                evidence = json.loads(a.magi_zero_evidence.read_text())
                if not evidence.get('complete'):
                    raise ValueError('Incomplete MAGI zero-sparsity validation')
                validate_zero_equivalence(evidence['baseline'], evidence['sparse'])
            check_backend(plan)
            status['hardware'] = hardware_candidates(plan)
            env = {**os.environ, **plan['env']}
            for tool in ('ffmpeg', 'ffprobe'):
                if not shutil.which(tool, path=env.get('PATH')):
                    raise ValueError(f'{tool} missing; install FFmpeg or set ARCQUANT_FFMPEG_BIN')
            report = audit_model(a.model, catalog['models'][a.model], override=a.model_path)
            if report['status'] != 'ready':
                raise ValueError(f'Model assets incomplete: {report}')
            report = audit_benchmark('vbench', catalog['benchmarks']['vbench'])
            if report['status'] != 'ready':
                raise ValueError(f'VBench assets incomplete: {report}')
            if a.stage != 'generate':
                report = audit_scorer(hash_contents=True)
                if report['status'] != 'ready':
                    raise ValueError(f'Scorer assets incomplete: {report}')
            for i, command in enumerate(plan['preflight']):
                run_child(command, {**env, 'CUDA_VISIBLE_DEVICES': ''}, plan['cwd'], out / f'preflight-{i}.log')
            for phase in plan['phases']:
                name = phase['name']
                if name.endswith('score'):
                    # In score-only mode, reject an unfinished suite before waiting for a card.
                    checked_result(dict(name=name.replace('-score', '-generate'),
                                        result=str(Path(phase['result']).with_name('gen_summary.json')),
                                        profile=phase['profile']))
                status['stages'][name] = 'waiting_for_gpus'
                save('waiting_for_gpus', current_stage=name)
                with wait_for_idle_gpus(phase['gpus'], phase['min_free_mib'], a.poll_seconds,
                                        candidates=phase['candidates']) as ids:
                    if build_plan(a, catalog)['identity'] != plan['identity']:
                        raise ValueError('Queue implementation changed while waiting; use a new output directory')
                    check_backend(plan)
                    status['stages'][name] = 'running'
                    save('running', gpu_ids=ids)
                    # Use a fresh process per phase; generator memory is released before scoring.
                    run_child(phase['command'], {**env, 'CUDA_VISIBLE_DEVICES': ids},
                              plan['cwd'], out / f'{name}.log')
                    checked_result(phase)
                    if a.model == 'magi-2-preview' and a.method != 'none' and name.endswith('generate'):
                        from benchmarks.magi_validation import validate_sparse_runtime
                        validate_sparse_runtime(Path(phase['result']).parent, a.method, a.sparsity)
                    if a.model == 'magi-2-preview' and name == 'tiny-score':
                        from benchmarks.magi_validation import validate_tiny_scores
                        validate_tiny_scores(Path(phase['result']).parent)
                status['stages'][name] = 'completed'
                save('between_stages', gpu_ids=None)
            save('completed', current_stage=None)
        except BaseException as exc:
            save('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def run(a):
    catalog = json.loads((ROOT / 'configs/experiments.json').read_text())
    plan = build_plan(a, catalog)
    if not a.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    execute(a, plan, catalog)
    return 0


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
