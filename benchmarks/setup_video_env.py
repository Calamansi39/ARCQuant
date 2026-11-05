"""Prepare isolated video environments using an existing compatible Torch Python.

Default: print the plan. --execute installs into a new --system-site-packages
venv, preserving the base Torch/torchvision versions. MAGI requires a prepared
official Hopper runtime (see docs/MULTI_SERVER_DEPLOYMENT.md).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.baseline_bridge import _revision
from benchmarks.fetch_assets import fetch_git
from benchmarks.paths import executable_path

DETECTRON = dict(url='https://github.com/facebookresearch/detectron2.git',
                revision='1e3e13bbf607b54f62205c4c33922521822fb298')
PROTECTED = ('torch', 'torchvision', 'torchaudio', 'triton')


def package_lines(path):
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')
            and re.split(r'[<>=!~\[; ]', line.strip())[0] not in PROTECTED]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['h3', 'scorer', 'magi'], required=True)
    p.add_argument('--base-python', required=True, help='existing Torch interpreter; MAGI: official Python 3.12 runtime')
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path)
    p.add_argument('--cuda-arch', choices=['8.0', '9.0'], help='required for scorer: A800=8.0, H20=9.0')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    if a.role == 'scorer' and not a.cuda_arch:
        p.error('--role scorer requires --cuda-arch 8.0 (A800) or 9.0 (H20)')
    base = executable_path(a.base_python)
    dest = a.destination.expanduser().resolve()
    catalog = json.loads((ROOT / 'configs/experiments.json').read_text())
    backend, runtime = ROOT / 'external/base_eval', ROOT / 'external/MAGI-2-preview'
    source = runtime if a.role == 'magi' else backend
    revision = catalog['models']['magi-2-preview']['runtime_source']['revision'] if a.role == 'magi' else catalog['backends']['base-eval']['revision']
    if _revision(source) != revision:
        raise ValueError('Prepare the pinned evaluator/runtime with arc.py provision first')
    requirements = source / {'h3': 'requirements-minimax-h3.txt',
        'scorer': 'requirements-vbench-a800.txt', 'magi': 'requirements.txt'}[a.role]
    packages = package_lines(requirements)
    if a.role == 'magi':
        packages += ['versioningit', 'ninja', 'depyf', 'graphviz', 'seaborn', 'debugpy',
                     'expecttest', 'nvidia-cutlass-dsl==4.4.2', 'quack-kernels==0.4.1']
    python = str(dest / 'bin/python')
    constraints = str(dest / 'torch-constraints.txt')
    commands = [[str(base), '-m', 'venv', '--system-site-packages', str(dest)],
                [python, '-m', 'pip', 'install', '-c', constraints, *packages]]
    detectron = ROOT / 'external/detectron2'
    if a.role == 'scorer':
        commands += [[python, '-m', 'pip', 'install', '--no-deps', 'pyiqa==0.1.13'],
                     [python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', str(detectron)]]
    check = ('import torch, flash_attn_interface, magi_attention, magi_compiler; '
             'from magi_attention.api import flex_flash_attn_func; '
             'from magi_attention import magi_attn_ext; '
             'from transformers import Qwen3_5TextModel') if a.role == 'magi' else (
             'import torch, torchvision, diffusers, transformers; from diffusers import ModularPipeline'
             if a.role == 'h3' else 'import torch, torchvision, detectron2._C, pyiqa; '
             'from importlib.metadata import version; assert version("transformers") == "4.33.2"')
    commands.append([python, '-c', check])
    plan = dict(role=a.role, base_python=str(base), destination=str(dest),
                cuda_arch=a.cuda_arch, cuda_home=str(a.cuda_home) if a.cuda_home else os.environ.get('CUDA_HOME'),
                source_revision=revision, requirements=packages, commands=commands,
                detectron=DETECTRON if a.role == 'scorer' else None)
    print(json.dumps(plan, indent=2), flush=True)
    if not a.execute:
        return
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    probe = ('import json, sys, torch, torchvision; from importlib.metadata import version, packages_distributions; '
             'print(json.dumps(dict(prefix=sys.prefix, cuda=torch.version.cuda, '
             'versions={n:version(n) for n in ' + repr(PROTECTED) + ' if n in packages_distributions()})))')
    info = json.loads(subprocess.check_output([str(base), '-c', probe], env=env, text=True))
    if not info['cuda'] or 'torch' not in info['versions'] or 'torchvision' not in info['versions']:
        raise ValueError('Base interpreter needs a matching CUDA Torch/torchvision pair')
    if dest == Path(info['prefix']).resolve():
        raise ValueError('Destination must differ from the base environment')
    if a.role == 'magi':
        subprocess.run([str(base), '-c', 'import sys, flash_attn_interface, magi_attention, magi_compiler; '
                        'assert sys.version_info[:2] == (3,12)'], env=env, check=True)
    if a.role == 'scorer':
        cuda = plan['cuda_home']
        if not cuda or not (Path(cuda) / 'bin/nvcc').is_file():
            raise ValueError('Scorer needs --cuda-home pointing to the CUDA toolkit matching base Torch')
        version = subprocess.check_output([str(Path(cuda) / 'bin/nvcc'), '--version'], text=True)
        match = re.search(r'release (\d+)\.(\d+)', version)
        if not match or '.'.join(match.groups()) != info['cuda']:
            raise ValueError(f"CUDA toolkit must match Torch CUDA {info['cuda']}; found {version}")
        env.update(CUDA_HOME=str(Path(cuda).resolve()), FORCE_CUDA='1',
                   TORCH_CUDA_ARCH_LIST=a.cuda_arch, MAX_JOBS=os.environ.get('MAX_JOBS', '4'))
    marker = dest / 'arcquant-env.json'
    identity = dict(role=a.role, base_python=str(base), source_revision=revision,
                    torch_versions=info['versions'], cuda_arch=a.cuda_arch, packages=packages)
    if dest.exists() and (not marker.is_file() or json.loads(marker.read_text()) != identity):
        raise ValueError('Existing destination is unmanaged or differs; choose a new environment directory')
    dest.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    Path(constraints).write_text(''.join(f'{n}=={v}\n' for n, v in info['versions'].items()))
    if a.role == 'scorer':
        fetch_git(DETECTRON, detectron)
    for command in commands:
        print('[install] ' + shlex.join(command), flush=True)
        subprocess.run(command, env=env, check=True)
    after = json.loads(subprocess.check_output([python, '-c', probe], env=env, text=True))
    if after['versions'] != info['versions']:
        raise RuntimeError('Base Torch versions changed in the new environment; do not start evaluation')
    variable = {'h3': 'ARCQUANT_H3_PYTHON', 'scorer': 'ARCQUANT_VBENCH_PYTHON', 'magi': 'ARCQUANT_MAGI2_PYTHON'}[a.role]
    print(f'export {variable}={shlex.quote(python)}', flush=True)


if __name__ == '__main__':
    main()
