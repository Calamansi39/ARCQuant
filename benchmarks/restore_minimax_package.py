"""Copied to restore.py in the A800 transfer package; verifies then restores Git repos.

Requires Python 3.10+ and Git. No downloads, environment installation or GPU use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(package):
    manifest = json.loads((package / 'MANIFEST.json').read_text())
    if manifest.get('format') != 'arcquant-minimax-a800-v1':
        raise ValueError('Unknown package format')
    for name, record in manifest['files'].items():
        path = (package / name).resolve()
        if not path.is_relative_to(package.resolve()):
            raise ValueError(f'File escapes package: {name}')
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Package file missing or checksum mismatch: {name}')
    print(f"[ok] {len(manifest['files'])} package files verified", flush=True)
    return manifest


def restore(package, workspace, manifest):
    destination = workspace.expanduser().resolve() / 'ARCQuant'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing repository: {destination}')
    workspace.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    bundle = package / 'bundles/ARCQuant.bundle'
    subprocess.run(['git', 'clone', str(bundle), str(destination)], check=True)
    revision = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != manifest['repositories']['arcquant']:
        raise ValueError('Restored ARCQuant commit differs from package manifest')
    subprocess.run(['git', '-C', str(destination), 'switch', '-c', 'deployment/minimax-a800'], check=True)
    subprocess.run(['git', '-C', str(destination), 'remote', 'set-url', 'origin',
                    'https://github.com/Calamansi39/ARCQuant.git'], check=True)
    # The CLI creates the evaluator checkout and portable sibling asset links.
    subprocess.run([sys.executable, str(destination / 'arc.py'), 'fetch', '--backend', 'base-eval',
                    '--source', str(package / 'bundles/base_eval.bundle')], cwd=destination, check=True)
    backend = destination / 'external/base_eval'
    revision = subprocess.check_output(['git', '-C', str(backend), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != manifest['repositories']['base_eval']:
        raise ValueError('Restored evaluator commit differs from package manifest')
    (destination / 'external/transfer-package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'[ok] Restored {destination}; next follow README.md for assets and environments', flush=True)
    return destination


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, help='parent directory; creates workspace/ARCQuant')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args(argv)
    if not a.check_only and a.workspace is None:
        p.error('--workspace is required unless --check-only')
    package = Path(__file__).resolve().parent
    manifest = verify(package)
    if not a.check_only:
        restore(package, a.workspace, manifest)


if __name__ == '__main__':
    main()
