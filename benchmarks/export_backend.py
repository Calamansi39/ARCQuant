"""Export the pinned evaluator as a private, self-contained Git bundle.

Only committed history reachable from the catalogued commit is included.
The source checkout is never modified. Transfer the bundle privately to your
other server; it does not belong in the public ARCQuant repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def export_bundle(source: Path, destination: Path, revision: str) -> dict:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(f'Bundle or manifest already exists: {destination}')
    git = ['git', '-c', f'safe.directory={source}', '-C', str(source)]
    actual = subprocess.check_output(git + ['rev-parse', revision + '^{commit}'], text=True).strip()
    if actual != revision:
        raise ValueError('Expected a full pinned commit hash')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='arcquant-export-') as temporary:
        clone = Path(temporary) / 'backend'
        subprocess.run(['git', '-c', f'safe.directory={source}', 'clone', '--shared',
                        '--no-checkout', str(source), str(clone)], check=True, capture_output=True)
        command = ['git', '-C', str(clone)]
        subprocess.run(command + ['branch', '-f', 'arcquant-pinned', revision], check=True)
        subprocess.run(command + ['symbolic-ref', 'HEAD', 'refs/heads/arcquant-pinned'], check=True)
        bundle = Path(temporary) / 'base_eval.bundle'
        subprocess.run(command + ['bundle', 'create', str(bundle), 'HEAD',
                                  'refs/heads/arcquant-pinned'], check=True)
        subprocess.run(command + ['bundle', 'verify', str(bundle)], check=True, capture_output=True)
        # The bundle must clone without access to source objects or a Git server.
        restored = Path(temporary) / 'restored'
        subprocess.run(['git', 'clone', '--no-checkout', str(bundle), str(restored)],
                       check=True, capture_output=True)
        head = subprocess.check_output(['git', '-C', str(restored), 'rev-parse', 'HEAD'], text=True).strip()
        if head != revision:
            raise ValueError('Round-trip bundle revision mismatch')
        content = bundle.read_bytes()
        with destination.open('xb') as output:
            output.write(content)
    manifest = dict(backend='base-eval', revision=revision, sha256=hashlib.sha256(content).hexdigest(),
                    bytes=len(content), file=destination.name, clone_verified=True)
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'external/base_eval')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    catalog = json.loads((ROOT / 'configs/experiments.json').read_text())
    print(json.dumps(export_bundle(a.source, a.output, catalog['backends']['base-eval']['revision']), indent=2))


if __name__ == '__main__':
    main()
