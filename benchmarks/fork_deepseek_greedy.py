"""Copy a consistent live greedy checkpoint into an independent method run.

Only the selected method list changes. Calibration inputs are propagated by
the unsparsified reference, so WINA and TEAL do not depend on each other's
search result. Never edit the parent's manifest or write into its directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from benchmarks.deepseek_greedy_strategy import SCHEMA, atomic_json, effective_sparsity


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def identity(path):
    s = Path(path).stat()
    return s.st_ino, s.st_size, s.st_mtime_ns


def fork_checkpoint(parent, destination, method, source):
    """Publish a fresh directory only after all ranks and prefixes validate.

    A live parent can advance during the copy. Such a snapshot is refused;
    callers may retry. Tensor files are copied, never hard-linked to live data.
    """
    import torch

    parent, destination, source = map(lambda p: Path(p).resolve(), (parent, destination, source))
    if destination.exists() or destination == parent or parent in destination.parents:
        raise ValueError('Use a fresh destination outside the parent run')
    raw = (parent / 'search_manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest.get('schema') != SCHEMA or method not in manifest.get('methods', []):
        raise ValueError('Method is absent from the parent search')
    if manifest.get('reference_step') != 'attn.wq_b cost; inverse-cost increments':
        raise ValueError('Unknown search semantics')
    for name, expected in manifest['implementation_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError(f'Frozen source mismatch: {name}')
    world = manifest['world_size']
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.fork-', dir=destination.parent))
    observed = {}
    files = {}

    def copy(relative):
        src, dst = parent / relative, temp / relative
        before = identity(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if identity(src) != before:
            raise RuntimeError(f'Parent advanced during snapshot: {relative}')
        observed[relative] = before
        files[relative] = digest(dst)

    try:
        boundary, offsets = None, None
        for rank in range(world):
            relative = f'inputs.rank{rank}.pt'
            copy(relative)
            state = torch.load(temp / relative, map_location='cpu', weights_only=True)
            if len(state['states']) != manifest['samples'] or len(state['offsets']) != manifest['samples']:
                raise ValueError('Input sample count differs from manifest')
            if boundary is None:
                boundary, offsets = state['next_layer'], state['offsets']
            elif boundary != state['next_layer'] or offsets != state['offsets']:
                raise RuntimeError('Ranks crossed a layer boundary during snapshot; retry')
            if not isinstance(boundary, int) or boundary < 0:
                raise ValueError('Invalid layer checkpoint')
            del state
        retained = []
        for layer in range(boundary + 1):
            layer_path = parent / method / f'layer-{layer}.json'
            required = layer < boundary
            # An in-progress layer is retained only once its calibration and
            # first search checkpoint have been published.
            if not layer_path.exists():
                if required:
                    raise ValueError(f'Missing completed prefix layer {layer}')
                continue
            relative = f'{method}/layer-{layer}.json'
            copy(relative)
            result = json.loads((temp / relative).read_text())
            if result['layer'] != layer or result['targets'] != manifest['targets']:
                raise ValueError('Layer identity or targets mismatch')
            if result['step'] != manifest['step'] or required and not result['complete']:
                raise ValueError('Incomplete/incompatible prefix layer')
            for profile in result['profiles']:
                if abs(effective_sparsity(profile['rates'], profile['weights']) - profile['target']) > 1e-6:
                    raise ValueError('Saved profile does not meet its exact budget')
            if required and {p['target'] for p in result['profiles']} != set(manifest['targets']):
                raise ValueError('Completed prefix lacks requested profiles')
            cost = f'calibration/layer-{layer}.costs.json'
            copy(cost)
            if json.loads((temp / cost).read_text())['weights'] != result['weights']:
                raise ValueError('Layer costs differ from search checkpoint')
            for rank in range(world):
                copy(f'calibration/layer-{layer}.rank{rank}.pt')
                coverage = f'calibration/layer-{layer}.rank{rank}.coverage.json'
                if (parent / coverage).exists():
                    copy(coverage)
            for suffix in ('replay', 'online'):
                diagnostic = f'{method}/layer-{layer}.{suffix}.json'
                if (parent / diagnostic).exists():
                    copy(diagnostic)
                elif required and suffix == 'online' and method == 'wina':
                    raise ValueError('Completed WINA layer lacks online check')
            retained.append(layer)
        if (parent / 'search_manifest.json').read_bytes() != raw:
            raise RuntimeError('Parent manifest changed during snapshot')
        # Inputs are atomically replaced at layer boundaries. Current-layer
        # JSON may keep advancing; the copied atomic version remains valid.
        for rank in range(world):
            relative = f'inputs.rank{rank}.pt'
            if identity(parent / relative) != observed[relative]:
                raise RuntimeError('Parent advanced to another layer; retry snapshot')
        selected = dict(manifest, methods=[method])
        atomic_json(temp / 'search_manifest.json', selected)
        record = dict(parent=str(parent), method=method, next_layer=boundary,
                      retained_layers=retained, parent_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                      created_at=datetime.now(timezone.utc).isoformat(), files_sha256=files,
                      changed_manifest_fields={'methods': {'before': manifest['methods'], 'after': [method]}},
                      reason='Independent method search on identical unsparsified reference inputs')
        atomic_json(temp / 'fork_provenance.json', record)
        # No stale combined calibration manifest or complete strategy is copied.
        # The resumed search regenerates and binds its own final artifacts.
        if destination.exists():
            raise FileExistsError(destination)
        temp.rename(destination)
        return record
    except BaseException:
        shutil.rmtree(temp)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--method', choices=('wina', 'teal'), required=True)
    a = p.parse_args(argv)
    result = fork_checkpoint(a.parent, a.destination, a.method, a.source)
    print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__':
    main()
