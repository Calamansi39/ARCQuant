"""Bounded-memory SVD preparation, shared by calibration and generation."""
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from integrations.magi_arc.protocol import digest, sha256
from vbench_generation_cache import atomic_write_json, cache_lock


def tensor_hash(tensor):
    import hashlib
    h = hashlib.sha256()
    rows=max(1,(32<<20)//max(1,tensor[0].numel()*tensor.element_size())) if len(tensor) else 1
    for chunk in tensor.detach().split(rows, dim=0):
        h.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def basis_for(weights, directory, name, *, device='cuda', batch=8):
    """weights: list of [E,N,K], concatenated logically along N for SVD.

    Keep bases on CPU: registering them as buffers would exceed preview VRAM
    after the official component .to(cuda). Only the current projection streams.
    """
    e, _, k = weights[0].shape
    identity = dict(schema='magi-svd-fp64-gram-v1', name=name,
        weights=[dict(shape=list(w.shape), sha256=tensor_hash(w)) for w in weights])
    folder = Path(directory) / digest(identity)
    path = folder / 'basis.safetensors'
    with cache_lock(folder, blocking=True):
        meta = folder / 'basis.json'
        if path.exists() and meta.exists():
            import json
            report = json.loads(meta.read_text())
            if report['identity'] != identity or report['sha256'] != sha256(path):
                raise ValueError(f'Corrupt SVD asset: {path}')
            return load_file(str(path))['basis'], report
        pieces, diagnostics = [], []
        for first in range(0, e, batch):
            count = min(batch, e-first)
            gram = torch.zeros((count, k, k), device=device, dtype=torch.float64)
            for w in weights:
                # Avoid a tall U and bound FP64 scratch by output-row chunks.
                for n in range(0, w.shape[1], 1024):
                    part = w[first:first+count, n:n+1024].to(device=device, dtype=torch.float64)
                    gram.baddbmm_(part.transpose(1, 2), part)
            eigenvalues, q = torch.linalg.eigh(gram)
            q = q.flip(2).contiguous()
            eigenvalues = eigenvalues.flip(1)
            # Canonical signs make independent rank/cache creation reproducible.
            pivots = q.abs().argmax(dim=1, keepdim=True)
            signs = q.gather(1, pivots).sign()
            q *= torch.where(signs == 0, 1, signs)
            eye = torch.eye(k, device=device, dtype=torch.float64)
            orth = (q.transpose(1,2) @ q - eye).abs().amax()
            residual = ((gram @ q - q * eigenvalues[:, None]).norm(dim=(1,2))
                        / gram.norm(dim=(1,2)).clamp_min(1e-30)).amax()
            if float(orth) > 1e-8 or float(residual) > 1e-8:
                raise ValueError('SVD accuracy check failed')
            diagnostics.append(dict(first=first, orthogonality=float(orth), relative_residual=float(residual)))
            pieces.append(q.float().cpu())
            del gram, q
        basis = torch.cat(pieces)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        save_file({'basis': basis}, str(tmp))
        os.replace(tmp, path)
        report = dict(identity=identity, path=str(path.resolve()), sha256=sha256(path), diagnostics=diagnostics)
        atomic_write_json(meta, report)
        return basis, report


@torch.no_grad()
def fold_(weights, basis, *, device='cuda', batch=8):
    for first in range(0, len(basis), batch):
        q = basis[first:first+batch].to(device=device, dtype=torch.float64)
        for w in weights:
            for n in range(0, w.shape[1], 1024):
                target = w[first:first+batch, n:n+1024]
                value = target.to(device=device, dtype=torch.float64) @ q
                target.copy_(value.to(device=target.device, dtype=target.dtype))


@torch.no_grad()
def column_norms(weights, *, device='cuda', batch=16):
    e, _, k = weights[0].shape
    result = torch.empty((e,k), dtype=torch.float32)
    for first in range(0,e,batch):
        acc = torch.zeros((min(batch,e-first), k), device=device, dtype=torch.float64)
        for w in weights:
            for n in range(0,w.shape[1],1024):
                acc += w[first:first+batch,n:n+1024].to(device=device,dtype=torch.float64).square().sum(1)
        result[first:first+batch] = acc.sqrt().float().cpu()
    return result
