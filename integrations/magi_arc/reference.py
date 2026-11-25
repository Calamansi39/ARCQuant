"""Independent, bounded PyTorch oracle for selection, quantization and counts."""
import math

import torch


STAGES = ('mask', 'selected_bf16', 'effective_native')


def keep_count(width, sparsity):
    if not math.isfinite(sparsity) or not 0 <= sparsity <= 1:
        raise ValueError('Sparsity must be finite and in [0, 1]')
    return max(0, min(width, math.floor(width * (1 - sparsity) + 1e-8)))


@torch.no_grad()
def select(x, keep, *, norms=None, quant='bf16', valid_rows=None, statistics=True):
    """x is [E,M,K]; ties retain smaller original channel indices.

    Scores are FP32, selection precedes BF16 rounding and HiFP4 W4A4.
    valid_rows excludes expert padding from all histograms.
    """
    from arcquant_fakequant import QUANT_FUNCS
    if x.ndim != 3 or x.shape[-1] % 4 or not 0 <= keep <= x.shape[-1]:
        raise ValueError('Expected [E,M,K], K divisible by four, and 0 <= keep <= K')
    if quant not in ('bf16', 'hifp4'):
        raise ValueError(quant)
    e, m, k = x.shape
    output = torch.empty_like(x, dtype=torch.bfloat16)
    hist = torch.zeros((e, 3, 5), dtype=torch.int64, device=x.device)
    lengths = [m]*e if valid_rows is None else valid_rows.cpu().tolist()
    for expert in range(e):
        valid = lengths[expert]
        for start in range(0, m, 64):
            values = x[expert, start:start + 64].float()
            if keep in (0,k):
                mask=torch.full_like(values,keep==k,dtype=torch.bool)
            else:
                scores = values.abs()
                if norms is not None:
                    scores = scores * norms[expert].float()
                order = scores.argsort(dim=-1, descending=True, stable=True)
                mask = torch.zeros_like(values, dtype=torch.bool)
                mask.scatter_(1, order[:, :keep], True)
            selected = values.masked_fill(~mask, 0).to(torch.bfloat16)
            result = (QUANT_FUNCS['hifp4'](selected.float()).to(torch.bfloat16)
                      if quant == 'hifp4' else selected)
            result[max(0, valid - start):] = 0
            output[expert, start:start + len(values)] = result
            length = max(0, min(len(values), valid - start))
            if statistics:
                for stage, data in enumerate((mask, selected != 0, result != 0)):
                    nnz = data[:length].reshape(-1, 4).sum(-1)
                    hist[expert, stage] += torch.bincount(nnz, minlength=5)
    return output, hist
