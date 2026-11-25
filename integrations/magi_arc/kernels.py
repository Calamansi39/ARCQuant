"""MAGI Hopper kernel: exact radix TopK + BF16/HiFP4 + natural 2:4 counts.

No 2:4 mask is imposed. cuBLAS still computes the selected BF16 GEMMs.
All addresses that can span long refiner sequences use 64-bit arithmetic.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _hifp4_row(values, B: tl.constexpr):
    # Identical group hierarchy and first-minimum lattice tie policy to the
    # baseline magi2_fast_quant._hifp4, applied AFTER selection/BF16 storage.
    groups: tl.constexpr = B // 64
    values = values.reshape(groups, 8, 2, 4)
    v16 = tl.max(tl.abs(values), 3)
    v8 = tl.max(v16, 2)
    maximum = tl.max(v8, 1)
    scale = maximum * (1. / 7.)
    scale = tl.where(scale == 0., 2. ** -48, scale)
    scale = tl.minimum(tl.maximum(scale, 2. ** -48), (2. ** 15) * 1.5)
    exponent = tl.floor(tl.log2(scale)).to(tl.int32)
    power = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    mantissa = tl.div_rn(scale, power) - 1.
    rounded = libdevice.nearbyint(mantissa * 4.) * .25
    overflow = rounded >= 1.
    rounded = tl.where(overflow, 0., rounded)
    exponent = tl.minimum(tl.maximum(exponent + overflow.to(tl.int32), -48), 15)
    rounded = tl.where((exponent == 15) & (rounded == .75), .5, rounded)
    power = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    scale = (1. + rounded) * power
    reciprocal = tl.div_rn(1., scale)
    e8 = v8 * reciprocal[:, None] >= 4.
    down8 = tl.where(e8, .5, 1.)
    e16 = (v16 * reciprocal[:, None, None]) * down8[:, :, None] >= 2.
    normalized = values * reciprocal[:, None, None, None]
    normalized = (normalized * down8[:, :, None, None]) * tl.where(e16[:, :, :, None], .5, 1.)
    best = tl.full((groups, 8, 2, 4), -1.75, tl.float32)
    distance = tl.abs(normalized + 1.75)
    for level in tl.static_range(1, 15):
        candidate = (level - 7) * .25
        new_distance = tl.abs(normalized - candidate)
        take = new_distance < distance
        best = tl.where(take, candidate, best)
        distance = tl.where(take, new_distance, distance)
    result = best * scale[:, None, None, None]
    result = (result * tl.where(e8[:, :, None, None], 2., 1.)) * tl.where(e16[:, :, :, None], 2., 1.)
    return result.reshape(B)


@triton.jit(do_not_specialize=['M'])
def _select(X, NORMS, VALID, Y, HIST, M, K: tl.constexpr, KEEP: tl.constexpr,
            WEIGHTED: tl.constexpr, PADDED: tl.constexpr, QUANT: tl.constexpr,
            STATS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    expert = tl.program_id(1)
    col = tl.arange(0, B)
    valid = row < (tl.load(VALID + expert) if PADDED else M)
    address = (expert.to(tl.int64) * M + row) * K + col
    values = tl.load(X + address, valid & (col < K), 0.).to(tl.float32)
    scores = tl.abs(values)
    if WEIGHTED:
        scores = scores * tl.load(NORMS + expert.to(tl.int64) * K + col, col < K, 0.)
    if KEEP == 0:
        mask = tl.full((B,), False, tl.int1)
    elif KEEP == K:
        mask = col < K
    else:
        # Nonnegative finite FP32 scores have monotone integer encodings.
        # Four radix histograms find the exact boundary; the prefix sum resolves
        # ties in original channel order, including all-zero rows.
        bits = scores.to(tl.uint32, bitcast=True)
        active = col < K
        remaining = tl.full((), KEEP, tl.int32)
        bins = tl.arange(0, 256)
        threshold = tl.full((), 0, tl.uint32)
        for shift in tl.static_range(24, -1, -8):
            digit = ((bits >> shift) & 255).to(tl.int32)
            hist = tl.histogram(tl.where(active, digit, 256), 256)
            cumulative = tl.cumsum(hist, reverse=True)
            boundary = tl.max(tl.where(cumulative >= remaining, bins, 0), 0)
            remaining -= tl.sum(tl.where(bins > boundary, hist, 0), 0)
            active = active & (digit == boundary)
            threshold = threshold | (boundary.to(tl.uint32) << shift)
        tied = (bits == threshold) & (col < K)
        tie_rank = tl.cumsum(tied.to(tl.int32))
        mask = (col < K) & ((bits > threshold) | (tied & (tie_rank <= remaining)))
    selected = tl.where(mask, values, 0.).to(tl.bfloat16).to(tl.float32)
    effective = _hifp4_row(selected, B) if QUANT else selected
    effective = effective.to(tl.bfloat16)
    tl.store(Y + address, effective, col < K)
    if STATS:
        groups = tl.arange(0, B // 4)
        nmask = tl.sum(mask.reshape(B // 4, 4).to(tl.int32), 1)
        nselected = tl.sum((selected != 0.).reshape(B // 4, 4).to(tl.int32), 1)
        neffective = tl.sum((effective != 0.).reshape(B // 4, 4).to(tl.int32), 1)
        stat = tl.arange(0, 8)
        h0 = tl.histogram(tl.where(valid & (groups < K // 4), nmask, 8), 8)
        h1 = tl.histogram(tl.where(valid & (groups < K // 4), nselected, 8), 8)
        h2 = tl.histogram(tl.where(valid & (groups < K // 4), neffective, 8), 8)
        tl.atomic_add(HIST + expert * 15 + stat, h0.to(tl.int64), stat < 5, sem='relaxed')
        tl.atomic_add(HIST + expert * 15 + 5 + stat, h1.to(tl.int64), stat < 5, sem='relaxed')
        tl.atomic_add(HIST + expert * 15 + 10 + stat, h2.to(tl.int64), stat < 5, sem='relaxed')


def select(x, keep, *, norms=None, quant='bf16', valid_rows=None, statistics=True):
    if (x.ndim != 3 or not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32)
            or x.shape[-1] % 4 or not 0 <= keep <= x.shape[-1]
            or x.shape[-1] > 32768 or quant not in ('bf16', 'hifp4')):
        raise ValueError('Expected CUDA BF16/FP32 [E,M,K], K <= 32768 divisible by four')
    x = x.contiguous()
    e, m, k = x.shape
    if norms is not None and (norms.shape != (e, k) or norms.dtype != torch.float32 or norms.device != x.device):
        raise ValueError('Expected FP32 norms [E,K] on the input device')
    if valid_rows is not None and (valid_rows.shape != (e,) or valid_rows.device != x.device):
        raise ValueError('Expected per-expert valid row counts on the input device')
    y = torch.empty_like(x, dtype=torch.bfloat16)
    hist = torch.zeros((e, 3, 5), dtype=torch.int64, device=x.device)
    if m:
        _select[(m, e)](x, norms, valid_rows, y, hist, m, k, keep, norms is not None,
                       valid_rows is not None, quant == 'hifp4', statistics,
                       max(64, triton.next_power_of_2(k)),
                       num_warps=4 if k <= 4096 else 8, enable_fp_fusion=False)
    return y, hist
