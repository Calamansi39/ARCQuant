"""Sparse input boundaries around the pinned MAGI batched expert operations."""

import os

import torch
import triton

from magi2_batched_moe import _merge, _pack, _write_contributions, swiglu_bf16
from magi2_fast_quant import quantize_batched_cuda


def _batch(controller, name, module, activation, probabilities, order, offsets, counts,
           contribution, head, first, batch, rows):
    tokens, heads, width = activation.shape
    topk = probabilities.shape[-1]
    packed = torch.empty((batch, rows, width), device=activation.device, dtype=activation.dtype)
    mapping = torch.empty((batch, rows), device=activation.device, dtype=torch.int64)
    _pack[(batch, triton.cdiv(rows, 32))](
        activation, order, offsets, counts, packed, mapping, *activation.stride(), head,
        first, head * tokens * topk, width, topk, rows,
        BD=triton.next_power_of_2(width), enable_fp_fusion=False)
    weights = slice(first, first + batch)
    norm = controller.norms.get(f"{name}.gate_up")
    packed = controller.mask(packed, f"{name}.gate_up",
                             norm=None if norm is None else norm[weights], valid_rows=mapping >= 0)
    if module._baseline_act:
        packed = quantize_batched_cuda(packed, module._baseline_quant)
    gate = torch.bmm(packed, module.W_gate[weights], out_dtype=torch.float32).clamp_(max=7.)
    up = torch.bmm(packed, module.W_up[weights], out_dtype=torch.float32).clamp_(min=-7., max=7.)
    hidden = swiglu_bf16(gate, up)
    del gate, up, packed
    norm = controller.norms.get(f"{name}.down")
    hidden = controller.mask(hidden, f"{name}.down",
                             norm=None if norm is None else norm[weights], valid_rows=mapping >= 0)
    if module._baseline_act:
        hidden = quantize_batched_cuda(hidden, module._baseline_quant)
    value = torch.bmm(hidden, module.W_down[weights], out_dtype=torch.float32)
    _write_contributions[(triton.cdiv(batch * rows, 32),)](
        value, mapping, probabilities[head], contribution, batch * rows, width,
        BD=triton.next_power_of_2(width), enable_fp_fusion=False)


@torch.no_grad()
def routed_batched(controller, name, module, activation, probabilities, indices):
    probabilities, indices = probabilities.contiguous(), indices.contiguous()
    tokens, heads, width = activation.shape
    if tokens == 0:
        return torch.empty_like(activation)
    experts = module.W_gate.shape[0] // heads
    topk = indices.shape[-1]
    global_ids = indices + torch.arange(heads, device=activation.device)[:, None, None] * experts
    order = global_ids.reshape(-1).argsort(stable=True)
    counts = torch.bincount(global_ids.reshape(-1), minlength=heads * experts)
    counts_cpu = counts.cpu().tolist()
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    output = torch.empty_like(activation)
    contribution = torch.empty((tokens * topk, width), device=activation.device, dtype=activation.dtype)
    batch_limit = int(os.environ.get("MAGI2_MOE_BATCH_EXPERTS", "16"))
    budget = int(os.environ.get("MAGI2_MOE_SCRATCH_MIB", "512")) * 2**20
    if batch_limit < 1 or budget < 1:
        raise ValueError("MAGI MoE batch and scratch limits must be positive")
    bytes_per_row = 24 * module.W_down.shape[1] + 24 * width + 8
    for head in range(heads):
        slots = indices[head].argsort(dim=-1, stable=True).contiguous()
        first, end = head * experts, (head + 1) * experts
        while first < end:
            batch = min(batch_limit, end - first)
            while True:
                rows = triton.cdiv(max(counts_cpu[first:first + batch]), 16) * 16
                if batch == 1 or batch * rows * bytes_per_row <= budget:
                    break
                batch = max(1, batch // 2)
            if rows:
                _batch(controller, name, module, activation, probabilities, order, offsets,
                       counts, contribution, head, first, batch, rows)
            first += batch
        _merge[(triton.cdiv(tokens, 32),)](
            contribution, slots, output, tokens, heads, head, width, topk,
            BD=triton.next_power_of_2(width), enable_fp_fusion=False)
    return output
