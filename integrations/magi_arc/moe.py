"""Sparse routed experts with bounded batches and deterministic baseline merge."""
import os

import torch
import triton

from magi2_batched_moe import _pack, _write_contributions, _merge, swiglu_bf16


@torch.no_grad()
def routed_experts(module, x, probs, indices):
    probs, indices = probs.contiguous(), indices.contiguous()
    s, heads, dim = x.shape
    wg,wu,wd = module.W_gate,module.W_up,module.W_down
    experts, topk = len(wg)//heads, indices.shape[-1]
    if not s:
        return torch.empty_like(x)
    ids = indices + torch.arange(heads,device=x.device)[:,None,None]*experts
    order = ids.reshape(-1).argsort(stable=True)
    counts = torch.bincount(ids.reshape(-1),minlength=len(wg))
    host_counts = counts.cpu().tolist()
    offsets = torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    output = torch.empty_like(x)
    contribution = torch.empty((s*topk,dim),device=x.device,dtype=x.dtype)
    max_batch = int(os.environ.get('MAGI2_MOE_BATCH_EXPERTS','16'))
    budget = int(os.environ.get('MAGI2_MOE_SCRATCH_MIB','512'))*2**20
    # Include FP32 WINA coordinates and both selection outputs in scratch.
    bytes_per_row = 32*wd.shape[1]+32*dim+8  # ARC may append up to K residual channels.
    for head in range(heads):
        slots = indices[head].argsort(dim=-1,stable=True).contiguous()
        first, end = head*experts,(head+1)*experts
        while first < end:
            batch = min(max_batch,end-first)
            while True:
                m = triton.cdiv(max(host_counts[first:first+batch]),16)*16
                if batch == 1 or batch*m*bytes_per_row <= budget:
                    break
                batch = max(1,batch//2)
            if m:
                packed = torch.empty((batch,m,dim),device=x.device,dtype=x.dtype)
                mapping = torch.empty((batch,m),device=x.device,dtype=torch.int64)
                _pack[(batch,triton.cdiv(m,32))](x,order,offsets,counts,packed,mapping,*x.stride(),
                    head,first,head*s*topk,dim,topk,m,BD=triton.next_power_of_2(dim),enable_fp_fusion=False)
                valid = counts[first:first+batch]
                inp = module._sparse_gate_up.apply(packed,first=first,valid_rows=valid)
                gate_weight=module._sparse_gate_up.expanded(wg[first:first+batch].transpose(1,2),0,first).transpose(1,2)
                gate = torch.bmm(inp,gate_weight,out_dtype=torch.float32).clamp_(max=7.)
                del gate_weight
                up_weight=module._sparse_gate_up.expanded(wu[first:first+batch].transpose(1,2),1,first).transpose(1,2)
                up = torch.bmm(inp,up_weight,out_dtype=torch.float32).clamp_(min=-7.,max=7.)
                del up_weight
                hidden = swiglu_bf16(gate,up)
                del packed,inp,gate,up
                selected = module._sparse_down.apply(hidden,first=first,valid_rows=valid)
                del hidden
                down_weight=module._sparse_down.expanded(wd[first:first+batch].transpose(1,2),0,first).transpose(1,2)
                value = torch.bmm(selected,down_weight,out_dtype=torch.float32)
                del down_weight
                _write_contributions[(triton.cdiv(batch*m,32),)](value,mapping,probs[head],contribution,
                    batch*m,dim,BD=triton.next_power_of_2(dim),enable_fp_fusion=False)
            first += batch
        _merge[(triton.cdiv(s,32),)](contribution,slots,output,s,heads,head,dim,topk,
            BD=triton.next_power_of_2(dim),enable_fp_fusion=False)
    return output
