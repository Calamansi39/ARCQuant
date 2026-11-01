"""Experimental WINA hooks for the pinned MAGI-2 preview runtime.

MAGI-2 stores routed expert matrices as ``[input, output]`` and executes them
through a custom expert kernel.  This module keeps that layout explicit and
implements WINA for dense ``nn.Linear`` modules and routed preview experts.
Grouped refiner linears are reported as pending instead of being silently
treated as ordinary dense layers.
"""

from __future__ import annotations

from collections import Counter
from types import MethodType

import torch


class Magi2WinaController:
    def __init__(self, sparsity: float):
        if not 0 <= sparsity < 1:
            raise ValueError("WINA sparsity must be in [0, 1)")
        self.sparsity = float(sparsity)
        self.stats = Counter()

    @torch.no_grad()
    def mask_linear(self, x, weight):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if flat.shape[-1] != weight.shape[-1]:
            raise ValueError("MAGI-2 WINA input/weight dimensions differ")
        keep = max(1, int(flat.shape[-1] * (1 - self.sparsity)))
        norm = weight.detach().float().norm(dim=0).to(flat.device)
        pieces = []
        for start in range(0, flat.shape[0], 4096):
            block = flat[start:start + 4096]
            indices = (block.float().abs() * norm).topk(keep, dim=-1).indices
            selected = torch.zeros_like(block, dtype=torch.bool).scatter_(-1, indices, True)
            pieces.append(block * selected)
            self.stats["masked_elements"] += int((~selected).sum().item())
            self.stats["total_elements"] += block.numel()
        return torch.cat(pieces, dim=0).view(shape)

    @torch.no_grad()
    def mask_expert(self, x, weight):
        # MAGI-2 logical expert weights are [input, output].
        if x.shape[-1] != weight.shape[0]:
            raise ValueError("MAGI-2 expert input/weight dimensions differ")
        flat = x.reshape(-1, x.shape[-1])
        keep = max(1, int(flat.shape[-1] * (1 - self.sparsity)))
        norm = weight.detach().float().norm(dim=1).to(flat.device)
        indices = (flat.float().abs() * norm).topk(keep, dim=-1).indices
        selected = torch.zeros_like(flat, dtype=torch.bool).scatter_(-1, indices, True)
        self.stats["masked_elements"] += int((~selected).sum().item())
        self.stats["total_elements"] += flat.numel()
        return (flat * selected).view_as(x)


def install_wina(model, sparsity: float):
    """Install WINA on supported MAGI-2 modules and return a report."""
    controller = Magi2WinaController(sparsity)
    report = {"method": "wina", "sparsity": float(sparsity),
              "dense_modules": [], "routed_modules": [], "grouped_pending": [],
              "stats": controller.stats}

    # Dense modules have the ordinary [out, input] orientation.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(part in name.lower() for part in ("embed", "adapter")):
                continue

            def hook(mod, inputs, ctl=controller):
                return (ctl.mask_linear(inputs[0], mod.weight),) + inputs[1:]

            handle = module.register_forward_pre_hook(hook)
            # Run before the quantization pre-hook installed by magi2_quant.
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.move_to_end(handle.id, last=False)
            report["dense_modules"].append(name)

        elif type(module).__name__ == "CoreMultiHeadMoE":
            original = getattr(module, "_flash_forward", None)
            if original is None:
                continue

            def expert_forward(self, x_heads, topk_probs, topk_indices,
                               ctl=controller):
                from magi2_quant import quantize_tensor
                wg, wu, wd = self.W_gate, self.W_up, self.W_down
                tokens, heads, dim = x_heads.shape
                experts = wg.shape[0] // heads
                output = torch.zeros_like(x_heads, dtype=torch.float32)
                global_ids = topk_indices + torch.arange(
                    heads, device=x_heads.device)[:, None, None] * experts
                order = global_ids.reshape(-1).argsort(stable=True)
                ids = torch.arange(tokens, device=x_heads.device)[None, :, None].expand_as(
                    topk_indices).reshape(-1)[order]
                scores = topk_probs.reshape(-1)[order]
                counts = torch.bincount(global_ids.reshape(-1),
                                        minlength=heads * experts).cpu().tolist()
                start = 0
                for expert, count in enumerate(counts):
                    if count == 0:
                        continue
                    head = expert // experts
                    selected = ids[start:start + count]
                    inp = x_heads[selected, head].contiguous()
                    gate_in = ctl.mask_expert(inp, wg[expert])
                    up_in = ctl.mask_expert(inp, wu[expert])
                    if getattr(self, "_baseline_act", False):
                        gate_in = quantize_tensor(gate_in, self._baseline_quant)
                        up_in = quantize_tensor(up_in, self._baseline_quant)
                    gate = torch.mm(gate_in.float(), wg[expert].float()).clamp(max=7.)
                    up = torch.mm(up_in.float(), wu[expert].float()).clamp(min=-7., max=7.)
                    hidden = (gate * torch.sigmoid(1.702 * gate) * (up + 1.)).to(wd.dtype)
                    hidden = ctl.mask_expert(hidden, wd[expert])
                    if getattr(self, "_baseline_act", False):
                        hidden = quantize_tensor(hidden, self._baseline_quant)
                    value = torch.mm(hidden.float(), wd[expert].float())
                    value = (value * scores[start:start + count, None]).to(x_heads.dtype).float()
                    output[:, head].index_add_(0, selected, value)
                    start += count
                return output.to(x_heads.dtype)

            module._flash_forward = MethodType(expert_forward, module)
            report["routed_modules"].append(name)

        elif type(module).__name__ in ("GroupedLinearBase", "CustomGroupedLinear", "BaseLinear", "NativeMoELinear"):
            report["grouped_pending"].append(name)
    report["stats"] = dict(controller.stats)
    return report
