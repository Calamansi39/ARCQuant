"""Architecture neutral WINA wrapper for video DiT linear layers.

MiniMax H3's pinned evaluator already owns the W4A4 quantizer.  This factory
adds the native WINA input rule before that quantizer's forward call, so the
text encoder, VAE, router and conditioning paths stay untouched.  TEAL is not
enabled here because video-model histograms are not interchangeable with the
DeepSeek calibration files.
"""

from __future__ import annotations

from collections import Counter

import torch


def make_wina_quantized_linear(base_class, sparsity: float, stats: Counter | None = None):
    if not 0 <= sparsity < 1:
        raise ValueError("WINA sparsity must be in [0, 1)")
    stats = stats if stats is not None else Counter()

    class WinaQuantizedLinear(base_class):
        def __init__(self, linear, quant_type, act_quant):
            norm = linear.weight.detach().float().norm(dim=0)
            super().__init__(linear, quant_type, act_quant)
            self.register_buffer("_arcquant_wina_norm", norm, persistent=False)

        @torch.no_grad()
        def forward(self, x):
            if self.training or sparsity == 0:
                return super().forward(x)
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            keep = max(1, int(flat.shape[-1] * (1 - sparsity)))
            norm = self._arcquant_wina_norm.to(flat.device)
            chunks = []
            masked = 0
            for start in range(0, flat.shape[0], 4096):
                block = flat[start:start + 4096]
                scores = block.float().abs() * norm
                indices = scores.topk(keep, dim=-1).indices
                mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
                chunks.append(block * mask)
                masked += int((~mask).sum().item())
            stats["masked_elements"] += masked
            stats["total_elements"] += flat.numel()
            return super().forward(torch.cat(chunks, dim=0).view(shape))

    WinaQuantizedLinear.__name__ = f"Wina{base_class.__name__}"
    return WinaQuantizedLinear, stats
