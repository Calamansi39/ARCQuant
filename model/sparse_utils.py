import os

import torch
import torch.nn as nn


def _interp(x, xp, fp):
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, len(xp) - 1)

    xp_left = xp[idx - 1]
    xp_right = xp[idx]
    fp_left = fp[idx - 1]
    fp_right = fp[idx]

    t = (x - xp_left) / (xp_right - xp_left)
    return fp_left + t * (fp_right - fp_left)


class HistogramDistribution:
    def __init__(self, histogram_file, hidden_type):
        histogram = torch.load(histogram_file, map_location="cpu")
        self.bin_centers = histogram[f"{hidden_type}_centers"]
        self.counts = histogram[hidden_type]
        self.total_count = self.counts.sum()
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    def icdf(self, q):
        target_count = q * self.total_count
        idx = torch.searchsorted(self.cumulative_counts, target_count)

        if idx == 0:
            return self.bin_centers[0]
        if idx == len(self.bin_centers):
            return self.bin_centers[-1]

        lower_count = self.cumulative_counts[idx - 1]
        upper_count = self.cumulative_counts[idx]
        lower_value = self.bin_centers[idx - 1]
        upper_value = self.bin_centers[idx]

        fraction = (target_count - lower_count) / (upper_count - lower_count)
        return lower_value + fraction * (upper_value - lower_value)


class ThresholdSparsifier(nn.Module):
    def __init__(self, distribution, sparsity=0.0, apply_prefill=True):
        super().__init__()
        self.distribution = distribution
        self.apply_prefill = apply_prefill
        self.register_buffer("threshold", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("zero_count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("total_count", torch.tensor(0.0, dtype=torch.float64))
        self.set_sparsity(sparsity)

    def set_sparsity(self, sparsity):
        self.sparsity = float(sparsity)
        if self.sparsity <= 0.0:
            self.threshold.zero_()
            return
        threshold = self.distribution.icdf(0.5 + self.sparsity / 2).item()
        self.threshold.fill_(float(threshold))

    def apply_mask(self, x):
        if self.sparsity <= 0.0:
            return x
        return x * x.abs().gt(self.threshold)

    @torch.no_grad()
    def update_stats(self, x):
        self.zero_count += float(x.numel() - torch.count_nonzero(x))
        self.total_count += float(x.numel())

    @torch.no_grad()
    def reset_stats(self):
        self.zero_count.zero_()
        self.total_count.zero_()

    def get_zero_ratio(self):
        if self.total_count.item() == 0:
            return 0.0
        return (self.zero_count / self.total_count).item()

    def forward(self, x):
        if self.sparsity <= 0.0:
            self.update_stats(x)
            return x

        if x.size(1) > 1 and self.apply_prefill:
            half_seq_len = x.size(1) // 2
            last_context = x[:, -half_seq_len:, :]
            masked_context = self.apply_mask(last_context)
            out = torch.cat((x[:, :-half_seq_len, :], masked_context), dim=1)
            self.update_stats(out)
            return out

        if x.size(1) > 1 and not self.apply_prefill:
            self.update_stats(x)
            return x

        out = self.apply_mask(x)
        self.update_stats(out)
        return out


def build_llama_teal_sparsifiers(layer_idx, histogram_root, sparsity, apply_prefill=True):
    layer_root = os.path.join(histogram_root, f"layer-{layer_idx}")
    attn_hist = os.path.join(layer_root, "self_attn", "histograms.pt")
    mlp_hist = os.path.join(layer_root, "mlp", "histograms.pt")

    attn_h1 = HistogramDistribution(attn_hist, "h1")
    attn_h2 = HistogramDistribution(attn_hist, "h2")
    mlp_h1 = HistogramDistribution(mlp_hist, "h1")
    mlp_h2 = HistogramDistribution(mlp_hist, "h2")

    return nn.ModuleDict(
        {
            "q": ThresholdSparsifier(attn_h1, sparsity=sparsity, apply_prefill=apply_prefill),
            "k": ThresholdSparsifier(attn_h1, sparsity=sparsity, apply_prefill=apply_prefill),
            "v": ThresholdSparsifier(attn_h1, sparsity=sparsity, apply_prefill=apply_prefill),
            "o": ThresholdSparsifier(attn_h2, sparsity=sparsity, apply_prefill=apply_prefill),
            "gate": ThresholdSparsifier(mlp_h1, sparsity=sparsity, apply_prefill=apply_prefill),
            "up": ThresholdSparsifier(mlp_h1, sparsity=sparsity, apply_prefill=apply_prefill),
            "down": ThresholdSparsifier(mlp_h2, sparsity=sparsity, apply_prefill=apply_prefill),
        }
    )


class WeightInformedTopKSparsifier(nn.Module):
    """Apply WINA's per-token top-k mask before ARCQuant activation quantization.

    The score is ``abs(activation) * column_l2_norm(weight)``. The original
    WINA top-k path masks the last 99% of prefill tokens and all decode tokens.
    This class implements the uniform-sparsity variant, without WINA's
    optional histogram and greedy per-matrix allocation stage.
    """

    def __init__(self, column_norm, sparsity=0.0, apply_prefill=True):
        super().__init__()
        if not 0 <= sparsity < 1:
            raise ValueError("WINA sparsity must be in [0, 1)")
        self.sparsity = float(sparsity)
        self.apply_prefill = apply_prefill
        self.register_buffer("column_norm", column_norm.detach().float().clone())
        self.register_buffer("zero_count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("total_count", torch.tensor(0.0, dtype=torch.float64))

    @torch.no_grad()
    def reset_stats(self):
        self.zero_count.zero_()
        self.total_count.zero_()

    def get_zero_ratio(self):
        if self.total_count.item() == 0:
            return 0.0
        return (self.zero_count / self.total_count).item()

    def forward(self, x):
        if x.shape[-1] != self.column_norm.numel():
            raise ValueError("WINA column norm dimension does not match activation width")
        if self.sparsity == 0 or (x.shape[1] > 1 and not self.apply_prefill):
            out = x
        else:
            start = x.shape[1] - max(1, int(0.99 * x.shape[1]))
            tail = x[:, start:, :]
            score = tail.float().abs() * self.column_norm
            keep = int(tail.shape[-1] * (1 - self.sparsity))
            indices = score.topk(keep, dim=-1).indices
            mask = torch.zeros_like(tail, dtype=torch.bool).scatter_(-1, indices, True)
            out = x.clone()
            out[:, start:, :] = tail * mask
        with torch.no_grad():
            self.zero_count += float(out.numel() - torch.count_nonzero(out))
            self.total_count += float(out.numel())
        return out


def build_llama_wina_sparsifiers(original_module, kind, sparsity, apply_prefill=True):
    """Mirror the weight-informed projections used in microsoft/wina.

    WINA's released uniform top-k setup weights the Llama K and gate scores by
    their input-column norms. Other projections use a unit norm.
    """
    projections = ("q", "k", "v", "o") if kind == "self_attn" else ("gate", "up", "down")
    informed = {"k"} if kind == "self_attn" else {"gate"}
    result = {}
    for name in projections:
        weight = getattr(original_module, f"{name}_proj").weight
        norm = weight.detach().float().norm(dim=0) if name in informed else torch.ones(weight.shape[1], device=weight.device)
        result[name] = WeightInformedTopKSparsifier(norm, sparsity, apply_prefill)
    return nn.ModuleDict(result)
