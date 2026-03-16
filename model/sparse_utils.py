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
