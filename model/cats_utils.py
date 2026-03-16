import torch
import torch.nn as nn


class CATSActivationSparsifier(nn.Module):
    def __init__(self, num_bins=20000, hist_min=-2.0, hist_max=2.0):
        super().__init__()
        self.num_bins = num_bins
        self.hist_min = hist_min
        self.hist_max = hist_max
        bin_edges = torch.linspace(hist_min, hist_max, num_bins - 2, dtype=torch.float32)
        bin_edges = torch.cat(
            [
                torch.tensor([-torch.inf], dtype=torch.float32),
                bin_edges,
                torch.tensor([torch.inf], dtype=torch.float32),
            ]
        )
        self.register_buffer("histogram_bins", bin_edges)
        self.register_buffer("post_act_hist_counts", torch.zeros(num_bins - 1, dtype=torch.float64))
        self.register_buffer("threshold", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("zero_count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("total_count", torch.tensor(0.0, dtype=torch.float64))
        self.collect_histogram = False
        self.sparse_enabled = False

    @torch.no_grad()
    def begin_calibration(self):
        self.collect_histogram = True
        self.sparse_enabled = False
        self.post_act_hist_counts.zero_()
        self.reset_stats()

    @torch.no_grad()
    def finish_calibration(self, sparsity):
        counts = self.post_act_hist_counts
        if counts.sum().item() == 0 or sparsity <= 0:
            self.threshold.zero_()
            self.collect_histogram = False
            self.sparse_enabled = False
            self.reset_stats()
            return

        cumulative = torch.cumsum(counts / counts.sum(), dim=0)
        idx = torch.searchsorted(
            cumulative,
            torch.tensor(
                float(sparsity),
                dtype=cumulative.dtype,
                device=cumulative.device,
            ),
            side="right",
        )
        idx = int(torch.clamp(idx, 0, self.histogram_bins.numel() - 1).item())
        self.threshold.copy_(self.histogram_bins[idx].to(self.threshold.dtype))
        self.collect_histogram = False
        self.sparse_enabled = True
        self.reset_stats()

    @torch.no_grad()
    def disable(self):
        self.sparse_enabled = False
        self.reset_stats()

    @torch.no_grad()
    def enable(self):
        self.sparse_enabled = True
        self.reset_stats()

    @torch.no_grad()
    def reset_stats(self):
        self.zero_count.zero_()
        self.total_count.zero_()

    @torch.no_grad()
    def get_zero_ratio(self):
        if self.total_count.item() == 0:
            return 0.0
        return (self.zero_count / self.total_count).item()

    @torch.no_grad()
    def _update_histogram(self, x):
        x = torch.abs(x.float().detach())
        hist = torch.cat(
            (
                (x < self.hist_min).sum().unsqueeze(0),
                torch.histc(
                    x,
                    bins=self.num_bins - 3,
                    min=self.hist_min,
                    max=self.hist_max,
                ),
                (x > self.hist_max).sum().unsqueeze(0),
            )
        )
        self.post_act_hist_counts += hist.to(
            device=self.post_act_hist_counts.device,
            dtype=self.post_act_hist_counts.dtype,
        )

    @torch.no_grad()
    def _update_stats(self, x):
        self.zero_count += float(x.numel() - torch.count_nonzero(x))
        self.total_count += float(x.numel())

    def forward(self, x):
        if self.collect_histogram:
            self._update_histogram(x)

        if self.sparse_enabled and self.threshold.item() > 0:
            x = x * x.abs().gt(self.threshold)

        self._update_stats(x)
        return x


def _iter_cats_sparsifiers(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "cats_gate_sparsifier"):
            yield layer.mlp.cats_gate_sparsifier


@torch.no_grad()
def begin_cats_calibration(model):
    for sparsifier in _iter_cats_sparsifiers(model):
        sparsifier.begin_calibration()


@torch.no_grad()
def finish_cats_calibration(model, sparsity):
    for sparsifier in _iter_cats_sparsifiers(model):
        sparsifier.finish_calibration(sparsity)


@torch.no_grad()
def enable_cats_sparsity(model):
    for sparsifier in _iter_cats_sparsifiers(model):
        sparsifier.enable()


@torch.no_grad()
def disable_cats_sparsity(model):
    for sparsifier in _iter_cats_sparsifiers(model):
        sparsifier.disable()
