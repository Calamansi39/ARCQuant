"""Apply a verified per-projection greedy strategy to generation masks."""
from __future__ import annotations
from benchmarks.deepseek_greedy_strategy import projection_group
import torch
from .sparse_utils import topk_keep_count


class DeepSeekBudgetInferenceMixin:
    def configure_budget(self, rates, provenance):
        expected = {projection_group(key) for key in self.sparse_modules.values()}
        if set(rates) != expected:
            raise ValueError(f'Greedy strategy projection coverage mismatch: missing={sorted(expected-set(rates))[:5]}, extra={sorted(set(rates)-expected)[:5]}')
        self.greedy_rates = rates.copy()
        self.greedy_provenance = provenance
        # A second strategy on the same runtime must not reuse old TEAL thresholds.
        if hasattr(self, 'sparse_thresholds'):
            self.sparse_thresholds.clear()

    def _row_parallel_wina(self, x, key, rate):
        """Choose top-k over the logical input, then return this TP slice."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        rows = (self._active_rows(shape[1], x.device).repeat(shape[0]) if x.ndim == 3
                else torch.ones(flat.shape[0], dtype=torch.bool, device=x.device))
        pieces = []
        for start in range(0, len(flat), self.sparse_row_chunk):
            block = flat[start:start + self.sparse_row_chunk]
            score = block.float().abs()
            gathered = [torch.empty_like(score) for _ in range(self.ds.world_size)]
            self.ds.dist.all_gather(gathered, score)
            full = torch.cat(gathered, -1)
            selected = full.topk(topk_keep_count(full.shape[-1], rate), dim=-1).indices
            keep = torch.zeros_like(full, dtype=torch.bool).scatter_(-1, selected, True)
            keep = keep.chunk(self.ds.world_size, -1)[self.ds.rank]
            active = rows[start:start + len(block)]
            keep = keep | ~active[:, None]
            pieces.append(block * keep)
            self.sparse_counts[key] += int((~keep).sum())
            self.sparse_eligible[key] += int(active.sum()) * block.shape[-1]
        return torch.cat(pieces).view(shape)

    def _sparse_mask(self, x, weight, key):
        if not hasattr(self, 'greedy_rates'):
            raise RuntimeError('Greedy runtime must load a verified strategy before forward')
        group = projection_group(key)
        rate = self.greedy_rates[group]
        if rate == 0:
            return x
        if (self.greedy_provenance.get('method') == 'wina' and self.ds.world_size > 1
                and key in getattr(self, 'greedy_row_parallel_keys', ())):
            return self._row_parallel_wina(x, key, rate)
        uniform_rate = self.sparse_sparsity
        self.sparse_sparsity = rate
        try:
            # Reuse existing TEAL/WINA masks, prefill rules, TP and MoE mapping.
            return super()._sparse_mask(x, weight, key)
        finally:
            self.sparse_sparsity = uniform_rate

    def stats(self):
        result = super().stats()
        result.update(allocation='greedy',greedy_strategy=self.greedy_provenance,
                      per_projection_sparsity=self.greedy_rates)
        return result
