"""Native threshold-search adapter for DeepSeek MLA/HC/MoE and TP.

Uses the released layer forward and ARC projection implementation. WINA
calibration keeps |x * column_norm| in FP16 and its empirical CDF, matching
the upstream grab_acts/Distribution/threshold search. TEAL keeps signed bins.
Routed experts share a rate per projection family; this is an explicit MoE
extension of the original dense-model per-projection search.
"""
from __future__ import annotations

from collections import Counter
import torch

from benchmarks.deepseek_greedy_strategy import projection_group
from .deepseek_experiment import (ExperimentRuntimeMixin, InputCalibration, prefill_rows,
                                 histogram_threshold, evenly_spaced_indices)


def empirical_threshold(centers, counts, quantile):
    cumulative = counts.float().cumsum(0)
    target = quantile * cumulative[-1]
    i = int(torch.searchsorted(cumulative, target))
    if i == 0:
        return float(centers[0])
    if i >= len(centers):
        return float(centers[-1])
    fraction = (target - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1])
    return float(centers[i - 1] + fraction * (centers[i] - centers[i - 1]))


class DeepSeekGreedyRuntimeMixin(ExperimentRuntimeMixin):
    def start_collection(self, layer_id, sample_limit=16384):
        self.collect, self.arc, self.method, self.sparsity = True, False, 'none', 0.
        self.quant_type, self.act_quant = 'bf16', False
        self.layer_id, self.sample_limit = layer_id, sample_limit
        self.observers, self.weighted_samples, self.norms = {}, {}, {}
        self.costs = Counter()
        self.supplementing = False
        self.rates = {}
        self._clear_caches()

    def _clear_caches(self):
        self.cache.clear()
        self.cache_bytes = 0
        self._frequent_keys.clear()
        self.arc_cache.clear()
        self.arc_cache_bytes = 0
        self.device_specs.clear()
        self.threshold_cache = {}

    def _column_importance(self, key, x, weight):
        if key not in self.norms:
            informed = key.endswith('.attn.wkv.weight') or key.endswith('.w1.weight')
            self.norms[key] = (self.dequantize_weight(weight).float().norm(dim=0)
                               if informed else torch.ones(x.shape[-1], device=x.device))
        return self.norms[key]

    def _observe(self, key, x):
        self.observers.setdefault(key, InputCalibration(x.shape[-1], self.sample_limit)).update(x)
        module = self.modules.get(key)
        grouped = module is None
        if grouped:
            module, _, groups = self.group_weights[key]
        else:
            groups = 1
        flat = x.detach().reshape(-1, x.shape[-1])
        score = (flat.float() * self._column_importance(key, flat, module.weight)).half().abs()
        if not torch.isfinite(score).all():
            raise ValueError(f'WINA FP16 calibration overflow at {key}')
        take = min(2048, score.numel())
        index = evenly_spaced_indices(score.numel(), take, score.device)
        items = self.weighted_samples.setdefault(key, [])
        items.append(score.flatten()[index].cpu())
        if sum(v.numel() for v in items) > 2 * self.sample_limit:
            merged = torch.cat(items)
            self.weighted_samples[key] = [merged[evenly_spaced_indices(len(merged), self.sample_limit)]]
        if not self.supplementing:
            # Count logical MACs, not packed FP4 storage bytes. Replicated
            # dense projections contribute once globally; sharded/expert ones sum.
            sharded = isinstance(module, (self.ds.RowParallelLinear, self.ds.ColumnParallelLinear))
            copies = 1 if sharded or '.experts.' in key else self.ds.world_size
            self.costs[projection_group(key)] += flat.shape[0] * module.in_features * (module.out_features // groups) / copies

    def finish_layer_collection(self):
        result = {}
        for key, observer in self.observers.items():
            item = observer.finish()
            values = torch.cat(self.weighted_samples[key])
            if values.numel() > self.sample_limit:
                values = values[evenly_spaced_indices(len(values), self.sample_limit)]
            centers, counts = torch.unique(values.float(), sorted=True, return_counts=True)
            item.update(wina_centers=centers, wina_counts=counts.float())
            result[key] = item
        return result

    def use_quantization(self, calibration, quantization):
        self.collect, self.arc = False, quantization == 'arc'
        self.quant_type, self.act_quant = 'hifp4', True
        self.calibration = calibration
        self._clear_caches()

    def set_search_rates(self, method, rates):
        self.method, self.rates = method, rates.copy()
        self.sparsity = max(rates.values(), default=0.)
        self.mask_mode = 'threshold'

    def _spec(self, key, x):
        # ARC channel bookkeeping must not depend on the changing search rate.
        if key not in self.device_specs:
            src = self.calibration[key]
            self.device_specs[key] = src['order'].to(x.device), int(src['select']), None
        return self.device_specs[key]

    def _mask(self, x, weight, key):
        if self.method == 'none':
            return x
        group = projection_group(key)
        if group not in self.rates:
            raise ValueError(f'Missing candidate budget for {group}')
        rate = self.rates[group]
        if rate == 0:
            return x
        if self.method == 'wina' and getattr(self, 'mask_mode', 'threshold') == 'topk':
            old = self.sparsity
            self.sparsity = rate
            try:
                return ExperimentRuntimeMixin._mask(self, x, weight, key)
            finally:
                self.sparsity = old
        flat = x.reshape(-1, x.shape[-1])
        if x.ndim == 3:
            rows = prefill_rows(x.shape[1], self.method, x.device).repeat(x.shape[0])
        elif self.row_mask is not None:
            rows = self.row_mask
        else:
            raise ValueError(f'Unknown routed token positions: {key}')
        if rows.numel() != flat.shape[0]:
            raise ValueError('Routed row mask mismatch')
        if key not in self.calibration:
            raise ValueError(f'No calibrated distribution for {key}')
        cache_key = (self.method, key, rate)
        if cache_key not in self.threshold_cache:
            data = self.calibration[key]
            self.threshold_cache[cache_key] = (histogram_threshold(data['centers'], data['counts'], rate)
                if self.method == 'teal' else empirical_threshold(data['wina_centers'], data['wina_counts'], rate))
        scores = flat.float().abs()
        if self.method == 'wina':
            scores = scores * self._column_importance(key, flat, weight)
        keep = (scores > self.threshold_cache[cache_key]) | ~rows[:, None]
        return (flat * keep).view_as(x)

    def _arc_weight(self, weight, key, order, select, numeric=None):
        # Layer-wise search revisits the same experts many times. A bounded
        # cache is valid because only activation masks change across candidates.
        if key in self.arc_cache:
            return self.arc_cache[key]
        result = super()._arc_weight(weight, key, order, select, numeric)
        size = result.numel() * result.element_size()
        if key not in self.arc_cache and self.arc_cache_bytes + size <= self.cache_limit:
            self.arc_cache[key] = result
            self.arc_cache_bytes += size
        return result


def reset_layer_state(layer):
    """Clear mutable KV/compressor state before every independent replay."""
    for module in layer.modules():
        for name in ('kv_cache', 'kv_state'):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor):
                value.zero_()
        value = getattr(module, 'score_state', None)
        if isinstance(value, torch.Tensor):
            value.fill_(-float('inf'))
