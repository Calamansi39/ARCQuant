"""DeepSeek whole-model quality runtime: HiF4, ARC, TEAL and WINA.

The quantizer reconstructs HiF4 values and uses dense GEMMs. This is an accuracy
implementation, not a sparse speed kernel. See docs/DEEPSEEK_EXPERIMENT.md for
the explicit mapping from the upstream dense-model methods to MLA and MoE.
"""
from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn.functional as F

from .deepseek_adapter import projection_kind


def reference_histogram(values, bins=10000):
    """Signed TEAL bins, including its 1% outlier edge bins."""
    values = values.float().cpu().flatten().sort().values
    n = values.numel()
    if not n or not torch.isfinite(values).all():
        raise ValueError("Empty/non-finite calibration sample")
    if values[0] == values[-1]:
        return values[:1], torch.tensor([n], dtype=torch.float32)
    tail = max(1, int(.01 * n))
    lo, hi = values[min(n-1, tail)], values[max(0, n-tail)]
    if lo >= hi:
        lo, hi = values[0], values[-1]
    first = torch.nextafter(values[0], torch.tensor(-float('inf'), dtype=torch.float32))
    last = torch.nextafter(values[-1], torch.tensor(float('inf'), dtype=torch.float32))
    edges = torch.cat((first[None], torch.linspace(lo.item(), hi.item(), bins-1,
                                                dtype=torch.float32), last[None]))
    edges = edges.unique_consecutive()
    counts = torch.histogram(values, bins=edges).hist
    return (edges[1:] + edges[:-1]) / 2, counts


def histogram_threshold(centers, counts, sparsity):
    cumulative = counts.cumsum(0)
    target = (.5 + sparsity / 2) * cumulative[-1]
    i = int(torch.searchsorted(cumulative, target))
    if i == 0:
        return float(centers[0])
    if i == len(centers):
        return float(centers[-1])
    fraction = (target - cumulative[i-1]) / (cumulative[i] - cumulative[i-1])
    return float(centers[i-1] + fraction * (centers[i] - centers[i-1]))


def prefill_rows(length, method, device):
    if length == 1:
        return torch.ones(1, dtype=torch.bool, device=device)
    count = length // 2 if method == 'teal' else max(1, int(.99 * length))
    return torch.arange(length, device=device) >= length - count


class InputCalibration:
    def __init__(self, width, sample_limit=16384):
        self.maximum = torch.zeros(width, dtype=torch.float32)
        self.samples = []
        self.sample_limit = sample_limit
        self.rows = self.elements = self.selected = 0

    def update(self, x):
        # Copy once to host: calibration is offline and this also makes CUDA
        # indexing errors surface at the actual preceding model operation.
        x = x.detach().float().reshape(-1, x.shape[-1]).cpu()
        if not x.numel():
            return
        if not torch.isfinite(x).all():
            raise ValueError('Non-finite calibration input')
        self.maximum = torch.maximum(self.maximum, x.abs().amax(0))
        # The signed max/8 rule is from utilize.search_select_proportions.
        self.selected += int((x > x.amax(-1, keepdim=True) * .125).sum())
        self.elements += x.numel()
        self.rows += x.shape[0]
        # Deterministic bounded sample from every observed event, rather than
        # retaining only the first prompt. All channels contribute to ARC max.
        flat = x.flatten()
        take = min(2048, flat.numel())
        indices = torch.linspace(0, flat.numel() - 1, take).floor().long().clamp_max(flat.numel() - 1)
        self.samples.append(flat[indices])
        if sum(t.numel() for t in self.samples) > self.sample_limit * 2:
            merged = torch.cat(self.samples)
            indices = torch.linspace(0, merged.numel() - 1, self.sample_limit).floor().long().clamp_max(merged.numel() - 1)
            self.samples = [merged[indices]]

    def finish(self):
        if not self.elements:
            raise ValueError('Projection received no calibration rows')
        values = torch.cat(self.samples)
        if values.numel() > self.sample_limit:
            values = values[torch.linspace(0, values.numel()-1, self.sample_limit).long()]
        centers, counts = reference_histogram(values)
        width = self.maximum.numel()
        selected = min(width, math.ceil(width * self.selected / self.elements / 64) * 64)
        return dict(order=self.maximum.argsort(stable=True), select=selected,
                    centers=centers, counts=counts, rows=self.rows,
                    histogram_elements=values.numel())


class ExperimentRuntimeMixin:
    """Mix into the external, version-recorded DeepSeek quantization runtime."""
    def configure(self, ds, model, *, calibration=None, method='none', sparsity=0.,
                  arc=False, collect=False):
        if method not in ('none', 'teal', 'wina') or not 0 <= sparsity < 1:
            raise ValueError('Invalid method/sparsity')
        self.ds, self.model = ds, model
        self.method, self.sparsity, self.arc, self.collect = method, sparsity, arc, collect
        self.calibration = calibration or {}
        self.names, self.modules, self.group_weights = {}, {}, {}
        self.observers, self.device_specs, self.norms = {}, {}, {}
        self.calls = Counter()
        self.sparse_counts = {}
        self.row_mask = None
        self.calibration_ffn_inputs = {}
        self.forward_rows = None
        for name, module in model.named_modules():
            if not isinstance(module, ds.Linear) or not name.startswith('layers.'):
                continue
            key = name + '.weight'
            projection_kind(key)  # Fail loudly if an upstream projection changes.
            self.names[module.weight.data_ptr()] = key
            self.modules[key] = module
            if key.endswith('.wo_a.weight'):
                attention = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                for group in range(attention.n_local_groups):
                    self.group_weights[f'{key}#group{group}'] = (module, group, attention.n_local_groups)
        self.expected = (set(self.modules) - {k for k in self.modules if k.endswith('.wo_a.weight')}) | set(self.group_weights)
        if (arc or method == 'teal') and not collect:
            missing = self.expected - set(self.calibration)
            if missing:
                raise ValueError(f'Missing calibration for {len(missing)} projections: {sorted(missing)[:3]}')
        # Keep all inherited quantized caches independent of the method and ARC
        # cache. configure is called only once for each loaded evaluation model.
        self.arc_cache = {}
        self.arc_cache_bytes = 0

    def begin_forward(self, tokens):
        self.forward_rows = prefill_rows(tokens.shape[1], self.method, tokens.device).repeat(tokens.shape[0])

    def _observe(self, key, x):
        obs = self.observers.setdefault(key, InputCalibration(x.shape[-1]))
        obs.update(x)

    def _spec(self, key, x):
        if key not in self.device_specs:
            src = self.calibration[key]
            self.device_specs[key] = (src['order'].to(x.device), int(src['select']),
                histogram_threshold(src['centers'], src['counts'], self.sparsity))
        return self.device_specs[key]

    def _mask(self, x, weight, key):
        if self.method == 'none' or self.sparsity == 0:
            return x
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if x.ndim == 3:
            rows = prefill_rows(shape[1], self.method, x.device).repeat(shape[0])
        elif x.ndim == 2 and self.row_mask is not None:
            rows = self.row_mask
        else:
            raise RuntimeError(f'{key}: no original token positions for routed rows')
        if rows.numel() != flat.shape[0]:
            raise ValueError(f'{key}: routed row positions mismatch')
        if self.method == 'teal':
            _, _, threshold = self._spec(key, x)
            keep = flat.abs() > threshold
        else:
            # MLA's fused KV corresponds to K/V; SwiGLU w1 is the gate.
            # These are complete matrices on a rank, so column norms need no
            # cross-rank reduction. Other projections use upstream unit norms.
            informed = key.endswith('.attn.wkv.weight') or key.endswith('.w1.weight')
            if key not in self.norms:
                self.norms[key] = (self.dequantize_weight(weight).float().norm(dim=0)
                                   if informed else torch.ones(flat.shape[-1], device=x.device, dtype=torch.float32))
            scores = flat.float().abs() * self.norms[key]
            module = self.modules.get(key)
            is_row = isinstance(module, self.ds.RowParallelLinear)
            if is_row and self.ds.world_size > 1:
                chunks = [torch.empty_like(scores) for _ in range(self.ds.world_size)]
                torch.distributed.all_gather(chunks, scores)
                scores = torch.cat(chunks, -1)
            indices = scores.topk(int(scores.shape[-1] * (1-self.sparsity)), dim=-1).indices
            keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
            if is_row and self.ds.world_size > 1:
                keep = keep.chunk(self.ds.world_size, -1)[self.ds.rank]
        keep = keep | ~rows[:, None]
        out = flat * keep
        counters = self.sparse_counts.setdefault(key, torch.zeros(3, device=x.device, dtype=torch.float64))
        counters[0] += (~keep).sum()
        counters[1] += out.numel() - out.count_nonzero()
        counters[2] += out.numel()
        return out.view(shape)

    def _arc_weight(self, weight, key, order, select, numeric=None):
        if key in self.arc_cache:
            return self.arc_cache[key]
        w = self.dequantize_weight(weight).float() if numeric is None else numeric.float()
        w = w.index_select(1, order)
        dtype = torch.float32 if weight.dtype == torch.float32 else torch.bfloat16
        result = self._fake_quant_2d(w, output_dtype=dtype)
        if select:
            result = torch.cat((result, self._fake_quant_2d(w[:, -select:].contiguous(), output_dtype=dtype)), -1)
        size = result.numel() * result.element_size()
        # Bounded cache, protecting always-used non-expert projections by only
        # caching those. Routed experts remain on-demand, as in the baseline.
        if '.experts.' not in key and self.arc_cache_bytes + size <= self.cache_limit:
            self.arc_cache[key] = result
            self.arc_cache_bytes += size
        return result

    def _project(self, x, weight, key, bias=None, numeric=None):
        self.calls[key] += 1
        if self.collect:
            self._observe(key, x)
        if not self.collect:
            x = self._mask(x, weight, key)
        if not self.arc or self.collect:
            if numeric is None:
                return super().linear(x, weight, bias)
            qx = self.quantize_activation(x)
            qw = self._fake_quant_2d(numeric, output_dtype=x.dtype)
            return F.linear(qx, qw, bias)
        order, select, _ = self._spec(key, x)
        reordered = x.index_select(-1, order).float()
        qx = self._fake_quant_2d(reordered)
        if select:
            residual = reordered[..., -select:] - qx[..., -select:]
            qx = torch.cat((qx, self._fake_quant_2d(residual.contiguous())), -1)
        qw = self._arc_weight(weight, key, order, select, numeric)
        return F.linear(qx.to(qw.dtype), qw, bias)

    def linear(self, x, weight, bias=None):
        if not hasattr(self, 'names'):
            return super().linear(x, weight, bias)
        key = self.names.get(weight.data_ptr())
        if key is None:
            raise RuntimeError('Unregistered DeepSeek linear weight')
        return self._project(x, weight, key, bias)

    def grouped_wo_a(self, x, module):
        key = self.names[module.weight.data_ptr()]
        groups = x.shape[2]
        numeric = self.dequantize_weight(module.weight).view(groups, -1, x.shape[-1])
        return torch.stack([self._project(x[:, :, g], module.weight, f'{key}#group{g}',
                                         numeric=numeric[g]) for g in range(groups)], 2)

    def finish_calibration(self):
        missing = self.expected - set(self.observers)
        if missing:
            raise ValueError(f'Unobserved calibration projections: {sorted(missing)[:5]}')
        return {key: obs.finish() for key, obs in self.observers.items()}

    def coverage(self):
        return dict(expected=len(self.expected), executed=len(set(self.calls) & self.expected),
                    unexecuted=sorted(self.expected-set(self.calls)),
                    calls=dict(self.calls), arc=self.arc, method=self.method,
                    sparse_counts={k: v.tolist() for k, v in self.sparse_counts.items()},
                    sparse_count_columns=['masked_elements', 'zero_elements', 'total_elements'])


def install_experiment_patches(ds, runtime):
    """Retain the reference attention computation; replace its grouped output."""
    # Use the already installed baseline attention implementation, replacing
    # exactly the grouped contraction. Compile a checked AST instead of a global
    # einsum hook. Unexpected upstream source layouts stop before model loading.
    import ast
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(ds.Attention.forward))
    tree = ast.parse(source)
    body = tree.body[0].body
    start = next((i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                  and ast.unparse(node.value) == 'runtime.quantize_activation(o)'), None)
    end = next((i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'torch.einsum'
                and node.value.args and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == 'bsgd,grd->bsgr'), None)
    if start is None or end is None or end <= start:
        raise RuntimeError('Unsupported external Attention.forward; grouped ARC patch refused')
    body[start:end+1] = ast.parse('o = runtime.grouped_wo_a(o, self.wo_a)').body
    ast.fix_missing_locations(tree)
    namespace = {**ds.Attention.forward.__globals__, 'ds': ds, 'runtime': runtime}
    exec(compile(tree, '<arcquant-grouped-attention>', 'exec'), namespace)
    ds.Attention.forward = namespace['forward']

    def moe(self, x, input_ids):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        if runtime.collect:
            runtime.calibration_ffn_inputs.setdefault(self.layer_id, x[:16].detach().clone())
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        previous = runtime.row_mask
        try:
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if not counts[i]:
                    continue
                idx, top = torch.where(indices == i)
                runtime.row_mask = runtime.forward_rows[idx]
                y[idx] += self.experts[i](x[idx], weights[idx, top, None])
            if ds.world_size > 1:
                ds.dist.all_reduce(y)
            runtime.row_mask = runtime.forward_rows
            y += self.shared_experts(x)
        finally:
            runtime.row_mask = previous
        return y.type_as(x).view(shape)

    ds.MoE.forward = moe
