"""ARC compensation for the native DeepSeek generation scheduler.

Reuse the audited ARC projection arithmetic; keep the generation mixin's
TEAL/WINA masks and MoE token-position tracking. Both ordinary attention and
ragged batch decode must route grouped wo_a through the same ARC projection.
"""
from __future__ import annotations

import ast
from collections import Counter
import inspect
import textwrap

import torch

from .deepseek_experiment import ExperimentRuntimeMixin


class DeepSeekGenerationArcMixin(ExperimentRuntimeMixin):
    """MRO: ARC, generation TEAL/WINA, external RequestQuantRuntime."""

    def configure_arc(self, ds, model, calibration, method):
        if self.quant_type != 'hifp4' or not self.act_quant:
            raise ValueError('ARC generation requires HiF4 W4A4')
        masked = self.sparse_counts
        # configure only registers layouts. It must not inspect checkpoint
        # values: the native scheduler loads weights after model construction.
        self.configure(ds, model, calibration=calibration, arc=True)
        self.sparse_counts = masked
        self.arc_method = method
        self.arc_attention_paths = Counter()
        self.arc_finite_logits_checks = 0
        self.arc_residual_calls = 0
        for key in self.expected:
            spec = self.calibration[key]
            module = self.modules.get(key)
            width = (module.in_features if module is not None
                     else self.group_weights[key][0].in_features)
            order = spec['order']
            selected = spec['select']
            if (order.ndim != 1 or order.numel() != width
                    or order.dtype not in (torch.int32, torch.int64)
                    or not torch.equal(order.cpu().sort().values, torch.arange(width, device='cpu'))
                    or type(selected) is not int or not 0 <= selected <= width
                    or selected % 64):
                raise ValueError(f'Invalid ARC channel calibration: {key}')

        def check_logits(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
                raise ValueError('Non-finite DeepSeek ARC logits')
            self.arc_finite_logits_checks += 1

        self.sparse_hooks.append(model.register_forward_hook(check_logits))

    def _spec(self, key, x):
        if key not in self.device_specs:
            src = self.calibration[key]
            self.device_specs[key] = (src['order'].to(x.device), src['select'], None)
        return self.device_specs[key]

    def _mask(self, x, weight, key):
        if not self.sparse_sparsity:
            return x
        return self._sparse_mask(x, weight, key)

    @torch.no_grad()
    def _project(self, x, weight, key, bias=None, numeric=None):
        if self.calibration[key]['select']:
            self.arc_residual_calls += 1
        return super()._project(x, weight, key, bias, numeric)

    def stats(self):
        base = super().stats()
        executed = set(self.calls) & self.expected
        base.update(arc=True, quantization='HiF4 W4A4 + ARC compensation',
                    sparse_method=self.arc_method,
                    arc_expected_projections=len(self.expected),
                    arc_executed_projections=len(executed),
                    arc_executed_families=sorted({
                        'grouped_wo_a' if '#group' in k else
                        'routed_expert' if '.experts.' in k else
                        'shared_expert' if '.shared_experts.' in k else 'attention'
                        for k in executed}),
                    arc_projection_calls=sum(self.calls.values()),
                    arc_residual_calls=self.arc_residual_calls,
                    arc_cache_gib=self.arc_cache_bytes / 2**30,
                    arc_attention_paths=dict(self.arc_attention_paths),
                    arc_finite_logits_checks=self.arc_finite_logits_checks,
                    kernel='HiF4-value-reconstruction+dense-GEMM')
        return base


def install_arc_grouped_attention(ds, runtime, path):
    """Checked replacement of the grouped contraction, retaining closures.

    The ragged implementation closes over the *already patched* ordinary
    forward. Dropping that closure would silently bypass or break prefill.
    """
    original = ds.Attention.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    body = function.body
    starts = [(i, node.targets[0].id) for i, node in enumerate(body)
              if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and isinstance(node.value, ast.Call)
              and ast.unparse(node.value.func) == 'runtime.quantize_activation']
    ends = [i for i, node in enumerate(body)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == 'torch.einsum'
            and node.value.args and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == 'bsgd,grd->bsgr']
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0][0]:
        raise RuntimeError(f'Unsupported {path} Attention.forward; ARC patch refused')
    start, variable = starts[0]
    operand = ast.unparse(body[start].value.args[0])
    body[start:ends[0] + 1] = ast.parse(
        f'runtime.arc_attention_paths[{path!r}] += 1\n'
        f'{variable} = runtime.grouped_wo_a({operand}, self.wo_a)').body
    ast.fix_missing_locations(tree)
    closure = inspect.getclosurevars(original)
    namespace = {**original.__globals__, **closure.nonlocals, 'runtime': runtime}
    exec(compile(tree, f'<arcquant-{path}-attention>', 'exec'), namespace)
    ds.Attention.forward = namespace[function.name]
