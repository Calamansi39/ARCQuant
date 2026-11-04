"""TEAL/WINA input masking for the pinned DeepSeek generation evaluator.

The normal DeepSeek experiment uses :class:`ExperimentRuntimeMixin` for the
four-choice GPQA runner.  LongBench and LiveCodeBench use the external
generation scheduler, which constructs the same reference model through a
different entry point.  This small mixin lets that scheduler reuse the native
WINA top-k rule without copying or editing the external checkout.

TEAL loads signed per-projection histograms from a separate calibration pass;
the generation runner never invents thresholds from the current prompt.
"""

from __future__ import annotations

from collections import Counter
import ast
import inspect
import textwrap

import torch

from .sparse_utils import topk_keep_count, validate_sparsity


class DeepSeekGenerationWinaMixin:
    """Mixin placed before the external quant runtime in the MRO."""

    def configure_sparse(self, ds, model, sparsity: float) -> None:
        import torch

        sparsity = validate_sparsity(sparsity, name="WINA sparsity")
        self.ds = ds
        self.sparse_sparsity = float(sparsity)
        self.sparse_modules = {}
        self.sparse_norms = {}
        self.sparse_counts = Counter()
        self.sparse_eligible = Counter()
        self.sparse_row_mask = None
        self.sparse_row_chunk = 4096
        self.sparse_groups = {}
        self.sparse_attention_stack = []
        self.sparse_linear_depth = 0
        self.sparse_hooks = []
        modules = dict(model.named_modules())
        for name, module in modules.items():
            if isinstance(module, ds.Linear) and name.startswith("layers."):
                key = name + ".weight"
                self.sparse_modules[module.weight.data_ptr()] = key
                if key.endswith(".attn.wo_a.weight"):
                    parent = modules.get(name.rsplit(".", 1)[0])
                    groups = int(getattr(parent, "n_local_groups", 0))
                    if groups > 0:
                        if module.out_features % groups:
                            raise ValueError(f"{key}: wo_a rows do not divide into groups")
                        # Configuration runs BEFORE checkpoint loading. Read
                        # layout only; never cache norms of uninitialized weights.
                        self.sparse_groups[key] = (groups, module.in_features)

                        def enter(_module, _args, projection=key):
                            self.sparse_attention_stack.append(projection)

                        def leave(_module, _args, _output, projection=key):
                            if not self.sparse_attention_stack or self.sparse_attention_stack.pop() != projection:
                                raise RuntimeError("Unbalanced sparse attention context")

                        self.sparse_hooks += [parent.register_forward_pre_hook(enter),
                            parent.register_forward_hook(leave, always_call=True)]
        self._install_moe_row_tracking(ds)

    @torch.no_grad()
    def quantize_activation(self, x):
        """Quantize activations and mask grouped ``wo_a`` inputs.

        Both the normal and ragged generation attention paths call this hook
        immediately before their grouped contraction.  Applying the mask here
        covers both paths without rewriting either external function.
        """
        if (getattr(self, "sparse_sparsity", 0.0) and x.ndim == 4
                and not getattr(self, "sparse_linear_depth", 0)):
            if not self.sparse_attention_stack:
                raise RuntimeError("Grouped sparse activation without attention context")
            x = self._mask_grouped_input(x, self.sparse_attention_stack[-1])
        # Match ordinary Linear: mask the original activation, then quantize.
        return super().quantize_activation(x)

    def _mask_grouped_input(self, x, key):
        groups, width = self.sparse_groups[key]
        if x.ndim != 4 or x.shape[2:] != (groups, width):
            raise ValueError(f"Unexpected grouped activation layout at {key}")
        # Upstream WINA uses unit importance for output projections. DeepSeek
        # splits its output projection into groups; each keeps that same rule.
        return torch.stack([self._sparse_mask(x[:, :, g, :], None, key + f"#group{g}")
                            for g in range(groups)], dim=2)

    def _install_moe_row_tracking(self, ds):
        """Preserve original token positions through expert dispatch.

        The reference MoE flattens and routes tokens before calling expert
        projections.  Without this wrapper every routed row would look like a
        decode row, which would apply the prefill rule to the wrong tokens.
        """
        runtime = self

        def moe(module, x, input_ids):
            shape = x.shape
            rows = (runtime._active_rows(shape[1], x.device).repeat(shape[0])
                    if x.ndim == 3 else torch.ones(x.reshape(-1, x.shape[-1]).shape[0],
                                                    dtype=torch.bool, device=x.device))
            flat = x.reshape(-1, module.dim)
            weights, indices = module.gate(flat, input_ids.flatten())
            y = torch.zeros_like(flat, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=module.n_routed_experts).tolist()
            previous = runtime.sparse_row_mask
            try:
                for expert_id in range(module.experts_start_idx, module.experts_end_idx):
                    if not counts[expert_id]:
                        continue
                    idx, top = torch.where(indices == expert_id)
                    runtime.sparse_row_mask = rows[idx]
                    y[idx] += module.experts[expert_id](flat[idx], weights[idx, top, None])
                if getattr(ds, "world_size", 1) > 1:
                    ds.dist.all_reduce(y)
                runtime.sparse_row_mask = rows
                y += module.shared_experts(flat)
            finally:
                runtime.sparse_row_mask = previous
            return y.type_as(x).view(shape)

        ds.MoE.forward = moe

    @staticmethod
    def _prefill_rows(length: int, device):
        import torch
        if length <= 1:
            return torch.ones(1, dtype=torch.bool, device=device)
        tail = max(1, int(0.99 * length))
        return torch.arange(length, device=device) >= length - tail

    @torch.no_grad()
    def _active_rows(self, length: int, device):
        return self._prefill_rows(length, device)

    @torch.no_grad()
    def _sparse_mask(self, x, weight, key):
        import torch
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if x.ndim == 3:
            rows = self._active_rows(shape[1], x.device).repeat(shape[0])
        elif self.sparse_row_mask is not None and key.startswith("layers.") and ".ffn." in key:
            rows = self.sparse_row_mask
        else:
            # The external scheduler only sends flattened routed rows through
            # linear; those rows are decode rows unless a caller explicitly
            # supplies a 3-D prefill tensor.
            rows = torch.ones(flat.shape[0], dtype=torch.bool, device=x.device)
        if key not in self.sparse_norms:
            informed = key.endswith(".attn.wkv.weight") or key.endswith(".w1.weight")
            if informed:
                numeric = self.dequantize_weight(weight).float()
                self.sparse_norms[key] = numeric.norm(dim=0)
            else:
                self.sparse_norms[key] = torch.ones(flat.shape[-1], device=x.device)
        keep = topk_keep_count(flat.shape[-1], self.sparse_sparsity)
        # A 128K-token prefill can make ``score`` several GiB for a single
        # projection.  Process rows in bounded chunks so top-k and mask
        # temporaries do not overlap across the whole prompt.  The result is
        # mathematically identical to one top-k over all rows.
        chunk_rows = max(1, int(getattr(self, "sparse_row_chunk", 4096)))
        norm = self.sparse_norms[key].to(flat.device)
        pieces = []
        masked = 0
        for start in range(0, flat.shape[0], chunk_rows):
            stop = min(flat.shape[0], start + chunk_rows)
            block = flat[start:stop]
            score = block.float().abs() * norm
            indices = score.topk(keep, dim=-1).indices
            block_mask = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, indices, True)
            block_rows = rows[start:stop]
            block_mask = block_mask | ~block_rows[:, None]
            pieces.append(block * block_mask)
            masked += int((~block_mask).sum().item())
            self.sparse_eligible[key] += int(block_rows.sum().item()) * flat.shape[-1]
        self.sparse_counts[key] += masked
        return torch.cat(pieces, dim=0).view(shape)

    _wina_mask = _sparse_mask

    @torch.no_grad()
    def grouped_wo_a(self, x, module):
        """Explicit grouped path with the same mask/quantization ordering."""
        key = self.sparse_modules[module.weight.data_ptr()]
        groups, _ = self.sparse_groups[key]
        if self.sparse_sparsity:
            x = self._mask_grouped_input(x, key)
        q_x = super().quantize_activation(x)
        q_w = self.get_weight(module.weight).view(groups, -1, x.shape[-1])
        return torch.einsum("bsgd,grd->bsgr", q_x.to(q_w.dtype), q_w)

    def linear(self, x, weight, bias=None):
        key = self.sparse_modules.get(weight.data_ptr())
        if key is not None and getattr(self, "sparse_sparsity", 0.0):
            x = self._sparse_mask(x, weight, key)
        self.sparse_linear_depth += 1
        try:
            return super().linear(x, weight, bias)
        finally:
            self.sparse_linear_depth -= 1

    def remove_sparse_hooks(self):
        for handle in self.sparse_hooks:
            handle.remove()
        self.sparse_hooks = []

    def stats(self):
        base = super().stats()
        eligible = {key: int(value) for key, value in self.sparse_eligible.items()}
        masked = {key: int(value) for key, value in self.sparse_counts.items()}
        base.update({"sparse_method": "wina" if hasattr(self, "sparse_sparsity") else "none",
                     "sparse_sparsity": getattr(self, "sparse_sparsity", 0.0),
                     "sparse_masked_elements": masked,
                     "sparse_eligible_elements": eligible,
                     "sparse_observed_zero_fraction": {
                         key: (masked[key] / value if value else 0.0)
                         for key, value in eligible.items()},
                     "sparse_target_zero_fraction": getattr(self, "sparse_sparsity", 0.0)})
        return base


def install_grouped_wo_a_patch(ds, runtime):
    """Replace the reference grouped einsum with the runtime-aware method."""
    source = textwrap.dedent(inspect.getsource(ds.Attention.forward))
    tree = ast.parse(source)
    body = tree.body[0].body
    start = next((i for i, node in enumerate(body)
                  if isinstance(node, ast.Assign) and ast.unparse(node.value) ==
                  'runtime.quantize_activation(o)'), None)
    end = next((i for i, node in enumerate(body)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'torch.einsum'
                and node.value.args and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == 'bsgd,grd->bsgr'), None)
    if start is None or end is None or end < start:
        raise RuntimeError('Unsupported external Attention.forward; grouped WINA patch refused')
    body[start:end + 1] = ast.parse(
        'o = runtime.grouped_wo_a(o, self.wo_a)').body
    ast.fix_missing_locations(tree)
    namespace = {**ds.Attention.forward.__globals__, 'runtime': runtime}
    exec(compile(tree, '<arcquant-grouped-wina>', 'exec'), namespace)
    ds.Attention.forward = namespace['forward']

class DeepSeekGenerationTealMixin(DeepSeekGenerationWinaMixin):
    """Native TEAL threshold masking for the generation scheduler.

    Thresholds are loaded from the same signed per-projection histograms used
    by the ARCQuant GPQA path.  No threshold is inferred from the generation
    prompt, so an omitted or incompatible calibration file fails loudly.
    """

    def configure_sparse(self, ds, model, sparsity: float, calibration=None) -> None:
        super().configure_sparse(ds, model, sparsity)
        if not calibration:
            raise ValueError("TEAL generation requires --calibration with rank*.pt")
        self.sparse_method = "teal"
        self.sparse_calibration = calibration
        self.sparse_thresholds = {}

    def _active_rows(self, length: int, device):
        if length <= 1:
            return torch.ones(1, dtype=torch.bool, device=device)
        count = max(1, length // 2)
        return torch.arange(length, device=device) >= length - count

    @torch.no_grad()
    def _sparse_mask(self, x, weight, key):
        if key not in self.sparse_thresholds:
            source = self.sparse_calibration.get(key)
            if source is None:
                raise KeyError(f"Missing TEAL calibration histogram: {key}")
            from model.deepseek_experiment import histogram_threshold
            centers = torch.as_tensor(source["centers"], dtype=torch.float32)
            counts = torch.as_tensor(source["counts"], dtype=torch.float32)
            self.sparse_thresholds[key] = histogram_threshold(centers, counts, self.sparse_sparsity)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if x.ndim == 3:
            rows = self._active_rows(shape[1], x.device).repeat(shape[0])
        elif self.sparse_row_mask is not None and ".ffn." in key:
            rows = self.sparse_row_mask
        else:
            rows = torch.ones(flat.shape[0], dtype=torch.bool, device=x.device)
        threshold = self.sparse_thresholds[key]
        mask = flat.abs() > threshold
        mask = mask | ~rows[:, None]
        out = (flat * mask).view(shape)
        self.sparse_counts[key] += int((~mask).sum().item())
        self.sparse_eligible[key] += int(rows.sum().item()) * flat.shape[-1]
        return out

    def stats(self):
        base = super().stats()
        base["sparse_method"] = "teal"
        base["teal_calibration_keys"] = len(self.sparse_calibration)
        return base
