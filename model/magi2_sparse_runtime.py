"""MAGI activation sparsity, applied before the pinned W4A4 boundaries.

This is a video architecture adaptation, not WINA's Llama reparameterization.
Fused QKV and gate/up use a shared input mask; routed gate/up use the norm of
their concatenated original matrices. Router/mHC/adapters remain untouched.
"""

from __future__ import annotations

from collections import Counter
import os
from types import MethodType

import torch
import torch.nn.functional as functional

from model.sparse_utils import topk_keep_count, validate_sparsity


GROUPED = ("GroupedLinearBase", "CustomGroupedLinear", "BaseLinear", "NativeMoELinear")
METRICS = ("total_elements", "input_zeros", "output_zeros", "masked_elements", "nonfinite")


@torch.no_grad()
def input_norm(weight, *, expert=False, device="cpu", chunk_elements=8_388_608):
    logical = weight.T if expert else weight
    total = torch.zeros(logical.shape[1], dtype=torch.float32, device=device)
    rows = max(1, chunk_elements // logical.shape[1])
    for first in range(0, logical.shape[0], rows):
        block = logical[first:first + rows].to(device=device, dtype=torch.float32)
        total.add_(block.square().sum(dim=0))
    return total.sqrt().cpu()


def projection_inventory(model):
    included, excluded = {}, []
    for name, module in model.named_modules():
        for control in ("mhc_phi_fused_attn", "mhc_phi_fused_mlp"):
            if hasattr(module, control):
                excluded.append(f"{name}.{control}")
        kind = type(module).__name__
        if kind == "CoreMultiHeadMoE":
            included[name] = ("routed", module)
            excluded.append(f"{name}.gate")
        elif kind in GROUPED or isinstance(module, torch.nn.Linear):
            if any(part in name.lower() for part in ("embed", "adapter")):
                excluded.append(name)
            else:
                included[name] = ("grouped" if kind in GROUPED else "dense", module)
    return included, excluded


@torch.no_grad()
def prepare_norms(model, device="cpu"):
    norms = {}
    included, _ = projection_inventory(model)
    for name, (kind, module) in included.items():
        if kind == "routed":
            shared, down = [], []
            for gate, up, output in zip(module.W_gate, module.W_up, module.W_down):
                gate_norm = input_norm(gate, expert=True, device=device)
                up_norm = input_norm(up, expert=True, device=device)
                shared.append((gate_norm.square() + up_norm.square()).sqrt())
                down.append(input_norm(output, expert=True, device=device))
            norms[f"{name}.gate_up"] = torch.stack(shared)
            norms[f"{name}.down"] = torch.stack(down)
        elif kind == "grouped":
            for expert, weight in enumerate(module.weight.chunk(module.num_experts, dim=0)):
                norms[f"{name}.group{expert}"] = input_norm(weight, device=device)
        else:
            norms[name] = input_norm(module.weight, device=device)
    return norms


class Magi2SparseController:
    def __init__(self, sparsity, *, method="wina", norms=None, calibration=None,
                 recorder=None, stage="preview", rows_per_chunk=256):
        self.sparsity = validate_sparsity(sparsity)
        if method not in ("wina", "teal", "calibrate"):
            raise ValueError(f"Unknown MAGI sparse method: {method}")
        if method == "teal" and calibration is None and self.sparsity:
            raise ValueError("TEAL requires video calibration; no online top-k fallback")
        if method == "calibrate" and (recorder is None or self.sparsity):
            raise ValueError("Calibration requires a recorder and zero sparsity")
        if rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be positive")
        self.method, self.stage = method, stage
        self.norms = norms or {}
        self.calibration, self.recorder = calibration, recorder
        self.rows_per_chunk = rows_per_chunk
        self.forward_index = -1
        self.video_index = -1
        self.counters = {}
        self.calls = Counter()
        self.zero_bypass = Counter()
        self.coverage = {}
        self.excluded = []
        self.handles = []
        self.output_nonfinite = None

    def begin_video(self):
        self.forward_index = -1
        self.video_index += 1

    def begin_forward(self, module, inputs):
        self.forward_index += 1

    def end_forward(self, module, inputs, output):
        if torch.is_tensor(output):
            count = (~torch.isfinite(output)).sum()
            self.output_nonfinite = count if self.output_nonfinite is None else self.output_nonfinite + count

    @property
    def bucket(self):
        return max(0, self.forward_index) // 10

    @property
    def stats(self):
        totals = Counter()
        for values in self.counters.values():
            totals.update(dict(zip(METRICS, values.cpu().tolist())))
        return totals

    def snapshot(self):
        modules, forwards = {}, {}
        for (name, forward), values in self.counters.items():
            counts = dict(zip(METRICS, values.cpu().tolist()))
            counts["calls"] = self.calls[(name, forward)]
            for table, key in ((modules, name), (forwards, str(forward))):
                table.setdefault(key, Counter()).update(counts)
        for counts in modules.values():
            total = counts["total_elements"]
            counts["observed_zero_rate"] = counts["output_zeros"] / total if total else 0
            counts["mask_rate"] = counts["masked_elements"] / total if total else 0
        return dict(method=self.method, variant="magi-projection-shared-mask-v1",
                    sparsity=self.sparsity, stage=self.stage,
                    norm_source="original-pre-quantization" if self.method == "wina" else "not-applicable",
                    mask_order="sparsify-then-activation-quantize", coverage=self.coverage,
                    excluded=self.excluded, modules=modules, forwards=forwards,
                    stats=dict(self.stats), zero_sparsity_native_bypass=dict(self.zero_bypass),
                    forward_index=self.forward_index, videos_started=self.video_index + 1,
                    output_nonfinite=0 if self.output_nonfinite is None else int(self.output_nonfinite.item()),
                    zero_rate_boundary="after sparsification, before activation fake quantization",
                    timing_axis="DiT forward invocation; includes CFG calls, not scheduler timestep")

    @torch.no_grad()
    def mask(self, activation, name, *, norm=None, valid_rows=None):
        if not activation.numel():
            return activation
        if self.method == "calibrate":
            self.recorder.observe(self.stage, name, self.bucket, activation, valid_rows)
        threshold = None
        if self.method == "teal" and self.sparsity:
            threshold = self.calibration.threshold(self.stage, name, self.bucket, self.sparsity)
        if self.method == "wina" and self.sparsity:
            norm = self.norms[name] if norm is None else norm
            norm = norm.to(device=activation.device, dtype=torch.float32)
            if norm.shape[-1] != activation.shape[-1]:
                raise ValueError(f"Input/norm width mismatch: {name}")
        batched = activation.ndim == 3 and valid_rows is not None
        shaped = activation if batched else activation.reshape(1, -1, activation.shape[-1])
        if norm is not None and norm.ndim == 2:
            norm = norm[:, None, :]
        result = torch.empty_like(shaped) if self.sparsity else shaped
        counts = torch.zeros(len(METRICS), dtype=torch.int64, device=activation.device)
        for first in range(0, shaped.shape[1], self.rows_per_chunk):
            block = shaped[:, first:first + self.rows_per_chunk]
            valid = (valid_rows[:, first:first + self.rows_per_chunk, None] if batched
                     else torch.ones((), dtype=torch.bool, device=activation.device))
            selected = torch.ones_like(block, dtype=torch.bool)
            if self.method == "wina" and self.sparsity:
                keep = topk_keep_count(block.shape[-1], self.sparsity)
                indices = (block.float().abs() * norm).topk(keep, dim=-1, sorted=False).indices
                selected.zero_().scatter_(-1, indices, True)
            elif threshold is not None:
                selected = block.abs().gt(threshold)
            masked = block.masked_fill(~selected, 0) if self.sparsity else block
            if self.sparsity:
                result[:, first:first + block.shape[1]] = masked
            counts += torch.stack((valid.expand_as(block).sum(),
                                   ((block == 0) & valid).sum(),
                                   ((masked == 0) & valid).sum(),
                                   ((~selected) & valid).sum(),
                                   ((~torch.isfinite(block)) & valid).sum()))
        key = (name, self.forward_index)
        if key not in self.counters:
            self.counters[key] = counts
        else:
            self.counters[key].add_(counts)
        self.calls[key] += 1
        return result.reshape_as(activation)


class Magi2WinaController(Magi2SparseController):
    def mask_linear(self, activation, weight):
        if activation.shape[-1] != weight.shape[-1]:
            raise ValueError("MAGI-2 WINA input/weight dimensions differ")
        return self.mask(activation, "linear", norm=input_norm(weight))

    def mask_expert(self, activation, weight):
        if activation.shape[-1] != weight.shape[0]:
            raise ValueError("MAGI-2 expert input/weight dimensions differ")
        return self.mask(activation, "expert", norm=input_norm(weight, expert=True))


def _grouped(controller, name, module, activation, splits=None, gather_ids=None):
    from magi2_quant import quantize_tensor

    if gather_ids is not None and module.num_experts > 1:
        activation = (activation.index_select(0, gather_ids) if gather_ids.ndim == 1
                      else torch.gather(activation, 0, gather_ids))
    if module.num_experts == 1:
        chunks = (activation,)
    else:
        if splits is None:
            raise ValueError("GroupedLinear requires modality splits")
        splits = splits.cpu().tolist() if torch.is_tensor(splits) else list(splits)
        if len(splits) != module.num_experts:
            raise ValueError("GroupedLinear split count differs from expert count")
        chunks = activation.split(splits, dim=0)
    weights = module.weight.chunk(module.num_experts, dim=0)
    biases = module.bias.chunk(module.num_experts, dim=0) if module.bias is not None else None
    outputs = []
    for expert, chunk in enumerate(chunks):
        value = controller.mask(chunk, f"{name}.group{expert}")
        if module._baseline_act:
            value = quantize_tensor(value, module._baseline_quant)
        outputs.append(functional.linear(value, weights[expert], None if biases is None else biases[expert]))
    return torch.cat(outputs, dim=0) if module.num_experts > 1 else outputs[0]


def _bind_grouped(controller, name, module):
    kind = type(module).__name__

    if kind == "GroupedLinearBase":
        def forward(self, input, cu_seqlens=None, m_splits=None, gather_ids=None):
            splits = m_splits if m_splits is not None else (cu_seqlens.diff() if cu_seqlens is not None else None)
            return _grouped(controller, name, self, input, splits, gather_ids)
    elif kind == "CustomGroupedLinear":
        def forward(self, input, modality_dispatcher=None, gather_ids=None):
            splits = getattr(modality_dispatcher, "group_size_cpu", modality_dispatcher)
            return _grouped(controller, name, self, input, splits, gather_ids)
    else:
        def forward(self, input, output_dtype=None, modality_dispatcher=None):
            from magi2_quant import quantize_tensor

            output_dtype = input.dtype if output_dtype is None else output_dtype
            chunks = modality_dispatcher.dispatch(input) if self.num_experts > 1 else (input,)
            weights = self.weight.chunk(self.num_experts, dim=0)
            biases = self.bias.chunk(self.num_experts, dim=0) if self.bias is not None else None
            outputs = []
            for expert, chunk in enumerate(chunks):
                value = controller.mask(chunk.to(torch.bfloat16), f"{name}.group{expert}")
                if self._baseline_act:
                    value = quantize_tensor(value, self._baseline_quant)
                value = torch.matmul(value, weights[expert].to(torch.bfloat16).T)
                if biases is not None:
                    value = value + biases[expert].to(torch.bfloat16)
                outputs.append(value.to(output_dtype))
            return modality_dispatcher.undispatch(*outputs) if self.num_experts > 1 else outputs[0]
    module.forward = MethodType(forward, module)


@torch.no_grad()
def routed_reference(controller, name, module, activation, probabilities, indices):
    from magi2_quant import _matmul_fp32, quantize_tensor

    tokens, heads, _ = activation.shape
    experts = module.W_gate.shape[0] // heads
    global_ids = indices + torch.arange(heads, device=activation.device)[:, None, None] * experts
    order = global_ids.reshape(-1).argsort(stable=True)
    token_ids = torch.arange(tokens, device=activation.device)[None, :, None].expand_as(indices).reshape(-1)[order]
    scores = probabilities.reshape(-1)[order]
    counts = torch.bincount(global_ids.reshape(-1), minlength=heads * experts).cpu().tolist()
    output = torch.zeros_like(activation, dtype=torch.float32)
    offset = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        head = expert // experts
        selected = token_ids[offset:offset + count]
        norm = controller.norms.get(f"{name}.gate_up")
        value = controller.mask(activation[selected, head], f"{name}.gate_up",
                                norm=None if norm is None else norm[expert])
        if module._baseline_act:
            value = quantize_tensor(value, module._baseline_quant)
        gate = _matmul_fp32(value, module.W_gate[expert]).clamp(max=7.)
        up = _matmul_fp32(value, module.W_up[expert]).clamp(min=-7., max=7.)
        hidden = (gate * torch.sigmoid(1.702 * gate) * (up + 1.)).to(module.W_down.dtype)
        norm = controller.norms.get(f"{name}.down")
        hidden = controller.mask(hidden, f"{name}.down", norm=None if norm is None else norm[expert])
        if module._baseline_act:
            hidden = quantize_tensor(hidden, module._baseline_quant)
        value = _matmul_fp32(hidden, module.W_down[expert]) * scores[offset:offset + count, None]
        output[:, head].index_add_(0, selected, value.to(activation.dtype).float())
        offset += count
    return output.to(activation.dtype)


def install_sparse(model, controller):
    included, controller.excluded = projection_inventory(model)
    if getattr(model, "_arcquant_sparse_installed", False):
        raise ValueError("MAGI sparsity already installed")
    model._arcquant_sparse_installed = True
    controller.handles.append(model.register_forward_pre_hook(controller.begin_forward))
    controller.handles.append(model.register_forward_hook(controller.end_forward))
    for name, (kind, module) in included.items():
        controller.coverage[name] = kind
        if kind == "dense":
            def hook(mod, inputs, key=name):
                return (controller.mask(inputs[0], key),) + inputs[1:]
            controller.handles.append(module.register_forward_pre_hook(hook, prepend=True))
        elif kind == "grouped":
            _bind_grouped(controller, name, module)
        else:
            original = module._flash_forward

            def forward(self, x_heads, topk_probs, topk_indices, key=name, native=original):
                if controller.sparsity == 0 and controller.method != "calibrate":
                    controller.zero_bypass[key] += 1
                    return native(x_heads, topk_probs, topk_indices)
                backend = os.environ.get("MAGI2_MOE_BACKEND", "batched")
                if backend not in ("batched", "reference"):
                    raise ValueError(f"Unknown MAGI expert backend: {backend}")
                if backend == "batched" and x_heads.is_cuda and x_heads.dtype == torch.bfloat16:
                    from model.magi2_sparse_moe import routed_batched
                    return routed_batched(controller, key, self, x_heads, topk_probs, topk_indices)
                return routed_reference(controller, key, self, x_heads, topk_probs, topk_indices)

            module._flash_forward = MethodType(forward, module)
    return controller


def install_wina(model, sparsity, *, norms=None):
    controller = Magi2WinaController(sparsity, norms=prepare_norms(model) if norms is None else norms)
    return install_sparse(model, controller)
