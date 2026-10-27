"""Projection-level ARCQuant adapter for DeepSeek-V4-Flash's released tensors.

This is the numeric W4A4 path. It deliberately does not install itself into the
reference Transformer: that requires a model-wide calibration and an explicit
replacement for Attention.wo_a's grouped einsum.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch

from .bridge import ArcQuantBridge
from .sparse_utils import HistogramDistribution, ThresholdSparsifier, WeightInformedTopKSparsifier


_ATTENTION = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b", "compressor.wkv",
              "compressor.wgate", "indexer.wq_b", "indexer.compressor.wkv",
              "indexer.compressor.wgate", "indexer.weights_proj"}
_EXPERT = {"w1", "w2", "w3"}
_ATTENTION_PATTERN = "|".join(re.escape(name) for name in sorted(_ATTENTION, key=len, reverse=True))
_LAYER_KEY = re.compile(r"layers\.(\d+)\.(?:attn\.((?:" + _ATTENTION_PATTERN + r"))|"
                        r"ffn\.(?:shared_experts|experts\.(\d+))\.(w1|w2|w3))\.weight")
_FP4_VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)


def projection_kind(key: str) -> str:
    """Validate the names used by the official 0731 safetensors index."""
    match = _LAYER_KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"Not a DeepSeek-V4 attention/expert projection: {key}")
    kind = match.group(2) or match.group(4)
    assert kind in _ATTENTION | _EXPERT
    return kind


def _calibration_key(key: str) -> str:
    base, separator, group = key.partition("#group")
    kind = projection_kind(base)
    if separator and (kind != "wo_a" or not group.isdecimal()):
        raise ValueError(f"Invalid grouped DeepSeek projection key: {key}")
    return kind


@torch.no_grad()
def dequantize_weight(weight: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Read one official FP8 or packed E2M1 FP4 matrix as numeric FP32.

    The released FP4 tensor stores two values per byte and one scale per 32
    logical input columns. FP8 scales cover rectangular blocks. Passing the
    explicit scale works for safetensors; module weights may carry ``.scale``.
    """
    scale = scale if scale is not None else getattr(weight, "scale", None)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype == torch.int8 or (fp4_dtype is not None and weight.dtype == fp4_dtype):
        if scale is None or weight.ndim != 2 or scale.ndim != 2:
            raise ValueError("Packed FP4 requires a two-dimensional scale grid")
        raw = weight.contiguous().view(torch.uint8)
        lookup = torch.tensor(_FP4_VALUES, device=weight.device, dtype=torch.float32)
        values = torch.stack((lookup[(raw & 15).long()], lookup[(raw >> 4).long()]), -1)
        values = values.reshape(raw.shape[0], raw.shape[1] * 2)
        if scale.shape[0] != values.shape[0] or values.shape[1] % scale.shape[1]:
            raise ValueError("Packed FP4 scale grid does not match logical matrix")
        return values * scale.float().repeat_interleave(values.shape[1] // scale.shape[1], -1)
    if str(weight.dtype).startswith("torch.float8_") and not str(weight.dtype).endswith("e8m0fnu"):
        if scale is None or weight.ndim != 2 or scale.ndim != 2:
            raise ValueError("FP8 requires a two-dimensional scale grid")
        rows, cols = weight.shape
        if not 0 < scale.shape[0] <= rows or not 0 < scale.shape[1] <= cols:
            raise ValueError("FP8 scale grid does not match matrix")
        row_block = (rows + scale.shape[0] - 1) // scale.shape[0]
        col_block = (cols + scale.shape[1] - 1) // scale.shape[1]
        expanded = scale.float().repeat_interleave(row_block, -2).repeat_interleave(col_block, -1)
        return weight.float() * expanded[:rows, :cols]
    if scale is not None:
        raise ValueError("An uncompressed weight must not have a quantization scale")
    if weight.ndim != 2:
        raise ValueError("Expected a two-dimensional projection weight")
    return weight.float()


class DeepSeekArcProjectionAdapter:
    """Apply TEAL/WINA to a named input, then ARC reorder + HiF4 W4A4.

    ``reorder_index`` and ``select_nums`` must come from target-model
    calibration, using official projection keys. Grouped ``wo_a`` uses one key
    per group, suffixed ``#group<N>``. No heuristic calibration is invented.
    """

    def __init__(self, reorder_index: Mapping[str, torch.Tensor],
                 select_nums: Mapping[str, int], *, method: str = "none",
                 sparsity: float = 0., histogram_files: Mapping[str, tuple[str, str]] | None = None,
                 informed_keys: set[str] | None = None, apply_prefill: bool = True):
        if method not in ("none", "teal", "wina"):
            raise ValueError(f"Unknown sparse method: {method}")
        if not 0 <= sparsity < 1:
            raise ValueError("sparsity must be in [0, 1)")
        if set(reorder_index) != set(select_nums):
            raise ValueError("ARC reorder and residual-channel keys must match exactly")
        for key in reorder_index:
            _calibration_key(key)
        self.bridge = ArcQuantBridge(dict(reorder_index), dict(select_nums), quant_type="HiF4")
        self.method = method
        self.sparsity = sparsity
        self.histogram_files = histogram_files or {}
        self.informed_keys = informed_keys or set()
        self.apply_prefill = apply_prefill
        self._sparsifiers = {}
        self._checked_calibrations = set()

    def _check_calibration(self, key: str, width: int):
        if (key, width) in self._checked_calibrations:
            return
        index = self.bridge.reorder_index[key]
        select = self.bridge.select_nums[key]
        if index.ndim != 1 or index.numel() != width or index.dtype not in (
                torch.int32, torch.int64):
            raise ValueError(f"{key}: reorder index must contain {width} integer channels")
        if not torch.equal(torch.sort(index.cpu().long()).values, torch.arange(width)):
            raise ValueError(f"{key}: reorder index is not a permutation")
        if isinstance(select, torch.Tensor):
            if select.numel() != 1:
                raise ValueError(f"{key}: residual channel count must be scalar")
            select = select.item()
        if isinstance(select, bool) or int(select) != select or not 0 <= select <= width:
            raise ValueError(f"{key}: residual channel count must be in [0, {width}]")
        self._checked_calibrations.add((key, width))

    def _sparsifier(self, key: str, numeric_weight: torch.Tensor):
        if key in self._sparsifiers:
            return self._sparsifiers[key]
        if self.method == "teal":
            if key not in self.histogram_files:
                raise FileNotFoundError(f"Missing DeepSeek TEAL histogram assignment: {key}")
            path, stage = self.histogram_files[key]
            distribution = HistogramDistribution(path, stage)
            sparse = ThresholdSparsifier(distribution, self.sparsity, self.apply_prefill)
        else:
            norm = (numeric_weight.float().norm(dim=0) if key in self.informed_keys
                    else torch.ones(numeric_weight.shape[1], device=numeric_weight.device))
            sparse = WeightInformedTopKSparsifier(norm, self.sparsity, self.apply_prefill)
        self._sparsifiers[key] = sparse
        return sparse

    @torch.no_grad()
    def linear(self, x: torch.Tensor, weight: torch.Tensor, key: str, *,
               scale: torch.Tensor | None = None, bias: torch.Tensor | None = None,
               sparse_rows: torch.Tensor | None = None) -> torch.Tensor:
        projection_kind(key)
        if not self.bridge.has_layer(key):
            raise KeyError(f"Missing DeepSeek ARC calibration for {key}")
        if x.ndim not in (2, 3):
            raise ValueError("Projection input must be [tokens, channels] or [batch, tokens, channels]")
        numeric = dequantize_weight(weight, scale)
        if numeric.shape[1] != x.shape[-1]:
            raise ValueError(f"{key}: logical weight width {numeric.shape[1]} != input {x.shape[-1]}")
        self._check_calibration(key, x.shape[-1])
        if self.method != "none" and self.sparsity:
            sparse = self._sparsifier(key, numeric)
            if x.ndim == 2:
                # Routed experts receive only the tokens selected by the router;
                # their row order no longer represents original sequence positions.
                # Caller must specify which rows follow the method's prefill rule.
                if sparse_rows is None or sparse_rows.dtype != torch.bool or sparse_rows.shape != x.shape[:1]:
                    raise ValueError(f"{key}: 2D routed inputs require an explicit sparse_rows boolean mask")
                if bool(sparse_rows.any()):
                    x = x.clone()
                    x[sparse_rows] = sparse(x[sparse_rows].unsqueeze(1)).squeeze(1)
            else:
                if sparse_rows is not None:
                    raise ValueError("sparse_rows applies only to 2D routed inputs")
                x = sparse(x)
        return self.bridge.linear(x, numeric.to(dtype=x.dtype), bias, key)

    @torch.no_grad()
    def grouped_wo_a(self, x: torch.Tensor, weight: torch.Tensor, key: str, *,
                     scale: torch.Tensor | None = None) -> torch.Tensor:
        """Replace the official ``bsgd,grd->bsgr`` contraction explicitly."""
        if projection_kind(key) != "wo_a" or x.ndim != 4:
            raise ValueError("grouped_wo_a expects a wo_a key and [batch, seq, groups, width]")
        numeric = dequantize_weight(weight, scale)
        groups = x.shape[2]
        if numeric.shape[0] % groups or numeric.shape[1] != x.shape[3]:
            raise ValueError("wo_a group/width dimensions do not match official weight")
        rows = numeric.shape[0] // groups
        parts = []
        for group in range(groups):
            group_key = f"{key}#group{group}"
            if not self.bridge.has_layer(group_key):
                raise KeyError(f"Missing DeepSeek ARC calibration for {group_key}")
            self._check_calibration(group_key, x.shape[-1])
            slice_x = x[:, :, group, :]
            slice_w = numeric[group * rows:(group + 1) * rows]
            if self.method != "none" and self.sparsity:
                slice_x = self._sparsifier(group_key, slice_w)(slice_x)
            parts.append(self.bridge.linear(slice_x, slice_w.to(slice_x.dtype), None, group_key))
        return torch.stack(parts, dim=2)
