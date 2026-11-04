"""Strict, per-row 2:4 activation masking and explicit projection selection.

This is a magnitude baseline, not TEAL/WINA or a trained xmask gate. Coverage
counts named linear projections, not parameters, FLOPs or transformer blocks.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import torch


def select_projections(inventory, coverage, scope="all", selected_names=None):
    coverage = float(coverage)
    if not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("layer coverage must be finite and in [0, 1]")
    if scope not in ("all", "attention", "ffn"):
        raise ValueError("scope must be all, attention or ffn")
    scoped = {k: v for k, v in inventory.items() if scope == "all" or v["area"] == scope}
    if not scoped:
        raise ValueError("No linear projections in requested scope")
    eligible = {k for k, v in scoped.items() if v["width"] > 0 and v["width"] % 4 == 0}
    # Stratify by area so many MoE expert matrices cannot hide a low Attention
    # coverage. With an explicit selection enforce the same lower bounds.
    areas = {v["area"] for v in scoped.values()}
    required_by_area = {area: math.ceil(coverage * sum(v["area"] == area for v in scoped.values()))
                        for area in areas}
    if selected_names is None:
        # Independent of module traversal order, TP duplication, and Python's
        # randomized hash seed. This is not a sensitivity-optimized selection.
        order = sorted(eligible, key=lambda k: (hashlib.sha256(k.encode()).hexdigest(), k))
        selected = set()
        for area in sorted(areas):
            selected.update([k for k in order if scoped[k]["area"] == area][:required_by_area[area]])
    else:
        if not isinstance(selected_names, list) or any(not isinstance(k, str) for k in selected_names):
            raise ValueError("selection must be a JSON list of projection names")
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("Duplicate projection names in selection")
        selected = set(selected_names)
        invalid = selected - eligible
        if invalid:
            raise ValueError(f"Unknown, out-of-scope or unsupported selection: {sorted(invalid)[:5]}")
    for area, required in required_by_area.items():
        count = sum(scoped[k]["area"] == area for k in selected)
        if count < required:
            raise ValueError(f"Requested {required} {area} sparse projections, only {count} selected/supported")
    return selected


def merge_inventories(inventories):
    """Collapse replicated TP projections; expert names remain distinct."""
    result = {}
    for inventory in inventories:
        for name, spec in inventory.items():
            if name in result and result[name] != spec:
                raise ValueError(f"Inconsistent TP projection layout: {name}")
            result[name] = spec
    return result


class Structured24Controller:
    def __init__(self, inventory, coverage=1., scope="all", selected_names=None, chunk_rows=4096):
        if not isinstance(chunk_rows, int) or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        self.inventory = inventory
        self.coverage = float(coverage)
        self.scope = scope
        self.selected = select_projections(inventory, coverage, scope, selected_names)
        self.chunk_rows = chunk_rows
        self.calls = Counter()
        self.counts = {}

    @torch.no_grad()
    def mask(self, x, name):
        if name not in self.inventory:
            raise KeyError(f"Unregistered projection: {name}")
        self.calls[name] += 1
        if name not in self.selected:
            return x
        if not x.is_floating_point() or x.ndim < 2 or x.shape[-1] != self.inventory[name]["width"]:
            raise ValueError(f"Invalid activation shape/dtype at {name}: {x.shape}, {x.dtype}")
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty_like(flat)
        counts = self.counts.setdefault(name, torch.zeros(7, dtype=torch.int64, device=x.device))
        for start in range(0, flat.shape[0], self.chunk_rows):
            block = flat[start:start + self.chunk_rows].reshape(-1, x.shape[-1] // 4, 4)
            if not bool(torch.isfinite(block).all()):
                raise ValueError(f"Non-finite activation before 2:4 mask at {name}")
            indices = block.abs().topk(2, dim=-1).indices
            mask = torch.zeros_like(block, dtype=torch.bool).scatter_(-1, indices, True)
            masked = torch.where(mask, block, 0)
            out[start:start + self.chunk_rows] = masked.reshape(-1, x.shape[-1])
            counts[0] += block.numel()
            counts[1] += (~mask).sum()
            counts[2] += block.numel() // 4
            counts[3] += (mask.sum(-1) != 2).sum()
        return out.reshape_as(x)

    @torch.no_grad()
    def observe_quantized(self, x, name):
        if name not in self.selected or x.numel() == 0:
            return
        # Quantization may introduce additional zeros. It must never resurrect
        # masked values and create more than two nonzeros in a group.
        flat = x.reshape(-1, x.shape[-1])
        for start in range(0, flat.shape[0], self.chunk_rows):
            groups = flat[start:start + self.chunk_rows].reshape(-1, 4)
            if not bool(torch.isfinite(groups).all()):
                raise ValueError(f"Non-finite quantized activation at {name}")
            violations = (groups.count_nonzero(dim=-1) > 2).sum()
            self.counts[name][4] += violations
            self.counts[name][5] += groups.numel() - groups.count_nonzero()
            self.counts[name][6] += groups.numel()
            if int(violations):
                raise ValueError(f"Quantizer broke 2:4 structure at {name}")

    def report(self):
        selected = self.selected
        scoped = {k for k, v in self.inventory.items() if self.scope == "all" or v["area"] == self.scope}
        measured = {k: v.tolist() for k, v in self.counts.items()}
        by_area = {}
        for area in ("attention", "ffn"):
            keys = {k for k, v in self.inventory.items() if v["area"] == area}
            by_area[area] = {"total": len(keys), "selected": len(keys & selected),
                             "configured_fraction": len(keys & selected) / len(keys) if keys else None}
        return {
            "pattern": "2:4", "selector": "per-row magnitude top2 within contiguous K groups of 4",
            "token_scope": "all prefill and decode rows; no dense prefix bypass",
            "coverage_unit": "unique named attention/FFN linear projection; expert projections counted separately",
            "scope": self.scope, "requested_scope_coverage": self.coverage,
            "coverage_policy": "ceil(target * projection_count) independently per in-scope area",
            "total_projections": len(self.inventory), "scope_projections": len(scoped),
            "configured_sparse_projections": len(selected),
            "configured_scope_coverage": len(selected) / len(scoped),
            "configured_all_coverage": len(selected) / len(self.inventory),
            "by_area": by_area,
            "selected_names": sorted(selected), "inventory": self.inventory,
            "executed_sparse_projections_on_this_rank": sum(v[0] > 0 for v in measured.values()),
            "unobserved_selected_on_this_rank": sorted(selected - {k for k, v in measured.items() if v[0]}),
            "calls_on_this_rank": dict(self.calls),
            "counter_columns": ["elements", "masked_elements", "groups", "mask_structure_violations",
                                "post_quant_structure_violations", "post_quant_zero_elements", "post_quant_observed_elements"],
            "counters_on_this_rank": measured,
            "mask_zero_fraction_on_this_rank": (sum(v[1] for v in measured.values()) /
                sum(v[0] for v in measured.values())) if sum(v[0] for v in measured.values()) else None,
            "post_quant_zero_fraction_on_this_rank": (sum(v[5] for v in measured.values()) /
                sum(v[6] for v in measured.values())) if sum(v[6] for v in measured.values()) else None,
            "acceptance_status": "not_evaluated",
            "implementation": "dense GEMM with masked inputs; no sparse kernel speed claim",
        }
