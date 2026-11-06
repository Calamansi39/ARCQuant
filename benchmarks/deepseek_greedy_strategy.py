"""Strict, portable artifacts for DeepSeek block-wise sparse budgets."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

SCHEMA = "deepseek-native-greedy-v1"
GROUPING = "per-layer-projection-family; routed-experts-share-family-rate"


def projection_group(key):
    base = key.split("#group", 1)[0]
    match = re.fullmatch(r"layers\.(\d+)\.(attn\..+|ffn\.(?:experts\.\d+|shared_experts)\.w[123])\.weight", base)
    if not match:
        raise ValueError(f"Unsupported budget projection: {key}")
    suffix = re.sub(r"ffn\.experts\.\d+\.", "ffn.experts.", match[2])
    return f"layers.{match[1]}.{suffix}"


def effective_sparsity(rates, weights):
    if set(rates) != set(weights) or not weights:
        raise ValueError("Budget rates/weights must cover the same nonempty groups")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Budget weights must be finite, nonnegative and not all zero")
    if any(not math.isfinite(s) or not 0 <= s < 1 for s in rates.values()):
        raise ValueError("Projection sparsity must be finite and in [0, 1)")
    if any(rates[k] != 0 for k in weights if weights[k] == 0):
        raise ValueError("Unobserved zero-cost groups must remain dense")
    return sum(rates[k] * weights[k] for k in weights) / sum(weights.values())


def load_strategy(path, *, method, sparsity, source_signature, world_size, quantization, n_layers):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    for key, expected in {"schema": SCHEMA, "method": method, "world_size": world_size,
                          "quantization": quantization, "source_signature": source_signature,
                          "grouping": GROUPING, "n_layers": n_layers,
                          "transform": False}.items():
        if data.get(key) != expected:
            raise ValueError(f"Greedy strategy {key} mismatch")
    if not data.get("complete"):
        raise ValueError("Incomplete/pilot greedy strategy cannot evaluate a full model")
    profiles = [p for p in data.get("profiles", []) if abs(p["target"] - sparsity) < 1e-8]
    if len(profiles) != 1:
        raise ValueError("Strategy must contain exactly the requested budget; no nearest-budget fallback")
    profile = profiles[0]
    if set(profile["layers"]) != {str(i) for i in range(n_layers)}:
        raise ValueError("Greedy strategy is missing model layers")
    rates = {}
    for layer, item in profile["layers"].items():
        if any(not k.startswith(f"layers.{layer}.") for k in item["rates"]):
            raise ValueError("Projection belongs to a different strategy layer")
        actual = effective_sparsity(item["rates"], item["weights"])
        if abs(actual - sparsity) > 1e-6:
            raise ValueError("Strategy does not satisfy its declared weighted budget")
        rates.update(item["rates"])
    return data, rates, hashlib.sha256(raw).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Workers on the same host may publish different heartbeat/manifest files
    # at the same time. A shared fixed temporary name lets one writer replace
    # another writer's temp file and causes FileNotFoundError on replace.
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
