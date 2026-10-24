"""Small, torch-free format for offline TEAL/WINA per-layer allocations."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


class StrategyError(ValueError):
    pass


def load_strategy(path: str | Path, *, method: str, layer_count: int | None = None,
                  target_sparsity: float | None = None,
                  model_config: str | Path | None = None) -> dict:
    source = Path(path).expanduser()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StrategyError(f"Cannot read sparse strategy {source}: {exc}") from exc
    if data.get("schema_version") != 1 or data.get("method") != method:
        raise StrategyError(f"{source} is not a v1 {method} strategy")
    if data.get("allocation") != "greedy":
        raise StrategyError(f"{source} must declare greedy allocation")
    if target_sparsity is not None:
        declared = data.get("target_sparsity")
        if isinstance(declared, bool) or not isinstance(declared, (float, int)) or abs(declared - target_sparsity) > 1e-6:
            raise StrategyError(f"{source} target sparsity differs from --sparsity {target_sparsity}")
    if model_config is not None and data.get("model_config_sha256"):
        try:
            actual = hashlib.sha256(Path(model_config).read_bytes()).hexdigest()
        except OSError as exc:
            raise StrategyError(f"Cannot check model config {model_config}: {exc}") from exc
        if data["model_config_sha256"] != actual:
            raise StrategyError(f"{source} was calibrated for a different model config")
    layers = data.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise StrategyError(f"{source} has no layer allocations")
    if layer_count is not None and set(layers) != {str(i) for i in range(layer_count)}:
        raise StrategyError(f"{source} does not contain exactly {layer_count} layers")
    for index, rates in layers.items():
        if not index.isdigit() or not isinstance(rates, dict) or set(rates) != set(PROJECTIONS):
            raise StrategyError(f"{source}: layer {index} must contain {', '.join(PROJECTIONS)}")
        for projection, value in rates.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
                raise StrategyError(f"{source}: invalid rate at layer {index}/{projection}")
    return data
