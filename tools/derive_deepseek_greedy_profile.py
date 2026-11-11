#!/usr/bin/env python3
"""Export a lower target from a completed native greedy search curve.

The native search is monotone: the prefix of the curve up to 40% is exactly
the path the same search would take when 40% is requested directly. This tool
only exports that already evaluated prefix; it never invents rates or reruns a
different sparse algorithm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import effective_sparsity


def export(source: Path, destination: Path, method: str, target: float):
    source = source.resolve()
    destination = destination.resolve()
    source_strategy = source / method / "strategy.json"
    if not source_strategy.is_file():
        raise FileNotFoundError(source_strategy)
    raw = json.loads(source_strategy.read_text())
    if not raw.get("complete") or raw.get("method") != method:
        raise ValueError(f"Source strategy is not a complete {method} strategy")
    old_targets = [float(p["target"]) for p in raw.get("profiles", [])]
    if not old_targets or target >= min(old_targets):
        raise ValueError(f"Target {target} must be below the first exported target {min(old_targets)}")

    layers = []
    for layer in range(int(raw["n_layers"])):
        path = source / method / f"layer-{layer}.json"
        data = json.loads(path.read_text())
        matches = [c for c in data.get("curve", [])
                   if abs(float(c["effective_sparsity"]) - target) <= 1e-6]
        if len(matches) != 1:
            raise ValueError(f"{path}: expected one exact curve point at {target}, got {len(matches)}")
        point = matches[0]
        rates, weights = point["rates"], point["candidates"][0].get("weights", data["weights"])
        # Curve points store a full rate map; the layer weights are stable and
        # are present in the source profile even when candidate metadata does
        # not repeat them.
        weights = data["profiles"][0]["weights"]
        actual = effective_sparsity(rates, weights)
        if abs(actual - target) > 1e-6:
            raise ValueError(f"{path}: exported rates have weighted sparsity {actual}")
        layers.append(dict(target=target, rates=rates, weights=weights,
                           error=point["error"], uniform_error=None,
                           derived_from_curve_iteration=point["iteration"]))

    destination.mkdir(parents=True, exist_ok=True)
    # Keep immutable calibration and search provenance available at the new
    # root while making the derived strategy the only modified artifact.
    for name in ("calibration",):
        src = source / name
        dst = destination / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src:
                continue
            raise FileExistsError(dst)
        dst.symlink_to(src, target_is_directory=True)
    manifest = json.loads((source / "search_manifest.json").read_text())
    manifest["targets"] = [target]
    manifest["derived_from_targets"] = old_targets
    manifest["profile_derivation"] = "exact monotone native greedy curve prefix"
    (destination / "search_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output = dict(raw)
    output["targets"] = [target]
    output["profiles"] = layers
    output["derived_from_targets"] = old_targets
    output["profile_derivation"] = "exact monotone native greedy curve prefix"
    (destination / method).mkdir(parents=True, exist_ok=True)
    (destination / method / "strategy.json").write_text(json.dumps(output, indent=2) + "\n")
    # Copying the source search manifest's code snapshots makes the derived
    # artifact portable even when the original search directory is archived.
    if (source / "sources").is_dir() and not (destination / "sources").exists():
        shutil.copytree(source / "sources", destination / "sources")
    print(json.dumps(dict(source=str(source), destination=str(destination), method=method,
                          target=target, layers=len(layers), old_targets=old_targets), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--method", choices=("wina", "teal"), required=True)
    p.add_argument("--target", type=float, default=0.4)
    args = p.parse_args()
    if not 0 < args.target < 1:
        p.error("target must be in (0,1)")
    export(args.source, args.destination, args.method, args.target)


if __name__ == "__main__":
    main()
