"""Inventory DeepSeek TP checkpoint projections without loading weight data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_inventory(path):
    from model.deepseek_adapter import projection_kind
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= 64 * 2**20:
            raise ValueError("Unexpected safetensors header size")
        raw = handle.read(size)
    header = json.loads(raw)
    inventory, excluded = {}, []
    for key, entry in header.items():
        if key == "__metadata__" or not key.endswith(".weight") or len(entry.get("shape", [])) != 2:
            continue
        try:
            projection_kind(key)
        except ValueError:
            excluded.append(key)
            continue
        # Safetensors F4 header reports logical K, unlike torch's packed tensor.
        name = key.removesuffix(".weight")
        inventory[name] = {"area": "attention" if ".attn." in key else "ffn",
                           "width": int(entry["shape"][-1])}
    return inventory, {"file": path.name, "header_sha256": hashlib.sha256(raw).hexdigest(),
                       "local_projection_count": len(inventory), "excluded_matrix_weights": excluded}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--converted-model", type=Path, required=True)
    p.add_argument("--layer-coverage", type=float, default=1.)
    p.add_argument("--scope", choices=("all", "attention", "ffn"), default="all")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write-selection", type=Path)
    a = p.parse_args(argv)
    from model.structured_sparsity import Structured24Controller, merge_inventories
    conversion = json.loads((a.converted_model / "conversion.json").read_text())
    if not conversion.get("completed"):
        raise ValueError("Checkpoint conversion is incomplete")
    world = int(conversion["world_size"])
    inventories, sources = [], []
    for rank in range(world):
        inventory, source = read_inventory(a.converted_model / f"model{rank}-mp{world}.safetensors")
        inventories.append(inventory)
        sources.append(source)
    controller = Structured24Controller(merge_inventories(inventories), a.layer_coverage, a.scope)
    report = {"kind": "checkpoint_header_inventory_only", "world_size": world,
              "sources": sources, "structured_sparsity": controller.report(),
              "note": "No tensor data loaded; no model execution or file-integrity certification. Runtime inventory remains authoritative."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    if a.write_selection:
        a.write_selection.parent.mkdir(parents=True, exist_ok=True)
        a.write_selection.write_text(json.dumps(sorted(controller.selected), indent=2) + "\n")
    summary = controller.report()
    print(json.dumps({k: summary[k] for k in ("total_projections", "configured_sparse_projections",
        "configured_all_coverage", "by_area")}, indent=2))


if __name__ == "__main__":
    main()
