"""CPU-only checks for the external checkpoints and benchmark data in this catalog.

These checks establish local file completeness for the selected experiments.
They do not establish model accuracy, inference compatibility, or GPU capacity.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.paths import asset_path, variant_path


MODEL_FILES = {
    "llama-3.1-8b": ("config.json", "tokenizer_config.json"),
    "qwen2.5-7b": ("config.json", "tokenizer_config.json"),
    "mixtral-8x7b": ("config.json", "tokenizer_config.json"),
    "deepseek-v4-flash": (
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "model.safetensors.index.json",
        "inference/model.py", "encoding/encoding_dsv4.py",
    ),
    "minimax-h3": (
        "model_index.json", "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json", "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors.index.json",
        "audio_vae/config.json", "audio_vae/diffusion_pytorch_model.safetensors",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        "processor/preprocessor_config.json", "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    ),
    "magi-2-preview": (
        "preview/model.safetensors.index.json", "refiner/model.safetensors.index.json",
        "text_encoder/config.json", "text_encoder/tokenizer.json",
        "text_encoder/model.safetensors.index.json", "vae/Wan2.2_VAE.pth",
        "stable-audio-open-1.0/model.safetensors", "turbo_vae/checkpoint.ckpt",
    ),
}


def _resolve(entry: dict, name: str, kind: str) -> Path:
    return asset_path(entry, name, kind)


def _safetensors_header_ok(path: Path) -> bool:
    if path.stat().st_size < 10:
        return False
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
    return 2 <= header_size <= 100_000_000 and path.stat().st_size >= header_size + 8


def audit_model(name: str, entry: dict) -> dict:
    root = _resolve(entry, name, "models")
    issues: list[str] = []
    referenced_shards: set[Path] = set()
    if root is None or not root.is_dir():
        return {"status": "missing", "path": str(root) if root else None,
                "issues": ["model directory is absent"]}
    for relative in MODEL_FILES.get(name, ()):
        if not (root / relative).is_file():
            issues.append(f"missing {relative}")
    for index in root.rglob("*.safetensors.index.json"):
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            referenced_shards.update(index.parent / filename for filename in weight_map.values())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"invalid index {index.relative_to(root)}: {exc}")
    for shard in sorted(referenced_shards):
        if not shard.is_file():
            issues.append(f"missing shard {shard.relative_to(root)}")
        elif not _safetensors_header_ok(shard):
            issues.append(f"invalid safetensors header {shard.relative_to(root)}")
    if name in ("llama-3.1-8b", "qwen2.5-7b", "mixtral-8x7b"):
        standalone = list(root.glob("*.safetensors"))
        if not referenced_shards and not standalone:
            issues.append("no model weight files or indexed shards")
        for shard in standalone:
            if not _safetensors_header_ok(shard):
                issues.append(f"invalid safetensors header {shard.relative_to(root)}")
    for relative in MODEL_FILES.get(name, ()):
        file = root / relative
        if file.is_file() and file.suffix == ".safetensors" and not _safetensors_header_ok(file):
            issues.append(f"invalid safetensors header {relative}")
    result = {"status": "ready" if not issues else "incomplete", "path": str(root),
              "indexed_shards": len(referenced_shards), "issues": issues}
    if name == "minimax-h3":
        result["scope"] = "T2VA/VBench components; reference-conditioned transformer_ref is optional and absent"
    if name == "deepseek-v4-flash":
        converted = variant_path(entry, "tp2")
        files = ("conversion.json", "model0-mp2.safetensors", "model1-mp2.safetensors")
        converted_issues = [f"missing {item}" for item in files if not (converted / item).is_file()]
        for item in files[1:]:
            shard = converted / item
            if shard.is_file() and not _safetensors_header_ok(shard):
                converted_issues.append(f"invalid safetensors header {item}")
        result["tp2"] = {"path": str(converted),
                         "status": "ready" if not converted_issues else "incomplete",
                         "issues": converted_issues}
    return result


def _count_records(path: Path, entry: dict) -> int:
    kind = entry["record_format"]
    if kind == "jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return sum(bool(line.strip()) for line in stream)
    if kind == "csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    if kind == "json":
        return len(json.loads(path.read_text(encoding="utf-8")))
    if kind == "unique_prompt_json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return len({row["prompt_en"] for row in data})
    if kind == "terminal_tasks":
        return sum((child / "task.yaml").is_file() for child in path.iterdir() if child.is_dir())
    raise ValueError(f"Unsupported record format: {kind}")


def audit_benchmark(name: str, entry: dict) -> dict:
    root = _resolve(entry, name, "datasets")
    relative = entry.get("data_file", ".")
    file = root / relative
    issues: list[str] = []
    if not file.exists():
        issues.append(f"missing {relative}")
    elif entry.get("expected_bytes") is not None and file.is_file():
        actual_bytes = file.stat().st_size
        if actual_bytes != entry["expected_bytes"]:
            issues.append(f"expected {entry['expected_bytes']} bytes, found {actual_bytes}")
    count = None
    if not issues and entry.get("record_format"):
        try:
            count = _count_records(file, entry)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"cannot read records: {exc}")
    expected = entry.get("expected_records")
    if count is not None and expected is not None and count != expected:
        issues.append(f"expected {expected} records, found {count}")
    return {"status": "ready" if not issues else "incomplete", "path": str(root),
            "file": str(file), "records": count, "expected_records": expected, "issues": issues}


def audit_catalog(catalog: dict) -> dict:
    return {
        "models": {name: audit_model(name, entry) for name, entry in catalog["models"].items()
                   if name in MODEL_FILES},
        "benchmarks": {name: audit_benchmark(name, entry)
                       for name, entry in catalog["benchmarks"].items()
                       if "data_file" in entry},
    }
