"""Portable asset paths shared by the CLI, inventory, and baseline bridge."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def asset_root() -> Path:
    raw = os.environ.get("ARCQUANT_ASSET_ROOT")
    return Path(raw).expanduser().resolve() if raw else REPOSITORY_ROOT / "assets"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def executable_path(value: str | Path) -> Path:
    """Keep a venv's bin/python symlink intact so Python finds its pyvenv.cfg."""
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else REPOSITORY_ROOT / path))


def asset_path(entry: dict, name: str, kind: str, *, override: str | Path | None = None) -> Path:
    """Explicit flag > per-asset env > catalog path > portable asset root."""
    env_key = entry.get("path_env") if kind == "models" else entry.get("data_env")
    raw = override or (os.environ.get(env_key) if env_key else None)
    if raw:
        return resolve_path(raw)
    if entry.get("default_path"):
        return resolve_path(entry["default_path"])
    return (asset_root() / kind / name).resolve()


def variant_path(entry: dict, variant: str, *, override: str | Path | None = None) -> Path:
    info = entry.get("variants", {}).get(variant)
    if info is None:
        raise KeyError(f"Unknown model variant: {variant}")
    if isinstance(info, str):
        return resolve_path(override or info)
    raw = override or os.environ.get(info.get("path_env", "")) or info.get("default_path")
    if raw:
        return resolve_path(raw)
    return (asset_root() / "models" / info.get("directory", variant)).resolve()
