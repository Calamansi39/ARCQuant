"""Fetch catalogued sources into a portable, Git-ignored asset directory."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.paths import REPOSITORY_ROOT, asset_path, asset_root


class FetchError(RuntimeError):
    pass


def fetch_huggingface(source: dict, destination: Path, *, dry_run: bool = False) -> dict:
    kind = source["type"]
    repo_type = "model" if kind == "hf_model" else "dataset"
    plan = {"type": kind, "repo_id": source["repo_id"], "requested_revision": source["revision"],
            "destination": str(destination), "allow_patterns": source.get("allow_patterns")}
    if dry_run:
        return plan
    if destination.exists() and not destination.is_symlink():
        raise FetchError(f"{destination} already exists; select another asset root or use that local directory")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise FetchError("Install the runtime dependencies first: python -m pip install -r requirements.txt") from exc
    api = HfApi()
    try:
        info = (api.model_info if repo_type == "model" else api.dataset_info)(
            source["repo_id"], revision=source["revision"]
        )
        revision = info.sha
        hub_cache = Path(os.environ.get("HF_HOME", str(asset_root() / "hf_home"))).expanduser() / "hub"
        snapshot = Path(snapshot_download(repo_id=source["repo_id"], repo_type=repo_type,
                                          revision=revision, cache_dir=str(hub_cache),
                                          allow_patterns=source.get("allow_patterns"))).resolve()
    except Exception as exc:
        raise FetchError(f"Cannot fetch {source['repo_id']} at {source['revision']}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != snapshot:
            destination.unlink()
    if not destination.exists():
        destination.symlink_to(Path(os.path.relpath(snapshot, destination.parent)), target_is_directory=True)
    record = {**plan, "resolved_revision": revision,
              "fetched_at_utc": datetime.now(timezone.utc).isoformat()}
    (destination / ".arcquant_asset.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def fetch_git(source: dict, destination: Path, *, dry_run: bool = False) -> dict:
    plan = {"type": "git", "url": source["url"], "requested_revision": source["revision"],
            "destination": str(destination)}
    if dry_run:
        return plan
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arcquant-git-") as temp:
        env = os.environ.copy()
        local_source = Path(source["url"]).expanduser()
        if local_source.is_dir():
            # A transferred checkout can be owned by a different server user.
            # Trust only this source, only for the child Git process; leave the
            # machine's global Git configuration untouched.
            safe_config = Path(temp) / "config"
            for trusted in (local_source.resolve(), (local_source / ".git").resolve()):
                subprocess.run(["git", "config", "--file", str(safe_config), "--add",
                                "safe.directory", str(trusted)], check=True)
            env["GIT_CONFIG_GLOBAL"] = str(safe_config)
        if not destination.exists():
            clone = ["git", "clone"]
            if not local_source.is_dir():
                clone.append("--filter=blob:none")
            subprocess.run(clone + [source["url"], str(destination)],
                           check=True, env=env)
        if not (destination / ".git").exists():
            raise FetchError(f"{destination} exists but is not a Git checkout")
        command = ["git", "-C", str(destination)]
        remote = subprocess.check_output(command + ["remote", "get-url", "origin"],
                                         text=True, env=env).strip()
        if remote != source["url"]:
            raise FetchError(f"{destination} has unexpected origin: {remote}")
        expected = source["revision"]
        current = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True, env=env).strip()
        if current != expected:
            if subprocess.check_output(command + ["status", "--porcelain"], text=True, env=env).strip():
                raise FetchError(f"{destination} has local changes; cannot check out {expected}")
            subprocess.run(command + ["fetch", "origin", expected], check=True, env=env)
            subprocess.run(command + ["checkout", "--detach", expected], check=True, env=env)
        plan["resolved_revision"] = subprocess.check_output(
            command + ["rev-parse", "HEAD"], text=True, env=env
        ).strip()
    return plan


def backend_link_status(evaluator: Path) -> dict[str, bool]:
    """Check the two links required by the pinned evaluator layout."""
    data = evaluator.parent / "data"
    activation = evaluator.parent / "activate_arcquant.sh"
    return {
        "data_link": data.is_symlink() and data.resolve() == asset_root().resolve(),
        "activation_link": activation.is_symlink() and activation.resolve() == (
            REPOSITORY_ROOT / "benchmarks" / "activate_base_eval.sh").resolve(),
    }


def ensure_backend_links(evaluator: Path) -> None:
    """Repair portable sibling links without replacing real files or directories."""
    links = (
        (evaluator.parent / "data", asset_root(), True),
        (evaluator.parent / "activate_arcquant.sh",
         REPOSITORY_ROOT / "benchmarks" / "activate_base_eval.sh", False),
    )
    for link, target, is_dir in links:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise FetchError(f"{link} exists as a real file or directory; cannot replace it")
        link.symlink_to(Path(os.path.relpath(target, link.parent)), target_is_directory=is_dir)


def fetch_entry(catalog: dict, kind: str, name: str, *, dry_run: bool = False,
                source_override: str | None = None) -> dict:
    if source_override and Path(source_override).expanduser().exists():
        source_override = str(Path(source_override).expanduser().resolve())
    if kind == "backend":
        if name != "base-eval":
            raise FetchError(f"Unknown backend: {name}")
        spec = catalog["backends"][name]
        source = {"type": "git", "url": source_override or spec["source_url"],
                  "revision": spec["revision"]}
        destination = REPOSITORY_ROOT / "external" / "base_eval"
    elif kind == "runtime":
        if name != "magi-2-preview":
            raise FetchError(f"Unknown runtime source: {name}")
        source = {**catalog["models"][name]["runtime_source"]}
        if source_override:
            source["url"] = source_override
        destination = REPOSITORY_ROOT / "external" / "MAGI-2-preview"
    else:
        if source_override:
            raise FetchError("--source is only supported for a Git backend or runtime")
        group = "models" if kind == "model" else "benchmarks"
        entry = catalog[group].get(name)
        if entry is None:
            raise FetchError(f"Unknown {kind}: {name}")
        source = entry.get("source")
        if source is None:
            raise FetchError(f"No verified download source is registered for {name}; supply its local path")
        destination = asset_path(entry, name, "models" if kind == "model" else "datasets")
    if source_override and source["type"] != "git":
        raise FetchError("--source is only supported for a Git backend or runtime")
    if source["type"] == "git":
        try:
            result = fetch_git(source, destination, dry_run=dry_run)
        except subprocess.CalledProcessError as exc:
            raise FetchError(f"Git fetch failed for {name}: {exc}") from exc
        if kind == "backend" and not dry_run:
            ensure_backend_links(destination)
        return result
    return fetch_huggingface(source, destination, dry_run=dry_run)
