"""Portable, hash-verified VBench scoring assets from the repository manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from benchmarks.paths import REPOSITORY_ROOT, asset_root


MANIFESTS = (REPOSITORY_ROOT / "configs/vbench_download_manifest.json",
             REPOSITORY_ROOT / "configs/vbench_bert_download_manifest.json")
AUXILIARY = REPOSITORY_ROOT / "configs/vbench_aux_download_manifest.json"


def scorer_root() -> Path:
    return Path(os.environ.get("VBENCH_CACHE_DIR", asset_root() / "vbench_cache")).expanduser().resolve()


def entries() -> list[tuple[dict, dict, Path]]:
    result = []
    for index, path in enumerate(MANIFESTS):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        prefix = Path("bert-base-uncased") if index else Path()
        result += [(manifest, item, prefix / item["path"]) for item in manifest["files"]]
    return result


def auxiliary_entries() -> list[dict]:
    return json.loads(AUXILIARY.read_text(encoding="utf-8"))["files"]


def verify_file(path: Path, spec: dict, *, hash_contents: bool = True) -> str:
    """Return ready/missing/size-mismatch/hash-mismatch for a manifest entry."""
    if not path.is_file():
        return "missing"
    if path.stat().st_size != spec["size"]:
        return "size-mismatch"
    if not hash_contents:
        return "size-only"
    sha = spec.get("sha256")
    blob = spec.get("git_blob")
    if not sha and not blob:
        raise ValueError(f"Scorer entry lacks a content digest: {spec['path']}")
    digest = hashlib.sha256() if sha else hashlib.sha1()
    if blob:
        digest.update(f"blob {spec['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return "ready" if digest.hexdigest() == (sha or blob) else "hash-mismatch"


def audit_scorer(*, hash_contents: bool = True) -> dict:
    root = scorer_root()
    files = []
    for _, spec, relative in entries():
        files.append({"path": str(relative), "status": verify_file(
            root / relative, spec, hash_contents=hash_contents)})
    for spec in auxiliary_entries():
        files.append({"path": spec["path"], "status": verify_file(
            root / spec["path"], spec, hash_contents=hash_contents)})
    indexed = {str(relative): spec for _, spec, relative in entries()}
    dino = root / "torch_hub/hub/checkpoints/dino_vitbase16_pretrain.pth"
    files.append({"path": "torch_hub/hub/checkpoints/dino_vitbase16_pretrain.pth",
                  "status": verify_file(dino, indexed["dino_model/dino_vitbase16_pretrain.pth"],
                                        hash_contents=hash_contents)})
    bert = json.loads(MANIFESTS[1].read_text(encoding="utf-8"))
    hub = asset_root() / "hf_home/hub/models--bert-base-uncased"
    ref = hub / "refs/main"
    files.append({"path": "hf_home/hub/models--bert-base-uncased/refs/main",
                  "status": "ready" if ref.is_file() and ref.read_text().strip() == bert["revision"]
                  else "missing"})
    for spec in bert["files"]:
        alias = hub / "snapshots" / bert["revision"] / spec["path"]
        files.append({"path": f"hf_home/hub/models--bert-base-uncased/snapshots/{bert['revision']}/{spec['path']}",
                      "status": verify_file(alias, spec, hash_contents=hash_contents)})
    issues = [item for item in files if item["status"] in (
        "missing", "size-mismatch", "hash-mismatch")]
    state = "incomplete" if issues else "ready" if hash_contents else "size-only"
    return {"status": state, "path": str(root), "file_count": len(files),
            "verified_count": sum(item["status"] == "ready" for item in files),
            "issues": issues}


def _link_verified(target: Path, source: Path, spec: dict) -> None:
    if target.exists() or target.is_symlink():
        if verify_file(target, spec) == "ready":
            return
        if not target.is_symlink():
            raise FileExistsError(f"Corrupt scoring asset exists; move it aside: {target}")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(Path(os.path.relpath(source, target.parent)))


def fetch_scorer() -> dict:
    """Download pinned files into HF cache, then link verified files into VBench."""
    root = scorer_root()
    cache = asset_root() / "hf_home" / "hub"
    download = None
    for manifest, spec, relative in entries():
        target = root / relative
        if verify_file(target, spec) == "ready":
            continue
        if download is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError("Install huggingface-hub before fetching scorer assets") from exc
            download = hf_hub_download
        source = Path(download(repo_id=manifest["repo_id"], filename=spec["path"],
                               revision=manifest["revision"], cache_dir=cache))
        state = verify_file(source, spec)
        if state != "ready":
            raise ValueError(f"Downloaded {relative} failed manifest validation: {state}")
        _link_verified(target, source, spec)
    for spec in auxiliary_entries():
        target = root / spec["path"]
        if verify_file(target, spec) == "ready":
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Corrupt tokenizer asset exists; move it aside: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".arcquant-vbench-", dir=target.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urllib.request.urlopen(spec["url"], timeout=120) as response:
                    shutil.copyfileobj(response, stream)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        if verify_file(temporary, spec) != "ready":
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Downloaded tokenizer failed manifest validation: {spec['path']}")
        temporary.replace(target)
    # The official scorer resolves these two weights through standard caches.
    indexed = {str(relative): spec for _, spec, relative in entries()}
    dino = root / "torch_hub/hub/checkpoints/dino_vitbase16_pretrain.pth"
    _link_verified(dino, root / "dino_model/dino_vitbase16_pretrain.pth",
                   indexed["dino_model/dino_vitbase16_pretrain.pth"])
    bert = next(manifest for manifest, _, relative in entries()
                if str(relative).startswith("bert-base-uncased/"))
    hub = cache / "models--bert-base-uncased"
    snapshot = hub / "snapshots" / bert["revision"]
    for _, spec, relative in entries():
        if not str(relative).startswith("bert-base-uncased/"):
            continue
        _link_verified(snapshot / spec["path"], root / relative, spec)
    (hub / "refs").mkdir(parents=True, exist_ok=True)
    (hub / "refs/main").write_text(bert["revision"], encoding="utf-8")
    return audit_scorer(hash_contents=True)
