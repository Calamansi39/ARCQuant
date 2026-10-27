"""Asset downloads should leave one cache copy and movable relative links."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import fetch_assets


class FetchAssetsTests(unittest.TestCase):
    def test_hf_snapshot_is_linked_from_asset_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / "hf_home" / "hub" / "datasets--example--data" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "test.jsonl").write_text("{}\n", encoding="utf-8")
            destination = root / "datasets" / "example"
            calls = []

            class FakeApi:
                def dataset_info(self, repo_id, revision):
                    return types.SimpleNamespace(sha="abc123")

            def fake_download(**kwargs):
                calls.append(kwargs)
                return str(snapshot)

            fake_hub = types.SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_download)
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}), \
                 patch.object(fetch_assets, "asset_root", return_value=root), \
                 patch.dict("os.environ", {"HF_HOME": ""}, clear=False):
                # Remove an empty override so the portable root is used.
                import os
                os.environ.pop("HF_HOME", None)
                result = fetch_assets.fetch_huggingface(
                    {"type": "hf_dataset", "repo_id": "example/data", "revision": "main"},
                    destination,
                )
            self.assertTrue(destination.is_symlink())
            self.assertFalse(Path(destination.readlink()).is_absolute())
            self.assertEqual((destination / "test.jsonl").read_text(), "{}\n")
            self.assertEqual(result["resolved_revision"], "abc123")
            self.assertEqual(json.loads((destination / ".arcquant_asset.json").read_text())["resolved_revision"],
                             "abc123")
            self.assertEqual(calls[0]["cache_dir"], str(root / "hf_home" / "hub"))

    def test_backend_fetch_links_sibling_data_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets = root / "assets"
            assets.mkdir()
            (root / "benchmarks").mkdir()
            (root / "benchmarks" / "activate_base_eval.sh").write_text("# test\n")
            catalog = {"backends": {"base-eval": {"source_url": "https://example.invalid/base.git",
                                                  "revision": "abc123"}}}
            with patch.object(fetch_assets, "REPOSITORY_ROOT", root), \
                 patch.object(fetch_assets, "asset_root", return_value=assets), \
                 patch.object(fetch_assets, "fetch_git", return_value={"resolved_revision": "abc123"}):
                fetch_assets.fetch_entry(catalog, "backend", "base-eval")
            self.assertEqual((root / "external" / "data").resolve(), assets)
            self.assertEqual((root / "external" / "activate_arcquant.sh").resolve(),
                             root / "benchmarks" / "activate_base_eval.sh")

    def test_backend_link_repair_preserves_real_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evaluator = root / "external/base_eval"
            evaluator.mkdir(parents=True)
            assets = root / "assets"
            assets.mkdir()
            (root / "benchmarks").mkdir()
            (root / "benchmarks/activate_base_eval.sh").touch()
            (root / "external/data").symlink_to(root / "wrong")
            (root / "external/activate_arcquant.sh").symlink_to(root / "wrong")
            with patch.object(fetch_assets, "REPOSITORY_ROOT", root), \
                 patch.object(fetch_assets, "asset_root", return_value=assets):
                fetch_assets.ensure_backend_links(evaluator)
                self.assertTrue(all(fetch_assets.backend_link_status(evaluator).values()))
                (root / "external/data").unlink()
                (root / "external/data").mkdir()
                with self.assertRaises(fetch_assets.FetchError):
                    fetch_assets.ensure_backend_links(evaluator)


if __name__ == "__main__":
    unittest.main()
