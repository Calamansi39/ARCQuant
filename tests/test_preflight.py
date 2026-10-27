import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arc
from benchmarks import preflight as checks
from benchmarks.provision import provision_plan
from benchmarks.scorer_assets import verify_file


class PreflightTests(unittest.TestCase):
    def test_converted_deepseek_requires_matching_source_and_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            converted = root / "tp2"
            source.mkdir()
            converted.mkdir()
            names = ("config.json", "model.safetensors.index.json", "inference/model.py",
                     "inference/convert.py")
            signature = {}
            for name in names:
                file = source / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(name)
                signature[name] = hashlib.sha256(name.encode()).hexdigest()
            sizes = {}
            for rank in range(2):
                name = f"model{rank}-mp2.safetensors"
                (converted / name).write_bytes(b"weight")
                sizes[name] = 6
            manifest = {"completed": True, "world_size": 2, "source_signature": signature,
                        "shard_bytes": sizes}
            (converted / "conversion.json").write_text(json.dumps(manifest))
            self.assertTrue(checks._converted_deepseek(source, converted)[0])
            (source / "config.json").write_text("changed")
            self.assertFalse(checks._converted_deepseek(source, converted)[0])

    def test_target_arcquant_run_remains_blocked(self):
        with patch.object(checks, "audit_model", return_value={"status": "ready", "issues": []}), \
             patch.object(checks, "audit_benchmark", return_value={"status": "ready", "issues": []}):
            result = checks.preflight(arc.read_catalog(), model="deepseek-v4-flash",
                                      benchmark="gpqa-diamond", backend="arcquant",
                                      method="wina", check_gpu=False)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("model_adapter", [item["name"] for item in result["checks"]
                       if item["status"] == "blocked"])

    def test_manifest_hash_and_git_blob_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "scorer.bin"
            file.write_bytes(b"test")
            sha = hashlib.sha256(b"test").hexdigest()
            blob = hashlib.sha1(b"blob 4\0test").hexdigest()
            self.assertEqual(verify_file(file, {"path": file.name, "size": 4,
                                                "sha256": sha, "git_blob": None}), "ready")
            self.assertEqual(verify_file(file, {"path": file.name, "size": 4,
                                                "sha256": None, "git_blob": blob}), "ready")
            self.assertEqual(verify_file(file, {"path": file.name, "size": 4,
                                                "sha256": "0" * 64, "git_blob": None}),
                             "hash-mismatch")
            self.assertEqual(verify_file(file, {"path": file.name, "size": 5,
                                                "sha256": sha, "git_blob": None}), "size-mismatch")

    def test_doctor_cli_returns_nonzero_for_blocked_prerequisites(self):
        with patch.object(arc, "preflight", return_value={"status": "blocked", "checks": []}) as call:
            result = arc.main(["doctor", "--model", "deepseek-v4-flash",
                               "--benchmark", "gpqa-diamond", "--method", "wina",
                               "--skip-gpu", "--native-dry-run"])
        self.assertEqual(result, 2)
        self.assertEqual(call.call_args.kwargs["method"], "wina")
        self.assertFalse(call.call_args.kwargs["check_gpu"])
        self.assertTrue(call.call_args.kwargs["native_dry_run"])

    def test_provision_lists_converted_shards_after_model_and_evaluator(self):
        with patch("benchmarks.provision.audit_model", return_value={"status": "ready", "path": "/tmp/model"}), \
             patch("benchmarks.provision.audit_benchmark", return_value={"status": "ready", "path": "/tmp/data"}), \
             patch("benchmarks.provision._revision", return_value=None), \
             patch("benchmarks.provision._converted_deepseek", return_value=(False, "missing")):
            steps = provision_plan(arc.read_catalog(), model="deepseek-v4-flash",
                                   benchmark="gpqa-diamond")
        self.assertEqual([step["step"] for step in steps],
                         ["model", "benchmark", "evaluator", "data_link", "activation_link",
                          "evaluator_launcher", "deepseek_tp2"])
        self.assertEqual(steps[-1]["status"], "needed")

    def test_doctor_rejects_missing_activation_link(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evaluator = root / "base_eval"
            (evaluator / "scripts").mkdir(parents=True)
            (evaluator / "scripts/run_all_generative.sh").touch()
            with patch.object(checks, "audit_model", return_value={"status": "ready", "issues": []}), \
                 patch.object(checks, "audit_benchmark", return_value={"status": "ready", "issues": []}), \
                 patch.object(checks, "_revision", return_value=arc.read_catalog()["backends"]["base-eval"]["revision"]), \
                 patch.object(checks, "_converted_deepseek", return_value=(True, "valid")):
                result = checks.preflight(arc.read_catalog(), model="deepseek-v4-flash",
                                          benchmark="gpqa-diamond", backend_override=str(evaluator),
                                          check_gpu=False, probe_runtime=False)
            blocked = {item["name"] for item in result["checks"] if item["status"] == "blocked"}
            self.assertIn("evaluator_activation_link", blocked)


if __name__ == "__main__":
    unittest.main()
