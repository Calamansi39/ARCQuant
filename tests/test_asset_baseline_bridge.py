import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import arc
from benchmarks.asset_inventory import audit_benchmark, audit_model
from benchmarks.baseline_bridge import BaselineError
from benchmarks.launch_deepseek_extended import launch


class AssetAndBaselineTests(unittest.TestCase):
    def test_missing_indexed_weight_shard_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model-00001.safetensors"}}))
            report = audit_model("deepseek-v4-flash", {
                "path_env": "", "default_path": str(root),
                "variants": {"tp2": str(root / "converted")},
            })
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("missing shard model-00001.safetensors", report["issues"])

    def test_dataset_count_and_size_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "test.jsonl"
            data.write_text('{"id": 1}\n{"id": 2}\n')
            entry = {"data_env": "", "default_path": str(root), "data_file": "test.jsonl",
                     "record_format": "jsonl", "expected_records": 2,
                     "expected_bytes": data.stat().st_size}
            self.assertEqual(audit_benchmark("sample", entry)["status"], "ready")
            self.assertEqual(audit_benchmark("sample", {**entry, "expected_records": 3})["status"], "incomplete")
            self.assertIn("dataset SHA256 mismatch test.jsonl",
                          audit_benchmark("sample", {**entry, "data_sha256": "0" * 64})["issues"])

    def test_model_source_signature_detects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_bytes(b"original")
            import hashlib
            entry = {"path_env": "", "default_path": str(root),
                     "source_signature": {"config.json": hashlib.sha256(b"original").hexdigest()}}
            self.assertEqual(audit_model("sample", entry)["status"], "ready")
            (root / "config.json").write_bytes(b"modified")
            self.assertIn("source signature mismatch config.json",
                          audit_model("sample", entry)["issues"])
            self.assertIn("model file config.json: expected 9 bytes, found 8",
                          audit_model("sample", {**entry, "source_signature": {},
                                                 "required_file_sizes": {"config.json": 9}})["issues"])

    def test_base_eval_plan_marks_external_protocol_and_forbids_sparsity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").touch()
            options = ["plan", "--backend", "base-eval", "--base-eval-root", str(root),
                       "--model", "deepseek-v4-flash", "--benchmark", "gpqa-diamond",
                       "--profile", "smoke", "--method", "none", "--quant-type", "HiF4",
                       "--gpu-ids", "0,1"]
            plan = arc.build_plan(arc._parser().parse_args(options))
            self.assertEqual(plan["env"]["QUANTS"], "hifp4")
            self.assertEqual(plan["command"][-2:], ["gpqa", "8"])
            self.assertIn("no ARCQuant residual channels", plan["protocol"])
            options[options.index("none")] = "teal"
            with self.assertRaises(BaselineError):
                arc.build_plan(arc._parser().parse_args(options))

    def test_magi_python_venv_symlink_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            venv = root / "venv"
            (base / "bin").mkdir(parents=True)
            (venv / "bin").mkdir(parents=True)
            (base / "bin/python").touch()
            (venv / "bin/python").symlink_to(base / "bin/python")
            with patch.dict("os.environ", {"ARCQUANT_MAGI2_PYTHON": str(venv / "bin/python")},
                            clear=False):
                options = arc._parser().parse_args([
                    "plan", "--backend", "base-eval", "--model", "magi-2-preview",
                    "--benchmark", "vbench", "--profile", "tiny", "--method", "none",
                    "--quant-type", "HiF4", "--gpu-ids", "0,1"])
                plan = arc.build_plan(options)
            self.assertEqual(plan["env"]["MAGI2_PYTHON"], str(venv / "bin/python"))

    def test_deepseek_extended_uses_explicit_python_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_root = root / "eval-env"
            (env_root / "bin").mkdir(parents=True)
            (env_root / "bin/python").symlink_to(sys.executable)
            evaluator = root / "external/base_eval"
            (evaluator / "scripts").mkdir(parents=True)
            script = evaluator / "scripts/run_all_deepseek_v4_extended.sh"
            script.write_text('#!/bin/bash\nset -euo pipefail\n'
                              'cd "$(dirname "$0")/.."\nsource ./env.sh\n'
                              'source activate arcquant 2>/dev/null || conda activate arcquant\n'
                              'command -v python > "$ARCQUANT_TEST_OUTPUT"\n')
            (evaluator / "env.sh").write_text(
                'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n')
            (root / "external/activate_arcquant.sh").symlink_to(
                arc.ROOT / "benchmarks/activate_base_eval.sh")
            output = root / "selected-python"
            with patch.dict("os.environ", {"ARCQUANT_BASE_EVAL_ENV": str(env_root),
                                         "ARCQUANT_TEST_OUTPUT": str(output)}):
                options = arc._parser().parse_args([
                    "plan", "--backend", "base-eval", "--base-eval-root", str(evaluator),
                    "--model", "deepseek-v4-flash",
                    "--benchmark", "longbench-v2", "--profile", "smoke", "--method", "none",
                    "--gpu-ids", "0,1"])
                plan = arc.build_plan(options)
                self.assertEqual(plan["evaluator_launcher"],
                                 str(evaluator / "scripts/run_all_deepseek_v4_extended.sh"))
                self.assertTrue(plan["command"][1].endswith("launch_deepseek_extended.py"))
                with self.assertRaisesRegex(ValueError, "has changed"):
                    launch(evaluator, ["longbench-v2", "0,1"])
                with patch("benchmarks.launch_deepseek_extended.PINNED_SHA256",
                           hashlib.sha256(script.read_bytes()).hexdigest()):
                    self.assertEqual(launch(evaluator, ["longbench-v2", "0,1"]), 0)
            self.assertEqual(output.read_text().strip(), str(env_root / "bin/python"))


if __name__ == "__main__":
    unittest.main()
