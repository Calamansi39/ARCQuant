import json
import tempfile
import unittest
from pathlib import Path

import arc
from benchmarks.asset_inventory import audit_benchmark, audit_model
from benchmarks.baseline_bridge import BaselineError


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


if __name__ == "__main__":
    unittest.main()
