import argparse
import unittest

from benchmarks.sparse_sweep import build_deepseek_plan, rates_for


class SparsitySweepTests(unittest.TestCase):
    def test_profiles_include_native_high_wina_point(self):
        self.assertEqual(rates_for("wina_native", "wina"), [0.25, 0.4, 0.5, 0.65])
        self.assertEqual(rates_for("high_sparsity", "teal"), [0.65])

    def test_formal_candidate_plan_separates_methods_and_rates(self):
        args = argparse.Namespace(
            profile="formal_candidate", methods="wina,teal", benchmark="math500",
            base_eval_root="/tmp/base", source_model="/tmp/source",
            converted_model="/tmp/tp2", calibration="/tmp/calibration",
            gpu_ids="2,3", limit=1, max_new_tokens=128,
            out_root="/tmp/sweep",
        )
        plan = build_deepseek_plan(args)
        self.assertEqual([(item["method"], item["sparsity"]) for item in plan["runs"]],
                         [("wina", 0.5), ("wina", 0.65), ("teal", 0.5), ("teal", 0.65)])
        self.assertTrue(all("sparsity_" not in item["output_dir"] for item in plan["runs"]))
        self.assertTrue(any("--sparsity 0.65" in item["shell_command"] for item in plan["runs"]))


if __name__ == "__main__":
    unittest.main()
