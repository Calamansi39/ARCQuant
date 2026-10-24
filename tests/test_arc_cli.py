import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arc


def args(**updates):
    values = dict(
        model="llama-3.1-8b", model_path="/tmp/Llama-3.1-8B",
        benchmark="wikitext2", profile="full", method="none", quant_type="HiF4",
        calibration_dataset="wikitext2", act_sort_metric="max", histogram_path=None,
        sparsity=0.0, disable_prefill_sparsity=False,
        kv_cache=False, seed=0, lm_eval_include_path=None,
        x_mask_ckpt=None, x_mask_tau=1.0, x_mask_alpha=1.0,
        x_mask_r_thr=-1.0, softmax_alpha_ckpt=None, rec=False,
    )
    values.update(updates)
    return argparse.Namespace(**values)


class ArcCliTests(unittest.TestCase):
    def test_teal_and_hif4_are_independent_options(self):
        plan = arc.build_plan(args(method="teal", sparsity=0.5, histogram_path="/tmp/teal-hist"))
        self.assertEqual(plan["method"], "teal")
        self.assertEqual(plan["quant_type"], "HiF4")
        self.assertEqual(plan["command"][plan["command"].index("--sparse_method") + 1], "teal")
        self.assertEqual(plan["command"][plan["command"].index("--quant_type") + 1], "HiF4")
        self.assertEqual(len(plan["cache_paths"]), 3)

    def test_wina_runs_with_hif4_without_histograms(self):
        plan = arc.build_plan(args(method="wina", sparsity=0.5))
        command = plan["command"]
        self.assertEqual(command[command.index("--sparse_method") + 1], "wina")
        self.assertEqual(command[command.index("--quant_type") + 1], "HiF4")
        self.assertNotIn("--teal_histogram_path", command)

    def test_greedy_plan_uses_one_default_strategy_path(self):
        for method in ("teal", "wina"):
            plan = arc.build_plan(args(method=method, sparsity=0.5, allocation="greedy"))
            self.assertEqual(plan["allocation"], "greedy")
            self.assertEqual(plan["command"][plan["command"].index("--sparse_strategy_path") + 1],
                             plan["strategy_path"])
            self.assertTrue(plan["strategy_path"].endswith(f"/{method}/strategy.json"))
            self.assertEqual("--teal_histogram_path" in plan["command"], method == "teal")

    def test_unsupported_model_and_benchmark_are_rejected(self):
        with self.assertRaisesRegex(arc.PlanError, "no ARCQuant model adapter"):
            arc.build_plan(args(model="magi-2-preview"))
        with self.assertRaisesRegex(arc.PlanError, "no evaluator adapter"):
            arc.build_plan(args(benchmark="longbench-v2"))

    def test_teal_is_limited_to_llama(self):
        with self.assertRaisesRegex(arc.PlanError, "Llama only"):
            arc.build_plan(args(model="qwen2.5-7b", model_path="/tmp/Qwen2.5-7B",
                                    method="teal", histogram_path="/tmp/teal-hist"))

    def test_prepare_cache_names_match_legacy_main(self):
        plan = arc.build_plan(args())
        self.assertEqual(Path(plan["cache_paths"][0]).name,
                         "llama-3.1-8b_reorder_index_wikitext2_max.pt")

    def test_ppl_does_not_launch_default_lm_eval_tasks(self):
        command = arc.build_plan(args())["command"]
        self.assertEqual(command[-2:], ["--eval_ppl", "--tasks"])
        self.assertNotIn("--eval_devices", command)

    def test_gpqa_and_math500_use_installed_task_names(self):
        for benchmark, task in (("gpqa-diamond", "gpqa_diamond_zeroshot"),
                                ("math500", "hendrycks_math500")):
            command = arc.build_plan(args(benchmark=benchmark, profile="smoke"))["command"]
            self.assertEqual(command[command.index("--tasks") + 1], task)
            self.assertEqual(command[command.index("--lm_eval_limit") + 1], "8")

    def test_teal_histogram_check_covers_every_model_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
            (root / "hist" / "layer-0" / "self_attn").mkdir(parents=True)
            (root / "hist" / "layer-0" / "self_attn" / "histograms.pt").touch()
            missing = arc.missing_teal_histograms(root / "hist", model)
            self.assertEqual(len(missing), 3)
            self.assertIn(root / "hist" / "layer-1" / "mlp" / "histograms.pt", missing)

    def test_xmask_routes_to_existing_main_interface(self):
        command = arc.build_plan(args(method="xmask", x_mask_ckpt="/tmp/mask.pt"))["command"]
        self.assertIn("--use_x_mask", command)
        self.assertEqual(command[command.index("--sparse_method") + 1], "none")
        self.assertEqual(command[command.index("--x_mask_ckpt") + 1], "/tmp/mask.pt")

    def test_runtime_check_rejects_transformers_api_drift(self):
        with patch.object(arc, "version", return_value="5.9.0"):
            with self.assertRaisesRegex(arc.PlanError, "requires transformers==4.46.3"):
                arc.check_runtime_compatibility()


if __name__ == "__main__":
    unittest.main()
