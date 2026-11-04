import argparse
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from benchmarks.structured_plan import build_plan
from benchmarks.structured_report import merge_reports
from model.structured_sparsity import Structured24Controller, select_projections, merge_inventories
from model.deepseek_structured_runtime import DeepSeekStructured24Mixin


class DenseRuntime:
    quant_type = "hifp4"
    act_quant = True

    def quantize_activation(self, x):
        self.quant_inputs.append(x.detach().clone())
        return x.round()  # Independent zero-preserving quantizer for boundary tests.

    def linear(self, x, weight, bias=None):
        return torch.nn.functional.linear(self.quantize_activation(x), weight, bias)

    def stats(self):
        return {}


class Runtime(DeepSeekStructured24Mixin, DenseRuntime):
    def __init__(self):
        self.quant_inputs = []


class Attention(nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.wo_a = nn.Linear(8, 4, bias=False)
        self.n_local_groups = 2
        self.runtime = runtime

    def forward(self, x, ragged=False, fail=False):
        if fail:
            raise RuntimeError("expected attention error")
        # Mirrors both external paths: they quantize grouped activation then
        # read wo_a weights directly, never calling wo_a.forward.
        values = self.runtime.quantize_activation(x)
        return torch.einsum("bsgd,grd->bsgr", values, self.wo_a.weight.view(2, 2, 8))


def toy_model(runtime):
    model = nn.Module()
    model.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.attn = Attention(runtime)
        layer.ffn = nn.Module()
        layer.ffn.w1 = nn.Linear(8, 4, bias=False)
        model.layers.append(layer)
    return model


class Structured24Tests(unittest.TestCase):
    def test_group_constraint_not_global_topk_and_chunk_equivalence(self):
        spec = {"x": {"width": 8, "area": "ffn"}}
        x = torch.tensor([[1., 2., 3., 4., 100., 200., 300., 400.]]).repeat(7, 1)
        small, large = Structured24Controller(spec, chunk_rows=1), Structured24Controller(spec)
        a, b = small.mask(x, "x"), large.mask(x, "x")
        expected = torch.tensor([[0., 0., 3., 4., 0., 0., 300., 400.]]).repeat(7, 1)
        self.assertTrue(torch.equal(a, expected))
        self.assertTrue(torch.equal(a, b))
        small.observe_quantized(a, "x")
        self.assertEqual(small.report()["mask_zero_fraction_on_this_rank"], .5)

    def test_ties_zeros_empty_noncontiguous_and_negative_values(self):
        c = Structured24Controller({"x": {"width": 8, "area": "ffn"}})
        x = torch.tensor([[-5., -1., 3., 2., 0., 0., 0., 0.]]).repeat(4, 1)[::2]
        out = c.mask(x, "x")
        c.observe_quantized(out, "x")
        self.assertEqual(out[0, 0], -5.)
        self.assertEqual(out[0, 2], 3.)
        ties = c.mask(torch.ones(3, 8), "x")
        self.assertTrue((ties.reshape(-1, 4).count_nonzero(-1) == 2).all())
        self.assertEqual(c.mask(torch.empty(0, 8), "x").shape, (0, 8))
        self.assertGreater(c.report()["post_quant_zero_fraction_on_this_rank"], 0)

    def test_zero_coverage_is_exact_identity(self):
        c = Structured24Controller({"x": {"width": 8, "area": "ffn"}}, coverage=0)
        x = torch.randn(2, 4, 8)
        self.assertIs(c.mask(x, "x"), x)
        self.assertIsNone(c.report()["mask_zero_fraction_on_this_rank"])

    def test_invalid_and_nonfinite_inputs_fail(self):
        spec = {"x": {"width": 8, "area": "ffn"}}
        for value in (float("nan"), float("inf"), -1., 1.01):
            with self.assertRaises(ValueError):
                Structured24Controller(spec, coverage=value)
        c = Structured24Controller(spec)
        with self.assertRaises(ValueError):
            c.mask(torch.full((1, 8), float("nan")), "x")
        with self.assertRaises(ValueError):
            c.mask(torch.ones(1, 4), "x")
        c.mask(torch.ones(1, 8), "x")
        with self.assertRaises(ValueError):
            c.observe_quantized(torch.ones(1, 8), "x")

    def test_coverage_counts_projections_and_preserves_unsupported_denominator(self):
        spec = {f"layer{i}": {"width": 8, "area": "ffn"} for i in range(10)}
        spec["layer9"]["width"] = 7
        c = Structured24Controller(spec, coverage=.8)
        self.assertEqual(len(c.selected), 8)
        self.assertEqual(c.report()["configured_all_coverage"], .8)
        self.assertNotIn("layer9", c.selected)
        self.assertEqual(select_projections(dict(reversed(list(spec.items()))), .8), c.selected)
        with self.assertRaises(ValueError):
            Structured24Controller(spec, coverage=1.)
        with self.assertRaises(ValueError):
            Structured24Controller(spec, coverage=.8, selected_names=["layer0"])

    def test_scope_does_not_inflate_all_model_coverage(self):
        spec = {"a": {"width": 8, "area": "attention"}, "b": {"width": 8, "area": "ffn"}}
        c = Structured24Controller(spec, scope="attention")
        self.assertEqual(c.report()["configured_scope_coverage"], 1.)
        self.assertEqual(c.report()["configured_all_coverage"], .5)

    def test_tp_merge_deduplicates_replicated_projections(self):
        a = {"shared": {"width": 8, "area": "attention"}, "expert0": {"width": 8, "area": "ffn"}}
        b = {"shared": a["shared"], "expert1": a["expert0"]}
        self.assertEqual(len(merge_inventories([a, b])), 3)
        with self.assertRaises(ValueError):
            merge_inventories([a, {"shared": {"width": 4, "area": "attention"}}])

    def test_moe_expert_count_cannot_hide_attention_coverage(self):
        inventory = {f"a{i}": {"width": 8, "area": "attention"} for i in range(3)}
        inventory.update({f"e{i}": {"width": 8, "area": "ffn"} for i in range(100)})
        c = Structured24Controller(inventory, coverage=.8)
        report = c.report()
        self.assertEqual(report["by_area"]["attention"]["selected"], 3)
        self.assertEqual(report["by_area"]["ffn"]["selected"], 80)
        with self.assertRaises(ValueError):
            Structured24Controller(inventory, coverage=.8, selected_names=[f"e{i}" for i in range(100)])

    def test_grouped_layer_identity_and_quantization_order(self):
        r = Runtime()
        model = toy_model(r)
        selected = ["layers.1.attn.wo_a", "layers.0.ffn.w1"]
        r.configure_structured(SimpleNamespace(Linear=nn.Linear), model, coverage=.5,
                               selected_names=selected)
        x = torch.arange(1., 33.).reshape(1, 2, 2, 8)
        model.layers[0].attn(x)
        model.layers[1].attn(x, ragged=True)
        self.assertTrue(torch.equal(r.quant_inputs[0], x))
        self.assertTrue((r.quant_inputs[1].reshape(-1, 4).count_nonzero(-1) == 2).all())
        r.linear(torch.ones(5, 8), model.layers[0].ffn.w1.weight)
        report = r.stats()["structured_sparsity"]
        self.assertEqual(report["executed_sparse_projections_on_this_rank"], 2)
        self.assertNotIn("layers.0.attn.wo_a", report["counters_on_this_rank"])
        self.assertEqual(report["mask_zero_fraction_on_this_rank"], .5)
        r.remove_structured_hooks()

    def test_attention_context_cleans_up_on_exception(self):
        r = Runtime()
        model = toy_model(r)
        r.configure_structured(SimpleNamespace(Linear=nn.Linear), model)
        with self.assertRaisesRegex(RuntimeError, "expected attention"):
            model.layers[0].attn(torch.ones(1, 1, 2, 8), fail=True)
        self.assertEqual(r._structured_active_attention, [])
        with self.assertRaisesRegex(RuntimeError, "without an attention"):
            r.quantize_activation(torch.ones(1, 1, 2, 8))
        r.remove_structured_hooks()

    def test_post_quant_extra_zeros_are_distinct_from_mask(self):
        r = Runtime()
        model = toy_model(r)
        r.configure_structured(SimpleNamespace(Linear=nn.Linear), model)
        r.linear(torch.full((2, 8), .1), model.layers[0].ffn.w1.weight)
        stats = r.stats()["structured_sparsity"]
        self.assertEqual(stats["mask_zero_fraction_on_this_rank"], .5)
        self.assertEqual(stats["post_quant_zero_fraction_on_this_rank"], 1.)
        r.remove_structured_hooks()

    def test_report_never_certifies_missing_ranks_or_unobserved_layers(self):
        c = Structured24Controller({"a": {"width": 8, "area": "ffn"},
                                    "b": {"width": 8, "area": "ffn"}})
        c.observe_quantized(c.mask(torch.ones(2, 8), "a"), "a")
        rank = {"rank": 0, "world_size": 2, "configuration": {},
                "evaluation_returned_successfully": True,
                "runtime": {"structured_sparsity": c.report()}}
        result = merge_reports([rank])
        self.assertEqual(result["missing_ranks"], [1])
        self.assertEqual(result["unobserved_selected"], ["b"])
        self.assertFalse(result["all_rank_evaluators_returned_successfully"])
        second = copy.deepcopy(rank)
        second["rank"] = 1
        complete = merge_reports([rank, second])
        self.assertTrue(complete["observed_structure_valid"])
        self.assertEqual(complete["observed_selected_projections"], 1)
        self.assertEqual(complete["acceptance_status"], "not_evaluated")
        with self.assertRaises(ValueError):
            merge_reports([rank, rank])

    def test_plan_separates_layer_coverage_and_zero_fraction(self):
        args = argparse.Namespace(benchmark="math500", base_eval_root="/tmp/base",
            source_model="/tmp/source", converted_model="/tmp/tp2", layer_coverage=.8,
            scope="all", selection=None, gpu_ids="2,3", limit=1, max_new_tokens=128,
            batch_size=1, smoke_steps=4, out_dir="validation/structured", dry_run=False)
        plan = build_plan(args)
        self.assertEqual(plan["layer_coverage"], .8)
        self.assertEqual(plan["element_mask_fraction"], .5)
        self.assertNotIn("--sparsity", plan["command"])
        args.dry_run = True
        self.assertNotIn("torch.distributed.run", build_plan(args)["command"])
        args.benchmark = "livecodebench"
        with self.assertRaises(ValueError):
            build_plan(args)
        args.benchmark = "gpqa-diamond"
        with self.assertRaisesRegex(ValueError, "1026"):
            build_plan(args)
        args.max_new_tokens = None
        command = build_plan(args)["command"]
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "8192")


if __name__ == "__main__":
    unittest.main()
