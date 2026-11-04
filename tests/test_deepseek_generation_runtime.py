import unittest
from collections import Counter
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from model.deepseek_generation_runtime import DeepSeekGenerationWinaMixin, DeepSeekGenerationTealMixin


if torch is not None:
    class BaseRuntime:
        def quantize_activation(self, x):
            self.quant_inputs.append(x.clone())
            return x.round()

        def dequantize_weight(self, weight):
            self.weight_reads += 1
            return weight

        def get_weight(self, weight):
            return weight.round()

        def linear(self, x, weight, bias=None):
            return torch.nn.functional.linear(self.quantize_activation(x), self.get_weight(weight), bias)

        def stats(self):
            return {}


    class Attention(torch.nn.Module):
        def __init__(self, runtime):
            super().__init__()
            self.wo_a = torch.nn.Linear(4, 4, bias=False)
            self.wkv = torch.nn.Linear(4, 4, bias=False)
            self.n_local_groups = 2
            self.runtime = runtime

        def forward(self, x, explicit=False):
            if explicit:
                return self.runtime.grouped_wo_a(x, self.wo_a)
            x = self.runtime.quantize_activation(x)
            return torch.einsum('bsgd,grd->bsgr', x, self.runtime.get_weight(self.wo_a.weight).view(2, 2, 4))


    def make_runtime(teal=False):
        mixin = DeepSeekGenerationTealMixin if teal else DeepSeekGenerationWinaMixin

        class Runtime(mixin, BaseRuntime):
            pass

        r = Runtime()
        r.quant_inputs, r.weight_reads = [], 0
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in model.layers:
            layer.attn = Attention(r)
            with torch.no_grad():
                layer.attn.wo_a.weight.fill_(float('nan'))
        class MoE:
            def forward(self, x, ids):
                raise NotImplementedError
        ds = SimpleNamespace(Linear=torch.nn.Linear, MoE=MoE, world_size=1)
        kw = {'calibration': {'placeholder': {}}} if teal else {}
        r.configure_sparse(ds, model, .5, **kw)
        with torch.no_grad():
            for layer in model.layers:
                layer.attn.wo_a.weight.fill_(1.)
        return r, model, ds


@unittest.skipIf(torch is None, "PyTorch required")
class DeepSeekGenerationSparseRuntimeTests(unittest.TestCase):
    def test_configure_does_not_read_unloaded_checkpoint(self):
        r, _, _ = make_runtime()
        self.assertEqual(r.weight_reads, 0)
        self.assertEqual(r.sparse_norms, {})

    def test_grouped_uses_current_layer_and_masks_before_quantization(self):
        r, model, _ = make_runtime()
        x = torch.tensor([.4, 1.4, 2.4, 3.4]).repeat(2).reshape(1, 1, 2, 4)
        for i in (0, 1):
            actual = model.layers[i].attn(x)
            self.assertTrue(torch.equal(r.quant_inputs[-1], x * torch.tensor([0, 0, 1, 1])))
            self.assertTrue(torch.equal(actual, torch.full((1, 1, 2, 2), 5.)))
        self.assertEqual(len(r.sparse_counts), 4)
        self.assertEqual(set(r.sparse_counts.values()), {2})
        self.assertEqual(r.sparse_attention_stack, [])

    def test_grouped_explicit_keeps_w4a4_and_matches_hook_path(self):
        r, model, _ = make_runtime()
        x = torch.tensor([.4, 1.4, 2.4, 3.4]).repeat(2).reshape(1, 1, 2, 4)
        torch.testing.assert_close(model.layers[1].attn(x), model.layers[1].attn(x, explicit=True))

    def test_teal_grouped_histograms_are_layer_specific(self):
        r, model, _ = make_runtime(teal=True)
        for layer in range(2):
            for group in range(2):
                r.sparse_thresholds[f'layers.{layer}.attn.wo_a.weight#group{group}'] = (1.5, 3.5)[layer]
        x = torch.arange(1., 5.).repeat(2).reshape(1, 1, 2, 4)
        self.assertTrue(torch.equal(model.layers[0].attn(x), torch.full((1, 1, 2, 2), 9.)))
        self.assertTrue(torch.equal(model.layers[1].attn(x), torch.full((1, 1, 2, 2), 4.)))

    def test_teal_moe_prefill_preserves_first_half(self):
        r, _, ds = make_runtime(teal=True)
        key = 'layers.0.ffn.experts.0.w1.weight'
        r.sparse_thresholds[key] = 2.5
        moe = ds.MoE()
        moe.dim = 4
        moe.n_routed_experts = 1
        moe.experts_start_idx, moe.experts_end_idx = 0, 1
        moe.gate = lambda x, ids: (torch.ones(len(x), 1), torch.zeros(len(x), 1, dtype=torch.long))
        moe.experts = [lambda x, weights: r._sparse_mask(x, None, key) * weights]
        moe.shared_experts = torch.zeros_like
        x = torch.arange(1., 5.).repeat(4).reshape(1, 4, 4)
        out = moe.forward(x, torch.zeros(1, 4, dtype=torch.long))
        torch.testing.assert_close(out[:, :2], x[:, :2])
        torch.testing.assert_close(out[:, 2:], x[:, 2:] * torch.tensor([0, 0, 1, 1]))
        self.assertIsNone(r.sparse_row_mask)

    def test_informed_norms_are_read_on_first_forward_after_loading(self):
        r, model, _ = make_runtime()
        module = model.layers[0].attn.wkv
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[10., 1., 1., 1.]]).repeat(4, 1))
        r.linear(torch.arange(1., 5.).reshape(1, 1, 4), module.weight)
        torch.testing.assert_close(r.quant_inputs[-1], torch.tensor([[[1., 0., 0., 4.]]]))

    def test_wina_65_reports_target_and_observed_mask_ratio(self):
        class Base:
            def stats(self):
                return {}

        class Runtime(DeepSeekGenerationWinaMixin, Base):
            def __init__(self):
                self.sparse_sparsity = 0.65
                self.sparse_norms = {}
                self.sparse_counts = Counter()
                self.sparse_eligible = Counter()
                self.sparse_row_mask = None
                self.sparse_row_chunk = 4096

            def dequantize_weight(self, weight):
                return weight

        runtime = Runtime()
        x = torch.arange(1.0, 21.0).reshape(1, 1, 20)
        out = runtime._sparse_mask(x, torch.ones(4, 20), "layers.0.attn.wkv.weight")
        self.assertEqual(out.count_nonzero().item(), 7)
        stats = runtime.stats()
        self.assertEqual(stats["sparse_target_zero_fraction"], 0.65)
        self.assertAlmostEqual(
            stats["sparse_observed_zero_fraction"]["layers.0.attn.wkv.weight"], 0.65)


if __name__ == "__main__":
    unittest.main()
