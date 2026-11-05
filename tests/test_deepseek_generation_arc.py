import ast
from collections import Counter
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest

import torch

from model.deepseek_generation_arc import DeepSeekGenerationArcMixin, install_arc_grouped_attention
from model.deepseek_generation_runtime import DeepSeekGenerationTealMixin, DeepSeekGenerationWinaMixin
from model.quantize import quantize_hif4_tensor, fake_reorder_quantize_x, fake_reorder_quantize_w
from benchmarks.run_deepseek_sparse_math import patched_prepare
from benchmarks.queue_sparse_baselines import arc_health


class Base:
    quant_type = 'hifp4'
    act_quant = True
    cache_limit = 0

    def dequantize_weight(self, weight):
        return weight

    def _fake_quant_2d(self, x, output_dtype=None):
        return quantize_hif4_tensor(x.contiguous()).to(output_dtype or x.dtype)

    def stats(self):
        return {}


class ArcTests(unittest.TestCase):
    def runtime(self, method, selected=64):
        mixin = DeepSeekGenerationTealMixin if method == 'teal' else DeepSeekGenerationWinaMixin
        r = type('Runtime', (DeepSeekGenerationArcMixin, mixin, Base), {})()
        r.arc, r.collect, r.method = True, False, 'none'
        r.sparsity, r.sparse_sparsity = .5, .5
        r.arc_method = method
        r.calls, r.device_specs, r.arc_cache = Counter(), {}, {}
        r.arc_cache_bytes, r.arc_residual_calls = 0, 0
        r.sparse_counts, r.sparse_eligible = Counter(), Counter()
        r.sparse_norms, r.sparse_thresholds = {}, {}
        r.sparse_row_mask = None
        r.sparse_row_chunk = 2
        r.calibration = {}
        r.sparse_calibration = r.calibration
        r.order = torch.arange(127, -1, -1)
        r.selected = selected
        return r

    def spec(self, r, key):
        r.calibration[key] = dict(order=r.order, select=r.selected,
                                  centers=torch.tensor([-1., 0., 1.]), counts=torch.ones(3))
        r.sparse_thresholds[key] = .7

    def test_sparse_arc_matches_repository_quantizers_in_original_mask_order(self):
        torch.manual_seed(64)
        x, w = torch.randn(2, 5, 128), torch.randn(16, 128)
        for method in ('teal', 'wina'):
            for selected in (0, 64, 128):
                with self.subTest(method=method, selected=selected):
                    r = self.runtime(method, selected)
                    key = 'layers.0.attn.wq_a.weight'
                    self.spec(r, key)
                    masked = r._sparse_mask(x, w, key)
                    qx = fake_reorder_quantize_x(masked.reshape(-1, 128), r.order, selected, dtype='HiF4')[0]
                    qw = fake_reorder_quantize_w(w, r.order, selected, dtype='HiF4')[0]
                    expected = torch.nn.functional.linear(qx, qw).view(2, 5, 16)
                    actual = r._project(x, w, key)
                    torch.testing.assert_close(actual, expected)
                    protected = 3 if method == 'teal' else 1
                    torch.testing.assert_close(masked[:, :protected], x[:, :protected])

    def test_grouped_attention_uses_each_calibration_and_masks_once(self):
        r = self.runtime('teal')
        key = 'layers.0.attn.wo_a.weight'
        module = torch.nn.Linear(128, 16, bias=False)
        r.names = {module.weight.data_ptr(): key}
        x = torch.randn(2, 1, 2, 128)
        for group in range(2):
            self.spec(r, key + f'#group{group}')
        r.calibration[key + '#group1']['select'] = 0
        actual = r.grouped_wo_a(x, module)
        expected = []
        for group, w in enumerate(module.weight.view(2, 8, 128)):
            selected = (64, 0)[group]
            masked = x[:, :, group] * (x[:, :, group].abs() > .7)
            qx = fake_reorder_quantize_x(masked.reshape(-1,128), r.order, selected, dtype='HiF4')[0]
            qw = fake_reorder_quantize_w(w, r.order, selected, dtype='HiF4')[0]
            expected.append(torch.nn.functional.linear(qx, qw).view(2, 1, 8))
            self.assertEqual(r.sparse_eligible[key + f'#group{group}'], 256)
        torch.testing.assert_close(actual, torch.stack(expected, 2))

    def test_zero_sparsity_still_uses_arc_compensation(self):
        r = self.runtime('wina')
        r.sparse_sparsity = 0.
        self.spec(r, 'key')
        x, w = torch.randn(2, 128), torch.randn(16, 128)
        expected = torch.nn.functional.linear(
            fake_reorder_quantize_x(x, r.order, 64, dtype='HiF4')[0],
            fake_reorder_quantize_w(w, r.order, 64, dtype='HiF4')[0])
        torch.testing.assert_close(r._project(x, w, 'key'), expected)
        self.assertFalse(r.sparse_counts)

    def test_patch_keeps_ragged_fallback_closure(self):
        original = lambda self, x, start_pos: x + 7
        ds = types.SimpleNamespace(_lcb_positions=None)
        runtime = types.SimpleNamespace(arc_attention_paths=Counter(), grouped_wo_a=lambda x, module: x + 13)

        def forward(self, x, start_pos):
            if ds._lcb_positions is None:
                return original(self, x, start_pos)
            output = runtime.quantize_activation(x)
            weights = runtime.get_weight(self.wo_a.weight)
            output = torch.einsum('bsgd,grd->bsgr', output, weights)
            return output

        ds.Attention = type('Attention', (), {'forward': forward})
        install_arc_grouped_attention(ds, runtime, 'ragged')
        a = ds.Attention()
        a.wo_a = object()
        self.assertEqual(a.forward(1, 0), 8)
        ds._lcb_positions = [2, 5]
        self.assertEqual(a.forward(1, 0), 14)
        self.assertEqual(runtime.arc_attention_paths['ragged'], 1)

    def test_native_protocol_identity_is_inserted_before_comparison(self):
        def prepare(a):
            protocol = {'quantization': {}}
            out = Path(a.out_dir)
            if a.previous is not None and protocol != a.previous:
                raise ValueError('different protocol')
            return protocol

        metadata = {'arc': True, 'sparsity': .5}
        a = types.SimpleNamespace(out_dir='unused', previous=None)
        patched = patched_prepare(prepare, metadata)
        a.previous = patched(a)
        self.assertEqual(patched(a)['arcquant'], metadata)
        with self.assertRaisesRegex(ValueError, 'different protocol'):
            patched_prepare(prepare, {'arc': False, 'sparsity': .5})(a)

    def test_arc_label_without_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            m = dict(arc=True, method='wina', sparsity=.5, runtime_sha256='test', calibration={'world_size': 2})
            (p / 'arcquant_sparse_runtime.json').write_text(json.dumps(m))
            (p / 'protocol.json').write_text(json.dumps({'arcquant':m}))
            self.assertFalse(arc_health(p)[0])
            stats = dict(arc=True, run_succeeded=True, runtime_sha256='test', sparse_method='wina',
                         sparse_sparsity=.5, arc_finite_logits_checks=2, arc_residual_calls=3,
                         arc_executed_families=['attention', 'grouped_wo_a', 'routed_expert', 'shared_expert'],
                         arc_attention_paths={'standard':2, 'ragged':1}, sparse_masked_elements={'key':128})
            for rank in range(2):
                (p / f'arcquant_runtime_rank{rank}.json').write_text(json.dumps(stats))
            self.assertTrue(arc_health(p, require_ragged=True)[0])
            stats['arc_attention_paths']['ragged'] = 0
            (p / 'arcquant_runtime_rank1.json').write_text(json.dumps(stats))
            self.assertFalse(arc_health(p, require_ragged=True)[0])


if __name__ == '__main__':
    unittest.main()
