import types
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from model.deepseek_experiment import (ExperimentRuntimeMixin, InputCalibration,
                                           prefill_rows, histogram_threshold)
    from model.quantize import quantize_hif4_tensor, fake_reorder_quantize_x, fake_reorder_quantize_w


@unittest.skipIf(torch is None, 'PyTorch required')
class WholeModelNumerics(unittest.TestCase):
    def test_arc_projection_matches_repository_reference(self):
        class Base:
            def dequantize_weight(self, w):
                return w
            def _fake_quant_2d(self, x, output_dtype=None):
                return quantize_hif4_tensor(x.contiguous()).to(output_dtype or x.dtype)
        class Runtime(ExperimentRuntimeMixin, Base):
            pass
        torch.manual_seed(20)
        x, w = torch.randn(2, 3, 128), torch.randn(16, 128)
        order = torch.randperm(128)
        for selected in (0, 64, 128):
            r = Runtime()
            r.arc, r.collect, r.method, r.sparsity = True, False, 'none', 0.
            r.calls, r.device_specs, r.arc_cache = {}, {}, {}
            from collections import Counter
            r.calls = Counter()
            r.cache_limit = 0
            r.arc_cache_bytes = 0
            r.calibration = {'projection': dict(order=order, select=selected,
                centers=torch.tensor([-1., 1.]), counts=torch.ones(2))}
            actual = r._project(x, w, 'projection')
            qx = fake_reorder_quantize_x(x.reshape(-1,128), order, selected, dtype='HiF4')[0]
            qw = fake_reorder_quantize_w(w, order, selected, dtype='HiF4')[0]
            expected = torch.nn.functional.linear(qx, qw).view(2,3,16)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_odd_prefill_native_tail_and_decode(self):
        self.assertEqual(prefill_rows(5, 'teal', 'cpu').tolist(), [False,False,False,True,True])
        self.assertEqual(prefill_rows(5, 'wina', 'cpu').tolist(), [False,True,True,True,True])
        self.assertEqual(prefill_rows(1, 'teal', 'cpu').tolist(), [True])

    def test_routed_rows_preserve_original_prefill_positions(self):
        class Runtime(ExperimentRuntimeMixin):
            pass
        r = Runtime()
        r.method, r.sparsity = 'wina', .5
        r.row_mask = torch.tensor([False, True])
        r.norms, r.modules, r.sparse_counts = {}, {}, {}
        r.ds = types.SimpleNamespace(RowParallelLinear=torch.nn.Linear, world_size=1)
        x = torch.tensor([[1.,2.,3.,4.], [1.,2.,3.,4.]])
        result = r._mask(x, torch.ones(4,4), 'unweighted')
        torch.testing.assert_close(result[0], x[0])
        self.assertEqual(result[1].tolist(), [0.,0.,3.,4.])

    def test_calibration_uses_all_events_for_arc_max(self):
        c = InputCalibration(128)
        c.update(torch.ones(2,128))
        last = torch.ones(1,128)
        last[0,3] = 200.
        c.update(last)
        value = c.finish()
        self.assertEqual(int(value['order'][-1]), 3)
        self.assertEqual(value['rows'], 3)
        self.assertEqual(value['select'] % 64, 0)
        self.assertTrue(torch.isfinite(torch.tensor(histogram_threshold(value['centers'],value['counts'],.5))))


if __name__ == '__main__':
    unittest.main()
