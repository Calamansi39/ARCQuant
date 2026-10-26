import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from model.deepseek_adapter import (DeepSeekArcProjectionAdapter, dequantize_weight,
                                        projection_kind)
    from model.bridge import ArcQuantBridge


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DeepSeekAdapterTests(unittest.TestCase):
    def setUp(self):
        self.key = "layers.0.attn.wkv.weight"
        self.index = torch.tensor([2, 3, 0, 1])
        self.weight = torch.tensor([[1., -2., 3., 4.], [2., 0., -1., 1.]])
        self.x = torch.tensor([[[1., 2., 3., 4.]]])

    def test_official_projection_names_and_exclusions(self):
        self.assertEqual(projection_kind(self.key), "wkv")
        self.assertEqual(projection_kind("layers.10.attn.indexer.compressor.wgate.weight"),
                         "indexer.compressor.wgate")
        self.assertEqual(projection_kind("layers.42.ffn.experts.255.w3.weight"), "w3")
        self.assertEqual(projection_kind("layers.0.ffn.shared_experts.w2.weight"), "w2")
        with self.assertRaises(ValueError):
            projection_kind("layers.0.ffn.gate.weight")

    def test_packed_fp4_expansion_follows_low_high_nibble_and_scales(self):
        packed = torch.tensor([[0x21, -0x7F]], dtype=torch.int8)
        scale = torch.tensor([[2., 4.]])
        decoded = dequantize_weight(packed, scale)
        # 0x21 -> [0.5, 1.0]; 0x81 -> [0.5, -0.0]
        torch.testing.assert_close(decoded, torch.tensor([[1., 2., 2., -0.]]))
        with self.assertRaises(ValueError):
            dequantize_weight(packed, torch.ones(2, 1))

    def test_fp8_scale_blocks(self):
        weight = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=torch.float8_e4m3fn)
        decoded = dequantize_weight(weight, torch.tensor([[2., 4.]]))
        torch.testing.assert_close(decoded, torch.tensor([[2., 4., 12., 16.], [10., 12., 28., 32.]]))

    def test_wina_mask_precedes_arc_residual_path(self):
        adapter = DeepSeekArcProjectionAdapter({self.key: self.index}, {self.key: 2},
                                               method="wina", sparsity=0.5,
                                               informed_keys={self.key})
        observed = adapter.linear(self.x, self.weight, self.key)
        norms = self.weight.norm(dim=0)
        scores = self.x.abs() * norms
        indices = scores.topk(2, dim=-1).indices
        masked = self.x * torch.zeros_like(self.x).scatter_(-1, indices, 1)
        expected = ArcQuantBridge({self.key: self.index}, {self.key: 2}, "HiF4").linear(
            masked, self.weight, None, self.key)
        torch.testing.assert_close(observed, expected)
        self.assertEqual(adapter._sparsifiers[self.key].get_zero_ratio(), 0.5)
        self.assertEqual(adapter.bridge._chunked_quantize_activation(masked.reshape(1, 4), 2).shape[-1], 6)

    def test_teal_histogram_masks_before_arc_quantization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "histograms.pt"
            torch.save({"h1_centers": torch.tensor([0., 1., 2., 3.]),
                        "h1": torch.ones(4)}, path)
            adapter = DeepSeekArcProjectionAdapter({self.key: self.index}, {self.key: 1},
                                                   method="teal", sparsity=0.5,
                                                   histogram_files={self.key: (str(path), "h1")})
            observed = adapter.linear(self.x, self.weight, self.key)
            masked = torch.tensor([[[0., 0., 3., 4.]]])
            expected = ArcQuantBridge({self.key: self.index}, {self.key: 1}, "HiF4").linear(
                masked, self.weight, None, self.key)
            torch.testing.assert_close(observed, expected)

    def test_grouped_wo_a_matches_separate_group_projection(self):
        key = "layers.0.attn.wo_a.weight"
        calibration = {f"{key}#group{i}": torch.arange(4) for i in range(2)}
        adapter = DeepSeekArcProjectionAdapter(calibration, {name: 1 for name in calibration})
        x = torch.tensor([[[[1., 2., 3., 4.], [4., 3., 2., 1.]]]])
        weight = torch.arange(1., 17.).reshape(4, 4)
        output = adapter.grouped_wo_a(x, weight, key)
        self.assertEqual(tuple(output.shape), (1, 1, 2, 2))
        for i in range(2):
            expected = adapter.bridge.linear(x[:, :, i], weight[i*2:(i+1)*2], None,
                                             f"{key}#group{i}")
            torch.testing.assert_close(output[:, :, i], expected)

    def test_missing_target_calibration_fails_closed(self):
        adapter = DeepSeekArcProjectionAdapter({}, {})
        with self.assertRaisesRegex(KeyError, "Missing DeepSeek ARC calibration"):
            adapter.linear(self.x, self.weight, self.key)

    def test_invalid_calibration_is_rejected_before_quantization(self):
        with self.assertRaisesRegex(ValueError, "must match exactly"):
            DeepSeekArcProjectionAdapter({self.key: self.index}, {})
        adapter = DeepSeekArcProjectionAdapter({self.key: torch.tensor([0, 0, 2, 3])},
                                               {self.key: 2})
        with self.assertRaisesRegex(ValueError, "not a permutation"):
            adapter.linear(self.x, self.weight, self.key)
        adapter = DeepSeekArcProjectionAdapter({self.key: torch.arange(4)}, {self.key: 5})
        with self.assertRaisesRegex(ValueError, "residual channel count"):
            adapter.linear(self.x, self.weight, self.key)


if __name__ == "__main__":
    unittest.main()
