import unittest
from collections import Counter

import torch

from model.video_sparse_runtime import make_wina_quantized_linear


class _BaseLinear(torch.nn.Module):
    def __init__(self, linear, quant_type, act_quant):
        super().__init__()
        self.weight = torch.nn.Parameter(linear.weight.detach().clone())
        self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class VideoSparseRuntimeTest(unittest.TestCase):
    def test_wina_masks_expected_input_columns(self):
        source = torch.nn.Linear(8, 4, bias=False)
        stats = Counter()
        wrapper, _ = make_wina_quantized_linear(_BaseLinear, 0.5, stats)
        layer = wrapper(source, "hifp4", False).eval()
        output = layer(torch.ones(2, 8))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(stats["masked_elements"], 8)
        self.assertEqual(stats["total_elements"], 16)

    def test_invalid_sparsity_rejected(self):
        with self.assertRaises(ValueError):
            make_wina_quantized_linear(_BaseLinear, 1.0)


if __name__ == "__main__":
    unittest.main()
