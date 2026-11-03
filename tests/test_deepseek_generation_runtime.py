import unittest
from collections import Counter

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from model.deepseek_generation_runtime import DeepSeekGenerationWinaMixin


@unittest.skipIf(torch is None, "PyTorch required")
class DeepSeekGenerationSparseRuntimeTests(unittest.TestCase):
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
