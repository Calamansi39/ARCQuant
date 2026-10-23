import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from sparse_utils import WeightInformedTopKSparsifier
    from qLlamaLayer import quantize_sparse_branch_input


@unittest.skipIf(torch is None, "PyTorch is not installed in this Python environment")
class WinaSparsifierTests(unittest.TestCase):
    def test_weight_norm_changes_which_activation_is_kept(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        informed = WeightInformedTopKSparsifier(torch.tensor([10.0, 1.0, 1.0, 1.0]), 0.5)
        plain = WeightInformedTopKSparsifier(torch.ones(4), 0.5)
        self.assertEqual(informed(x).tolist(), [[[1.0, 0.0, 0.0, 4.0]]])
        self.assertEqual(plain(x).tolist(), [[[0.0, 0.0, 3.0, 4.0]]])
        self.assertEqual(informed.get_zero_ratio(), 0.5)

    def test_prefill_switch_keeps_context_but_masks_decode(self):
        sparse = WeightInformedTopKSparsifier(torch.ones(4), 0.5, apply_prefill=False)
        context = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]])
        self.assertTrue(torch.equal(sparse(context), context))
        self.assertEqual(sparse(context[:, :1]).count_nonzero().item(), 2)

    def test_sparse_input_reaches_hif4_quantization(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        sparse = WeightInformedTopKSparsifier(torch.tensor([10.0, 1.0, 1.0, 1.0]), 0.5)
        qx, _, _, bsz, q_len = quantize_sparse_branch_input(
            x, sparse, torch.arange(4), 0, "HiF4", True,
        )
        self.assertEqual((bsz, q_len), (1, 1))
        self.assertEqual(qx.shape, (1, 4))
        self.assertEqual(qx[0, 1:3].count_nonzero().item(), 0)
        self.assertNotEqual(qx[0, 0].item(), 0)


if __name__ == "__main__":
    unittest.main()
