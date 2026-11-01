import unittest

import torch

from model.magi2_sparse_runtime import Magi2WinaController


class Magi2SparseRuntimeTest(unittest.TestCase):
    def test_expert_orientation_uses_input_axis(self):
        controller = Magi2WinaController(0.5)
        # MAGI-2 expert storage is [input, output].
        x = torch.ones(2, 4)
        weight = torch.ones(4, 3)
        masked = controller.mask_expert(x, weight)
        self.assertEqual(tuple(masked.shape), (2, 4))
        self.assertEqual(controller.stats["masked_elements"], 4)

    def test_invalid_sparsity_rejected(self):
        with self.assertRaises(ValueError):
            Magi2WinaController(1.0)


if __name__ == "__main__":
    unittest.main()
