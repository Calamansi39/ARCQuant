import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from sparse_utils import build_llama_teal_sparsifiers


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TealSparsifierTests(unittest.TestCase):
    def test_histogram_threshold_masks_decode_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "layer-0"
            for part in ("self_attn", "mlp"):
                folder = root / part
                folder.mkdir(parents=True)
                histogram = {
                    "h1_centers": torch.tensor([0.0, 1.0, 2.0, 3.0]),
                    "h1": torch.ones(4),
                    "h2_centers": torch.tensor([0.0, 1.0, 2.0, 3.0]),
                    "h2": torch.ones(4),
                }
                torch.save(histogram, folder / "histograms.pt")
            sparsifiers = build_llama_teal_sparsifiers(0, tmp, 0.5)
            x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
            out = sparsifiers["q"](x)
            self.assertEqual(out.tolist(), [[[0.0, 0.0, 3.0, 4.0]]])
            self.assertEqual(sparsifiers["q"].get_zero_ratio(), 0.5)


if __name__ == "__main__":
    unittest.main()
