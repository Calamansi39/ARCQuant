import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from benchmarks.deepseek_activation_probe import ActivationSummary, _apply_mask, _build_masks


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DeepSeekActivationProbeTests(unittest.TestCase):
    def test_summary_reports_wina_and_teal_diagnostics(self):
        summary = ActivationSummary(sample_limit=32)
        summary.update(torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 0.25, -0.75, 1.5]]))
        report = summary.to_dict(torch.ones(4))
        self.assertEqual(report["elements"], 8)
        self.assertIn("teal", report["sparsity"]["0.5"])
        self.assertIn("wina", report["sparsity"]["0.5"])

    def test_teal_mask_preserves_prefill_prefix(self):
        summary = ActivationSummary(sample_limit=32)
        summary.update(torch.tensor([[0.1, 0.2], [3.0, 4.0], [0.3, 0.4], [5.0, 6.0]]))
        masks = _build_masks({"x": summary}, {}, "teal", 0.5)
        x = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
        masked = _apply_mask(x, masks["x"])
        torch.testing.assert_close(masked[:, :2], x[:, :2])
        self.assertLess(torch.count_nonzero(masked[:, 2:]), torch.count_nonzero(x[:, 2:]))


if __name__ == "__main__":
    unittest.main()
