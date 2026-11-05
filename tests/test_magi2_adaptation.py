import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/base_eval"))
try:
    import magi2_quant
except ModuleNotFoundError:
    magi2_quant = None
from model.magi2_calibration import ActivationRecorder, VideoCalibration, histogram_quantile, merge_calibration, signed_histogram
from model.magi2_sparse_runtime import Magi2SparseController, install_sparse, prepare_norms, routed_reference


class GroupedLinearBase(torch.nn.Module):
    def __init__(self, groups=3):
        super().__init__()
        self.num_experts, self.in_features, self.out_features = groups, 8, 4
        self.weight = torch.nn.Parameter(torch.randn(groups * 4, 8))
        self.bias = torch.nn.Parameter(torch.randn(groups * 4))


class CustomGroupedLinear(GroupedLinearBase):
    pass


class BaseLinear(GroupedLinearBase):
    pass


class NativeMoELinear(GroupedLinearBase):
    pass


class CoreMultiHeadMoE(torch.nn.Module):
    def __init__(self, width=8, hidden=12):
        super().__init__()
        self.W_gate = torch.nn.Parameter(torch.randn(6, width, hidden) * .1)
        self.W_up = torch.nn.Parameter(torch.randn(6, width, hidden) * .1)
        self.W_down = torch.nn.Parameter(torch.randn(6, hidden, width) * .1)
        self.gate = torch.nn.Parameter(torch.randn(2, width, 3))


class Dispatcher:
    group_size_cpu = [2, 0, 3]

    def dispatch(self, activation):
        return list(activation.split(self.group_size_cpu))

    def undispatch(self, *chunks):
        return torch.cat(chunks)


@unittest.skipIf(magi2_quant is None, "requires pinned base_eval checkout")
class MagiAdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_grouped_zero_matches_pinned_with_empty_modality_and_gather(self):
        for module_type in (GroupedLinearBase, CustomGroupedLinear, BaseLinear, NativeMoELinear):
            for groups in (1, 3):
                with self.subTest(module_type=module_type, groups=groups):
                    module = module_type(groups)
                    model = torch.nn.ModuleDict({"projection": module})
                    norms = prepare_norms(model)
                    magi2_quant.apply_magi2_quant(model, "bf16")
                    activation = torch.randn(5, 8)
                    if module_type == GroupedLinearBase:
                        kwargs = dict(m_splits=Dispatcher.group_size_cpu,
                                      gather_ids=torch.tensor([4, 3, 2, 1, 0]))
                    elif module_type == CustomGroupedLinear:
                        kwargs = dict(modality_dispatcher=Dispatcher(), gather_ids=torch.tensor([4, 3, 2, 1, 0]))
                    else:
                        kwargs = dict(modality_dispatcher=Dispatcher(), output_dtype=torch.float32)
                    expected = module(activation, **kwargs)
                    controller = install_sparse(model, Magi2SparseController(0, norms=norms))
                    torch.testing.assert_close(module(activation, **kwargs), expected, rtol=0, atol=0)
                    self.assertEqual(controller.stats["masked_elements"], 0)
                    self.assertEqual(controller.stats["total_elements"], 40)

    def test_grouped_real_sparsity_records_only_nonempty_inputs(self):
        module = CustomGroupedLinear()
        model = torch.nn.ModuleDict({"projection": module})
        norms = prepare_norms(model)
        magi2_quant.apply_magi2_quant(model, "bf16")
        controller = install_sparse(model, Magi2SparseController(.5, norms=norms))
        module(torch.ones(5, 8), Dispatcher())
        self.assertEqual(controller.stats["masked_elements"], 20)
        self.assertNotIn("projection.group1", controller.snapshot()["modules"])

    def test_original_norms_and_shared_routed_gate_up(self):
        module = CoreMultiHeadMoE()
        model = torch.nn.ModuleDict({"moe": module, "dense": torch.nn.Linear(8, 4)})
        norms = prepare_norms(model)
        expected = (module.W_gate.float().square().sum(-1) + module.W_up.float().square().sum(-1)).sqrt()
        torch.testing.assert_close(norms["moe.gate_up"], expected)
        module.W_gate.data.zero_()
        self.assertGreater(float(norms["moe.gate_up"].sum()), 0)

    def test_dense_mask_precedes_activation_quantization_and_excludes_adapter(self):
        model = torch.nn.ModuleDict({"projection": torch.nn.Linear(8, 4), "pre_adapter": torch.nn.Linear(8, 4)})
        norms = prepare_norms(model)
        magi2_quant.apply_magi2_quant(model, "bf16")
        observed = []
        model["projection"].register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0].clone()))
        controller = install_sparse(model, Magi2SparseController(.5, norms=norms))
        model["projection"](torch.ones(3, 8))
        self.assertEqual(int((observed[0] == 0).sum()), 12)
        self.assertNotIn("pre_adapter", controller.coverage)
        self.assertIn("pre_adapter", controller.excluded)
        with self.assertRaises(ValueError):
            install_sparse(model, controller)

    def test_routed_zero_native_and_reference_match(self):
        module = CoreMultiHeadMoE()
        model = torch.nn.ModuleDict({"moe": module, "dense": torch.nn.Linear(8, 4)})
        magi2_quant.apply_magi2_quant(model, "bf16")
        activation = torch.randn(5, 2, 8)
        indices = torch.tensor([[[0, 2]] * 5, [[2, 0]] * 5])
        probabilities = torch.rand(2, 5, 2)
        expected = module._flash_forward(activation, probabilities, indices)
        controller = Magi2SparseController(0)
        torch.testing.assert_close(routed_reference(controller, "moe", module, activation, probabilities, indices), expected, rtol=0, atol=0)
        install_sparse(model, controller)
        torch.testing.assert_close(module._flash_forward(activation, probabilities, indices), expected, rtol=0, atol=0)
        self.assertEqual(controller.zero_bypass["moe"], 1)

    def test_padding_is_not_counted_and_nan_is_reported(self):
        controller = Magi2SparseController(.5)
        activation = torch.ones(2, 3, 8)
        activation[0, 0, 0] = float("nan")
        valid = torch.tensor([[True, False, False], [True, True, False]])
        controller.mask(activation, "batch", norm=torch.ones(2, 8), valid_rows=valid)
        self.assertEqual(controller.stats["total_elements"], 24)
        self.assertEqual(controller.stats["masked_elements"], 12)
        self.assertEqual(controller.stats["nonfinite"], 1)

    def test_signed_histogram_and_teal_rule_not_abs_quantile(self):
        centers, counts = signed_histogram(torch.linspace(-3, 5, 2000))
        threshold = histogram_quantile(centers, counts, .75)
        self.assertAlmostEqual(threshold, 3., places=1)
        artifact = dict(schema="magi-teal-signed-v1", complete=True, histograms={
            "preview/projection/bucket0": dict(centers=centers, counts=counts)})
        controller = Magi2SparseController(.5, method="teal", calibration=VideoCalibration(artifact))
        masked = controller.mask(torch.tensor([[-4., -2., 0., 2., 4.]]), "projection")
        torch.testing.assert_close(masked, torch.tensor([[-4., 0., 0., 0., 4.]]))
        with self.assertRaisesRegex(ValueError, "Uncalibrated"):
            controller.mask(torch.ones(1, 5), "unknown")

    def test_calibration_bounded_rng_and_rank_completion(self):
        recorder = ActivationRecorder(per_key=32, per_event=16)
        state = torch.random.get_rng_state()
        for stage in ("preview", "refiner"):
            for _ in range(6):
                recorder.observe(stage, "proj", 0, torch.arange(100.).reshape(10, 10))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(all(sample.numel() == 32 for sample in recorder.samples.values()))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            shard = dict(**recorder.state(), identity={"revision": "test"}, complete=True)
            torch.save(shard, folder / "calibration-rank0.pt")
            torch.save(dict(shard, complete=False), folder / "calibration-rank1.pt")
            with self.assertRaisesRegex(ValueError, "did not finish"):
                merge_calibration(folder, folder / "merged.pt")
            torch.save(shard, folder / "calibration-rank1.pt")
            merge_calibration(folder, folder / "merged.pt")
            VideoCalibration.load(folder / "merged.pt", expected={"revision": "test"})
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                VideoCalibration.load(folder / "merged.pt", expected={"revision": "other"})

    def test_calibration_samples_noncontiguous_without_changing_values(self):
        activation = torch.arange(120.).reshape(10, 12)[:, ::2]
        self.assertFalse(activation.is_contiguous())
        strided, contiguous = ActivationRecorder(), ActivationRecorder()
        strided.observe("preview", "proj", 0, activation)
        contiguous.observe("preview", "proj", 0, activation.contiguous())
        torch.testing.assert_close(strided.samples["preview/proj/bucket0"],
                                   contiguous.samples["preview/proj/bucket0"], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a leased CUDA GPU")
    def test_cuda_batched_preserves_reference_routing_and_padding(self):
        module = CoreMultiHeadMoE(width=128, hidden=128).to("cuda", dtype=torch.bfloat16)
        model = torch.nn.ModuleDict({"moe": module, "dense": torch.nn.Linear(128, 128).to("cuda", dtype=torch.bfloat16)})
        norms = prepare_norms(model, device="cuda")
        magi2_quant.apply_magi2_quant(model, "hifp4", device="cuda")
        activation = torch.randn(11, 2, 128, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([[[0, 2]] * 11, [[2, 0]] * 11], device="cuda")
        probabilities = torch.rand(2, 11, 2, device="cuda")
        from model.magi2_sparse_moe import routed_batched
        for sparsity in (0., .5, .6, .7, .8, .9):
            controller = Magi2SparseController(sparsity, norms=norms)
            reference = routed_reference(controller, "moe", module, activation, probabilities, indices)
            batched_controller = Magi2SparseController(sparsity, norms=norms)
            actual = routed_batched(batched_controller, "moe", module, activation, probabilities, indices)
            torch.testing.assert_close(actual, reference, rtol=.025, atol=.02)
            self.assertEqual(controller.stats, batched_controller.stats)
            if sparsity == 0:
                torch.testing.assert_close(actual, module._flash_forward(activation, probabilities, indices), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a leased CUDA GPU")
    def test_cuda_hif4_grouped_zero_equivalence(self):
        for module_type in (GroupedLinearBase, CustomGroupedLinear, BaseLinear, NativeMoELinear):
            module = module_type()
            module.in_features = module.out_features = 128
            module.weight = torch.nn.Parameter(torch.randn(384, 128, device="cuda", dtype=torch.bfloat16))
            module.bias = torch.nn.Parameter(torch.randn(384, device="cuda", dtype=torch.bfloat16))
            model = torch.nn.ModuleDict({"projection": module})
            norms = prepare_norms(model, device="cuda")
            magi2_quant.apply_magi2_quant(model, "hifp4", device="cuda")
            kwargs = dict(m_splits=Dispatcher.group_size_cpu) if module_type == GroupedLinearBase else dict(modality_dispatcher=Dispatcher())
            activation = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
            expected = module(activation, **kwargs)
            install_sparse(model, Magi2SparseController(0., norms=norms))
            torch.testing.assert_close(module(activation, **kwargs), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
