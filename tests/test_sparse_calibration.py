"""Small model checks for the offline artifacts and Llama sparsity wiring."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import transformers
except ImportError:
    torch = None
    transformers = None

if torch is not None and transformers is not None and transformers.__version__ == "4.46.3":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from transformers import LlamaConfig, LlamaForCausalLM
    from calibrate_sparse import (build_layer_masks, dense_layer_pass, forward_layer, main,
                                  greedy_allocate_layer, reference_histogram,
                                  save_layer_histograms)
    from model_utils import reorder_model_llama
    from sparse_utils import PROJECTIONS, build_llama_teal_sparsifiers


@unittest.skipUnless(torch is not None and transformers is not None and
                     transformers.__version__ == "4.46.3", "requires the pinned Llama runtime")
class SparseCalibrationTests(unittest.TestCase):
    def setUp(self):
        config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64)
        config._attn_implementation = "sdpa"
        self.model = LlamaForCausalLM(config).eval()
        self.tokens = torch.randint(0, 128, (1, 8))

    def test_layer_calibration_matches_huggingface_forward(self):
        with torch.inference_mode():
            embedded = self.model.model.embed_tokens(self.tokens)
            outputs, captured = dense_layer_pass(self.model.model.layers[0], [embedded],
                                                  torch.device("cpu"), collect_histograms=True)
            final = self.model.model(self.tokens).last_hidden_state
            self.assertTrue(torch.allclose(self.model.model.norm(outputs[0]), final,
                                           atol=1e-5, rtol=1e-5))
            self.assertEqual({part: set(groups) for part, groups in captured.items()},
                             {"self_attn": {"h1", "h2"}, "mlp": {"h1", "h2"}})
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                save_layer_histograms(captured, root, 0)
                masks = build_llama_teal_sparsifiers(0, root / "histograms", 0.5)
                self.assertEqual(set(masks), set(PROJECTIONS))
                for part in captured.values():
                    for tensors in part.values():
                        _, counts = reference_histogram(tensors)
                        self.assertEqual(int(counts.sum().item()), tensors[0].numel())
                centers, counts = reference_histogram([torch.tensor([-2., -1., 0., 1., 2.])])
                self.assertLess(centers.min().item(), 0)
                self.assertEqual(int(counts.sum().item()), 5)

    def test_wina_greedy_reaches_weighted_budget(self):
        layer = self.model.model.layers[0]
        with torch.inference_mode():
            state = self.model.model.embed_tokens(self.tokens)
            targets, _ = dense_layer_pass(layer, [state], torch.device("cpu"))
            masks = build_layer_masks(layer, "wina", 0, Path("/tmp"))
            rates = greedy_allocate_layer(layer, [state], targets, masks,
                                          0.25, 0.1, 0.25, torch.device("cpu"))
        modules = {"q": layer.self_attn.q_proj, "k": layer.self_attn.k_proj,
                   "v": layer.self_attn.v_proj, "o": layer.self_attn.o_proj,
                   "gate": layer.mlp.gate_proj, "up": layer.mlp.up_proj,
                   "down": layer.mlp.down_proj}
        effective = sum(rates[name] * module.weight.numel() for name, module in modules.items())
        effective /= sum(module.weight.numel() for module in modules.values())
        self.assertAlmostEqual(effective, 0.25, places=5)

    def test_zero_wina_rate_preserves_quantized_llama_output(self):
        reorder = {}
        select = {}
        for part, names in (("self_attn", ("q", "k", "v", "o")),
                            ("mlp", ("gate", "up", "down"))):
            for name in names:
                key = f"layers.0.{part}.{name}_proj.input"
                reorder[key] = torch.arange(64 if name == "down" else 32, dtype=torch.int16)
                select[key] = 0
        kwargs = dict(device=torch.device("cpu"), kv_cache=False, reorder_index=reorder,
                      select_nums=select, quant_type="HiF4")
        with torch.inference_mode():
            dense = reorder_model_llama(copy.deepcopy(self.model), **kwargs)
            sparse = reorder_model_llama(copy.deepcopy(self.model), **kwargs,
                                         sparse_config={"method": "wina", "sparsity": 0.0,
                                                        "histogram_path": None,
                                                        "apply_prefill": True,
                                                        "layer_sparsities": {}})
            state = self.model.model.embed_tokens(self.tokens)
            dense_out = forward_layer(dense.model.layers[0], state, torch.device("cpu"))
            sparse_out = forward_layer(sparse.model.layers[0], state, torch.device("cpu"))
        self.assertTrue(torch.equal(dense_out, sparse_out))

    def test_calibration_command_writes_reusable_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_path = root / "tiny-llama"
            self.model.save_pretrained(model_path)
            with patch("calibrate_sparse.get_loaders",
                       return_value=([(self.tokens, None)], None, None)):
                code = main(["--model-path", str(model_path), "--method", "teal",
                             "--allocation", "greedy", "--output-dir", str(root / "out"),
                             "--device", "cpu", "--samples", "1", "--seqlen", "8",
                             "--sparsity", "0.25", "--step", "0.25"])
            self.assertEqual(code, 0)
            self.assertTrue((root / "out" / "histograms" / "layer-0" / "self_attn" / "histograms.pt").is_file())
            self.assertTrue((root / "out" / "strategy.json").is_file())


if __name__ == "__main__":
    unittest.main()
