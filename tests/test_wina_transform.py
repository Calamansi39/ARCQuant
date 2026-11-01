"""Numerical contract checks for the optional native WINA transform."""

import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})

if torch is not None:
    from model.wina_transform import apply_wina_transform


class _Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(width) + 0.5)

    def forward(self, x):
        return x * self.weight


class _Layer(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.input_layernorm = _Norm(width)
        self.post_attention_layernorm = _Norm(width)
        self.self_attn = SimpleNamespace(
            q_proj=nn.Linear(width, width, bias=False),
            k_proj=nn.Linear(width, width, bias=False),
            v_proj=nn.Linear(width, width, bias=False),
            o_proj=nn.Linear(width, width, bias=False),
        )
        self.mlp = SimpleNamespace(
            gate_proj=nn.Linear(width, intermediate, bias=False),
            up_proj=nn.Linear(width, intermediate, bias=False),
            down_proj=nn.Linear(intermediate, width, bias=False),
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WinaTransformTests(unittest.TestCase):
    def test_transform_attaches_square_residual_rotations(self):
        # Register the SimpleNamespace children explicitly so parameters are
        # visible to the transform's dtype/device checks.
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(32, 8)
                self.model.norm = _Norm(8)
                self.model.layers = nn.ModuleList()
                for _ in range(2):
                    layer = nn.Module()
                    layer.input_layernorm = _Norm(8)
                    layer.post_attention_layernorm = _Norm(8)
                    layer.self_attn = nn.Module()
                    layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
                    layer.self_attn.k_proj = nn.Linear(8, 8, bias=False)
                    layer.self_attn.v_proj = nn.Linear(8, 8, bias=False)
                    layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
                    layer.mlp = nn.Module()
                    layer.mlp.gate_proj = nn.Linear(8, 16, bias=False)
                    layer.mlp.up_proj = nn.Linear(8, 16, bias=False)
                    layer.mlp.down_proj = nn.Linear(16, 8, bias=False)
                    self.model.layers.append(layer)
                self.lm_head = nn.Linear(8, 32, bias=False)

        model = Model()
        manifest = apply_wina_transform(model, device="cpu")
        self.assertEqual(manifest["layers"], 2)
        self.assertTrue(model._arcquant_wina_transformed)
        for layer in model.model.layers:
            self.assertEqual(tuple(layer.attn_shortcut_Q.shape), (8, 8))
            self.assertEqual(tuple(layer.mlp_shortcut_Q.shape), (8, 8))
            self.assertTrue(torch.allclose(
                layer.attn_shortcut_Q.T @ layer.attn_shortcut_Q,
                torch.eye(8), atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
