"""One-GPU 2:4/HiF4 boundary probe using three real DeepSeek TP2 weights.

Inputs are seeded synthetic activations, NOT recorded model activations. This
checks numerical plumbing/structure, not model accuracy or native speedup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-eval-root", type=Path, required=True)
    p.add_argument("--source-model", type=Path, required=True)
    p.add_argument("--converted-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(a.base_eval_root.resolve()))
    import torch
    from torch import nn
    from safetensors import safe_open
    from deepseek_v4_quant import DeepSeekV4QuantRuntime
    from model.deepseek_structured_runtime import DeepSeekStructured24Mixin

    class Runtime(DeepSeekStructured24Mixin, DeepSeekV4QuantRuntime):
        pass

    class Projection(nn.Module):
        def __init__(self, weight, width):
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.in_features = width

    config = json.loads((a.source_model / "config.json").read_text())
    groups = int(config["o_groups"]) // 2
    width = int(config["hidden_size"])
    runtime = Runtime("hifp4", act_quant=True, cache_gib=.5)
    dense = DeepSeekV4QuantRuntime("hifp4", act_quant=True, cache_gib=.5)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_local_groups = groups

        def forward(self, x):
            values = runtime.quantize_activation(x)
            weight = runtime.get_weight(self.wo_a.weight).view(groups, -1, x.shape[-1])
            return torch.einsum("bsgd,grd->bsgr", values.to(weight.dtype), weight)

    model = nn.Module()
    layer = nn.Module()
    layer.attn = Attention()
    layer.ffn = nn.Module()
    layer.ffn.experts = nn.ModuleList([nn.Module()])
    model.layers = nn.ModuleList([layer])
    weights = {}
    shard = a.converted_model / "model0-mp2.safetensors"
    names = ("layers.0.attn.wq_a", "layers.0.attn.wo_a", "layers.0.ffn.experts.0.w1")
    with safe_open(str(shard), framework="pt", device="cuda:0") as handle:
        for name in names:
            weight = handle.get_tensor(name + ".weight")
            logical_width = (int(config["num_attention_heads"]) * int(config["head_dim"]) //
                             int(config["o_groups"])) if name.endswith("wo_a") else width
            module = Projection(weight, logical_width)
            if name + ".scale" in handle.keys():
                module.weight.scale = handle.get_tensor(name + ".scale")
            parent, key = name.rsplit(".", 1)
            model.get_submodule(parent).add_module(key, module)
            weights[name] = {"stored_shape": list(weight.shape), "stored_dtype": str(weight.dtype),
                             "logical_input_width": logical_width}
    runtime.configure_structured(SimpleNamespace(Linear=Projection), model, chunk_rows=3)
    torch.manual_seed(20261104)
    records = []
    with torch.inference_mode():
        for name in names:
            module = model.get_submodule(name)
            for length in (1, 8):
                shape = (1, length, groups, module.in_features) if name.endswith("wo_a") else (
                    (length, module.in_features) if ".experts." in name else (1, length, module.in_features))
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                if name.endswith("wo_a"):
                    output = layer.attn(x)
                    qw = dense.get_weight(module.weight).view(groups, -1, module.in_features)
                    reference = torch.einsum("bsgd,grd->bsgr", dense.quantize_activation(x).to(qw.dtype), qw)
                else:
                    output = runtime.linear(x, module.weight)
                    reference = dense.linear(x, module.weight)
                finite = bool(torch.isfinite(output).all())
                if not finite:
                    raise ValueError(f"Nonfinite output at {name}")
                rel = float((output.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
                records.append({"projection": name, "input_shape": list(shape), "output_shape": list(output.shape),
                                "finite": finite, "relative_l2_to_hif4_dense_synthetic_input": rel})
    torch.cuda.synchronize()
    report = {"kind": "real_weight_synthetic_input_projection_probe", "model": "deepseek-v4-flash",
              "seed": 20261104, "device": torch.cuda.get_device_name(0), "quantization": "HiF4 W4A4",
              "checkpoint": str(shard), "weights": weights, "records": records,
              "implementation_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__), ROOT / "model/structured_sparsity.py", ROOT / "model/deepseek_structured_runtime.py")},
              "runtime": runtime.stats(), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "benchmark_score": None, "acceptance_status": "not_evaluated"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "cases": len(records), "all_finite": True,
                      "mask_zero_fraction": report["runtime"]["structured_sparsity"]["mask_zero_fraction_on_this_rank"],
                      "peak_allocated_gib": report["peak_allocated_gib"]}, indent=2))
    runtime.remove_structured_hooks()


if __name__ == "__main__":
    main()
