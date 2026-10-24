"""Offline Llama calibration for native TEAL/WINA masks in ARCQuant.

TEAL histograms use the signed h1/h2 activations and the binning rule from
FasterDecoding/TEAL. Optional greedy allocation uses the layer-output error
criterion from the TEAL/WINA reference implementations, evaluated with the
same masks that ARCQuant will use online.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from datautils import get_loaders
from sparse_utils import (
    PROJECTIONS,
    build_llama_teal_sparsifiers,
    build_llama_wina_sparsifiers,
)


HISTOGRAM_HOOKS = {
    "self_attn": {"h1": "q_proj", "h2": "o_proj"},
    "mlp": {"h1": "gate_proj", "h2": "down_proj"},
}


def reference_histogram(tensors: list[torch.Tensor], bins: int = 10_000,
                        outlier_fraction: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """TEAL's signed histogram with 1% tails and outlier edge bins."""
    if not tensors:
        raise ValueError("No calibration activations were recorded")
    values = torch.cat([x.detach().reshape(-1).float().cpu() for x in tensors])
    values = torch.sort(values).values
    if values.numel() == 0:
        raise ValueError("Calibration activations are empty")
    if values[0] == values[-1]:
        return values[:1].clone(), torch.tensor([values.numel()], dtype=torch.float32)
    n = values.numel()
    tail = max(1, int(outlier_fraction * n))
    lower = values[min(n - 1, tail)]
    upper = values[max(0, n - tail)]
    if lower >= upper:
        lower, upper = values[0], values[-1]
    first = torch.nextafter(values[0], torch.tensor(float("-inf")))
    last = torch.nextafter(values[-1], torch.tensor(float("inf")))
    edges = torch.cat((first.view(1), torch.linspace(lower.item(), upper.item(), bins - 1),
                       last.view(1)))
    if not torch.all(edges[1:] > edges[:-1]):
        edges = torch.linspace(first.item(), last.item(), bins + 1)
    counts = torch.histogram(values, bins=edges).hist.float()
    centers = (edges[:-1] + edges[1:]) / 2
    return centers.float(), counts


def save_layer_histograms(captured: dict, destination: Path, layer_index: int) -> None:
    for part, groups in captured.items():
        output = {}
        for name, tensors in groups.items():
            centers, counts = reference_histogram(tensors)
            output[name] = counts
            output[f"{name}_centers"] = centers
        folder = destination / "histograms" / f"layer-{layer_index}" / part
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(output, folder / "histograms.pt")


def forward_layer(layer, state: torch.Tensor, device: torch.device) -> torch.Tensor:
    hidden = state.to(device)
    position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0).expand(hidden.shape[0], -1)
    position_embeddings = layer.self_attn.rotary_emb(hidden, position_ids)
    result = layer(hidden, attention_mask=None, position_ids=position_ids,
                   position_embeddings=position_embeddings, use_cache=False,
                   output_attentions=False)
    return result[0]


def dense_layer_pass(layer, states: list[torch.Tensor], device: torch.device,
                     *, collect_histograms: bool = False):
    captured = {part: {name: [] for name in groups} for part, groups in HISTOGRAM_HOOKS.items()}
    hooks = []
    if collect_histograms:
        for part, groups in HISTOGRAM_HOOKS.items():
            module = getattr(layer, part)
            for name, projection in groups.items():
                def save_input(_module, inputs, *, part=part, name=name):
                    captured[part][name].append(inputs[0].detach().to("cpu", dtype=torch.float32))
                hooks.append(getattr(module, projection).register_forward_pre_hook(save_input))
    try:
        outputs = [forward_layer(layer, state, device).detach().cpu() for state in states]
    finally:
        for hook in hooks:
            hook.remove()
    return outputs, captured


def build_layer_masks(layer, method: str, layer_index: int, histogram_root: Path):
    if method == "teal":
        all_masks = build_llama_teal_sparsifiers(layer_index, str(histogram_root), 0.0)
        masks = {name: all_masks[name] for name in PROJECTIONS}
    else:
        attn = build_llama_wina_sparsifiers(layer.self_attn, "self_attn", 0.0)
        mlp = build_llama_wina_sparsifiers(layer.mlp, "mlp", 0.0)
        masks = {name: attn[name] for name in ("q", "k", "v", "o")}
        masks.update({name: mlp[name] for name in ("gate", "up", "down")})
    for mask in masks.values():
        mask.track_stats = False
        mask.to(next(layer.parameters()).device)
    return masks


def greedy_allocate_layer(layer, states: list[torch.Tensor], targets: list[torch.Tensor],
                          masks: dict, target_sparsity: float, step: float,
                          last_fraction: float, device: torch.device) -> dict[str, float]:
    """Choose one projection at a time by reference-style block output error."""
    modules = {
        "q": layer.self_attn.q_proj, "k": layer.self_attn.k_proj,
        "v": layer.self_attn.v_proj, "o": layer.self_attn.o_proj,
        "gate": layer.mlp.gate_proj, "up": layer.mlp.up_proj,
        "down": layer.mlp.down_proj,
    }
    weights = {name: float(module.weight.numel()) for name, module in modules.items()}
    total_weight = sum(weights.values())
    unit_weight = weights["q"]
    rates = {name: 0.0 for name in PROJECTIONS}

    def apply_rates(current):
        for name, value in current.items():
            masks[name].set_sparsity(value)

    def error(current):
        apply_rates(current)
        total = 0.0
        for state, target in zip(states, targets):
            actual = forward_layer(layer, state, device).float()
            target = target.to(device, dtype=torch.float32)
            start = int(actual.shape[1] * (1 - last_fraction))
            total += torch.linalg.vector_norm(actual[:, start:] - target[:, start:], dim=1).mean().item()
        return total / len(states)

    hooks = []
    for name, module in modules.items():
        def sparsify(_module, inputs, *, name=name):
            return (masks[name](inputs[0]), *inputs[1:])
        hooks.append(module.register_forward_pre_hook(sparsify))
    try:
        # The reference greedy search increments a Q-sized matrix by `step`;
        # smaller/larger matrices use inverse-size rate steps so each choice
        # spends the same number of removable multiply weights.
        budget_step = step * unit_weight / total_weight
        max_steps = math.ceil(target_sparsity / budget_step) + len(PROJECTIONS) + 2
        for _ in range(max_steps):
            effective = sum(rates[name] * weights[name] for name in PROJECTIONS) / total_weight
            if effective >= target_sparsity - 1e-7:
                break
            candidates = []
            remaining = target_sparsity - effective
            for name in PROJECTIONS:
                delta = min(1.0 - 1e-6 - rates[name],
                            step * unit_weight / weights[name],
                            remaining * total_weight / weights[name])
                if delta <= 1e-8:
                    continue
                candidate = rates.copy()
                candidate[name] += delta
                candidates.append((error(candidate), name, candidate))
            if not candidates:
                raise RuntimeError("Cannot reach requested effective sparsity with these projections")
            _, _, rates = min(candidates, key=lambda row: (row[0], row[1]))
        else:
            raise RuntimeError("Greedy allocation exceeded its iteration limit")
    finally:
        for hook in hooks:
            hook.remove()
    return {name: round(value, 8) for name, value in rates.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline TEAL/WINA calibration for ARCQuant Llama")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--method", choices=["teal", "wina"], required=True)
    parser.add_argument("--allocation", choices=["uniform", "greedy"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["wikitext2", "c4", "arc_mix"], default="wikitext2")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--step", type=float, default=0.025,
                        help="Greedy rate increment for a Q-sized weight matrix (reference default: 0.025)")
    parser.add_argument("--last-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.seqlen < 2 or not 0 < args.sparsity < 1 or not 0 < args.step < 1:
        parser.error("samples >= 1, seqlen >= 2, sparsity and step in (0, 1) are required")
    if not 0 < args.last_fraction <= 1:
        parser.error("--last-fraction must be in (0, 1]")
    if args.method == "wina" and args.allocation != "greedy":
        parser.error("Uniform WINA needs no offline calibration")
    if args.method == "teal" and args.dataset != "wikitext2":
        print("TEAL histograms will use the selected calibration dataset; record it when comparing runs.", flush=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA is unavailable for --device {args.device}")
    torch.manual_seed(args.seed)
    model_path = args.model_path.expanduser().resolve()
    config_file = model_path / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") != "llama":
        parser.error("This native calibrator currently supports Hugging Face Llama only")
    trainloader, _, _ = get_loaders(args.dataset, nsamples=args.samples, seed=args.seed,
                                    seqlen=args.seqlen, model=str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", trust_remote_code=False
    )
    model.eval()
    model.config.use_cache = False
    embedding = model.model.embed_tokens.to(device)
    with torch.inference_mode():
        states = [embedding(input_ids.to(device)).detach().cpu() for input_ids, _ in trainloader]
    embedding.cpu()
    output_dir = args.output_dir.expanduser().resolve()
    prior = output_dir / "calibration.json"
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    if prior.exists():
        old = json.loads(prior.read_text(encoding="utf-8"))
        if any(old.get(key) != value for key, value in (
            ("model_config_sha256", config_hash), ("method", args.method),
            ("allocation", args.allocation), ("dataset", args.dataset),
            ("samples", args.samples), ("seqlen", args.seqlen), ("seed", args.seed),
            ("target_sparsity", args.sparsity), ("step", args.step),
            ("last_fraction", args.last_fraction),
        )):
            parser.error(f"{output_dir} contains artifacts from a different calibration; choose another --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_layers = {}
    with torch.inference_mode():
        for index, layer in enumerate(model.model.layers):
            layer.to(device)
            print(f"Calibrating layer {index + 1}/{len(model.model.layers)}", flush=True)
            outputs, captured = dense_layer_pass(layer, states, device,
                                                 collect_histograms=args.method == "teal")
            if args.method == "teal":
                save_layer_histograms(captured, output_dir, index)
            if args.allocation == "greedy":
                masks = build_layer_masks(layer, args.method, index, output_dir / "histograms")
                strategy_layers[str(index)] = greedy_allocate_layer(
                    layer, states, outputs, masks, args.sparsity, args.step, args.last_fraction, device
                )
                print(f"  allocation: {strategy_layers[str(index)]}", flush=True)
                del masks
            states = outputs
            layer.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    provenance = {"schema_version": 1, "method": args.method, "allocation": args.allocation,
                  "target_sparsity": args.sparsity, "model_name": model_path.name,
                  "model_config_sha256": config_hash,
                  "dataset": args.dataset, "samples": args.samples, "seqlen": args.seqlen,
                  "seed": args.seed, "step": args.step,
                  "step_unit": "q_projection_weight_count", "last_fraction": args.last_fraction}
    (output_dir / "calibration.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if strategy_layers:
        strategy = {**provenance, "layers": strategy_layers}
        (output_dir / "strategy.json").write_text(
            json.dumps(strategy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Calibration artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
