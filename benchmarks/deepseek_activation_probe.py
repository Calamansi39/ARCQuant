"""Collect real DeepSeek-V4 projection activations on a short prompt.

This is a diagnostic experiment, not an ARCQuant result.  It loads the pinned
official TP checkpoint, runs one real forward pass, and records bounded
activation summaries for the projections that are useful for a first TEAL/WINA
integration.  No activation tensor is written to disk.

The script intentionally imports the model/runtime from ``--base-eval-root``;
the large external evaluator is not copied into this repository.  Launch it
with torchrun and two GPUs for the released DeepSeek TP2 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


CORE_PROJECTION = re.compile(
    r"^layers\.\d+\.(?:attn\.(?:wq_a|wq_b|wkv|wo_b|compressor\.wkv|compressor\.wgate|"
    r"indexer\.wq_b|indexer\.weights_proj)|ffn\.shared_experts\.(?:w1|w2|w3))$"
)
_DEFAULT_TEXT = (
    "Explain why structural sparsity in a fine-grained W4A4 quantization path can "
    "preserve model quality. Give a short answer with one concrete example."
)


class ActivationSummary:
    """Streaming scalar statistics plus a small sample for quantile diagnostics."""

    def __init__(self, sample_limit: int = 32768):
        self.sample_limit = sample_limit
        self.count = 0
        self.zero_count = 0
        self.abs_sum = 0.0
        self.abs_sq_sum = 0.0
        self.abs_max = 0.0
        self._sample_rows = []
        self.width = None
        self.events = 0

    def update(self, tensor):
        import torch

        values = tensor.detach().float().reshape(-1, tensor.shape[-1])
        if values.numel() == 0:
            return
        absolute = values.abs()
        self.width = int(values.shape[-1])
        self.events += 1
        self.count += int(values.numel())
        self.zero_count += int(torch.count_nonzero(values == 0).item())
        self.abs_sum += float(absolute.sum().item())
        self.abs_sq_sum += float(absolute.square().sum().item())
        self.abs_max = max(self.abs_max, float(absolute.max().item()))
        sampled_rows = sum(row.shape[0] for row in self._sample_rows)
        if sampled_rows < self.sample_limit:
            take = min(self.sample_limit - sampled_rows, values.shape[0])
            # A deterministic stride avoids allocating a second large copy.
            stride = max(1, values.shape[0] // take)
            self._sample_rows.append(values[::stride][:take].cpu())

    def to_dict(self, weight_norm=None, sparsities=(0.25, 0.5, 0.75)):
        import torch

        sample = (torch.cat(self._sample_rows, dim=0)
                  if self._sample_rows else torch.empty((0, self.width or 0), dtype=torch.float32))
        flat = sample.reshape(-1)
        absolute = flat.abs()
        result = {
            "events": self.events,
            "elements": self.count,
            "zero_fraction": self.zero_count / self.count if self.count else 0.0,
            "abs_mean": self.abs_sum / self.count if self.count else 0.0,
            "abs_rms": (self.abs_sq_sum / self.count) ** 0.5 if self.count else 0.0,
            "abs_max": self.abs_max,
            "sample_elements": int(flat.numel()),
        }
        if sample.numel() == 0:
            result["sparsity"] = {}
            return result
        quantiles = (0.5, 0.9, 0.99)
        q = torch.quantile(absolute, torch.tensor(quantiles, dtype=torch.float32)).tolist()
        result["abs_quantiles"] = {f"p{int(x * 100)}": float(y) for x, y in zip(quantiles, q)}

        norm = None
        if weight_norm is not None:
            norm = weight_norm.detach().float().cpu().reshape(-1)
            if norm.numel() != sample.shape[-1]:
                norm = None
        masks = {}
        for sparsity in sparsities:
            target = float(sparsity)
            teal_threshold = float(torch.quantile(
                absolute, torch.tensor(target, dtype=torch.float32)).item())
            teal_keep = absolute > teal_threshold
            # Ties at zero are counted as removed, matching the signed-threshold
            # behavior used by the native TEAL sparsifier.
            teal_removed = flat.masked_fill(teal_keep, 0.0)
            total_energy = float(flat.square().sum().item())
            masks[f"{target:g}"] = {
                "teal": {
                    "threshold_abs": teal_threshold,
                    "keep_fraction": float(teal_keep.float().mean().item()),
                    "removed_l2_fraction": (float(teal_removed.square().sum().item()) / total_energy) ** 0.5
                    if total_energy else 0.0,
                }
            }
            if norm is not None:
                scores = sample.abs() * norm
                keep = max(1, min(scores.shape[-1], int(scores.shape[-1] * (1.0 - target))))
                indices = scores.topk(keep, dim=-1).indices
                wina_keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
                wina_removed = sample.masked_fill(wina_keep, 0.0)
                masks[f"{target:g}"]["wina"] = {
                    "keep_fraction": float(wina_keep.float().mean().item()),
                    "removed_l2_fraction": (float(wina_removed.square().sum().item()) / total_energy) ** 0.5
                    if total_energy else 0.0,
                }
        result["sparsity"] = masks
        return result

    def sample_tensor(self):
        import torch

        return (torch.cat(self._sample_rows, dim=0)
                if self._sample_rows else torch.empty((0, self.width or 0), dtype=torch.float32))



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval-root", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--converted-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default=_DEFAULT_TEXT)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=32768)
    parser.add_argument("--quant", choices=("bf16", "hifp4"), default="hifp4")
    parser.add_argument("--cache-gib", type=float, default=2.0)
    parser.add_argument("--method", choices=("none", "teal", "wina"), default="none")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _setup_distributed():
    import os
    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 2:
        raise RuntimeError("DeepSeek TP2 activation probe requires torchrun with two processes")
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return world, rank, local_rank


def _load(args, world, rank):
    import os
    import torch

    sys.path.insert(0, str(args.base_eval_root.expanduser().resolve()))
    from deepseek_v4_support import load_reference_model, make_model_args, reference_config
    from deepseek_v4_quant import DeepSeekV4QuantRuntime, install_reference_model_patches
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    source = args.source_model.expanduser().resolve()
    converted = args.converted_model.expanduser().resolve()
    ds = load_reference_model(source / "inference")
    runtime = DeepSeekV4QuantRuntime(args.quant, act_quant=args.quant != "bf16",
                                     cache_gib=args.cache_gib)
    install_reference_model_patches(ds, runtime)
    config = reference_config(source)
    model_args = make_model_args(ds, config, args.max_length + 16)
    with torch.device("cuda"):
        model = ds.Transformer(model_args)
    shard = converted / f"model{rank}-mp{world}.safetensors"
    missing, unexpected = load_model(model, str(shard), strict=False)
    if missing or any(not name.startswith("mtp.") for name in unexpected):
        raise RuntimeError(f"DeepSeek shard mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
    return ds, runtime, model, tokenizer


def _register_hooks(model, sample_limit, summaries=None, masks=None):
    collected = summaries if summaries is not None else {}
    handles = []
    for name, module in model.named_modules():
        if not CORE_PROJECTION.fullmatch(name):
            continue
        summary = collected.get(name) if summaries is not None else ActivationSummary(sample_limit)
        if summaries is None:
            collected[name] = summary

        def capture(_module, inputs, *, summary=summary, name=name):
            if inputs:
                if masks is None:
                    summary.update(inputs[0])
                if masks and name in masks:
                    return (_apply_mask(inputs[0], masks[name]), *inputs[1:])

        handles.append(module.register_forward_pre_hook(capture))
    if not collected:
        raise RuntimeError("No DeepSeek core projections matched; inspect official module names")
    return collected, handles


def _build_masks(summaries, norms, method, sparsity):
    import torch

    masks = {}
    for name, summary in summaries.items():
        sample = summary.sample_tensor()
        if sample.numel() == 0:
            continue
        if method == "teal":
            threshold = float(torch.quantile(
                sample.abs().reshape(-1), torch.tensor(sparsity, dtype=torch.float32)).item())
            masks[name] = {"method": method, "threshold": threshold}
        elif method == "wina" and name in norms and norms[name].numel() == sample.shape[-1]:
            masks[name] = {"method": method, "norm": norms[name].detach().float().cpu(),
                           "sparsity": sparsity}
    return masks


def _apply_mask(tensor, spec):
    import torch

    if tensor.ndim < 2:
        return tensor
    sequence_dim = tensor.ndim - 2
    if tensor.shape[sequence_dim] > 1:
        if spec["method"] == "teal":
            start = tensor.shape[sequence_dim] // 2
        else:
            start = tensor.shape[sequence_dim] - max(1, int(0.99 * tensor.shape[sequence_dim]))
        prefix = tensor.narrow(sequence_dim, 0, start)
        tail = tensor.narrow(sequence_dim, start, tensor.shape[sequence_dim] - start)
    else:
        prefix = None
        tail = tensor
    if spec["method"] == "teal":
        masked = tail * tail.abs().gt(spec["threshold"])
    else:
        norm = spec["norm"].to(device=tail.device, dtype=torch.float32)
        scores = tail.float().abs() * norm
        keep = max(1, min(scores.shape[-1], int(scores.shape[-1] * (1.0 - spec["sparsity"]))))
        indices = scores.topk(keep, dim=-1).indices
        keep_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
        masked = tail * keep_mask
    if prefix is None:
        return masked
    return torch.cat((prefix, masked), dim=sequence_dim)


def _weight_norms(summaries, model, runtime):
    import torch

    norms = {}
    modules = dict(model.named_modules())
    for name in summaries:
        module = modules[name]
        try:
            numeric = runtime.dequantize_weight(module.weight)
            norms[name] = numeric.float().norm(dim=0).cpu()
            del numeric
        except (RuntimeError, ValueError, AttributeError):
            # A malformed or unsupported auxiliary projection should not hide
            # its activation statistics; WINA is simply omitted for that key.
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return norms


def run(args):
    import datetime
    import torch
    import torch.distributed as dist

    if not 0 <= args.sparsity < 1:
        raise ValueError("--sparsity must be in [0, 1)")
    torch.manual_seed(args.seed)
    # The released reference constructor relies on the evaluator's BF16
    # default.  Without this, auxiliary compressed-attention tensors are made
    # in FP32 and the first indexer einsum fails before hooks can observe data.
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    world, rank, local_rank = _setup_distributed()
    ds, runtime, model, tokenizer = _load(args, world, rank)
    summaries, handles = _register_hooks(model, args.sample_limit)
    try:
        encoded = tokenizer(args.text, return_tensors="pt", truncation=True,
                            max_length=args.max_length).input_ids
        if encoded.shape[1] < 2:
            raise ValueError("Prompt must tokenize to at least two tokens")
        encoded = encoded.cuda(local_rank)
        # The external evaluator normally requests only the final logit vector.
        # Keep the full sequence here so the comparison below really covers the
        # complete last-token vocabulary distribution.
        ds._baseline_full_logits = True
        with torch.inference_mode():
            dense_logits = model(encoded, 0).detach().float()
        prediction = dense_logits[:, -1].argmax(dim=-1).cpu().tolist()
    finally:
        for handle in handles:
            handle.remove()

    norms = _weight_norms(summaries, model, runtime)
    sparse_prediction = None
    comparison = None
    if args.method != "none":
        masks = _build_masks(summaries, norms, args.method, args.sparsity)
        _, sparse_handles = _register_hooks(model, args.sample_limit, summaries=summaries, masks=masks)
        try:
            with torch.inference_mode():
                sparse_logits = model(encoded, 0).detach().float()
        finally:
            for handle in sparse_handles:
                handle.remove()
        ds._baseline_full_logits = False
        dense_last = dense_logits[:, -1]
        sparse_last = sparse_logits[:, -1]
        delta = sparse_last - dense_last
        sparse_prediction = sparse_last.argmax(dim=-1).cpu().tolist()
        dense_norm = float(torch.linalg.vector_norm(dense_last).item())
        comparison = {
            "masked_projection_count": len(masks),
            "dense_prediction_token": prediction,
            "sparse_prediction_token": sparse_prediction,
            "last_token_max_abs_logit_delta": float(delta.abs().max().item()),
            "last_token_mean_abs_logit_delta": float(delta.abs().mean().item()),
            "last_token_relative_l2_logit_delta": float(torch.linalg.vector_norm(delta).item()) / dense_norm
            if dense_norm else 0.0,
            "last_token_kl_dense_to_sparse": float(torch.nn.functional.kl_div(
                torch.log_softmax(sparse_last, dim=-1), torch.softmax(dense_last, dim=-1),
                reduction="batchmean").item()),
        }
    payload = {
        "schema_version": 1,
        "kind": "deepseek_activation_probe",
        "model": str(args.source_model.expanduser().resolve()),
        "converted_model": str(args.converted_model.expanduser().resolve()),
        "quant": args.quant,
        "method": args.method,
        "sparsity": args.sparsity,
        "world_size": world,
        "rank": rank,
        "prompt_tokens": int(encoded.shape[1]),
        "prediction_token": prediction,
        "sparse_prediction_token": sparse_prediction,
        "comparison": comparison,
        "runtime": runtime.stats(),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "projections": {
            name: summary.to_dict(norms.get(name)) for name, summary in sorted(summaries.items())
        },
    }
    rank_path = args.output.with_suffix(f".rank{rank}.json")
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dist.barrier()
    if rank == 0:
        rank_payloads = [json.loads(args.output.with_suffix(f".rank{i}.json").read_text(encoding="utf-8"))
                         for i in range(world)]
        combined = {
            "schema_version": 1,
            "kind": "deepseek_activation_probe",
            "model": payload["model"],
            "converted_model": payload["converted_model"],
            "quant": args.quant,
            "method": args.method,
            "sparsity": args.sparsity,
            "world_size": world,
            "prompt_tokens": payload["prompt_tokens"],
            "prediction_token": payload["prediction_token"],
            "sparse_prediction_token": payload["sparse_prediction_token"],
            "comparison": payload["comparison"],
            "created_at_utc": payload["created_at_utc"],
            "ranks": rank_payloads,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[probe] saved {args.output}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
