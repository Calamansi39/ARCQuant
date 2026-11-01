"""The optional tensor transform used by the native WINA implementation.

WINA's ``--transform`` path folds the RMSNorm scales into the following
linear layers and rotates every residual stream with an SVD-derived orthogonal
matrix.  The original project performs these operations in-place on CUDA and
stores two matrices on every decoder layer so the residual additions remain
in the rotated coordinate system.  This module keeps the same mathematics,
but makes the device explicit and leaves the model on its original device
after every operation, which is important for ARCQuant's layer-by-layer
wrapper.

The transform is deliberately opt-in.  ARCQuant reorder indices and ARC
residual-channel counts are coordinate dependent and must be generated after
this function has run.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def _linear_weight(module: nn.Module) -> torch.Tensor:
    if not hasattr(module, "weight") or module.weight.ndim != 2:
        raise TypeError(f"Expected a two-dimensional linear module, got {type(module)!r}")
    return module.weight


def _attention_inputs(layer: nn.Module) -> list[nn.Module]:
    return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]


def _mlp_inputs(layer: nn.Module) -> list[nn.Module]:
    return [layer.mlp.gate_proj, layer.mlp.up_proj]


@torch.no_grad()
def fuse_rmsnorm_linear(norm: nn.Module, linears: Iterable[nn.Module]) -> None:
    """Fold an RMSNorm scale into all immediately following projections."""
    if not hasattr(norm, "weight"):
        raise TypeError(f"The transform expects an RMSNorm-like module: {type(norm)!r}")
    scale = norm.weight.detach().double()
    if scale.ndim != 1:
        raise ValueError("RMSNorm weight must be one-dimensional")
    for linear in linears:
        weight = _linear_weight(linear)
        if weight.shape[1] != scale.numel():
            raise ValueError(f"Norm width {scale.numel()} does not match projection input {weight.shape}")
        linear.weight.data = (weight.detach().double() * scale[None, :]).to(weight.dtype)
        # Llama RMSNorm has no bias.  Supporting a bias here also makes the
        # helper safe for compatible RMSNorm implementations.
        if getattr(linear, "bias", None) is not None:
            linear.bias.data = linear.bias.data.detach().double().to(linear.bias.dtype)
    norm.weight.data = torch.ones_like(norm.weight.data)


@torch.no_grad()
def fuse_layernorms(model: nn.Module) -> None:
    """Apply the exact WINA preprocessing to a Llama-style model."""
    embeddings = model.model.embed_tokens
    # This mean subtraction is part of WINA's reference ``fuse_layernorm``
    # routine.  It is harmless for RMSNorm and is retained for provenance.
    ew = embeddings.weight.detach().double()
    embeddings.weight.data = (ew - ew.mean(dim=-1, keepdim=True)).to(embeddings.weight.dtype)
    for layer in model.model.layers:
        fuse_rmsnorm_linear(layer.input_layernorm, _attention_inputs(layer))
        fuse_rmsnorm_linear(layer.post_attention_layernorm, _mlp_inputs(layer))
    fuse_rmsnorm_linear(model.model.norm, [model.lm_head])


@torch.no_grad()
def _svd_rotation(linear: nn.Module, device: torch.device) -> torch.Tensor:
    """Return WINA's ``V`` matrix for a projection weight."""
    weight = _linear_weight(linear)
    # full_matrices=True is intentional: WINA uses a square hidden-width
    # rotation even when the projection has fewer output rows.
    _, _, vh = torch.linalg.svd(weight.detach().to(device=device, dtype=torch.float64),
                                full_matrices=True)
    return vh.transpose(0, 1).contiguous()


@torch.no_grad()
def _right_rotate(linear: nn.Module, rotation: torch.Tensor, device: torch.device) -> None:
    weight = _linear_weight(linear)
    if weight.shape[1] != rotation.shape[0]:
        raise ValueError(f"Rotation width {rotation.shape} does not match {weight.shape}")
    linear.weight.data = (weight.detach().to(device=device, dtype=torch.float64) @ rotation).to(weight.dtype).cpu()


@torch.no_grad()
def _left_rotate(linear: nn.Module, rotation: torch.Tensor, device: torch.device) -> None:
    weight = _linear_weight(linear)
    if weight.shape[0] != rotation.shape[0]:
        raise ValueError(f"Rotation width {rotation.shape} does not match {weight.shape}")
    linear.weight.data = (rotation.transpose(0, 1) @ weight.detach().to(device=device, dtype=torch.float64)).to(weight.dtype).cpu()
    if getattr(linear, "bias", None) is not None:
        bias = linear.bias.detach().to(device=device, dtype=torch.float64)
        linear.bias.data = (rotation.transpose(0, 1) @ bias).to(linear.bias.dtype).cpu()


@torch.no_grad()
def apply_wina_transform(model: nn.Module, device: str | torch.device = "cuda") -> dict:
    """Transform a Llama model in-place and attach rotated residual matrices.

    The returned manifest is JSON serialisable and should be saved alongside
    the transformed model's ARC calibration.  Calling this twice is rejected
    because the second rotation would describe a different model.
    """
    if getattr(model, "_arcquant_wina_transformed", False):
        raise ValueError("WINA tensor transform has already been applied")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("WINA tensor transform expects a Llama-style causal LM")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for transform device {target}")
    layers = list(model.model.layers)
    if not layers:
        raise ValueError("Model has no decoder layers")

    fuse_layernorms(model)
    # WINA chooses the first layer's K projection as the initial orientation.
    previous = _svd_rotation(layers[0].self_attn.k_proj, target)
    hidden = previous.shape[0]
    _right_rotate(model.model.embed_tokens, previous, target)
    for index, layer in enumerate(layers):
        if previous.shape != (hidden, hidden):
            raise ValueError("WINA rotations must be square hidden-width matrices")
        _right_rotate(layer.self_attn.q_proj, previous, target)
        _right_rotate(layer.self_attn.k_proj, previous, target)
        _right_rotate(layer.self_attn.v_proj, previous, target)

        current = _svd_rotation(layer.mlp.gate_proj, target)
        # Residual before attention is in the previous orientation; attention
        # output is in the current orientation.
        layer.attn_shortcut_Q = nn.Parameter((previous.transpose(0, 1) @ current).to(torch.float32), requires_grad=False)
        _left_rotate(layer.self_attn.o_proj, current, target)
        _right_rotate(layer.mlp.gate_proj, current, target)
        _right_rotate(layer.mlp.up_proj, current, target)

        next_rotation = (_svd_rotation(layers[index + 1].self_attn.k_proj, target)
                         if index + 1 < len(layers) else current)
        layer.mlp_shortcut_Q = nn.Parameter((current.transpose(0, 1) @ next_rotation).to(torch.float32), requires_grad=False)
        _left_rotate(layer.mlp.down_proj, next_rotation, target)
        previous = next_rotation

    _right_rotate(model.lm_head, previous, target)
    model._arcquant_wina_transformed = True
    return {
        "schema_version": 1,
        "method": "wina",
        "transform": "fuse_rmsnorm_then_sequential_svd_rotation",
        "device": str(target),
        "layers": len(layers),
        "hidden_size": hidden,
        "mean_subtracted_embeddings": True,
        "residual_coordinates": "attn_shortcut_Q and mlp_shortcut_Q",
    }
