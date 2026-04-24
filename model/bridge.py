import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from quantize import fake_reorder_quantize_w, fake_reorder_quantize_x


def _normalize_model_name(model_name: str) -> str:
    if model_name.endswith("/"):
        model_name = model_name[:-1]
    return os.path.basename(model_name).lower()


def _to_int(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        raise ValueError("Expected scalar tensor for select_num.")
    return int(value)


@dataclass(frozen=True)
class _CacheKey:
    layer_key: str
    device: str
    dtype: str
    data_ptr: int


class ArcQuantBridge:
    def __init__(self, reorder_index: Dict[str, torch.Tensor], select_nums: Dict[str, int], quant_type: str = "NVFP4"):
        self.reorder_index = reorder_index
        self.select_nums = select_nums
        self.quant_type = quant_type
        self.weight_chunk_rows = 128
        self.act_chunk_rows = 64
        self.cache_weights = False
        self._weight_cache: Dict[_CacheKey, torch.Tensor] = {}
        self._index_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._arange_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    @classmethod
    def from_saved(
        cls,
        model_name: str,
        saved_dir: str,
        dataset: str = "wikitext2",
        metric: str = "max",
        quant_type: str = "NVFP4",
    ):
        model_key = _normalize_model_name(model_name)
        reorder_path = os.path.join(saved_dir, f"{model_key}_reorder_index_{dataset}_{metric}.pt")
        select_num_path = os.path.join(saved_dir, f"{model_key}_select_num_{dataset}_{metric}.pt")
        if not os.path.isfile(reorder_path):
            raise FileNotFoundError(f"ARC reorder index not found: {reorder_path}")
        if not os.path.isfile(select_num_path):
            raise FileNotFoundError(f"ARC select_num not found: {select_num_path}")
        reorder_index = torch.load(reorder_path, weights_only=False)
        select_nums = torch.load(select_num_path, weights_only=False)
        return cls(reorder_index=reorder_index, select_nums=select_nums, quant_type=quant_type)

    def has_layer(self, layer_key: str) -> bool:
        return layer_key in self.reorder_index and layer_key in self.select_nums

    def _device_index(self, layer_key: str, device: torch.device) -> torch.Tensor:
        cache_key = (layer_key, str(device))
        if cache_key not in self._index_cache:
            self._index_cache[cache_key] = self.reorder_index[layer_key].to(device=device, dtype=torch.int32)
        return self._index_cache[cache_key]

    def _device_arange(self, size: int, device: torch.device) -> torch.Tensor:
        cache_key = (size, str(device))
        if cache_key not in self._arange_cache:
            self._arange_cache[cache_key] = torch.arange(size, device=device)
        return self._arange_cache[cache_key]

    def _chunked_quantize_weight(self, reordered_weight: torch.Tensor, select_num: int) -> torch.Tensor:
        reorder_index = self._device_arange(reordered_weight.shape[1], reordered_weight.device)
        chunks = []
        for start in range(0, reordered_weight.shape[0], self.weight_chunk_rows):
            chunk = reordered_weight[start : start + self.weight_chunk_rows]
            quantized_chunk, _, _ = fake_reorder_quantize_w(
                chunk,
                reorder_index,
                select_num,
                dtype=self.quant_type,
            )
            chunks.append(quantized_chunk)
        return torch.cat(chunks, dim=0)

    def _linear_with_chunked_weights(
        self,
        quantized_x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        layer_key: str,
    ) -> torch.Tensor:
        index = self._device_index(layer_key, weight.device)
        select_num = _to_int(self.select_nums[layer_key])
        outputs = []
        for start in range(0, weight.shape[0], self.weight_chunk_rows):
            weight_chunk = weight[start : start + self.weight_chunk_rows]
            bias_chunk = None if bias is None else bias[start : start + self.weight_chunk_rows]
            reordered_weight_chunk = torch.index_select(weight_chunk, 1, index)
            quantized_weight_chunk = self._chunked_quantize_weight(reordered_weight_chunk, select_num)
            outputs.append(F.linear(quantized_x, quantized_weight_chunk, bias_chunk))
        return torch.cat(outputs, dim=-1)

    def _chunked_quantize_activation(self, reordered_x: torch.Tensor, select_num: int) -> torch.Tensor:
        reorder_index = self._device_arange(reordered_x.shape[1], reordered_x.device)
        chunks = []
        for start in range(0, reordered_x.shape[0], self.act_chunk_rows):
            chunk = reordered_x[start : start + self.act_chunk_rows]
            quantized_chunk, _, _ = fake_reorder_quantize_x(
                chunk,
                reorder_index,
                select_num,
                dtype=self.quant_type,
            )
            chunks.append(quantized_chunk)
        return torch.cat(chunks, dim=0)

    def _quantized_weight(self, weight: torch.Tensor, layer_key: str) -> torch.Tensor:
        cache_key = _CacheKey(
            layer_key=layer_key,
            device=str(weight.device),
            dtype=str(weight.dtype),
            data_ptr=weight.data_ptr(),
        )
        if self.cache_weights and cache_key in self._weight_cache:
            return self._weight_cache[cache_key]

        index = self._device_index(layer_key, weight.device)
        select_num = _to_int(self.select_nums[layer_key])
        reordered_weight = torch.index_select(weight, 1, index)
        quantized_weight = self._chunked_quantize_weight(reordered_weight, select_num)
        if self.cache_weights:
            self._weight_cache[cache_key] = quantized_weight
        return quantized_weight

    @torch.no_grad()
    def linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, layer_key: str) -> torch.Tensor:
        if not self.has_layer(layer_key):
            raise KeyError(f"Layer key not found in ARC assets: {layer_key}")

        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        index = self._device_index(layer_key, x.device)
        select_num = _to_int(self.select_nums[layer_key])
        reordered_x = torch.index_select(x_2d, 1, index)
        quantized_x = self._chunked_quantize_activation(reordered_x, select_num)
        if self.cache_weights:
            quantized_weight = self._quantized_weight(weight, layer_key)
            out = F.linear(quantized_x, quantized_weight, bias)
        else:
            out = self._linear_with_chunked_weights(quantized_x, weight, bias, layer_key)
        return out.reshape(*original_shape, weight.shape[0])
