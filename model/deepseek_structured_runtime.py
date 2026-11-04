"""2:4 boundary hooks for DeepSeek's normal and ragged HiF4 evaluators."""
from __future__ import annotations

import torch
from .structured_sparsity import Structured24Controller, merge_inventories


class DeepSeekStructured24Mixin:
    def configure_structured(self, ds, model, *, coverage=1., scope="all", selected_names=None,
                             chunk_rows=4096):
        if self.quant_type != "hifp4" or not self.act_quant:
            raise ValueError("structured24 currently requires HiF4 W4A4")
        self._structured_active_attention = []
        self._structured_linear_depth = 0
        self._structured_names = {}
        self._structured_groups = {}
        self._structured_hooks = []
        self._structured_excluded = []
        local = {}
        modules = dict(model.named_modules())
        for name, module in modules.items():
            if not isinstance(module, ds.Linear):
                continue
            area = ("attention" if name.startswith("layers.") and ".attn." in name else
                    "ffn" if name.startswith("layers.") and ".ffn." in name else None)
            if area is None:
                self._structured_excluded.append(name)
                continue
            width = int(module.in_features)  # Logical width, not packed FP4 storage width.
            local[name] = {"area": area, "width": width}
            self._structured_names[module.weight.data_ptr()] = name
            if name.endswith(".attn.wo_a"):
                attention = modules[name.rsplit(".", 1)[0]]
                groups = int(attention.n_local_groups)
                if groups < 1:
                    raise ValueError(f"Invalid grouped projection at {name}")
                self._structured_groups[name] = groups

                def enter(_module, _args, key=name):
                    self._structured_active_attention.append(key)

                def leave(_module, _args, _output, key=name):
                    if not self._structured_active_attention or self._structured_active_attention.pop() != key:
                        raise RuntimeError("Unbalanced structured attention context")

                self._structured_hooks += [attention.register_forward_pre_hook(enter),
                    attention.register_forward_hook(leave, always_call=True)]
        inventories = [local]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            inventories = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(inventories, local)
        self.structured = Structured24Controller(merge_inventories(inventories), coverage,
                                                 scope, selected_names, chunk_rows)

    def linear(self, x, weight, bias=None):
        name = getattr(self, "_structured_names", {}).get(weight.data_ptr())
        if name is None:
            return super().linear(x, weight, bias)
        x = self.structured.mask(x, name)
        # Keep the external weight quantizer/cache and matmul semantics. The
        # active name lets quantize_activation audit the exact GEMM operand.
        previous = getattr(self, "_structured_linear_name", None)
        self._structured_linear_name = name
        self._structured_linear_depth += 1
        try:
            return super().linear(x, weight, bias)
        finally:
            self._structured_linear_depth -= 1
            self._structured_linear_name = previous

    def quantize_activation(self, x):
        if not hasattr(self, "structured"):
            return super().quantize_activation(x)
        if self._structured_linear_depth:
            result = super().quantize_activation(x)
            self.structured.observe_quantized(result, self._structured_linear_name)
            return result
        # wo_a bypasses Linear.forward in BOTH normal and ragged attention.
        # Attribute it using the module currently executing, not tensor shape.
        if x.ndim == 4:
            if not self._structured_active_attention:
                raise RuntimeError("Grouped activation without an attention module context")
            name = self._structured_active_attention[-1]
            if x.shape[2] != self._structured_groups[name]:
                raise ValueError(f"Unexpected grouped activation layout at {name}")
            x = self.structured.mask(x, name)
            result = super().quantize_activation(x)
            self.structured.observe_quantized(result, name)
            return result
        return super().quantize_activation(x)

    def stats(self):
        base = super().stats()
        if hasattr(self, "structured"):
            base["structured_sparsity"] = self.structured.report()
            base["structured_sparsity"]["excluded_linear_names"] = self._structured_excluded
            base["structured_sparsity"]["excluded_scope"] = "router (not ds.Linear), embedding, head and HC plumbing"
        return base

    def remove_structured_hooks(self):
        for handle in getattr(self, "_structured_hooks", []):
            handle.remove()
        self._structured_hooks = []
