import json
import os

import numpy as np
import torch
import torch.nn as nn

from input_sparsity_log import record_input_sparsity
from optional_agemm import HAS_AGEMM, require_agemm
from qLinearLayer import QLinearLayer
from quantize import fake_reorder_quantize_x


RS_PARSE_MODULE_SPECS = [
    ("self_attn", "q_proj", "q", "q_svd_path", "q_threshold", "q_low_rank"),
    ("self_attn", "k_proj", "k", "k_svd_path", "k_threshold", "k_low_rank"),
    ("self_attn", "v_proj", "v", "v_svd_path", "v_threshold", "v_low_rank"),
    ("self_attn", "o_proj", "o", "o_svd_path", "o_threshold", "o_low_rank"),
    ("mlp", "gate_proj", "gate", "gate_svd_path", "gate_threshold", "gate_low_rank"),
    ("mlp", "up_proj", "up", "up_svd_path", "up_threshold", "up_low_rank"),
    ("mlp", "down_proj", "down", "down_svd_path", "down_threshold", "down_low_rank"),
]


def _resolve_svd_path(config_file, path, low_rank_root=None):
    if os.path.isabs(path):
        return path
    if low_rank_root is not None:
        tail = path.split("low_rank_models/", 1)[-1] if "low_rank_models/" in path else os.path.basename(path)
        return os.path.join(low_rank_root, tail)
    config_dir = os.path.dirname(os.path.abspath(config_file))
    resolved = os.path.abspath(os.path.join(config_dir, path))
    if os.path.exists(resolved):
        return resolved
    if "low_rank_models/" in path:
        tail = path.split("low_rank_models/", 1)[-1]
        fallback_root = os.path.abspath(os.path.join(config_dir, "..", "..", "low_rank_models"))
        fallback = os.path.join(fallback_root, tail)
        if os.path.exists(fallback):
            return fallback
    return resolved


def _projection_dims(model_config, component, projection):
    if component == "self_attn":
        return model_config.hidden_size, model_config.hidden_size
    if projection == "down_proj":
        return model_config.intermediate_size, model_config.hidden_size
    return model_config.hidden_size, model_config.intermediate_size


def _compute_budgeted_rank(in_features, out_features, target_sparsity, sparse_ratio):
    channels = max(int(in_features * (1 - target_sparsity) * sparse_ratio), 1)
    overall_budget = in_features * out_features * (1 - target_sparsity)
    sparse_budget = channels * out_features
    low_rank_budget = max(overall_budget - sparse_budget, 0.0)
    if low_rank_budget <= 0:
        return 0
    return max(int(low_rank_budget / (in_features + out_features)), 1)


def build_rsparse_module_configs(model_config, config_file, search_file, low_rank_root, target_sparsity, sparse_ratio, prefill_ratio):
    with open(config_file, "r") as f:
        config_data = json.load(f)

    search_values = None
    if search_file is not None:
        search_values = np.loadtxt(search_file)
        if search_values.ndim != 1:
            search_values = search_values.reshape(-1)

    settings = {}
    search_idx = 0

    for layer_idx in range(model_config.num_hidden_layers):
        for component, projection, short_name, path_key, threshold_key, rank_key in RS_PARSE_MODULE_SPECS:
            in_features, out_features = _projection_dims(model_config, component, projection)

            if search_values is not None:
                alpha = float(search_values[search_idx])
                search_target = float(search_values[search_idx + 1])
                search_idx += 2
                module_sparse_ratio = alpha
                module_target_sparsity = 1 - (1 - search_target) * alpha
                module_rank = _compute_budgeted_rank(
                    in_features,
                    out_features,
                    search_target,
                    alpha,
                )
            else:
                module_sparse_ratio = float(sparse_ratio)
                module_target_sparsity = 1 - (1 - target_sparsity) * module_sparse_ratio
                module_rank = _compute_budgeted_rank(
                    in_features,
                    out_features,
                    float(target_sparsity),
                    module_sparse_ratio,
                )

            module_name = f"layers.{layer_idx}.{component}.{projection}"
            settings[module_name] = {
                "threshold": float(config_data[threshold_key][layer_idx]),
                "rank": int(module_rank if search_values is not None else config_data[rank_key][layer_idx]),
                "sparse_ratio": module_sparse_ratio,
                "target_sparsity": float(module_target_sparsity),
                "prefill_ratio": float(prefill_ratio),
                "svd_path": _resolve_svd_path(config_file, config_data[path_key][layer_idx], low_rank_root),
            }

    return settings


def _iter_rsparse_modules(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return
    for layer in model.model.layers:
        for component in ("self_attn", "mlp"):
            if not hasattr(layer, component):
                continue
            module = getattr(layer, component)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                if hasattr(module, projection):
                    proj_module = getattr(module, projection)
                    if isinstance(proj_module, QRSparseLinear):
                        yield proj_module


@torch.no_grad()
def begin_rsparse_calibration(model):
    for module in _iter_rsparse_modules(model):
        module.begin_threshold_calibration()


@torch.no_grad()
def finish_rsparse_calibration(model):
    for module in _iter_rsparse_modules(model):
        module.finish_threshold_calibration()


def _reorder_quantize_x(x, reorder_index, select_num, quant_type="NVFP4"):
    if quant_type == "NVFP4" and HAS_AGEMM:
        scale = torch.max(x.abs()).float() / (448.0 * 6.0)
        qx, scale_x = require_agemm().reorder_quantize_x(x / scale, reorder_index, select_num)
        return qx, scale_x, scale
    return fake_reorder_quantize_x(x, reorder_index, select_num, dtype=quant_type)


def _quantize_branch_input(x, reorder_index, select_num, quant_type):
    bsz, q_len, _ = x.shape
    x = x.reshape(bsz * q_len, -1).contiguous().detach()
    qx, scale_x, scale = _reorder_quantize_x(x, reorder_index, select_num, quant_type)
    if qx.is_cuda:
        torch.cuda.synchronize()
    return qx, scale_x, scale, bsz, q_len


class QRSparseLinear(nn.Module):
    def __init__(
        self,
        original_layer,
        select_num,
        reorder_index,
        quant_type,
        rsparse_config,
        log_tag=None,
    ):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.quant_type = quant_type
        self.select_num = select_num
        self.prefill_ratio = float(rsparse_config["prefill_ratio"])
        self.sparse_ratio = float(rsparse_config["sparse_ratio"])
        self.target_sparsity = float(rsparse_config["target_sparsity"])
        self.rank = int(rsparse_config["rank"])
        self.log_tag = log_tag

        self.qlinear = QLinearLayer(
            original_layer,
            select_num=select_num,
            reorder_index=reorder_index,
            quant_type=quant_type,
        )
        self.register_buffer("reorder_index", reorder_index.to(torch.int16).clone())
        self.register_buffer("threshold", torch.tensor(float(rsparse_config["threshold"]), dtype=torch.float32))
        self.register_buffer("zero_count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("total_count", torch.tensor(0.0, dtype=torch.float64))

        self.calibrating_threshold = False
        self.calibration_finished = False

        self._load_low_rank_module(rsparse_config["svd_path"])
        self._set_mode()

    def _load_low_rank_module(self, path):
        if self.rank <= 0:
            self.low_rank_vs = None
            self.low_rank_u_t = None
            self.scale = None
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"R-Sparse low-rank factor not found: {path}")

        u, s, v, scale = torch.load(path, map_location="cpu")
        u = u[:, : self.rank].to(self.qlinear.W.dtype)
        s = s[: self.rank].to(self.qlinear.W.dtype)
        v = v[:, : self.rank].to(self.qlinear.W.dtype)
        vs = v * s.unsqueeze(0)
        self.register_buffer("low_rank_vs", vs.contiguous())
        self.register_buffer("low_rank_u_t", u.transpose(0, 1).contiguous())
        self.register_buffer("scale", scale.to(self.qlinear.W.dtype))

    def _set_mode(self):
        if self.rank > 0 and 0 < self.sparse_ratio < 1:
            self.mode = "r_sparse"
        elif self.sparse_ratio >= 1:
            self.mode = "sparse"
        elif self.rank > 0:
            self.mode = "low_rank"
        else:
            self.mode = "dense"

    @torch.no_grad()
    def begin_threshold_calibration(self):
        self.calibrating_threshold = True
        self.calibration_finished = False
        self.reset_stats()

    @torch.no_grad()
    def finish_threshold_calibration(self):
        self.calibrating_threshold = False
        self.calibration_finished = True
        self.reset_stats()

    @torch.no_grad()
    def reset_stats(self):
        self.zero_count.zero_()
        self.total_count.zero_()

    def get_zero_ratio(self):
        if self.total_count.item() == 0:
            return 0.0
        return (self.zero_count / self.total_count).item()

    @torch.no_grad()
    def _update_stats(self, x):
        self.zero_count += float(x.numel() - torch.count_nonzero(x))
        self.total_count += float(x.numel())

    def _dense_forward(self, x):
        if x.numel() == 0:
            return x.new_empty(x.shape[0], x.shape[1], self.out_features)
        if self.log_tag is not None:
            record_input_sparsity(self.log_tag, x)
        return self.qlinear(
            _quantize_branch_input(
                x,
                self.reorder_index,
                self.select_num,
                self.quant_type,
            )
        )

    def _low_rank_forward(self, x):
        if x.numel() == 0:
            return x.new_empty(x.shape[0], x.shape[1], self.out_features)
        if self.rank <= 0 or self.low_rank_vs is None or self.low_rank_u_t is None:
            return torch.zeros(
                x.shape[0],
                x.shape[1],
                self.out_features,
                dtype=x.dtype,
                device=x.device,
            )
        flat_x = x.reshape(-1, x.shape[-1]).to(self.low_rank_vs.dtype)
        out = flat_x @ self.low_rank_vs
        out = out @ self.low_rank_u_t
        return out.reshape(x.shape[0], x.shape[1], self.out_features)

    def _mask_metric(self, x):
        if self.sparse_ratio == 1 or self.scale is None:
            return x.abs()
        return (x * self.scale.unsqueeze(0).unsqueeze(0)).abs()

    def _calibrate_threshold(self, x):
        nelements = x.numel()
        if nelements == 0 or self.target_sparsity <= 0:
            self.threshold.zero_()
            return

        metric = self._mask_metric(x)
        kth = int(nelements * self.target_sparsity)
        kth = max(min(kth, nelements - 1), 0)
        threshold = torch.topk(metric.reshape(-1), kth + 1, largest=False)[0][-1]
        self.threshold.copy_(threshold.to(self.threshold.dtype))

    def _hybrid_forward(self, x):
        if x.numel() == 0:
            return x.new_empty(x.shape[0], x.shape[1], self.out_features)

        if self.mode == "dense":
            self._update_stats(x)
            return self._dense_forward(x)

        if self.mode == "low_rank":
            zeros = torch.zeros_like(x)
            self._update_stats(zeros)
            out = self._low_rank_forward(x)
            if self.qlinear.bias is not None:
                out = out + self.qlinear.bias
            return out

        metric = self._mask_metric(x)
        s_mask = metric.gt(self.threshold).to(x.dtype)
        sparse_input = x * s_mask
        self._update_stats(sparse_input)
        sparse_output = self._dense_forward(sparse_input)

        if self.mode == "sparse":
            return sparse_output

        low_rank_input = x * (1 - s_mask)
        low_rank_output = self._low_rank_forward(low_rank_input)
        return sparse_output + low_rank_output

    @torch.no_grad()
    def forward(self, x):
        if self.calibrating_threshold:
            self._calibrate_threshold(x)
            self.calibrating_threshold = False
            return self._dense_forward(x)

        num_tokens = x.size(1)

        if self.prefill_ratio == 1:
            if num_tokens > 1:
                self._update_stats(x)
                return self._dense_forward(x)
            return self._hybrid_forward(x)

        if num_tokens <= 1:
            return self._hybrid_forward(x)

        sparse_tokens = int(num_tokens * (1 - self.prefill_ratio))
        sparse_tokens = max(min(sparse_tokens, num_tokens), 0)
        input_prefill = x[:, :sparse_tokens, :]
        input_decoding = x[:, sparse_tokens:, :]

        output_prefill = self._dense_forward(input_prefill)
        output_decoding = self._hybrid_forward(input_decoding)
        return torch.cat([output_prefill, output_decoding], dim=1)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.qlinear = self.qlinear.to(*args, **kwargs)
        if self.low_rank_vs is not None:
            self.low_rank_vs = self.low_rank_vs.to(*args, **kwargs)
        if self.low_rank_u_t is not None:
            self.low_rank_u_t = self.low_rank_u_t.to(*args, **kwargs)
        if self.scale is not None:
            self.scale = self.scale.to(*args, **kwargs)
        self.reorder_index = self.reorder_index.to(*args, **kwargs)
        self.threshold = self.threshold.to(*args, **kwargs)
        self.zero_count = self.zero_count.to(*args, **kwargs)
        self.total_count = self.total_count.to(*args, **kwargs)
        return self
