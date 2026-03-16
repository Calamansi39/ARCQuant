import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import *
from optional_agemm import (
    AGEMM_IMPORT_ERROR,
    DEVICE_SM,
    HAS_AGEMM,
    require_agemm,
    warn_agemm_fallback_once,
)

import math
import random


def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def NVFP4_reorder_quantize_w(w, reorder_index, select_num):
    scale = torch.max(w.abs()).float() / (448.0*6.0)
    qw, scale_w = require_agemm().reorder_quantize_w(w / scale, reorder_index, select_num)
    return qw, scale_w, scale
    
class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        select_num, 
        reorder_index,
        out_reorder_index=None,
        quant_type='NVFP4',
    ):
        super().__init__()
      
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
    
        
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        
        self.select_num = select_num
        self.quant_type = quant_type
        self.use_agemm = bool(self.quant_type == "NVFP4" and HAS_AGEMM)

        if self.use_agemm:
            self.W, self.scale_w, self.scale = NVFP4_reorder_quantize_w(
                originalLayer.weight.data,
                reorder_index.to(torch.int16).cuda(),
                select_num,
            )
        else:
            if self.quant_type == "NVFP4" and not HAS_AGEMM:
                reason = f"SM{DEVICE_SM} detected" if DEVICE_SM is not None else "no CUDA device detected"
                if AGEMM_IMPORT_ERROR is not None:
                    reason = f"import failed: {AGEMM_IMPORT_ERROR}"
                warn_agemm_fallback_once(reason)

            self.W, self.scale_w, self.scale = fake_reorder_quantize_w(
                originalLayer.weight.data,
                reorder_index.to(device=originalLayer.weight.device, dtype=torch.long),
                select_num,
                dtype=quant_type,
            )
        
        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        qx, scale_x, scale, bsz, q_len = x

        if self.use_agemm:
            y = require_agemm().matmul(qx, self.W, scale_x, self.scale_w, scale * self.scale)
        else:
            y = F.linear(qx, self.W)
        
        if qx.is_cuda:
            torch.cuda.synchronize()
        if self.bias is not None:
            y = y + self.bias

        if bsz is not None:
            y = y.reshape(bsz, q_len, -1)
        else:
            y = y.reshape(q_len, -1)
        return y

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.W = self.W.to(*args, **kwargs)
        self.scale_w = self.scale_w.to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    
