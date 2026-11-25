"""Adapters around the pinned dense loader; the dense source stays unchanged."""
from contextlib import contextmanager
import importlib
import os
import sys

import torch


def environment(args):
    import gen_vbench_magi2
    gen_vbench_magi2.configure_kernel_cache()
    if int(os.environ.get('WORLD_SIZE','0')) != 2 or args.num_gpus != 2:
        raise ValueError('MAGI sparse experiments require torchrun with CP=EP=2')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    if torch.cuda.get_device_capability()[0] != 9:
        raise ValueError('The pinned MAGI runtime requires Hopper')
    if os.environ.get('SKIP_LOAD_MODEL','').lower() in ('1','true','yes','on'):
        raise ValueError('Synthetic weights cannot be used for experiments')
    sys.path.insert(0,str(args.repo.resolve()))
    os.environ.update(MAGI2_CKPT_ROOT=str(args.model.resolve()),TORCH_COMPILE_DISABLE='1',
        MAGI2_DISABLE_MAGI_COMPILE='1',MAGI2_ROUTER_BIAS_SOURCE='ema',MAGI2_DETERMINISTIC='1',
        MAGI2_MOE_BACKEND='batched',MAGI2_MOE_BATCH_EXPERTS=str(args.moe_batch_experts),
        MAGI2_MOE_SCRATCH_MIB=str(args.moe_scratch_mib))
    import magi_compiler
    magi_compiler.magi_compile = lambda *a, **kw: (lambda cls: cls)


def initialize_distributed(args):
    environment(args)
    from inference.common.magi2_config import load_config
    from inference.pipeline.entry import _init_distributed, _enable_deterministic
    config = load_config(str(args.repo/'configs/magi2_refiner.json'))
    config.engine_config.cp_size = config.engine_config.ep_size = 2
    config.engine_config.seed = args.seed
    torch.set_float32_matmul_precision('highest')
    _enable_deterministic(args.seed)
    _init_distributed(config)
    return config


@contextmanager
def installation(context=None, after_load=None):
    import magi2_quant
    original = magi2_quant.apply_magi2_quant
    if context is None:
        from .runtime import Context
        context=Context(os.environ['MAGI_ARC_METHOD'],'hifp4',{},
            os.environ['MAGI_ARC_ASSETS'],rank=int(os.environ['RANK']))
        context.observer.enabled=False
    if context is not None:
        context.quantizer = original
    def apply(model, quant, act_quant=True, device='cpu'):
        component = 'refiner' if model.__class__.__module__.endswith('magi2_refiner') else 'preview'
        if context is not None:
            from integrations.magi_arc.runtime import install
            report = install(model,component,context)
            model.register_forward_pre_hook(lambda mod, inp: context.observer.before_component(component))
        else:
            report = original(model,quant,act_quant,device)
        if after_load is not None:
            after_load(component,model)
        def finite(mod,inputs,output):
            if not torch.isfinite(output).all():
                raise ValueError(f'Nonfinite {component} output')
        model.register_forward_hook(finite)
        return report
    magi2_quant.apply_magi2_quant = apply
    try:
        yield
    finally:
        magi2_quant.apply_magi2_quant = original


def load_block(args, config, component, index):
    """Read only one layer, using the exact official EP slice/EMA reader."""
    from inference.infra.checkpoint.magi2_checkpointing import (
        read_safetensors_weight_map, _load_by_shard, _read_ep_slice, _apply_router_bias_ema)
    module = importlib.import_module('inference.model.magi2_'+component)
    cfg = config.arch_config if component == 'preview' else config.magi2_refiner_arch_config
    # The official constructors register tensors/custom ops; instantiate on CPU.
    block = module.TransformerLayer(cfg,index)
    prefix = f'block.layers.{index}.'
    targets = {prefix+k:v for k,v in block.state_dict().items()}
    mapping = read_safetensors_weight_map(str(args.model/component))
    mapping = {k:v for k,v in mapping.items() if k in targets}
    state = _apply_router_bias_ema(_load_by_shard(mapping,_read_ep_slice,'Loading sparse block',targets))
    state = {k[len(prefix):]:v.to(dtype=targets[k].dtype) for k,v in state.items()}
    block.load_state_dict(state,strict=True)
    return block.eval()


def release_block(block):
    # The official MoE custom-op registry owns modules strongly.
    module = sys.modules.get('inference.model.magi2_preview')
    if module is not None:
        for child in block.modules():
            if type(child).__name__ == 'CoreMultiHeadMoE':
                module._magi2_moe_registry.pop(child._moe_key,None)


def seed_all(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
