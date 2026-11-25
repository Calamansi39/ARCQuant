"""GPU module/offload and real-weight ARC checks; never benchmark scores."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-eval-root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    os.environ.update(ARC_BASE_EVAL=str(a.base_eval_root.resolve()),TORCH_COMPILE_DISABLE='1',
        MAGI2_DISABLE_MAGI_COMPILE='1',MAGI2_MOE_BACKEND='batched')
    sys.path[:0]=[str(a.base_eval_root),str(a.repo)]
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    if torch.cuda.get_device_capability()[0]!=9:raise ValueError('Validate on the target Hopper runtime')
    from benchmarks.run_deepseek_arc_extended import release_previous,hold_for_successor
    release_previous()
    import magi_compiler
    magi_compiler.magi_compile=lambda *args,**kwargs:(lambda cls:cls)
    from inference.model import magi2_preview as preview,magi2_refiner as refiner
    from inference.common.offload_utils import OffloadUtils,OffloadMode
    from .runtime import Context,install
    from .assets import tensor_hash
    from .arithmetic import Observer,activation,weight,expanded_weight
    from model.quantize import fake_reorder_quantize_x,fake_reorder_quantize_w
    from safetensors import safe_open
    from vbench_generation_cache import atomic_write_json
    torch.manual_seed(419);torch.set_float32_matmul_precision('highest')
    reports=[]
    with tempfile.TemporaryDirectory(prefix='magi-arc-check-') as directory:
        for component,source in (('preview',preview),('refiner',refiner)):
            original=torch.nn.Module()
            original.linear_qkv=source.create_linear(128,192,num_experts=3,bias=True).bfloat16()
            original.down_proj=source.create_linear(256,128,num_experts=1,bias=True).bfloat16()
            if component=='preview':
                original.moe_mlp=preview.CoreMultiHeadMoE(preview.CoreMultiHeadMoEConfig(
                    hidden_size=256,num_heads=2,num_experts=5,top_k=3,expert_intermediate_size=128,
                    num_layers=1,params_dtype=torch.bfloat16))
            for value in original.parameters():
                with torch.no_grad():value.normal_(0,.05)
            x=torch.randn(17,128,device='cuda',dtype=torch.bfloat16)
            down=torch.randn(17,256,device='cuda',dtype=torch.bfloat16)
            dispatcher=source.ModalityDispatcher(torch.tensor([0]*11+[1]*3+[2]*3,device='cuda'),3)
            route_x=torch.randn(31,2,128,device='cuda',dtype=torch.bfloat16)
            indices=torch.rand(2,31,5,device='cuda').argsort()[...,:3].contiguous()
            probs=torch.softmax(torch.randn(2,31,3,device='cuda'),-1)
            def forward(model):
                result=[model.linear_qkv(x,modality_dispatcher=dispatcher),model.down_proj(down)]
                if component=='preview':result.append(model.moe_mlp._flash_forward(route_x,probs,indices))
                return result
            for method in ('teal','wina'):
                calibration=copy.deepcopy(original)
                collect=Context(method,'hifp4',{},directory,world=1,collect=True)
                install(calibration,component,collect);calibration.cuda()
                with torch.inference_mode():forward(calibration)
                specs={}
                for name,projection in collect.registry.items():
                    specs[name]=projection.collector.finish()
                    specs[name]['basis_sha256']=projection.asset['sha256'] if projection.asset else None
                del calibration,collect
                model=copy.deepcopy(original)
                ctx=Context(method,'hifp4',{},directory,world=1,specs=specs)
                install(model,component,ctx)
                OffloadUtils.setup(model,OffloadMode.ROUNDTRIP,'cuda');model.cuda()
                for rate in (0.,.4,.8,1.):
                    ctx.configure(dict.fromkeys(ctx.registry,rate));ctx.observer.reset();ctx.backend='reference'
                    with torch.inference_mode():expected=forward(model)
                    old=ctx.observer.export(0,ctx.registry)
                    ctx.observer.reset();ctx.backend='triton'
                    with torch.inference_mode():actual=forward(model)
                    if not all(torch.equal(x,y) and torch.isfinite(y).all() for x,y in zip(expected,actual)) or old!=ctx.observer.export(0,ctx.registry):
                        raise ValueError(f'ARC reference/fast adapter mismatch: {component}/{method}/{rate}')
                    reports.append(dict(component=component,method=method,rate=rate,bit_exact=True))
                OffloadUtils.offload(model);model.cuda()
                with torch.inference_mode():again=forward(model)
                if not all(torch.equal(x,y) for x,y in zip(actual,again)):raise ValueError('Offload changed ARC computation')
                del model,ctx
        for component,key in [('preview','block.layers.0.attention.linear_qkv.weight'),
                              ('preview','block.layers.2.mlp.moe_mlp.W_gate'),
                              ('refiner','block.layers.0.mlp.down_proj.weight')]:
            mapping=json.loads((a.model/component/'model.safetensors.index.json').read_text())['weight_map']
            with safe_open(str(a.model/component/mapping[key]),framework='pt',device='cpu') as f:
                view=f.get_slice(key);shape=view.get_shape()
                raw=view[0].T.contiguous() if len(shape)==3 else view[:min(shape[0],512)].contiguous()
            raw=raw.bfloat16();k=raw.shape[-1]
            x=torch.randn(1,11,k,device='cuda',dtype=torch.bfloat16)*.2
            observer=Observer(1,k);observer.update(x);spec=observer.finish()
            main=raw[None].clone();residual=weight(main,spec,'triton','cuda')
            ax=activation(x,spec,backend='triton');aw=expanded_weight(main.cuda(),residual,spec)
            rx=fake_reorder_quantize_x(x[0],spec['order'][0].cuda(),int(spec['select'][0]),dtype='HiF4')[0]
            rw=fake_reorder_quantize_w(raw.cuda(),spec['order'][0].cuda(),int(spec['select'][0]),dtype='HiF4')[0]
            if not torch.equal(ax[0],rx) or not torch.equal(aw[0],rw):raise ValueError('Real-weight ARC oracle mismatch')
            reports.append(dict(real_weight=key,weight_sha256=tensor_hash(raw),full_input_width=k,bit_exact=True))
    path=a.output/f'rank{os.environ.get("RANK","0")}.json'
    atomic_write_json(path,dict(complete=True,cases=reports,device=torch.cuda.get_device_name(),
        scope='Synthetic official modules + sampled real checkpoint matrices; no video quality scores'))
    print(f'[ARC validated] {len(reports)} GPU cases',flush=True)
    hold_for_successor()


if __name__=='__main__':main()
