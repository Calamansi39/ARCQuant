"""Method-specific ARC channel statistics from real, verified teacher inputs.

Old HiF4 trajectories provide calibration inputs only. ARC capture and greedy
search are new computations and cannot reuse the old no-ARC search decisions.
"""
import json
import os
from pathlib import Path

import torch

from .protocol import sources,sha256,digest,ROOT
from vbench_generation_cache import atomic_write_json,cache_lock


def manifest_path():
    return Path(os.environ['MAGI_ARC_CALIBRATION'])/'manifest.json'


def arc_identity(method=None):
    path=manifest_path()
    data=json.loads(path.read_text())
    coordinates=lambda name:'svd' if name=='wina' else 'original'
    if (data.get('schema')!='magi-arc-channel-calibration-v1' or not data.get('complete')
            or data['source_sha256']!=sources() or (method is not None and coordinates(data['method'])!=coordinates(method))):
        raise ValueError('ARC calibration identity/method/source mismatch')
    return dict(schema='magi-arc-v1',enabled=True,method_coordinates=coordinates(data['method']),
        calibration_manifest_sha256=sha256(path),rank_sha256=data['rank_sha256'],
        policy='signed max/8; ceil residual count to 64; stable ascending channel absmax',
        arithmetic='mask then reorder; HiF4 main + quantized selected activation residuals',
        kernel='HiF4 reconstructed values + dense GEMM')


def load_specs(method,rank):
    identity=arc_identity(method)
    path=manifest_path().parent/f'rank{rank}.pt'
    if sha256(path)!=identity['rank_sha256'][rank]: raise ValueError('Changed ARC rank statistics')
    return torch.load(path,map_location='cpu',weights_only=True)


def validate_generation(config,formal=True):
    data=json.loads(manifest_path().read_text())
    if formal and data['debug']: raise ValueError('Debug ARC calibration cannot produce formal results')
    for key,value in data['generation'].items():
        if config[key]!=value: raise ValueError('ARC calibration generation mismatch: '+key)


def collect(opt,args,teacher_root):
    """Replay ordinary HiF4 teacher prefixes; observe each raw student block."""
    import gc
    import torch.distributed as dist
    import gen_vbench_magi2 as native
    from magi2_quant import apply_magi2_quant
    from .session import initialize_distributed,load_block,release_block
    from .snapshots import read,decode
    from .runtime import Context,install
    from .assets import tensor_hash
    from .calibrate import block_names
    config=initialize_distributed(args)
    rank=dist.get_rank()
    dense=teacher_root/'dense/hifp4'
    teacher=json.loads((dense/'identity.json').read_text())
    # Verify every recorded numerical source without depending on its import namespace.
    from .protocol import BASE
    for name,expected in teacher['source_sha256'].items():
        if sha256(BASE/name)!=expected: raise ValueError('Teacher source changed: '+name)
    _,_,_,generation,_,_=native.prepare(args)
    for key in ('model','checkpoint_manifest_sha256','checkpoint_files','source_revision','resolution',
                'preview_steps','refiner_steps','cp_size','ep_size','runtime_versions','quant','mode'):
        if teacher['generation'][key]!=generation[key]: raise ValueError('Teacher configuration differs: '+key)
    if teacher['debug'] and not opt.debug: raise ValueError('Formal statistics cannot use debug trajectories')
    count=opt.debug_prompts
    records={i:json.loads((dense/f'prompt-{i:02d}'/f'rank{rank}/complete.json').read_text()) for i in range(count)}
    if any(r['identity']!=teacher for r in records.values()): raise ValueError('Teacher snapshot identity differs')
    destination=manifest_path().parent
    identity=dict(schema='magi-arc-channel-calibration-v1',method=opt.method,
        teacher_identity_sha256=sha256(dense/'identity.json'),source_sha256=sources(),
        generation={k:generation[k] for k in ('checkpoint_manifest_sha256','checkpoint_files','source_revision',
            'runtime_versions','quant','mode','resolution','preview_steps','refiner_steps','cp_size','ep_size')},
        prompts=count,debug=opt.debug)
    selected=set(block_names(opt,args))
    result={}
    active=None
    with cache_lock(destination/f'.rank{rank}'):
        for component,index in [('preview',i) for i in range(40)]+([('refiner',i) for i in range(30)] if args.resolution=='1080p' else []):
            if not any(c==component and i>=index for c,i in selected): continue
            if active!=component:
                active=component
                paths=[dense/f'prompt-{i:02d}'/f'rank{rank}'/component/'block.layers.0.safetensors' for i in range(count)]
                for i,path in enumerate(paths):
                    if sha256(path)!=records[i]['paths'][f'{component}.block.layers.0']['sha256']: raise ValueError('Changed teacher input')
                working=[read(path) for path in paths]
            prefix=f'{component}.block.layers.{index}'
            teacher_block=load_block(args,config,component,index)
            apply_magi2_quant(teacher_block,'hifp4',True,device='cuda')
            teacher_block.to('cuda')
            targets=[]
            for i,payload in enumerate(working):
                item=decode(*payload)
                with torch.inference_mode(): target=teacher_block(*item['args'],**item['kwargs']).cpu()
                if tensor_hash(target)!=records[i]['outputs'][prefix]['sha256']: raise ValueError('Teacher replay mismatch')
                targets.append(target)
            release_block(teacher_block)
            del teacher_block,item
            gc.collect();torch.cuda.empty_cache()
            if (component,index) in selected:
                path=destination/'blocks'/f'{prefix}.rank{rank}.pt'
                meta=path.with_suffix('.json')
                if path.exists() and meta.exists():
                    saved=json.loads(meta.read_text())
                    if saved['identity']!=identity or saved['sha256']!=sha256(path): raise ValueError('Changed partial ARC statistics')
                    result.update(torch.load(path,weights_only=True))
                else:
                    block=load_block(args,config,component,index)
                    ctx=Context(opt.method,'hifp4',{},Path(os.environ['MAGI_ARC_ASSETS']),rank=rank,collect=True)
                    ctx.observer.enabled=False
                    install(block,prefix,ctx)
                    block.to('cuda')
                    for payload in working:
                        item=decode(*payload)
                        with torch.inference_mode(): output=block(*item['args'],**item['kwargs'])
                        if not torch.isfinite(output).all(): raise ValueError('Nonfinite calibration block')
                    specs={}
                    for name,projection in ctx.registry.items():
                        obs=projection.collector
                        if not projection.ep_sharded:
                            for key,op in (('maximum',dist.ReduceOp.MAX),('rows',dist.ReduceOp.SUM),('selected',dist.ReduceOp.SUM)):
                                value=getattr(obs,key).cuda();dist.all_reduce(value,op=op);setattr(obs,key,value.cpu())
                        spec=obs.finish()
                        spec['basis_sha256']=projection.asset['sha256'] if projection.asset else None
                        specs[name]=spec
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_suffix('.tmp');torch.save(specs,temp);temp.replace(path)
                    atomic_write_json(meta,dict(identity=identity,sha256=sha256(path)))
                    result.update(specs)
                    release_block(block);del block,ctx,output,item
                    gc.collect();torch.cuda.empty_cache()
            for (tree,tensors),target in zip(working,targets):
                tensors[tree['dict']['args']['tuple'][0]['tensor']]=target
            print(f'[ARC statistics] rank={rank} {prefix}',flush=True)
        if len(selected)!=(70 if args.resolution=='1080p' else 40):
            raise ValueError('Partial statistics saved; all blocks required for manifest')
        target=destination/f'rank{rank}.pt'
        temp=target.with_suffix('.tmp');torch.save(result,temp);temp.replace(target)
        hashes=[None]*2;dist.all_gather_object(hashes,sha256(target))
        if rank==0: atomic_write_json(manifest_path(),dict(identity,complete=True,rank_sha256=hashes))
    dist.destroy_process_group()
