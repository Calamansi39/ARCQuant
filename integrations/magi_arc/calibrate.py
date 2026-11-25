"""Ten full dense trajectories, distributed block greedy search and path extraction."""
import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path

from integrations.magi_arc.protocol import BASE, ROOT, DEFAULT_ROOT, SCHEMA, sources, sha256, candidates, replay, bind_dense_baseline
from vbench_generation_cache import atomic_write_json, cache_lock
from integrations.magi_arc.sharding import parse_shard, partition, shard_name


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('phase',choices=['stats','capture','search','extract'],nargs='?')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--method',choices=['none','teal','wina'],default='teal')
    p.add_argument('--target',type=float,default=.6)
    p.add_argument('--step',type=float,default=.01,help='Greedy parameter-budget step (default: 0.01 for TEAL and WINA)')
    p.add_argument('--extract-target',type=float,default=.4)
    blocks = p.add_mutually_exclusive_group()
    blocks.add_argument('--blocks',help='Debug/subset: comma-separated preview:0,refiner:0')
    blocks.add_argument('--block-shard',type=parse_shard,help='Search disjoint block subset i/n (zero-based)')
    capture_subset = p.add_mutually_exclusive_group()
    capture_subset.add_argument('--prompt-index',type=int,help='Capture one of the ten trajectories')
    capture_subset.add_argument('--prompt-shard',type=parse_shard,help='Capture disjoint prompt subset i/n (zero-based)')
    p.add_argument('--max-rounds',type=int,help='Stop search at this round; incomplete paths cannot be extracted')
    p.add_argument('--debug',action='store_true',help='Permit shortened trajectories in an isolated root')
    p.add_argument('--debug-prompts',type=int,default=10,help='With --debug only: fewer trajectories for integration tests')
    p.add_argument('--help',action='store_true')
    opt,remaining = p.parse_known_args(argv)
    import gen_vbench_magi2 as native
    args = native.parse_args(['--protocol','tiny16','--num-inference-steps','50',*remaining])
    if opt.help:
        p.print_help()
        raise SystemExit()
    if opt.phase is None:
        p.error('Specify capture, search or extract')
    if args.quant != 'hifp4' or args.num_gpus != 2 or args.mode != 'w4a4':
        raise ValueError('Calibration requires BF16/HiFP4 W4A4, CP=EP=2')
    if not opt.debug and (args.resolution != '1080p' or args.num_inference_steps != 50 or args.refiner_num_inference_steps != 5):
        raise ValueError('Formal calibration requires the completed dense baseline: 1080p / 50+5 steps')
    if opt.debug and opt.root.resolve() == DEFAULT_ROOT.resolve():
        raise ValueError('Debug calibration must use a separate --root')
    if not 1<=opt.debug_prompts<=10 or (not opt.debug and opt.debug_prompts!=10):
        raise ValueError('Formal calibration requires exactly ten prompts')
    if not 0 < opt.target < 1 or not 0 < opt.step <= opt.target:
        raise ValueError('Invalid greedy target/step')
    if opt.prompt_index is not None and not 0 <= opt.prompt_index < opt.debug_prompts:
        raise ValueError('--prompt-index must be within the calibration prompt count')
    if (opt.prompt_index is not None or opt.prompt_shard is not None) and opt.phase!='capture':
        raise ValueError('Prompt subsets apply only to capture')
    if (opt.blocks or opt.block_shard is not None) and opt.phase not in ('search','stats'):
        raise ValueError('Block subsets apply only to search')
    if opt.max_rounds is not None and opt.max_rounds < 1:
        raise ValueError('--max-rounds must be positive')
    args.model,args.repo = args.model.resolve(),args.repo.resolve()
    return opt,args


def prompts():
    from vbench_protocol import load_prompts, STANDARD_DIMS
    data = json.loads((ROOT/'prompts.json').read_text())
    tiny = {p for v in load_prompts(STANDARD_DIMS,-1,'tiny16').values() for p in v}
    rows = data['prompts']
    if len(rows) != 10 or len({r['prompt'] for r in rows}) != 10 or any(r['prompt'] in tiny for r in rows):
        raise ValueError('Calibration prompts must be ten distinct non-Tiny prompts')
    return rows


def dense_identity(args,opt):
    import gen_vbench_magi2 as native
    _,_,_,config,_,_ = native.prepare(args)
    from .calibration import arc_identity,validate_generation
    validate_generation(config,formal=not opt.debug)
    config['arcquant']=arc_identity(opt.method)
    from vbench_generation_cache import generation_identity
    return dict(schema=SCHEMA,kind='dense-calibration',generation=generation_identity(config),
                source_sha256=sources(),prompts_sha256=sha256(ROOT/'prompts.json'),debug=opt.debug,
                baseline_reference=None if opt.debug else bind_dense_baseline(config),
                calibration_prompts=opt.debug_prompts,
                capture='full first-block inputs plus every dense block output SHA256; exact dense-prefix replay; evenly spaced component forward indices')


def capture(opt,args):
    import torch
    import torch.distributed as dist
    from magi2_runtime import initialize_pipeline, StageTimer
    from magi2_vbench_worker import inference_arguments
    from integrations.magi_arc.session import environment, installation, seed_all
    from integrations.magi_arc.snapshots import Capture
    environment(args)
    identity = dense_identity(args,opt)
    folder = opt.root/'dense'/args.quant
    rank = int(os.environ['RANK'])
    selected = partition(range(opt.debug_prompts),opt.prompt_shard) if opt.prompt_index is None else [opt.prompt_index]
    with ExitStack() as resources:
        # A whole-precision lock would serialize disjoint four-GPU shards.
        for i in sorted(selected):
            resources.enter_context(cache_lock(folder/f'prompt-{i:02d}'/f'.capture-rank{rank}'))
        config_file = folder/'identity.json'
        with cache_lock(folder/'.identity',blocking=True):
            if config_file.exists() and json.loads(config_file.read_text()) != identity:
                raise ValueError('Dense calibration identity differs; use another root')
            if not config_file.exists():
                atomic_write_json(config_file,identity)
        pending = []
        for i in selected:
            valid = True
            for r in range(2):
                path = folder/f'prompt-{i:02d}'/f'rank{r}'/'complete.json'
                try:
                    data = json.loads(path.read_text())
                    valid &= data['identity']==identity and len(data['outputs'])==(70 if args.resolution=='1080p' else 40)
                    valid &= len(data['paths'])==(2 if args.resolution=='1080p' else 1)
                    valid &= all(sha256(p['path'])==p['sha256'] for p in data['paths'].values())
                except (OSError,KeyError,ValueError):
                    valid = False
            if not valid:
                pending.append(i)
        if not pending:
            print('[capture] Selected trajectories are complete',flush=True)
            return
        recorder = Capture(folder,identity,dict(preview=args.num_inference_steps,
            **({'refiner':2*args.refiner_num_inference_steps} if args.resolution=='1080p' else {})))
        worker = shard_name(opt.prompt_shard) if opt.prompt_index is None else f'prompt-{opt.prompt_index:02d}'
        timer = StageTimer(folder/'workers'/worker/f'timing-rank{rank}',args.profile_timing)
        with installation(after_load=recorder.install):
            pipe,_,_ = initialize_pipeline(args,timer)
        rows = prompts()
        for i in pending:
            recorder.begin(i)
            seed_all(rows[i]['seed'])
            prefix = str(folder/f'prompt-{i:02d}'/'dense')
            with torch.inference_mode():
                pipe.run_offline(**inference_arguments(args,rows[i]['prompt'],prefix))
            recorder.finish()
            dist.barrier()
            if rank == 0:
                print(f'[capture] completed prompt {i+1}/10',flush=True)
        dist.destroy_process_group()


def block_names(opt,args):
    names = [('preview',i) for i in range(40)]
    if args.resolution=='1080p':
        names += [('refiner',i) for i in range(30)]
    if opt.blocks:
        requested = [(s.split(':')[0],int(s.split(':')[1])) for s in opt.blocks.split(',')]
        if set(requested)-set(names) or len(set(requested)) != len(requested):
            raise ValueError('Invalid/duplicate --blocks')
        names = requested
    return partition(names,opt.block_shard)


def search(opt,args):
    import torch
    import torch.distributed as dist
    from integrations.magi_arc.session import initialize_distributed, load_block, release_block
    from integrations.magi_arc.runtime import Context, install
    from integrations.magi_arc.snapshots import read,decode
    from integrations.magi_arc.assets import tensor_hash
    from magi2_quant import apply_magi2_quant
    config = initialize_distributed(args)
    rank = dist.get_rank()
    identity = dense_identity(args,opt)
    dense = opt.root/'dense'/args.quant
    if json.loads((dense/'identity.json').read_text()) != identity:
        raise ValueError('Search does not match the dense calibration identity')
    complete_records = {}
    count=opt.debug_prompts
    for i in range(count):
        for r in range(2):
            complete = json.loads((dense/f'prompt-{i:02d}'/f'rank{r}'/'complete.json').read_text())
            if complete['identity'] != identity:
                raise ValueError('Incomplete/incompatible ten-prompt calibration')
            complete_records[i,r] = complete
    folder = opt.root/'search'/f'{opt.method}-{args.quant}'
    selected=set(block_names(opt,args))
    with ExitStack() as resources:
        for component,index in sorted(selected):
            resources.enter_context(cache_lock(folder/f'.{component}-{index}-rank{rank}'))
        active_component=None
        working=None
        # Replay the original dense prefix even when earlier searches are done.
        # Ten complete documents live on CPU; one moves to CUDA at a time.
        for component,index in [('preview',i) for i in range(40)]+([('refiner',i) for i in range(30)] if args.resolution=='1080p' else []):
            if not any(c==component and i>=index for c,i in selected):
                continue
            if active_component!=component:
                active_component=component
                snapshot_paths=[dense/f'prompt-{i:02d}'/f'rank{rank}'/component/'block.layers.0.safetensors' for i in range(count)]
                working=[read(p) for p in snapshot_paths]
                for i,p in enumerate(snapshot_paths):
                    if sha256(p)!=complete_records[i,rank]['paths'][f'{component}.block.layers.0']['sha256']:
                        raise ValueError('Changed dense component entry snapshot')
            prefix = f'{component}.block.layers.{index}'
            path = folder/(prefix+'.json')
            history = json.loads(path.read_text()) if (component,index) in selected and path.exists() else None
            step = opt.step
            snapshot_hashes={f'prompt-{i:02d}/rank{r}':dict(
                entry=complete_records[i,r]['paths'][f'{component}.block.layers.0']['sha256'],
                output=complete_records[i,r]['outputs'][prefix]['sha256']) for i in range(count) for r in range(2)}
            header = dict(schema=SCHEMA,identity=identity,method=opt.method,quant=args.quant,
                          block=prefix,target=opt.target,step=step,snapshot_sha256=snapshot_hashes)
            if history and any(history[k]!=v for k,v in header.items()):
                raise ValueError(f'Changed search identity: {path}')
            teacher=load_block(args,config,component,index)
            teacher_context=Context(opt.method,args.quant,{},Path(os.environ['MAGI_ARC_ASSETS']),rank=rank)
            teacher_context.observer.enabled=False
            install(teacher,prefix,teacher_context)
            teacher.to('cuda')
            targets=[]
            for i,(tree,tensors) in enumerate(working):
                payload=decode(tree,tensors)
                with torch.inference_mode():
                    output=teacher(*payload['args'],**payload['kwargs']).cpu()
                if tensor_hash(output)!=complete_records[i,rank]['outputs'][prefix]['sha256']:
                    raise ValueError(f'Dense prefix replay differs from trajectory: {prefix}/prompt-{i:02d}/rank{rank}')
                targets.append(output)
                del payload
            release_block(teacher)
            del teacher,teacher_context
            gc.collect()
            torch.cuda.empty_cache()
            def advance():
                for (tree,tensors),target in zip(working,targets):
                    tensors[tree['dict']['args']['tuple'][0]['tensor']]=target
            if (component,index) not in selected or (history and history.get('complete')):
                if history and history.get('complete'):
                    replay(history,opt.target)
                advance()
                continue
            block = load_block(args,config,component,index)
            context = Context(opt.method,args.quant,{},Path(os.environ['MAGI_ARC_ASSETS']),rank=rank)
            context.observer.enabled = False
            install(block,prefix,context)
            block.to('cuda')
            sizes = context.sizes()
            if history and history['sizes'] != sizes:
                raise ValueError('Projection family sizes differ')
            widths={n:p.width for n,p in context.registry.items()}
            runtime_assets=[None]*2
            dist.all_gather_object(runtime_assets,context.fingerprints())
            state = history or dict(header,sizes=sizes,widths=widths,runtime_assets=runtime_assets,history=[],complete=False)
            if state['widths']!=widths:
                raise ValueError('Calibrated projection widths differ')
            if state['runtime_assets']!=runtime_assets:
                raise ValueError('SVD bases or actual weight norms differ from calibration')
            if not state.get('preflight'):
                audits=[]
                for sample_index in range(min(2,count)):
                    payload=decode(*working[sample_index])
                    target=targets[sample_index].to('cuda')
                    for rate in (0.,.4):
                        context.configure(dict.fromkeys(sizes,rate))
                        context.backend='reference'
                        with torch.inference_mode():
                            reference=block(*payload['args'],**payload['kwargs'])
                        context.backend='triton'
                        with torch.inference_mode():
                            fast=block(*payload['args'],**payload['kwargs'])
                        same=torch.equal(reference,fast)
                        zero_same=torch.equal(fast,target)
                        if not same or (rate==0 and opt.method=='teal' and not zero_same):
                            raise ValueError(f'Block preflight failed: {prefix}, rate={rate}, accelerated_equal={same}, dense_equal={zero_same}')
                        audits.append(dict(prompt_index=sample_index,rate=rate,bit_exact=same,dense_equal=zero_same))
                        del reference,fast
                    del payload,target
                state['preflight']=audits
            current = dict.fromkeys(sizes,0.)
            for row in state['history']:
                options = candidates(sizes,current,opt.target,step)
                errors = row['errors']
                if set(errors)!=set(options) or not all(__import__('math').isfinite(v) and v>=0 for v in errors.values()):
                    raise ValueError('Corrupt partial greedy candidate errors')
                best = min(errors,key=lambda n:(errors[n],n))
                if row['selected']!=best or row['allocation']!=options[best]:
                    raise ValueError('Corrupt partial greedy search')
                current = options[best]
            while options := candidates(sizes,current,opt.target,step):
                if opt.max_rounds and len(state['history']) >= opt.max_rounds:
                    break
                keys = sorted(options)
                totals = torch.zeros(len(keys),device='cuda',dtype=torch.float64)
                for sample_index in range(count):
                    payload = decode(*working[sample_index])
                    target = targets[sample_index].to('cuda')
                    errors = torch.zeros((len(keys),2),device='cuda',dtype=torch.float64)
                    for j,name in enumerate(keys):
                        context.configure(options[name])
                        with torch.inference_mode():
                            out = block(*payload['args'],**payload['kwargs'])
                        if not torch.isfinite(out).all():
                            raise ValueError(f'Nonfinite candidate {prefix}/{name}')
                        # Bounded reduction: never materialize a full FP64 activation.
                        error = torch.zeros((),device='cuda',dtype=torch.float64)
                        for a,b in zip(out.split(512),target.split(512)):
                            error += (a.double()-b.double()).square().sum()
                        errors[j,0] = error
                        errors[j,1] = out.numel()
                        del out
                    dist.all_reduce(errors)
                    totals += errors[:,0]/errors[:,1]
                    del payload,target
                errors = dict(zip(keys,(totals/count).cpu().tolist()))
                best = min(keys,key=lambda n:(errors[n],n))
                current = options[best]
                budget = sum(sizes[n]*current[n] for n in sizes)/sum(sizes.values())
                state['history'].append(dict(selected=best,errors=errors,allocation=current,budget=budget))
                state['complete'] = abs(budget-opt.target)<1e-9
                if rank==0:
                    atomic_write_json(path,state)
                    print(f'[search] {prefix} {opt.method}/{args.quant} round={len(state["history"])} budget={budget:.6f}',flush=True)
                dist.barrier()
            release_block(block)
            advance()
            del block,context
            gc.collect()
            torch.cuda.empty_cache()
    dist.destroy_process_group()


def extract(opt,args):
    if opt.blocks or opt.block_shard:
        raise ValueError('Formal allocation extraction requires all model blocks')
    folder = opt.root/'search'/f'{opt.method}-{args.quant}'
    layers,records,sizes,widths = {},{},{},{}
    generation=None
    runtime_assets=[{},{}]
    output = opt.root/'allocations'/f'{opt.method}-{args.quant}-s{opt.extract_target:g}.json'
    for component,index in block_names(opt,args):
        prefix = f'{component}.block.layers.{index}'
        path = folder/(prefix+'.json')
        data = json.loads(path.read_text())
        if (data['identity']['source_sha256'] != sources() or data['quant']!=args.quant
                or data['method']!=opt.method or not data['complete'] or not data.get('preflight') or data['identity']['debug']):
            raise ValueError('Only complete, current, formal calibration can be extracted')
        layers.update(replay(data,opt.extract_target))
        sizes.update(data['sizes'])
        widths.update(data['widths'])
        if generation is not None and generation!=data['identity']['generation']:
            raise ValueError('Calibration blocks have different dense generation identities')
        generation=data['identity']['generation']
        for rank in range(2):
            runtime_assets[rank].update(data['runtime_assets'][rank])
        records[prefix] = dict(path=os.path.relpath(path,output.parent),sha256=sha256(path))
    from integrations.magi_arc.reference import keep_count
    actual=sum(sizes[n]*(1-keep_count(widths[n],rate)/widths[n]) for n,rate in layers.items())/sum(sizes.values())
    atomic_write_json(output,dict(schema=SCHEMA,method=opt.method,quant=args.quant,target=opt.extract_target,
        source_sha256=sources(),layers=layers,search_files=records,calibration_root=str(opt.root.resolve()),
        generation=generation,prompts_sha256=sha256(ROOT/'prompts.json'),
        baseline_reference=data['identity']['baseline_reference'],
        runtime_assets=runtime_assets,
        audit=dict(parameters=sum(sizes.values()),parameter_weighted_mask_sparsity=actual,
                   nominal_sparsity=sum(sizes[n]*p for n,p in layers.items())/sum(sizes.values()))))
    print(output)


def main(argv=None):
    opt,args = arguments(argv)
    {'capture':capture,'search':search,'extract':extract}[opt.phase](opt,args)


if __name__=='__main__':
    main()
