"""Resumable MAGI Tiny generation with calibrated sparsity and complete statistics."""
import argparse
from contextlib import ExitStack
import gzip
import json
import os
from pathlib import Path

import gen_vbench_magi2 as native
from integrations.magi_arc.protocol import SCHEMA, DEFAULT_ROOT, sources, sha256, load_allocation, digest, bind_dense_baseline
from vbench_generation_cache import (atomic_write_json, cache_lock, cache_video_path,
    generation_identity, prepare_cache, record_cached_video, video_is_complete, video_job_lock)
from vbench_protocol import video_seed
from integrations.magi_arc.sharding import parse_shard, partition, shard_name


def arguments(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--method',choices=['none','teal','wina'],required=True)
    p.add_argument('--allocation',type=Path)
    p.add_argument('--sparsity',type=float,default=.4)
    p.add_argument('--asset-dir',type=Path,default=DEFAULT_ROOT/'svd')
    p.add_argument('--backend',choices=['triton','reference'],default='triton')
    p.add_argument('--job-shard',type=parse_shard,help='Generate disjoint video subset i/n (zero-based)')
    p.add_argument('--defer-finalize',action='store_true',help='Coordinator publishes after all shards finish')
    p.add_argument('--debug-uniform',action='store_true',help='Uncalibrated smoke tests only; cannot publish a Tiny suite')
    opt,remaining = p.parse_known_args(argv)
    args = native.parse_args(['--protocol','tiny16','--num-inference-steps','50',*remaining])
    if args.finalize_only and (opt.job_shard is not None or opt.defer_finalize):
        raise ValueError('--finalize-only must validate the entire suite, without sharding/deferred publication')
    if args.quant!='hifp4' or args.num_gpus!=2 or args.mode!='w4a4':
        raise ValueError('Requires BF16/HiFP4 W4A4, CP=EP=2')
    from integrations.magi_arc.reference import keep_count
    keep_count(256,opt.sparsity)
    if opt.debug_uniform:
        if opt.allocation or args.protocol!='custom10' or not args.out_dir:
            raise ValueError('--debug-uniform requires custom10 and an explicit debug --out-dir, without allocation')
    elif opt.method=='none':
        if opt.sparsity!=0 or opt.allocation:
            raise ValueError('ARC dense uses --method none --sparsity 0 without allocation')
    elif not opt.allocation:
        raise ValueError('Formal generation requires a greedy --allocation')
    args.tag = args.tag or f'arc-{args.quant}-{opt.method}-s{opt.sparsity:g}-v1'
    return opt,args


def prepare(opt,args):
    out,cache,tag,config,by_dim,jobs = native.prepare(args)
    from .calibration import arc_identity,validate_generation
    validate_generation(config,formal=not opt.debug_uniform)
    config['arcquant']=arc_identity(opt.method)
    allocation = load_allocation(opt.allocation,opt.method,args.quant) if opt.allocation else None
    if allocation:
        baseline=bind_dense_baseline(config)
        if baseline!=allocation['baseline_reference']:
            raise ValueError('Calibration is bound to a different dense baseline')
        config['dense_baseline_reference']=baseline
        if abs(allocation['target']-opt.sparsity)>1e-9:
            raise ValueError('Requested rate differs from allocation')
        for field in ('model','checkpoint_revision','checkpoint_files','source_revision','resolution',
                      'preview_steps','refiner_steps','cp_size','ep_size','mode','quant','expert_backend',
                      'moe_execution','runtime_versions'):
            if allocation['generation'][field] != config[field]:
                raise ValueError(f'Generation differs from calibration: {field}')
    config['sparsity'] = dict(schema=SCHEMA,method=opt.method,target=opt.sparsity,
        allocation_sha256=sha256(opt.allocation) if allocation else None,
        allocation=allocation['layers'] if allocation else ('dense-zero' if opt.method=='none' else 'debug-uniform'),backend=opt.backend,
        selection='exact per-token TopK; equal scores retain lower original channel indices',
        allocation_unit='logical projection family; shared rate across modalities/experts',
        scope='preview/refiner Attention QKV/O and FFN dense/shared/routed/split/merge projections',
        excluded='routers, attention gates, mHC, norms, embeddings/adapters, text encoder, VAEs',
        wina='SVD of fused QKV/gate-up; local xQ and WQ; output projections unrotated; norms of ARC main weights mapped back before selection',
        statistics='every executed token and forward; padding excluded; CP tokens sum, EP expert IDs disjoint')
    config['source_sha256'].update(sources())
    return out,cache,tag,config,by_dim,jobs,allocation


def complete(path,config,**kwargs):
    if not video_is_complete(path,**kwargs):
        return False
    try:
        record = json.loads(path.with_suffix('.mp4.json').read_text())['magi_sparsity']
        stats = path.with_suffix('.two_four.json.gz')
        return (record['identity']==digest(generation_identity(config)) and
                record['statistics_sha256']==sha256(stats))
    except (OSError,ValueError,TypeError,KeyError):
        return False


def write_statistics(path,data):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8') as f:
        json.dump(data,f,allow_nan=False,separators=(',',':'))
    os.replace(tmp,path)


def finalize(args,out,cache,tag,config,by_dim,jobs):
    from integrations.magi_arc.metrics import summarize
    aggregate = None
    for prompt,sample in jobs:
        path = cache_video_path(cache,prompt,sample)
        if not complete(path,config,prompt=prompt,sample=sample,seed=video_seed(args.seed,prompt,sample)):
            raise ValueError(f'Incomplete video/statistics: {path}')
        with gzip.open(path.with_suffix('.two_four.json.gz'),'rt') as f:
            data = json.load(f)['merged']
        if aggregate is None:
            aggregate = data
        else:
            if set(aggregate['layers'])!=set(data['layers']):
                raise ValueError('Video projection coverage differs')
            for name,row in data['layers'].items():
                target = aggregate['layers'][name]
                if target['metadata']!=row['metadata'] or set(target['experts'])!=set(row['experts']):
                    raise ValueError('Video projection metadata differs')
                for expert,hist in row['experts'].items():
                    for s in range(3):
                        target['experts'][expert][s]=[a+b for a,b in zip(target['experts'][expert][s],hist[s])]
    result = dict(videos=len(jobs),**summarize(aggregate),identity=digest(generation_identity(config)))
    atomic_write_json(out/'two_four_summary.json',result)
    import csv
    with (out/'two_four_summary.csv').open('w',newline='') as f:
        fields=['stage','groups','element_sparsity','eligible_2of4_fraction','exactly_2_nonzero_fraction',
                'parameter_weighted_eligible_2of4','mac_weighted_eligible_2of4']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for stage,row in result['stages'].items():
            writer.writerow(dict(stage=stage,**{k:row[k] for k in fields[1:]}))
    native.publish_summary(args,out,cache,tag,config,by_dim,jobs)
    path=out/'gen_summary.json'
    data=json.loads(path.read_text())
    data['sparsity']=config['sparsity']
    data['two_four_summary']='two_four_summary.json'
    atomic_write_json(path,data)


def unfinished(args,cache,config,jobs):
    return [(p,s) for p,s in jobs if not complete(cache_video_path(cache,p,s),config,
            prompt=p,sample=s,seed=video_seed(args.seed,p,s))]


def publish_if_complete(args,out,cache,tag,config,by_dim,jobs,*,required=False):
    # Workers hold shared cache reservations. Completed videos are immutable;
    # this separate lock serializes publishers without blocking other shards.
    with cache_lock(cache/'.publish',blocking=True):
        missing = unfinished(args,cache,config,jobs)
        if missing:
            if required:
                raise ValueError(f'{len(missing)} videos/statistics still missing; refusing incomplete publication')
            print(f'[partial] {len(missing)} videos remain; no complete suite published',flush=True)
            return False
        finalize(args,out,cache,tag,config,by_dim,jobs)
        return True


def main(argv=None):
    opt,args=arguments(argv)
    out,cache,tag,config,by_dim,jobs,allocation=prepare(opt,args)
    selected = partition(jobs,opt.job_shard)
    if args.dry_run:
        print(f'[checked] {opt.method}/{args.quant} target={opt.sparsity}; {len(selected)}/{len(jobs)} videos')
        return
    rank=int(os.environ.get('RANK','0'))
    with ExitStack() as resources:
        if rank==0:
            resources.enter_context(cache_lock(cache,shared=True,blocking=True))
            with cache_lock(cache/'.prepare',blocking=True):
                prepare_cache(cache,generation_identity(config))
                path=out/'generation_config.json'
                if path.exists() and json.loads(path.read_text())!=config:
                    raise ValueError('Incompatible generation resume')
                atomic_write_json(path,config)
        if args.prepare_only:
            return
        if args.finalize_only:
            if rank==0:
                publish_if_complete(args,out,cache,tag,config,by_dim,jobs,required=True)
            return
        # Decide the resume list once. Independent rank checks can diverge if
        # another worker finishes a video between the two filesystem reads.
        missing=unfinished(args,cache,config,selected) if rank==0 else None
        if int(os.environ.get('WORLD_SIZE','1'))==1:
            if missing:
                raise RuntimeError('Launch generation under torchrun --nproc_per_node=2')
            if not opt.defer_finalize:
                publish_if_complete(args,out,cache,tag,config,by_dim,jobs)
            return
        import torch
        import torch.distributed as dist
        from magi2_runtime import initialize_pipeline,StageTimer
        from magi2_vbench_worker import inference_arguments
        from integrations.magi_arc.session import environment,installation,seed_all
        from integrations.magi_arc.runtime import Context
        from integrations.magi_arc.metrics import merge_ranks,summarize
        environment(args)
        dist.init_process_group(backend='nccl')
        payload=[missing]
        dist.broadcast_object_list(payload,src=0)
        missing=payload[0]
        if not missing:
            if rank==0 and not opt.defer_finalize:
                publish_if_complete(args,out,cache,tag,config,by_dim,jobs)
            dist.barrier()
            dist.destroy_process_group()
            return
        context=Context(opt.method,args.quant,allocation['layers'] if allocation else {},opt.asset_dir,
                        rank=rank,backend=opt.backend)
        worker=out/'workers'/f'{shard_name(opt.job_shard)}-{os.environ.get("MASTER_PORT","single")}'
        timer=StageTimer(worker/f'timings-rank{rank}',args.profile_timing)
        with installation(context):
            pipe,reports,text_stats=initialize_pipeline(args,timer)
        wanted=allocation['layers'] if allocation else dict.fromkeys(context.registry,opt.sparsity)
        context.configure(wanted)
        if allocation and context.fingerprints()!=allocation['runtime_assets'][rank]:
            raise ValueError('Generation SVD bases/weight norms differ from the calibrated artifacts')
        if rank==0:
            atomic_write_json(worker/'quantization_report.json',reports)
        generated=0
        for index,(prompt,sample) in enumerate(missing):
            with ExitStack() as claim:
                final=cache_video_path(cache,prompt,sample)
                seed=video_seed(args.seed,prompt,sample)
                decision=[False]
                if rank==0:
                    claim.enter_context(video_job_lock(cache,prompt,sample,blocking=True))
                    decision[0]=not complete(final,config,prompt=prompt,sample=sample,seed=seed)
                dist.broadcast_object_list(decision,src=0)
                if not decision[0]:
                    continue
                context.observer.reset()
                seed_all(seed)
                prefix=str(final.with_suffix(''))+f'.{os.environ.get("MASTER_PORT","worker")}.tmp'
                with torch.inference_mode(),timer.stage('video.total'):
                    pipe.run_offline(**inference_arguments(args,prompt,prefix))
                atomic_write_json(worker/f'arc_execution_rank{rank}.json',context.arc_report())
                local=context.observer.export(rank,context.registry)
                records=[None]*2
                dist.all_gather_object(records,local)
                generated+=1
                if rank==0:
                    merged=merge_ranks(records)
                    expected=dict(preview=args.num_inference_steps,
                        **({'refiner':args.refiner_num_inference_steps*2} if args.resolution=='1080p' else {}))
                    if merged['component_forward_calls']!=expected:
                        raise ValueError('Incomplete denoiser statistics')
                    stats_path=final.with_suffix('.two_four.json.gz')
                    write_statistics(stats_path,dict(identity=digest(generation_identity(config)),
                        prompt=prompt,sample=sample,seed=seed,merged=merged,summary=summarize(merged)))
                    os.replace(prefix+'.mp4',final)
                    pw,ph={'272p':(448,256),'540p':(896,512),'1080p':(1920,1080)}[args.resolution]
                    record_cached_video(final,prompt=prompt,sample=sample,seed=seed,
                        expected=dict(width=pw,height=ph,fps=25 if args.resolution=='1080p' else 12.5,
                                      **({'num_frames':249} if args.resolution=='1080p' else {})),
                        extra={'magi_sparsity':dict(identity=digest(generation_identity(config)),
                            statistics_sha256=sha256(stats_path))})
                    atomic_write_json(worker/'progress.json',dict(generated_this_run=generated,planned=len(jobs),
                        shard=opt.job_shard,shard_planned=len(selected),remaining_in_pass=len(missing)-index-1,text_cache=text_stats))
                    print(f'[sparse generate] {index+1}/{len(missing)} {prompt[:60]} sample={sample}',flush=True)
                dist.barrier()
            if args.max_new_videos and generated>=args.max_new_videos:
                break
        if rank==0 and not opt.defer_finalize:
            publish_if_complete(args,out,cache,tag,config,by_dim,jobs)
        dist.barrier()
        dist.destroy_process_group()


if __name__=='__main__':
    main()
