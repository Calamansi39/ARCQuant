"""TP DeepSeek offline TEAL/WINA calibration and native-style greedy search.

Run with torchrun. This is block replay, not benchmark generation. Each layer
is calibrated on independent text, searched on its saved inputs, checkpointed,
and propagated with the unsparsified quantized reference. No test answers enter
the optimizer. See docs/DEEPSEEK_GREEDY.md for the explicit MoE budget mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import SCHEMA, GROUPING, atomic_json, projection_group
from model.native_greedy import search, native_activation_error


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base-eval-root', 'source-model', 'converted-model', 'calibration-data', 'out-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--methods', default='wina,teal')
    p.add_argument('--targets', default='0.5,0.6,0.7,0.8,0.9')
    p.add_argument('--quantization', choices=('arc', 'hif4'), default='arc')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--search-samples', type=int, default=20)
    p.add_argument('--search-batch-size', type=int, default=20, help='batch independent replay samples without changing the error average')
    p.add_argument('--length', type=int, default=512)
    p.add_argument('--step', type=float, default=.025)
    p.add_argument('--last-fraction', type=float, default=.25)
    p.add_argument('--histogram-sample-limit', type=int, default=16384)
    p.add_argument('--cache-gib', type=float, default=12.)
    p.add_argument('--layer-streaming', action='store_true', help='keep model in CPU RAM and move only the current block to GPU')
    p.add_argument('--gpu-memory-gib', type=float, help='optional PyTorch allocator cap per rank, including cache')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--layers', type=int, help='diagnostic prefix only; incomplete strategies cannot evaluate the whole model')
    p.add_argument('--max-steps', type=int, help='diagnostic candidate-search step limit per invocation')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def validate(a):
    if not 1 <= a.search_samples <= a.samples or a.search_batch_size < 1 or a.length < 2 or not 0 < a.step < 1:
        raise ValueError('Invalid calibration/search sample count, sequence length or step')
    if not 0 < a.last_fraction <= 1 or a.histogram_sample_limit < 2048 or a.cache_gib < 0:
        raise ValueError('Invalid error window, histogram limit or cache')
    if a.gpu_memory_gib is not None and a.gpu_memory_gib <= 0:
        raise ValueError('GPU memory cap must be positive')
    if a.layers is not None and a.layers < 1 or a.max_steps is not None and a.max_steps < 1:
        raise ValueError('Diagnostic limits must be positive')
    methods = a.methods.split(',')
    if not methods or len(set(methods)) != len(methods) or any(m not in ('wina', 'teal') for m in methods):
        raise ValueError('Methods must be distinct wina/teal entries')
    targets = sorted(set(float(s) for s in a.targets.split(',')))
    if not targets or any(not 0 < s < 1 for s in targets):
        raise ValueError('Targets must be in (0, 1)')
    for name in ('base_eval_root', 'source_model', 'converted_model', 'calibration_data'):
        if not getattr(a, name).exists():
            raise FileNotFoundError(getattr(a, name))
    return methods, targets


def export_strategy(out, method, metadata, targets, n_layers):
    layers = {}
    for path in sorted((out / method).glob('layer-*.json')):
        if not path.stem.removeprefix('layer-').isdigit():
            continue
        item = json.loads(path.read_text())
        layers[str(item['layer'])] = item
    profiles = []
    for target in targets:
        entries = {}
        for layer, result in layers.items():
            matches = [r for r in result['profiles'] if abs(r['target'] - target) < 1e-8]
            if matches:
                entries[layer] = matches[0]
        profiles.append(dict(target=target, layers=entries))
    complete = all(set(p['layers']) == {str(i) for i in range(n_layers)} for p in profiles)
    atomic_json(out / method / 'strategy.json', dict(metadata, schema=SCHEMA, method=method,
        n_layers=n_layers, transform=False, grouping=GROUPING, complete=complete, profiles=profiles))


def reference_batches(samples, search_samples, batch_size):
    """Match reference/candidate batch boundaries, including a partial last batch."""
    for begin, stop in ((0, search_samples), (search_samples, samples)):
        for start in range(begin, stop, batch_size):
            yield start, min(stop, start + batch_size)


def main(argv=None):
    a = parser().parse_args(argv)
    methods, targets = validate(a)
    sys.path.insert(0, str(a.base_eval_root.resolve()))
    from deepseek_v4_support import source_signature, runtime_signature, reference_config
    signature, external_signature = source_signature(a.source_model), runtime_signature()
    conversion = json.loads((a.converted_model / 'conversion.json').read_text())
    world = int(os.environ.get('WORLD_SIZE', conversion['world_size']))
    if not conversion['completed'] or conversion['source_signature'] != signature or conversion['world_size'] != world:
        raise ValueError('TP model provenance mismatch')
    for name, size in conversion['shard_bytes'].items():
        if (a.converted_model / name).stat().st_size != size:
            raise ValueError('TP model shard missing or incomplete')
    metadata = dict(schema=SCHEMA, source_signature=signature, runtime_signature=external_signature,
        world_size=world, quantization=a.quantization, samples=a.samples, search_samples=a.search_samples,
        search_batch_size=a.search_batch_size,
        length=a.length, seed=a.seed, targets=targets, methods=methods, step=a.step,
        last_fraction=a.last_fraction, histogram_sample_limit=a.histogram_sample_limit,
        layer_streaming=a.layer_streaming,
        data_sha256=digest(a.calibration_data), budget_metric='calibration-executed-linear-MACs',
        reference_step='attn.wq_b cost; inverse-cost increments',
        search_mask='native-threshold', evaluation_mask=dict(wina='topk', teal='threshold'),
        implementation_sha256={name: digest(ROOT / name) for name in
            ('benchmarks/deepseek_greedy_search.py', 'benchmarks/deepseek_greedy_strategy.py',
             'model/native_greedy.py', 'model/deepseek_greedy_runtime.py', 'model/deepseek_experiment.py')})
    if a.dry_run:
        print(json.dumps(dict(metadata, status='cpu_preflight_passed', gpu_execution=False), indent=2))
        return 0
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from deepseek_v4_support import load_reference_model, make_model_args
    from deepseek_v4_quant import DeepSeekV4QuantRuntime, install_reference_model_patches
    from run_deepseek_v4_baseline import setup_distributed
    from model.deepseek_experiment import install_experiment_patches, reference_histogram
    from model.deepseek_greedy_runtime import DeepSeekGreedyRuntimeMixin, reset_layer_state
    world, rank, local_rank = setup_distributed()
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(a.seed)
    device = torch.device('cuda', local_rank)
    if a.gpu_memory_gib is not None:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., a.gpu_memory_gib * 2**30 / total), device)
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'search_manifest.json'
    existing_error = [None]
    if rank == 0:
        if manifest_path.exists() and (not a.resume or json.loads(manifest_path.read_text()) != metadata):
            existing_error[0] = 'Existing output requires --resume with identical data/code/config'
    dist.broadcast_object_list(existing_error, src=0)
    if existing_error[0]:
        raise ValueError(existing_error[0])
    if rank == 0:
        atomic_json(manifest_path, metadata)
        for name in metadata['implementation_sha256']:
            snapshot=out/'sources'/name
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes((ROOT/name).read_bytes())
    dist.barrier()

    def status(phase, **details):
        if rank == 0:
            item = dict(phase=phase, pid=os.getpid(), updated_at=time.time(), **details)
            atomic_json(out / 'status.json', item)
            print('[greedy] ' + json.dumps(item, ensure_ascii=False), flush=True)

    status('loading_model')
    ds = load_reference_model(a.source_model / 'inference')
    class Runtime(DeepSeekGreedyRuntimeMixin, DeepSeekV4QuantRuntime):
        pass
    runtime = Runtime('bf16', act_quant=False, cache_gib=a.cache_gib)
    install_reference_model_patches(ds, runtime)
    install_experiment_patches(ds, runtime)
    config = make_model_args(ds, reference_config(a.source_model), max(a.length, 512))
    config.max_batch_size = min(a.search_samples, a.search_batch_size)
    with torch.device('cpu' if a.layer_streaming else device):
        model = ds.Transformer(config)
    missing, extra = load_model(model, str(a.converted_model / f'model{rank}-mp{world}.safetensors'), strict=False)
    if missing or any(not k.startswith('mtp.') for k in extra):
        raise ValueError(f'Model mismatch: {missing[:3]}, {extra[:3]}')
    model.eval()
    if not a.layer_streaming:
        runtime.configure(ds, model, collect=True)
    status('loading_calibration_text', layers=config.n_layers)
    tokenizer = AutoTokenizer.from_pretrained(str(a.source_model), trust_remote_code=True)
    data = load_dataset('parquet', data_files=str(a.calibration_data), split='train')
    ids = tokenizer.encode('\n\n'.join(data['text']), add_special_tokens=False)
    if len(ids) <= a.length:
        raise ValueError('Calibration corpus too short')
    rng = random.Random(a.seed)
    offsets = [rng.randrange(len(ids)-a.length) for _ in range(a.samples)]
    tokens = [torch.tensor([ids[n:n+a.length]], dtype=torch.long) for n in offsets]
    state_path = out / f'inputs.rank{rank}.pt'
    first_layer = 0
    with torch.inference_mode():
        if a.resume and state_path.exists():
            saved = torch.load(state_path, map_location='cpu', weights_only=True)
            first_layer, states = saved['next_layer'], saved['states']
            if saved['offsets'] != offsets:
                raise ValueError('Resume calibration samples mismatch')
        else:
            if a.layer_streaming:
                model.embed.to(device)
            states = [model.embed(t.to(device)).unsqueeze(2).repeat(1, 1, model.hc_mult, 1).cpu() for t in tokens]
            if a.layer_streaming:
                model.embed.cpu()
                torch.cuda.empty_cache()
        resume_layers = [None] * world
        dist.all_gather_object(resume_layers, first_layer)
        if len(set(resume_layers)) != 1:
            raise ValueError(f'Per-rank input checkpoints disagree: {resume_layers}; restore a consistent checkpoint')
        limit = min(config.n_layers, a.layers or config.n_layers)
        for layer_id in range(first_layer, limit):
            layer = model.layers[layer_id]
            if a.layer_streaming:
                status('loading_layer', layer=layer_id)
                layer.to(device)
                # Compressed weight.scale is a Python alias, not a buffer.
                # Rebind it after device moves to the registered scale parameter.
                for module in layer.modules():
                    if isinstance(module, ds.Linear) and module.scale is not None:
                        module.weight.scale = module.scale
                runtime.configure(ds, model, collect=True)
            def save_inputs(next_layer, values):
                temp = state_path.with_suffix('.tmp')
                torch.save(dict(next_layer=next_layer, states=values, offsets=offsets), temp)
                temp.replace(state_path)
            save_inputs(layer_id, states)
            runtime.start_collection(layer_id, a.histogram_sample_limit)
            layer_cal_path = out / 'calibration' / f'layer-{layer_id}.rank{rank}.pt'
            cost_path = out / 'calibration' / f'layer-{layer_id}.costs.json'
            layer_cal_path.parent.mkdir(exist_ok=True)
            reusable = [None] * world
            dist.all_gather_object(reusable, bool(a.resume and layer_cal_path.exists() and cost_path.exists()))
            if all(reusable):
                calibration = torch.load(layer_cal_path, map_location='cpu', weights_only=True)
                weights = json.loads(cost_path.read_text())['weights']
            else:
                status('collecting', layer=layer_id, samples=a.samples)
                for index, (state, t) in enumerate(zip(states, tokens)):
                    reset_layer_state(layer)
                    runtime.begin_forward(t.to(device))
                    layer(state.to(device), 0, t.to(device))
                    if (index+1) % 8 == 0 or index == 0:
                        status('collecting', layer=layer_id, completed=index+1, samples=a.samples)
                groups_local = {projection_group(k) for k in runtime.expected if k.startswith(f'layers.{layer_id}.')}
                runtime.supplementing = True
                supplemented = []
                for expert_id in range(layer.ffn.experts_start_idx, layer.ffn.experts_end_idx):
                    key = f'layers.{layer_id}.ffn.experts.{expert_id}.w2.weight'
                    if key not in runtime.observers:
                        layer.ffn.experts[expert_id](runtime.calibration_ffn_inputs[layer_id])
                        supplemented.append(expert_id)
                calibration = runtime.finish_layer_collection()
                missing = {k for k in runtime.expected if k.startswith(f'layers.{layer_id}.')} - set(calibration)
                if missing:
                    raise ValueError(f'Incomplete layer calibration: {sorted(missing)[:5]}')
                # Share the distribution of input-sharded projections across TP.
                for key in sorted(k for k in calibration if isinstance(runtime.modules.get(k), ds.RowParallelLinear)):
                    local = (torch.cat(runtime.observers[key].samples), torch.cat(runtime.weighted_samples[key]))
                    gathered = [None] * world
                    dist.all_gather_object(gathered, local)
                    calibration[key]['centers'], calibration[key]['counts'] = reference_histogram(torch.cat([g[0] for g in gathered]))
                    centers, counts = torch.unique(torch.cat([g[1].float() for g in gathered]), sorted=True, return_counts=True)
                    calibration[key].update(wina_centers=centers, wina_counts=counts.float())
                gathered = [None] * world
                dist.all_gather_object(gathered, (dict(runtime.costs), sorted(groups_local)))
                total_cost = Counter()
                groups = set()
                for costs, names in gathered:
                    total_cost.update(costs)
                    groups.update(names)
                weights = {k: float(total_cost[k]) for k in sorted(groups)}
                torch.save(calibration, layer_cal_path)
                if rank == 0:
                    atomic_json(cost_path, dict(weights=weights, budget_metric=metadata['budget_metric'],
                                                unobserved_protected_groups=[k for k,v in weights.items() if v == 0]))
                atomic_json(out/'calibration'/f'layer-{layer_id}.rank{rank}.coverage.json',
                    dict(supplemented_experts=supplemented, projections=len(calibration),
                         min_observed_rows=min(v['rows'] for v in calibration.values())))
            runtime.use_quantization(calibration, a.quantization)
            runtime.method = 'none'
            runtime.observers.clear()
            runtime.weighted_samples.clear()
            status('reference_outputs', layer=layer_id)
            outputs = []
            for begin, end in reference_batches(a.samples, a.search_samples, a.search_batch_size):
                t = torch.cat(tokens[begin:end]).to(device)
                reset_layer_state(layer)
                runtime.begin_forward(t)
                value = layer(torch.cat(states[begin:end]).to(device), 0, t)
                if not torch.isfinite(value).all():
                    raise ValueError('Non-finite unsparsified quantized reference')
                outputs.extend(value.cpu().split(1))
            reference_weight = weights.get(f'layers.{layer_id}.attn.wq_b') or max(weights.values())
            incomplete_methods = []
            for method in methods:
                destination = out / method / f'layer-{layer_id}.json'
                resumed = json.loads(destination.read_text()) if a.resume and destination.exists() else None
                if resumed and resumed.get('complete'):
                    continue
                evaluation_count = 0
                def evaluate(rates, mask_mode='threshold'):
                    nonlocal evaluation_count
                    runtime.set_search_rates(method, rates)
                    runtime.mask_mode = mask_mode
                    error = torch.zeros((), device=device, dtype=torch.float64)
                    for i in range(0, a.search_samples, a.search_batch_size):
                        end = min(a.search_samples, i + a.search_batch_size)
                        batch_tokens = torch.cat(tokens[i:end]).to(device)
                        reset_layer_state(layer)
                        runtime.begin_forward(batch_tokens)
                        actual = layer(torch.cat(states[i:end]).to(device), 0, batch_tokens)
                        error += native_activation_error(torch.cat(outputs[i:end]).to(device), actual, a.last_fraction).double() * (end - i)
                    dist.all_reduce(error)
                    evaluation_count += 1
                    result = float(error / world / a.search_samples)
                    if evaluation_count == 1 or evaluation_count % 8 == 0:
                        status('searching_candidates', layer=layer_id, method=method, mask_mode=mask_mode,
                               candidate_evaluations=evaluation_count, last_error=result)
                    return result
                zero_error = evaluate(dict.fromkeys(weights, 0.))
                reference_norm = sum(float(native_activation_error(torch.zeros_like(v), v, a.last_fraction))
                                     for v in outputs[:a.search_samples]) / a.search_samples
                if zero_error > max(1e-5, reference_norm * .01):
                    raise ValueError(f'Zero-sparsity replay differs excessively from reference: {zero_error}, norm={reference_norm}')
                if rank == 0:
                    atomic_json(out/method/f'layer-{layer_id}.replay.json',
                        dict(zero_sparsity_error=zero_error, reference_norm=reference_norm,
                             search_batch_size=a.search_batch_size, search_samples=a.search_samples))
                exported_profiles = -1
                def checkpoint(state):
                    nonlocal exported_profiles
                    state['layer'] = layer_id
                    if rank == 0:
                        atomic_json(destination, state)
                        # Full strategy changes only when a target is reached.
                        # Keep each-step checkpoints without rereading every
                        # completed layer on the shared filesystem every step.
                        if len(state['profiles']) != exported_profiles or state['complete']:
                            export_strategy(out, method, metadata, targets, config.n_layers)
                            exported_profiles = len(state['profiles'])
                    status('searching', layer=layer_id, method=method, iterations=state['iterations'],
                           effective_sparsity=state['curve'][-1]['effective_sparsity'] if state['curve'] else 0,
                           elapsed_seconds=state['elapsed_seconds'], complete=state['complete'])
                result = search(weights, evaluate, targets=targets, step=a.step,
                    reference_weight=reference_weight, checkpoint=checkpoint, resume=resumed, max_steps=a.max_steps)
                if method == 'wina':
                    # The upstream workflow searches thresholds but evaluates
                    # top-k. Check that actual online rule on the same block.
                    from benchmarks.deepseek_greedy_strategy import effective_sparsity
                    points = result['profiles'] or [dict(target=effective_sparsity(result['rates'], weights), rates=result['rates'])]
                    online_checks = []
                    for point in points:
                        target = point['target']
                        online_checks.append(dict(target=target,
                            greedy_error=evaluate(point['rates'], 'topk'),
                            uniform_error=evaluate({g: target if weights[g] else 0. for g in weights}, 'topk')))
                    if rank == 0:
                        atomic_json(out/method/f'layer-{layer_id}.online.json', dict(mask='topk', checks=online_checks))
                if not result['complete']:
                    incomplete_methods.append(method)
            if incomplete_methods:
                status('pilot_paused', layer=layer_id, methods=incomplete_methods,
                       note='Diagnostic limit reached; same config --resume can continue')
                dist.barrier()
                dist.destroy_process_group()
                return 0
            save_inputs(layer_id+1, outputs)
            states = outputs
            runtime.norms.clear()
            runtime.calibration_ffn_inputs.clear()
            runtime._clear_caches()
            if a.layer_streaming:
                layer.cpu()
            torch.cuda.empty_cache()
        full = limit == config.n_layers
        if full:
            combined = {}
            for i in range(config.n_layers):
                combined.update(torch.load(out/'calibration'/f'layer-{i}.rank{rank}.pt',map_location='cpu',weights_only=True))
            torch.save(combined, out/'calibration'/f'rank{rank}.pt')
            rank_hashes = [None] * world
            dist.all_gather_object(rank_hashes, digest(out/'calibration'/f'rank{rank}.pt'))
            dist.barrier()
            if rank == 0:
                atomic_json(out/'calibration'/'manifest.json',dict(metadata, dataset='WikiText-2 train',
                    offsets=offsets, arc_policy='per-input-channel absmax order; signed max/8 count rounded up to 64',
                    reference='unsparsified layer inputs with HiF4/ARC baseline propagation', complete=True,
                    calibration_rank_sha256=rank_hashes))
                for method in methods:
                    export_strategy(out, method, dict(metadata, calibration_rank_sha256=rank_hashes), targets, config.n_layers)
        status('completed' if full else 'diagnostic_layers_completed', layers=limit, full_model=full)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
