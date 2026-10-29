"""Calibrate and evaluate the whole DeepSeek model using local, pinned data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(path):
    p = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def gpqa_items(path, seed):
    """lm-eval gpqa_diamond_zeroshot v1 prompt and choice processing."""
    def clean(text):
        return re.sub(r'\[.*?\]', '', text.strip().replace(' [title]', '. ')).replace('  ', ' ')
    rng = random.Random(seed)
    items = []
    with open(path, newline='') as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            options = [clean(row[f'Incorrect Answer {j}']) for j in (1, 2, 3)] + [clean(row['Correct Answer'])]
            permutation = list(range(4))
            rng.shuffle(permutation)
            prompt = 'What is the correct answer to this question:' + row['Question'] + '\nChoices:\n'
            prompt += '\n'.join(f'({chr(65+j)}) {options[k]}' for j, k in enumerate(permutation))
            prompt += '\nAnswer:'
            items.append(dict(id=i, question_sha256=hashlib.sha256(row['Question'].encode()).hexdigest(),
                              prompt=prompt, target=permutation.index(3), permutation=permutation))
    return items


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['calibrate', 'evaluate'], required=True)
    p.add_argument('--base-eval-root', type=Path, required=True)
    p.add_argument('--source-model', type=Path, required=True)
    p.add_argument('--converted-model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--calibration-data', type=Path)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--calibration-length', type=int, default=256)
    p.add_argument('--benchmark', choices=['gpqa-diamond'], default='gpqa-diamond')
    p.add_argument('--data', type=Path)
    p.add_argument('--variant', choices=['bf16', 'hif4', 'arc', 'teal', 'wina'], default='arc')
    p.add_argument('--sparsity', type=float, default=.5)
    p.add_argument('--limit', type=int, default=-1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--cache-gib', type=float, default=8.)
    p.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.samples < 1 or args.calibration_length < 2 or args.max_length < args.calibration_length:
        raise ValueError('Invalid calibration/context lengths')
    if args.limit != -1 and args.limit < 1:
        raise ValueError('limit must be -1 or positive')
    if not 0 <= args.sparsity < 1 or args.cache_gib < 0:
        raise ValueError('Invalid sparsity/cache budget')
    if args.stage == 'calibrate' and not args.calibration_data:
        raise ValueError('--calibration-data must be a WikiText-2 train parquet')
    if args.stage == 'evaluate' and not args.data:
        raise ValueError('--data must be the GPQA Diamond CSV')
    sys.path.insert(0, str(args.base_eval_root.resolve()))
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from safetensors.torch import load_model
    from deepseek_v4_support import load_reference_model, make_model_args, reference_config, source_signature, runtime_signature
    from deepseek_v4_quant import DeepSeekV4QuantRuntime, install_reference_model_patches
    from run_deepseek_v4_baseline import setup_distributed
    from model.deepseek_experiment import ExperimentRuntimeMixin, install_experiment_patches, reference_histogram

    rank_file = args.calibration / f'rank{os.environ.get("RANK", "0")}.pt'
    source_sig = source_signature(args.source_model)
    runtime_sig = runtime_signature()
    metadata = dict(schema_version=1, model='deepseek-v4-flash', benchmark=args.benchmark,
                    source_signature=source_sig, runtime_signature=runtime_sig,
                    arcquant_revision=revision(ROOT), external_revision=revision(args.base_eval_root),
                    implementation_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                        (Path(__file__), ROOT/'model/deepseek_experiment.py')},
                    kernel='HiF4-value-reconstruction+dense-GEMM', seed=args.seed)
    # Validate source/TP signatures before allocating model parameters.
    conversion = json.loads((args.converted_model/'conversion.json').read_text())
    world_requested = int(os.environ.get('WORLD_SIZE', '1'))
    if conversion['source_signature'] != source_sig or conversion['world_size'] != world_requested or not conversion['completed']:
        raise ValueError('TP checkpoint provenance mismatch')
    for name, size in conversion['shard_bytes'].items():
        if (args.converted_model/name).stat().st_size != size:
            raise ValueError('TP checkpoint shard size mismatch')
    calibration = None
    if args.stage == 'evaluate' and args.variant in ('arc', 'teal', 'wina'):
        manifest = json.loads((args.calibration/'manifest.json').read_text())
        if manifest['source_signature'] != source_sig or manifest['world_size'] != world_requested:
            raise ValueError('Calibration belongs to another model/TP layout')
        if manifest['runtime_signature'] != runtime_sig:
            raise ValueError('Calibrate and evaluate with the same runtime implementation')
        calibration = torch.load(rank_file, map_location='cpu', weights_only=True)
        metadata['calibration_manifest_sha256'] = digest(args.calibration/'manifest.json')
    if args.stage == 'evaluate':
        items = gpqa_items(args.data, args.seed)
        if len(items) != 198:
            raise ValueError('Expected all 198 GPQA Diamond rows')
        items = items if args.limit == -1 else items[:args.limit]
        metadata.update(dataset_sha256=digest(args.data), protocol='gpqa_diamond_zeroshot_likelihood_v1',
                        prompt_template='lm-eval gpqa/zeroshot v1', target_delimiter=' ',
                        metric='accuracy', selection='CSV order, first N; seed controls choice permutation',
                        requested_questions=len(items), full_dataset=len(items)==198,
                        variant=args.variant, quant='bf16' if args.variant=='bf16' else 'HiF4-W4A4',
                        sparsity=args.sparsity if args.variant in ('teal', 'wina') else 0.,
                        arc=args.variant in ('arc', 'teal', 'wina'))
        if args.output.exists():
            raise FileExistsError(f'Refusing to overwrite {args.output}; choose a new output')
    world, rank, local_rank = setup_distributed()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    ds = load_reference_model(args.source_model/'inference')

    class Runtime(ExperimentRuntimeMixin, DeepSeekV4QuantRuntime):
        pass

    collecting = args.stage == 'calibrate'
    quant = 'bf16' if collecting or args.variant == 'bf16' else 'hifp4'
    runtime = Runtime(quant, act_quant=quant!='bf16', cache_gib=args.cache_gib)
    install_reference_model_patches(ds, runtime)
    install_experiment_patches(ds, runtime)
    model_args = make_model_args(ds, reference_config(args.source_model), args.max_length)
    model_args.max_batch_size = 1 if collecting else 4
    with torch.device('cuda'):
        model = ds.Transformer(model_args)
    missing, extra = load_model(model, str(args.converted_model/f'model{rank}-mp{world}.safetensors'), strict=False)
    if missing or any(not name.startswith('mtp.') for name in extra):
        raise ValueError(f'Checkpoint mismatch: {missing[:3]}, {extra[:3]}')
    model.eval()
    runtime.configure(ds, model, calibration=calibration, collect=collecting,
                      arc=not collecting and args.variant in ('arc', 'teal', 'wina'),
                      method=args.variant if args.variant in ('teal', 'wina') else 'none',
                      sparsity=args.sparsity)
    model.register_forward_pre_hook(lambda _m, inputs: runtime.begin_forward(inputs[0]))
    tok = AutoTokenizer.from_pretrained(str(args.source_model), trust_remote_code=True)
    device = torch.device('cuda', local_rank)
    started = time.monotonic()
    if rank == 0:
        print(f'[experiment] loaded stage={args.stage} variant={args.variant}', flush=True)
    with torch.inference_mode():
        if collecting:
            from datasets import load_dataset
            data = load_dataset('parquet', data_files=str(args.calibration_data), split='train')
            token_ids = tok.encode('\n\n'.join(data['text']), add_special_tokens=False)
            if len(token_ids) <= args.calibration_length:
                raise ValueError('Calibration corpus is too short')
            rng = random.Random(args.seed)
            offsets = [rng.randrange(len(token_ids)-args.calibration_length) for _ in range(args.samples)]
            for i, offset in enumerate(offsets):
                ids = torch.tensor([token_ids[offset:offset+args.calibration_length]], device=device)
                model(ids, 0)
                if rank == 0:
                    print(f'[calibration] {i+1}/{args.samples}, elapsed={time.monotonic()-started:.1f}s', flush=True)
            # Rare/unselected experts still need calibrated w2 inputs. Evaluate
            # only unseen experts on recorded real inputs from their own layer,
            # without feeding their outputs into the network or using test data.
            supplemented = []
            for layer in model.layers:
                ffn = layer.ffn
                for i in range(ffn.experts_start_idx, ffn.experts_end_idx):
                    key = f'layers.{layer.layer_id}.ffn.experts.{i}.w2.weight'
                    if key not in runtime.observers:
                        ffn.experts[i](runtime.calibration_ffn_inputs[layer.layer_id])
                        supplemented.append(key)
            artifacts = runtime.finish_calibration()
            # wo_b shards partition the input channels. TEAL uses a single
            # signed activation distribution across those channels.
            for key in sorted(k for k in artifacts if k.endswith('.attn.wo_b.weight')):
                local = torch.cat(runtime.observers[key].samples)
                gathered = [None] * world
                dist.all_gather_object(gathered, local)
                artifacts[key]['centers'], artifacts[key]['counts'] = reference_histogram(torch.cat(gathered))
            args.calibration.mkdir(parents=True, exist_ok=True)
            torch.save(artifacts, rank_file)
            report = dict(rank=rank, projections=len(artifacts), supplemented_experts=supplemented,
                          min_rows=min(a['rows'] for a in artifacts.values()),
                          max_residual_fraction=max(a['select']/len(a['order']) for a in artifacts.values()),
                          mean_residual_fraction=sum(a['select']/len(a['order']) for a in artifacts.values())/len(artifacts),
                          elapsed_seconds=time.monotonic()-started)
            reports = [None] * world
            dist.all_gather_object(reports, report)
            if rank == 0:
                manifest = dict(metadata, world_size=world, dataset='WikiText-2 train',
                                data_sha256=digest(args.calibration_data), offsets=offsets,
                                samples=args.samples, length=args.calibration_length,
                                reference='official checkpoint dequantized BF16',
                                arc_policy='per-input-channel absmax order; signed max/8 count rounded up to 64',
                                teal_policy='signed 10000-bin histogram, bounded deterministic sample per projection',
                                ranks=reports)
                atomic_json(args.calibration/'manifest.json', manifest)
                atomic_json(args.output, manifest)
        else:
            # All choices have the same token length for this tokenizer. One
            # batch evaluates four independent full contexts, avoiding any
            # reuse of mutated compression/KV state between answer candidates.
            choices = [tok.encode(' '+f'({c})', add_special_tokens=False) for c in 'ABCD']
            lengths = {len(c) for c in choices}
            if len(lengths) != 1:
                raise ValueError('This batched evaluator requires equal-length answer encodings')
            n = len(choices[0])
            ds._baseline_full_logits = True
            model.head.register_forward_pre_hook(lambda _m, inputs: (inputs[0][:, -n:], *inputs[1:]))
            results = []
            torch.cuda.reset_peak_memory_stats(device)
            for item in items:
                ctx = tok.encode(item['prompt'], add_special_tokens=False)
                if len(ctx)+n-1 > args.max_length:
                    raise ValueError(f'Question {item["id"]} exceeds context; increase --max-length (no silent truncation)')
                tokens = torch.tensor([ctx+c[:-1] for c in choices], device=device)
                torch.cuda.synchronize()
                before = time.monotonic()
                logits = model(tokens, 0).float()
                if logits.shape[:2] != (4, n) or not torch.isfinite(logits).all():
                    raise ValueError(f'Invalid logits shape/values: {logits.shape}')
                targets = torch.tensor(choices, device=device)
                scores = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).sum(-1)
                scores = scores.cpu().tolist()
                predicted = max(range(4), key=scores.__getitem__)
                elapsed = time.monotonic()-before
                results.append(dict(id=item['id'], question_sha256=item['question_sha256'],
                                    permutation=item['permutation'], target=item['target'], predicted=predicted,
                                    correct=predicted==item['target'], scores=scores,
                                    prompt_tokens=len(ctx), seconds=elapsed))
                if rank == 0:
                    correct = sum(r['correct'] for r in results)
                    print(f'[gpqa] {len(results)}/{len(items)} correct={correct} time={elapsed:.1f}s', flush=True)
                    atomic_json(args.output, dict(metadata, status='running', completed_questions=len(results),
                                                  correct=correct, accuracy=correct/len(results), rows=results))
            coverage_path = args.output.with_suffix(f'.rank{rank}.coverage.json')
            atomic_json(coverage_path, runtime.coverage())
            metrics = dict(rank=rank, peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
                           elapsed_seconds=time.monotonic()-started,
                           runtime=runtime.stats(), arc_cache_gib=runtime.arc_cache_bytes/2**30)
            metrics_all = [None] * world
            dist.all_gather_object(metrics_all, metrics)
            if rank == 0:
                correct = sum(r['correct'] for r in results)
                atomic_json(args.output, dict(metadata, status='completed', completed_questions=len(results),
                                              correct=correct, accuracy=correct/len(results), rows=results,
                                              ranks=metrics_all))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
