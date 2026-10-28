# DeepSeek first real smoke run

This note records the first end to end checks on the DeepSeek V4 Flash TP2
checkpoint. The files under `runs/` are intentionally ignored by Git because
they contain machine paths and evaluator output; the commands below recreate
them on another machine.

## What was run

The official two way converted checkpoint was loaded on two H20 GPUs with the
pinned external evaluator. The first run used HiF4 W4A4 and one GPQA Diamond
question:

```bash
source external/local_env.sh
CUDA_VISIBLE_DEVICES=4,7 \
  "$ARCQUANT_BASE_EVAL_ENV/bin/torchrun" --standalone --nproc-per-node=2 \
  /path/to/base_eval/run_deepseek_v4_baseline.py \
  --source-model "$ARCQUANT_ASSET_ROOT/models/deepseek-v4-flash" \
  --converted-model "$ARCQUANT_ASSET_ROOT/models/deepseek-v4-flash-tp2" \
  --quant hifp4 --mode w4a4 --tasks gpqa_diamond_zeroshot \
  --limit 1 --no-ppl --max-gen-toks 8 --seqlen 512 \
  --context-length 2048 --cache-gib 4
```

On this server the result was saved as
`runs/deepseek_smoke/DeepSeek-V4-Flash-0731/gpqa/hifp4_gpqa_smoke_g47_20261029_003403.json`.
The single smoke question scored `acc=1.0` and `acc_norm=1.0`. This is only a
startup check, not a benchmark claim. Rank 0 reached a 79.64 GiB peak and the
runtime completed 31,485 weight-cache misses while streaming the routed
experts.

The first attempted pair used a busy GPU and failed with an out of memory
allocation while quantizing one expert. Selecting GPUs only after checking
their current free memory fixed that issue. A stable run therefore needs a
pair of cards with roughly 80 GiB or more free per rank, plus a conservative
`--cache-gib` value.

## Activation probe

The repository now contains a real activation diagnostic:

```bash
python arc.py probe \
  --model deepseek-v4-flash \
  --base-eval-root /path/to/base_eval \
  --converted-model "$ARCQUANT_ASSET_ROOT/models/deepseek-v4-flash-tp2" \
  --gpu-ids 4,7 \
  --output runs/deepseek_smoke/activation_probe.json \
  --max-length 64 --sample-limit 2048 --quant hifp4 --cache-gib 2
```

It executes a real official-model forward pass. It does not save full
activations. For 425 attention and shared-expert projections on each TP rank,
it stores bounded samples and reports the element count, exact zero fraction,
mean/RMS/max, p50/p90/p99 absolute values, and two diagnostic masks:

* TEAL: an absolute activation threshold chosen at the requested target
  sparsity;
* WINA: a per-token top-k mask weighted by the projection's input-column
  weight norm when that norm can be decoded safely.

The first HiF4 probe used a 29-token prompt. All values were finite. Across
both ranks, the median projection maximum was 2.28 and the largest maximum was
88; at 50% target sparsity the median removed input L2 fraction was about
0.25 for both masks. A BF16 control probe had a median maximum of 2.30 and a
largest maximum of 98, with a median HiF4/BF16 RMS ratio of 1.00. These are
reasonable diagnostics, but the prompt is too short to choose final TEAL
histograms or claim an accuracy result.

The JSON schema is deliberately independent of the external evaluator. The
next implementation step is to replace these diagnostic thresholds with
calibration files collected over a representative corpus, then install the
same masks in the official DeepSeek projection calls and compare full-model
logits against the BF16 control.

## What the first sparse pass tells us

The two-stage probe was also run with a 50% target sparsity. It applied the
mask to all 425 observed attention/shared-expert projections and compared the
full last-token vocabulary logits before and after masking:

| diagnostic | relative logit L2 change | KL (dense to sparse) | greedy token |
|---|---:|---:|---|
| TEAL, 25% | 22.7% | 0.175 | changed (223 → 539) |
| TEAL, 50% | 27.6% | 0.478 | changed (223 → 539) |
| WINA, 50% | 32.1% | 0.841 | changed (223 → 343) |

The values are finite and the process is stable, but the quality change is too
large for a final experiment. This is an expected diagnostic outcome: the
probe currently zeros masked inputs and does not yet add ARC residual channels
or use corpus-level calibration. We therefore treat these runs as a concrete
debug baseline, not as TEAL/WINA benchmark results. The next implementation
must calibrate on representative prompts, reserve and reorder ARC channels, and
then repeat this exact logit comparison before spending time on full
benchmarks.
