#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
SAMPLES="${SAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
METRIC="${METRIC:-max}"

mkdir -p /mnt/data2/lbc/ARCQuant/saved
cd /mnt/data2/lbc/ARCQuant
CUDA_VISIBLE_DEVICES="${GPU}" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 python -u reorder_indices.py \
  --model "${MODEL}" \
  --samples "${SAMPLES}" \
  --seqlen "${SEQLEN}" \
  --dataset wikitext2 \
  --act_sort_metric "${METRIC}"
