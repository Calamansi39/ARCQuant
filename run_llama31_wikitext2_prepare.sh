#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-7}"
MODEL="${MODEL:-${ARCQUANT_LLAMA_MODEL:-}}"
SAMPLES="${SAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
METRIC="${METRIC:-max}"

if [[ -z "${MODEL}" ]]; then
  echo "Set MODEL or ARCQUANT_LLAMA_MODEL to a local Llama model directory." >&2
  exit 2
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 python -u arc.py prepare \
  --model llama-3.1-8b \
  --model-path "${MODEL}" \
  --samples "${SAMPLES}" \
  --seqlen "${SEQLEN}" \
  --dataset wikitext2 \
  --act-sort-metric "${METRIC}"
