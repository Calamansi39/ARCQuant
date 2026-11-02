#!/usr/bin/env bash
# Wait for genuinely free H20s, then run the two missing DeepSeek MATH-500
# ARCQuant methods.  It never kills or shares an existing CUDA process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE_EVAL_ROOT="${ARCQUANT_BASE_EVAL_ROOT:-/mnt/kpfs_qyang/zhaolianchao/base_eval}"
SOURCE_MODEL="${ARCQUANT_DEEPSEEK_MODEL:-/mnt/kpfs_qyang/djs/models/DeepSeek-V4-Flash-0731}"
TP_MODEL="${ARCQUANT_DEEPSEEK_TP2:-/mnt/kpfs_qyang/zhaolianchao/models/DeepSeek-V4-Flash-0731-mp2}"
PYTHON="${ARCQUANT_BASE_EVAL_ENV:-/mnt/kpfs_qyang/zhaolianchao/miniconda3/envs/arcquant}/bin/python"
TORCHRUN="${ARCQUANT_BASE_EVAL_ENV:-/mnt/kpfs_qyang/zhaolianchao/miniconda3/envs/arcquant}/bin/torchrun"
CALIBRATION="${ARCQUANT_TEAL_CALIBRATION:-$ROOT/runs/deepseek_formal/calibration80}"
MIN_FREE_MIB="${ARCQUANT_MIN_FREE_MIB:-138000}"
POLL_SECONDS="${ARCQUANT_GPU_POLL_SECONDS:-60}"
LOG_DIR="${ARCQUANT_SPARSE_LOG_DIR:-$ROOT/runs/deepseek_sparse_queue}"
mkdir -p "$LOG_DIR"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v min="$MIN_FREE_MIB" '{gsub(/[[:space:]]/,"",$1); gsub(/[[:space:]]/,"",$2); gsub(/[[:space:]]/,"",$3); if ($2 >= min && $3 <= 5) print $1}'
}

wait_for_gpus() {
  local needed="$1" max_polls="${2:-0}" ids="" polls=0
  while :; do
    ids="$(free_gpus | head -n "$needed" | paste -sd, -)"
    if [ "$(printf '%s' "$ids" | awk -F, '{print NF}')" = "$needed" ]; then
      printf '%s\n' "$ids"
      return 0
    fi
    if [ "$max_polls" -gt 0 ] && [ "$polls" -ge "$max_polls" ]; then
      return 1
    fi
    polls=$((polls + 1))
    echo "[$(date -Is)] waiting for $needed free H20s (need >=${MIN_FREE_MIB} MiB free); available: ${ids:-none}" | tee -a "$LOG_DIR/queue.log" >&2
    sleep "$POLL_SECONDS"
  done
}

run_method() {
  local method="$1" pair="$2" out="$ROOT/runs/deepseek_formal/math500_arcquant_${method}"
  echo "[$(date -Is)] starting MATH-500 method=$method CUDA_VISIBLE_DEVICES=$pair" | tee -a "$LOG_DIR/queue.log"
  CUDA_VISIBLE_DEVICES="$pair" "$TORCHRUN" --standalone --nproc-per-node=2 \
    "$ROOT/benchmarks/run_deepseek_sparse_math.py" \
    --base-eval-root "$BASE_EVAL_ROOT" --sparse-method "$method" --sparsity 0.5 \
    --calibration "$CALIBRATION" --model deepseek --benchmark math500 \
    --source-model "$SOURCE_MODEL" --converted-model "$TP_MODEL" \
    --quant hifp4 --mode w4a4 --batch-size 20 --max-new-tokens 8192 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --n 1 --seed 20261011 \
    --limit -1 --out-dir "$out" \
    >"$LOG_DIR/math500_${method}.log" 2>&1
  echo "[$(date -Is)] finished MATH-500 method=$method" | tee -a "$LOG_DIR/queue.log"
}

if [ ! -x "$TORCHRUN" ] || [ ! -x "$PYTHON" ]; then
  echo "missing ARCQuant evaluation environment: $TORCHRUN" >&2
  exit 2
fi
for required in "$SOURCE_MODEL" "$TP_MODEL" "$CALIBRATION/rank0.pt" "$CALIBRATION/rank1.pt"; do
  [ -e "$required" ] || { echo "missing required asset: $required" >&2; exit 2; }
done

# Use four cards concurrently when available; fall back to two cards and run
# the methods sequentially when only one TP2 pair is free.
PAIRS="$(wait_for_gpus 4 10)" || true
if [ "$(printf '%s\n' "$PAIRS" | awk -F, 'END {print NF}')" = 4 ]; then
  IFS=, read -r g0 g1 g2 g3 <<<"$PAIRS"
  run_method wina "$g0,$g1" & p1=$!
  run_method teal "$g2,$g3" & p2=$!
  wait "$p1" "$p2"
else
  pair="$(wait_for_gpus 2)"
  run_method wina "$pair"
  pair="$(wait_for_gpus 2)"
  run_method teal "$pair"
fi
echo "[$(date -Is)] both MATH-500 sparse methods finished" | tee -a "$LOG_DIR/queue.log"
