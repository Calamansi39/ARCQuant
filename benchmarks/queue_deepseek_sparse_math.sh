#!/usr/bin/env bash
# Wait for genuinely free H20s, then run DeepSeek MATH-500 ARCQuant methods at
# one explicit sparsity point. It never kills or shares an existing CUDA process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE_EVAL_ROOT="${ARCQUANT_BASE_EVAL_ROOT:-/mnt/kpfs_qyang/zhaolianchao/base_eval}"
SOURCE_MODEL="${ARCQUANT_DEEPSEEK_MODEL:-/mnt/kpfs_qyang/djs/models/DeepSeek-V4-Flash-0731}"
TP_MODEL="${ARCQUANT_DEEPSEEK_TP2:-/mnt/kpfs_qyang/zhaolianchao/models/DeepSeek-V4-Flash-0731-mp2}"
PYTHON="${ARCQUANT_BASE_EVAL_ENV:-/mnt/kpfs_qyang/zhaolianchao/miniconda3/envs/arcquant}/bin/python"
TORCHRUN="${ARCQUANT_BASE_EVAL_ENV:-/mnt/kpfs_qyang/zhaolianchao/miniconda3/envs/arcquant}/bin/torchrun"
CALIBRATION="${ARCQUANT_TEAL_CALIBRATION:-$ROOT/runs/deepseek_formal/calibration80}"
SPARSITY="${ARCQUANT_SPARSITY:-0.5}"
LIMIT="${ARCQUANT_LIMIT:--1}"
MAX_NEW_TOKENS="${ARCQUANT_MAX_NEW_TOKENS:-8192}"
BATCH_SIZE="${ARCQUANT_BATCH_SIZE:-20}"
MIN_FREE_MIB="${ARCQUANT_MIN_FREE_MIB:-138000}"
POLL_SECONDS="${ARCQUANT_GPU_POLL_SECONDS:-60}"
# A short second observation reduces the chance that a newly starting external
# process wins the race between nvidia-smi and torchrun.  It cannot reserve
# GPUs from unrelated users, so the launch is still checked immediately before
# each method.
SETTLE_SECONDS="${ARCQUANT_GPU_SETTLE_SECONDS:-3}"
FOUR_GPU_WAIT_POLLS="${ARCQUANT_FOUR_GPU_WAIT_POLLS:-1}"
LOG_DIR="${ARCQUANT_SPARSE_LOG_DIR:-$ROOT/runs/deepseek_sparse_queue}"
FIXED_GPU_IDS="${ARCQUANT_GPU_IDS:-}"
mkdir -p "$LOG_DIR"

free_gpus() {
  local id free util apps
  while IFS=, read -r id free util; do
    id="${id//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [ -n "$id" ] && [ -n "$free" ] && [ -n "$util" ] || continue
    # A card can briefly report high free memory while a process is creating
    # its CUDA context.  Exclude cards with any compute process explicitly.
    apps="$(nvidia-smi -i "$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [ -z "$apps" ] || continue
    [ "$free" -ge "$MIN_FREE_MIB" ] && [ "$util" -le 5 ] && echo "$id"
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
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

pair_available() {
  local pair="$1" id apps line free util
  IFS=, read -r -a ids <<<"$pair"
  for id in "${ids[@]}"; do
    apps="$(nvidia-smi -i "$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [ -z "$apps" ] || return 1
    line="$(nvidia-smi -i "$id" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
    free="${line%%,*}"
    util="${line##*,}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [ -n "$free" ] && [ -n "$util" ] || return 1
    [ "$free" -ge "$MIN_FREE_MIB" ] && [ "$util" -le 5 ] || return 1
  done
  return 0
}

reserve_pair() {
  local pair
  while :; do
    pair="$(wait_for_gpus 2)"
    # Observe twice so a process that appeared just after wait_for_gpus is
    # caught before model construction allocates tens of GiB.
    if pair_available "$pair"; then
      [ "$SETTLE_SECONDS" -gt 0 ] && sleep "$SETTLE_SECONDS"
    fi
    if pair_available "$pair"; then
      printf '%s\n' "$pair"
      return 0
    fi
    echo "[$(date -Is)] candidate pair $pair was occupied before launch; retrying" | tee -a "$LOG_DIR/queue.log" >&2
    sleep 2
  done
}

run_method() {
  local method="$1"
  local pair="$2" tag="${SPARSITY#0.}"
  local out="$ROOT/runs/deepseek_formal/math500_arcquant_${method}"
  if [ "$SPARSITY" != "0.5" ]; then
    out="${out}_sparsity_${tag}"
  fi
  echo "[$(date -Is)] starting MATH-500 method=$method sparsity=$SPARSITY CUDA_VISIBLE_DEVICES=$pair" | tee -a "$LOG_DIR/queue.log"
  CUDA_VISIBLE_DEVICES="$pair" "$TORCHRUN" --standalone --nproc-per-node=2 -- \
    "$ROOT/benchmarks/run_deepseek_sparse_math.py" \
    --base-eval-root "$BASE_EVAL_ROOT" --sparse-method "$method" --sparsity "$SPARSITY" \
    --calibration "$CALIBRATION" --model deepseek --benchmark math500 \
    --source-model "$SOURCE_MODEL" --converted-model "$TP_MODEL" \
    --quant hifp4 --mode w4a4 --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 --top-p 0.95 --top-k 20 --n 1 --seed 20261011 \
    --limit "$LIMIT" --out-dir "$out" \
    >"$LOG_DIR/math500_${method}_s${tag}.log" 2>&1
  echo "[$(date -Is)] finished MATH-500 method=$method sparsity=$SPARSITY" | tee -a "$LOG_DIR/queue.log"
}

if [ ! -x "$TORCHRUN" ] || [ ! -x "$PYTHON" ]; then
  echo "missing ARCQuant evaluation environment: $TORCHRUN" >&2
  exit 2
fi
for required in "$SOURCE_MODEL" "$TP_MODEL" "$CALIBRATION/rank0.pt" "$CALIBRATION/rank1.pt"; do
  [ -e "$required" ] || { echo "missing required asset: $required" >&2; exit 2; }
done

# If a caller has reserved a known pair, use it sequentially.  This avoids a
# race with another scheduler between nvidia-smi's free-memory check and
# torchrun startup.  The caller should set ARCQUANT_GPU_IDS only after checking
# that the pair has no compute processes.
if [ -n "$FIXED_GPU_IDS" ]; then
  if ! pair_available "$FIXED_GPU_IDS"; then
    echo "requested ARCQUANT_GPU_IDS=$FIXED_GPU_IDS is already occupied" >&2
    exit 3
  fi
  run_method wina "$FIXED_GPU_IDS"
  run_method teal "$FIXED_GPU_IDS"
  echo "[$(date -Is)] both MATH-500 sparse methods finished" | tee -a "$LOG_DIR/queue.log"
  exit 0
fi

# Use four cards concurrently when available; fall back to two cards and run
# the methods sequentially when only one TP2 pair is free.
PAIRS="$(wait_for_gpus 4 "$FOUR_GPU_WAIT_POLLS")" || true
if [ "$(printf '%s\n' "$PAIRS" | awk -F, 'END {print NF}')" = 4 ]; then
  IFS=, read -r g0 g1 g2 g3 <<<"$PAIRS"
  [ "$SETTLE_SECONDS" -gt 0 ] && sleep "$SETTLE_SECONDS"
  if pair_available "$g0,$g1" && pair_available "$g2,$g3"; then
    run_method wina "$g0,$g1" & p1=$!
    run_method teal "$g2,$g3" & p2=$!
    wait "$p1" "$p2"
  else
    echo "[$(date -Is)] four-card candidate was occupied before launch; falling back to reserved pairs" | tee -a "$LOG_DIR/queue.log"
    pair="$(reserve_pair)"
    run_method wina "$pair"
    pair="$(reserve_pair)"
    run_method teal "$pair"
  fi
else
  pair="$(reserve_pair)"
  run_method wina "$pair"
  pair="$(reserve_pair)"
  run_method teal "$pair"
fi
echo "[$(date -Is)] both MATH-500 sparse methods finished" | tee -a "$LOG_DIR/queue.log"
