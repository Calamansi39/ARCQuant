#!/usr/bin/env bash
set -u

# Wait for a stable, genuinely free TP2 pair before starting the auxiliary
# native ARC greedy search. This does not reserve or kill other users' GPUs;
# it retries after a transient race or an OOM during model loading.
ROOT=/mnt/kpfs_qyang/LBC/ARCQuant
RUNS="$ROOT/runs/deepseek_greedy_aux_h20_2_20261106"
PY=/mnt/kpfs_qyang/zhaolianchao/miniconda3/envs/arcquant/bin/python
LOG="$RUNS/waiter.log"
mkdir -p "$RUNS"
exec >>"$LOG" 2>&1

free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | head -1
}

primary_ready() {
  python3 - <<'PY'
import json
from pathlib import Path
roots = [
    (Path('/mnt/kpfs_qyang/LBC/ARCQuant/runs/deepseek_method_split_20261105_2040/wina-search'), 'wina'),
    (Path('/mnt/kpfs_qyang/LBC/ARCQuant/runs/deepseek_greedy_race_20261106/teal-search'), 'teal'),
]
for root, method in roots:
    path = root / method / 'strategy.json'
    if path.is_file():
        try:
            if json.loads(path.read_text()).get('complete'):
                raise SystemExit(0)
        except (OSError, ValueError):
            pass
raise SystemExit(1)
PY
}

while :; do
  if primary_ready; then
    echo "$(date -Is) primary greedy is complete; tiny evaluation has priority"
    exit 0
  fi
  f5=$(free_mib 5 || echo 0)
  f6=$(free_mib 6 || echo 0)
  if [ "${f5:-0}" -ge 138000 ] && [ "${f6:-0}" -ge 138000 ]; then
    echo "$(date -Is) pair looks free; holding 90s for a stable check"
    sleep 90
    f5=$(free_mib 5 || echo 0)
    f6=$(free_mib 6 || echo 0)
    if [ "${f5:-0}" -ge 138000 ] && [ "${f6:-0}" -ge 138000 ]; then
      stamp=$(date +%Y%m%d_%H%M%S)
      out="$RUNS/attempt_$stamp"
      mkdir -p "$out"
      echo "$(date -Is) launching $out with free=${f5},${f6} MiB"
      "$PY" "$ROOT/benchmarks/queue_deepseek_greedy.py" \
        --gpu-ids 5,6 --out-dir "$out" \
        --base-eval-root /mnt/kpfs_qyang/zhaolianchao/base_eval \
        --source-model /mnt/kpfs_qyang/djs/models/DeepSeek-V4-Flash-0731 \
        --converted-model /mnt/kpfs_qyang/zhaolianchao/models/DeepSeek-V4-Flash-0731-mp2 \
        --calibration-data "$ROOT/assets/datasets/wikitext2/wikitext-2-raw-v1/train-00000-of-00001.parquet"
      rc=$?
      echo "$(date -Is) attempt exited rc=$rc"
      [ "$rc" -eq 0 ] && exit 0
      sleep 120
    fi
  fi
  sleep 30
done
