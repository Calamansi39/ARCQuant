#!/usr/bin/env bash
# Copy to external/local_env.sh, edit the paths for your machine, then run:
#   source external/local_env.sh
# Keep this file free of credentials and machine-specific values.

# Large files live outside Git. Omit this line to use this repo's assets/.
export ARCQUANT_ASSET_ROOT="/path/to/arcquant-assets"

# The pinned external evaluator is checked out by `arc.py provision --execute`.
# Point each variable at the actual environment on this machine.
export ARCQUANT_BASE_EVAL_ENV="/path/to/deepseek-eval-env"
export ARCQUANT_H3_PYTHON="/path/to/h3-env/bin/python"
export ARCQUANT_VBENCH_PYTHON="/path/to/vbench-env/bin/python"
export ARCQUANT_MAGI2_PYTHON="/path/to/magi2-env/bin/python"
export ARCQUANT_FFMPEG_BIN="/path/to/ffmpeg/bin"

# Optional if the official Hugging Face endpoint is unavailable.
# export HF_ENDPOINT="https://hf-mirror.com"
