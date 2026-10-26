#!/usr/bin/env bash
# Sourced by the pinned base_eval launcher from external/activate_arcquant.sh.
# An explicit environment prefix makes the copied evaluator independent of
# the original server user's conda installation and shell startup files.
if [ -n "${ARCQUANT_BASE_EVAL_ENV:-}" ]; then
    if [ ! -x "$ARCQUANT_BASE_EVAL_ENV/bin/python" ]; then
        echo "ARCQUANT_BASE_EVAL_ENV has no bin/python: $ARCQUANT_BASE_EVAL_ENV" >&2
        return 1
    fi
    export PATH="$ARCQUANT_BASE_EVAL_ENV/bin:$PATH"
    export CONDA_PREFIX="$ARCQUANT_BASE_EVAL_ENV"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || return $?
    conda activate arcquant || return $?
    export PATH="$CONDA_PREFIX/bin:$PATH"
else
    echo "Set ARCQUANT_BASE_EVAL_ENV to the installed DeepSeek evaluation environment" >&2
    return 1
fi
hash -r
