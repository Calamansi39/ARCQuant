"""Run the pinned DeepSeek extended evaluator with portable activation.

Only the evaluator's hardcoded conda activation is changed in memory. The
external checkout remains clean and its benchmark protocol is untouched.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


PINNED_SHA256 = "cefbf62c4db568470fbece5a5894e21a4738f4c5301e83ae5ec8e49f56ef3ea7"
ORIGINAL = "source activate arcquant 2>/dev/null || conda activate arcquant"
PORTABLE = 'source "$PROJECT_ROOT/activate_arcquant.sh"'


def launch(backend_root: Path, arguments: list[str]) -> int:
    script = backend_root / "scripts/run_all_deepseek_v4_extended.sh"
    if not script.is_file():
        raise FileNotFoundError(f"Pinned evaluator launcher missing: {script}")
    source = script.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINNED_SHA256:
        raise ValueError(f"Pinned DeepSeek launcher has changed: {script}")
    code = source.decode("utf-8")
    if code.count(ORIGINAL) != 1:
        raise ValueError("Expected exactly one DeepSeek environment activation line")
    # With `bash -c`, the first argument becomes $0, retaining the original
    # script's `cd "$(dirname "$0")/.."` behavior.
    return subprocess.run(["bash", "-c", code.replace(ORIGINAL, PORTABLE),
                           str(script), *arguments], cwd=backend_root, check=False).returncode


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("Usage: launch_deepseek_extended.py BASE_EVAL_ROOT TASK GPU_IDS [options]")
    raise SystemExit(launch(Path(sys.argv[1]).expanduser().resolve(), sys.argv[2:]))
