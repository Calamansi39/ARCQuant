"""Launch MiniMax H3 VBench with an ARCQuant WINA DiT wrapper.

The official ``base_eval/gen_vbench_minimax.py`` remains the owner of prompt
sampling, video caching and VBench protocol.  This launcher only replaces its
DiT ``QuantizedLinear`` class before import.  Use ``--dry-run`` on a new
machine to validate the external model and prompt assets without loading GPU
weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sparse-method", choices=("wina",), required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--base-eval-root", type=Path, required=True)
    sparse, remaining = parser.parse_known_args(argv)
    if not 0 <= sparse.sparsity < 1:
        parser.error("--sparsity must be in [0, 1)")
    sys.path.insert(0, str(sparse.base_eval_root.expanduser().resolve()))
    import fake_quant
    from collections import Counter
    from model.video_sparse_runtime import make_wina_quantized_linear

    stats = Counter()
    wrapped, _ = make_wina_quantized_linear(fake_quant.QuantizedLinear,
                                            sparse.sparsity, stats)
    fake_quant.QuantizedLinear = wrapped
    # The external script uses argparse directly. Add an isolated tag unless
    # the caller supplied one, keeping sparse and dense video caches separate.
    if "--tag" not in remaining:
        remaining += ["--tag", f"arcquant-wina-{sparse.sparsity:g}"]
    import gen_vbench_minimax as reference
    old_argv = sys.argv
    sys.argv = [str(Path(reference.__file__))] + remaining
    try:
        result = reference.main()
    finally:
        sys.argv = old_argv
    print(f"[arcquant] method=wina sparsity={sparse.sparsity:g} "
          f"masked_elements={stats.get('masked_elements', 0)} "
          f"total_elements={stats.get('total_elements', 0)}", flush=True)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
