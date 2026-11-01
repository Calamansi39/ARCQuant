"""Launch MAGI-2 VBench with the supported ARCQuant WINA components.

The pinned MAGI-2 evaluator remains responsible for distributed setup,
conditioning, denoising and VBench files.  The wrapper installs WINA on dense
linear modules and routed preview experts.  Grouped refiner linears are listed
as pending in the quantization report until their dispatcher contract is
calibrated on the target MAGI-2 revision.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import argparse

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
    # External checkouts are often unpacked by a different service account on
    # shared servers.  This scoped git setting avoids changing the user's
    # global git configuration while allowing the pinned revision check.
    repo_hint = next((remaining[i + 1] for i, value in enumerate(remaining[:-1])
                      if value == "--repo"), "*")
    git_slot = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    os.environ["GIT_CONFIG_COUNT"] = str(git_slot + 1)
    os.environ[f"GIT_CONFIG_KEY_{git_slot}"] = "safe.directory"
    os.environ[f"GIT_CONFIG_VALUE_{git_slot}"] = (
        str(Path(repo_hint).expanduser().resolve()) if repo_hint != "*" else "*")
    import magi2_quant
    from model.magi2_sparse_runtime import install_wina

    original_apply = magi2_quant.apply_magi2_quant

    def apply(model, quant, act_quant=True, device="cpu"):
        report = original_apply(model, quant, act_quant, device)
        report["arcquant_sparse"] = install_wina(model, sparse.sparsity)
        return report

    magi2_quant.apply_magi2_quant = apply
    if "--tag" not in remaining:
        remaining += ["--tag", f"arcquant-wina-{sparse.sparsity:g}"]
    import gen_vbench_magi2 as reference
    old_argv = sys.argv
    sys.argv = [str(Path(reference.__file__))] + remaining
    try:
        result = reference.main()
    finally:
        sys.argv = old_argv
        magi2_quant.apply_magi2_quant = original_apply
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
