"""Print only complete, protocol-checked MAGI Tiny scores."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.magi_validation import validate_sparse_runtime, validate_tiny_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [("BF16", "none", 0., args.baseline_root / "bf16-50p5/tiny"),
            ("HiF4 W4A4 (no ARC)", "none", 0., args.baseline_root / "hifp4-w4a4-50p5/tiny")]
    rows += [("HiF4 W4A4 (no ARC)", method, sparsity, args.campaign / f"{method}-s{sparsity:g}-tiny/tiny")
             for sparsity in (.5, .6, .7, .8, .9) for method in ("wina", "teal")]
    print("模型 | benchmark/协议 | 精度 | 方法 | 稀疏度 | 总分 | 质量 | 语义 | 状态")
    print("--- | --- | --- | --- | --- | --- | --- | --- | ---")
    for precision, method, sparsity, folder in rows:
        try:
            scores = validate_tiny_scores(folder)
            if method != "none":
                validate_sparse_runtime(folder, method, sparsity)
            values = [f"{100 * scores[key]:.4f}%" for key in ("total", "quality", "semantic")]
            state = "完整 Tiny；非官方全集"
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            values = ["—"] * 3
            state = "待完成" if isinstance(exc, FileNotFoundError) else f"未通过校验：{exc}"
        label = f"{method} uniform / mask-only" if method != "none" else method
        print(" | ".join(["MAGI-2-preview", "VBench tiny16", precision, label,
                          f"{sparsity:.0%}", *values, state]))


if __name__ == "__main__":
    main()
