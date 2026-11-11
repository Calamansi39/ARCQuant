#!/usr/bin/env python3
"""Build the separate 40% DeepSeek tiny result table."""
from __future__ import annotations

import sys

import build_deepseek_tiny_results as base

base.SWEEP = base.ROOT / "runs/deepseek_40_tiny_sweep_20261111"
base.RATES = (0.4,)
base.OUTPUT_STEM = "deepseek_40_sparsity_results"
base.TABLE_TITLE = "DeepSeek 40% 稀疏度 Tiny Bench 结果"


def main():
    base.main()


if __name__ == "__main__":
    main()
