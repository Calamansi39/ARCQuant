#!/usr/bin/env python3
"""Refresh the single canonical workbook; all inputs are existing artifacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.verified_results import main

if __name__ == '__main__':
    raise SystemExit(main())
