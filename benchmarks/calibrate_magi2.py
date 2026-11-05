"""Merge complete distributed MAGI calibration shards; never accept partial runs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.magi2_calibration import merge_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(merge_calibration(args.input, args.output, args.ranks), indent=2))


if __name__ == "__main__":
    main()
