#!/usr/bin/env python3
"""Bind complete scoring outputs, audit them, then update the canonical tables."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-eval', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('longbench','lcb64','lcb128'), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.base_eval))
    from lcb_protocol import atomic_json
    from benchmarks.verified_results import digest, main as publish
    directory = args.run_dir/args.stage
    if args.stage == 'longbench':
        from rescore_longbench_v2 import rescore_file
        rescore_file(directory/'hifp4-w4a4.jsonl', update_summary=True)
    else:
        name = 'hifp4' if args.stage == 'lcb128' else 'hifp4-w4a4'
        raw = directory/(name+'.jsonl')
        summary = Path(str(raw)+'.summary.json')
        atomic_json(directory/'arcquant_score_binding.json',
            dict(raw_sha256=digest(raw), summary_sha256=digest(summary)))
    publish(['--snapshot-only', '--base-eval', str(args.base_eval),
             '--refresh-extended-run', str(args.run_dir)])


if __name__ == '__main__':
    main()
