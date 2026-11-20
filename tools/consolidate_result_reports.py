#!/usr/bin/env python3
"""Remove an explicit list of retired presentation files after audit succeeds.

Never traverses runs, assets, model weights, videos, or other users' base_eval.
Without --apply it only lists candidate files. Git retains tracked-file history.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.verified_results import DEFAULT_OUT, matrices, read

STEMS = ('completed_benchmarks', 'deepseek_40_sparsity_results',
         'deepseek_tiny_sparsity_results', 'deepseek_tiny_truncation_results')
NAMES = {f'{stem}.{ext}' for stem in STEMS for ext in ('md', 'xlsx', 'csv', 'tsv')}
NAMES.update({'completed_benchmarks_colored.html', 'completed_benchmarks_legacy.xlsx',
              'RESULTS_PRESENTATION.md', 'COMPLETED_BENCHMARKS.md', 'REPORTABLE_RESULTS_NOW.md',
              'MAGI_EVAL_STATUS_PLAN.md', 'MAGI_EVAL_STATUS_PLAN.csv', 'video_tiny_reference.csv',
              'CURRENT_EVAL_STATUS.md', 'BASELINE_PROGRESS.md', 'DEEPSEEK_FORMAL_RESULTS.md'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    snapshot = read(DEFAULT_OUT / 'audit.json')
    matrices(snapshot)
    if not (DEFAULT_OUT / '实验结果.xlsx').is_file() or not (DEFAULT_OUT / 'README.md').is_file():
        raise SystemExit('Build and validate the canonical workbook first')
    roots = [ROOT, *sorted((ROOT.parent / 'magi_work').glob('*/ARCQuant'))]
    candidates = [root / 'docs' / name for root in roots for name in sorted(NAMES)
                  if (root / 'docs' / name).is_file()]
    # A known derived snapshot outside docs; do not scan or touch the run artifacts.
    old_snapshot = ROOT.parent / 'magi_runs/10-0-2-54/validation-20261105/tiny-results-snapshot.md'
    if old_snapshot.is_file():
        candidates.append(old_snapshot)
    records = []
    for path in candidates:
        if not path.resolve().is_relative_to(ROOT.parent.resolve()):
            raise ValueError(f'Outside authorized workspace: {path}')
        records.append(dict(path=str(path.relative_to(ROOT.parent)), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(json.dumps(dict(apply=args.apply, count=len(records), files=[r['path'] for r in records]), ensure_ascii=False, indent=2))
    if args.apply:
        # Publish the deletion audit before unlinking any file.
        audit = DEFAULT_OUT / 'cleanup.json'
        previous = read(audit) if audit.exists() else {}
        known = {r['path']: r for r in previous.get('deleted', [])}
        known.update({r['path']: r for r in records})
        previous.update(at=datetime.now(timezone.utc).isoformat(), deleted=list(known.values()),
                        scope='Retired derived presentation files only; raw experiments retained')
        audit.write_text(json.dumps(previous,
                                    ensure_ascii=False, indent=2) + '\n')
        for path in candidates:
            path.unlink()


if __name__ == '__main__':
    main()
