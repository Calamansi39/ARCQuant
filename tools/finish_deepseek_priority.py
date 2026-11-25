#!/usr/bin/env python3
"""Score a finished extended job and publish only fully audited result cells."""
import argparse
import fcntl
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--base-eval',type=Path,required=True)
    p.add_argument('--stage',choices=('longbench','lcb64'))
    p.add_argument('--directory',type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.base_eval))
    if a.stage:
        if a.directory is None:p.error('stage requires directory')
        from lcb_protocol import atomic_json
        from benchmarks.verified_results import digest
        if a.stage=='longbench':
            from rescore_longbench_v2 import rescore_file
            rescore_file(a.directory/'hifp4-w4a4.jsonl',update_summary=True)
        else:
            raw=a.directory/'hifp4-w4a4.jsonl'
            atomic_json(a.directory/'arcquant_score_binding.json',dict(raw_sha256=digest(raw),
                summary_sha256=digest(str(raw)+'.summary.json')))
        from benchmarks.deepseek_arc_extended_audit import audit
        audit(a.directory,a.stage,a.base_eval)
    from benchmarks.verified_results import main as publish
    with (ROOT/'reports/verified/.update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        publish(['--snapshot-only','--base-eval',str(a.base_eval),'--refresh-priority-plan',str(a.plan)])


if __name__=='__main__':main()
