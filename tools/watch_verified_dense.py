#!/usr/bin/env python3
"""Refresh the single verified report and worklog when dense progress changes."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.verified_results import main as publish
from tools.queue_deepseek_dense import complete, result_directory

START, END = '<!-- DS_DENSE_PROGRESS_START -->', '<!-- DS_DENSE_PROGRESS_END -->'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--notes', type=Path)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    plan = json.loads((a.run_dir / 'plan.json').read_text())
    previous = None
    with (a.run_dir / 'report-watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            rows = []
            for job in plan['jobs']:
                summary = json.loads((result_directory(job) / 'summary.json').read_text())
                rows.append((job['benchmark'], summary['completed_candidates'], job['expected'], complete(job)))
            signature = tuple(rows)
            if signature != previous:
                try:
                    publish(['--snapshot-only', '--refresh-dense-run', str(a.run_dir)])
                except ValueError as exc:
                    # Summary can finish before both rank execution records land.
                    print('Waiting for complete execution evidence:', exc, flush=True)
                else:
                    if a.notes:
                        text = a.notes.read_text()
                        block = START + '\n\n## DS ARC HiF4 dense 全量进度（自动更新）\n\n'
                        block += f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。\n\n'
                        block += '\n'.join(f'- {bench}：已保存 {n}/{total} 题，' +
                            ('完整已核查。' if done else '全量未完成，不报告中途准确率。') for bench,n,total,done in rows)
                        block += '\n\n其中 GPQA 32 题、MATH 100 题复用原有同协议 Tiny 答案。成绩只维护在 '
                        block += '[统一结果目录](ARCQuant/reports/verified/README.md)。\n\n' + END
                        if START in text and END in text:
                            begin, end = text.index(START), text.index(END) + len(END)
                            text = text[:begin] + block + text[end:]
                        else:
                            heading, rest = text.split('\n', 1)
                            text = heading + '\n\n' + block + '\n' + rest
                        temp = a.notes.with_suffix('.md.tmp')
                        temp.write_text(text)
                        temp.replace(a.notes)
                    previous = signature
            if a.once or (all(row[-1] for row in rows) and signature == previous):
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
