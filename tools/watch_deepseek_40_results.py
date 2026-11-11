#!/usr/bin/env python3
"""Refresh the separate 40% result files until all four tasks finish."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import build_deepseek_40_results as report


def completed(rows):
    return sum("%（" in cell for row in rows for cell in row[4:6])


def main():
    root = report.base.SWEEP
    lock_path = root / "results-watch.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A 40% results watcher already holds the lock")
        previous = None
        while True:
            rows = report.base.make_rows()
            signature = json.dumps(rows, ensure_ascii=False)
            count = completed(rows)
            if signature != previous:
                report.base.write_text_tables(rows)
                report.base.write_xlsx(rows)
                previous = signature
                print(f"{datetime.now().astimezone().isoformat()} refreshed: {count}/4", flush=True)
            status = dict(schema="deepseek-40-results-watch-v1", phase="completed" if count == 4 else "watching",
                          checked_at=datetime.now(timezone.utc).isoformat(), completed=count, total=4,
                          sweep=str(root))
            (root / "results-watch.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
            if count == 4:
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
