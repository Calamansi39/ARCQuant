#!/usr/bin/env python3
"""Refresh this sweep's verified reports until all 20 evaluations complete.

CPU only. The existing queue owns GPU jobs and retries; this process only reads
its evidence and writes reports. It never changes generation or scoring.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import time

import build_deepseek_tiny_results as report
import build_deepseek_tiny_truncation_results as truncation_report

BEGIN = "<!-- deepseek-tiny-live:begin -->"
END = "<!-- deepseek-tiny-live:end -->"


def completed_count(rows):
    return sum("%（" in cell for row in rows for cell in row[4:6])


def warnings_for(rows):
    warnings = []
    for row in rows:
        method = row[1].split("（")[0].lower()
        rate = int(row[2].rstrip("%")) / 100
        for (benchmark, _), cell in zip(report.BENCHMARKS, row[4:6]):
            if "%（" in cell:
                continue
            task = f"{method}-{rate:g}-{benchmark}"
            path = report.SWEEP / "claims" / task / "state.json"
            if not path.exists():
                warnings.append(f"{task}：尚无任务状态，请检查队列。")
                continue
            state = report.read(path)
            if cell in ("待核查", "待修复") or state.get("phase") == "failed":
                warnings.append(f"{task}：失败或证据未通过核验，请检查调度器日志。")
            elif state.get("phase") in ("running", "launching"):
                updated = datetime.fromisoformat(state["updated_at"])
                if (datetime.now(timezone.utc) - updated).total_seconds() > 300:
                    warnings.append(f"{task}：调度心跳超过 5 分钟未更新，请核查原主机。")
                logs = list(path.parent.glob("attempt-*.log"))
                if logs and time.time() - max(p.stat().st_mtime for p in logs) > 1800:
                    warnings.append(f"{task}：日志超过 30 分钟无更新，请检查是否停滞。")
    return warnings


def update_notes(path, rows, warnings, interval):
    """Replace only the marked live block, retaining the explanatory history."""
    complete = completed_count(rows)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [BEGIN, "## DeepSeek 本轮 tiny 实时汇总", "",
             f"结果更新时间：{timestamp}。**已通过核验 {complete}/20 项。**", "",
             "本轮继续使用相同题目、ARC HiF4 W4A4、greedy 和 8192 token 上限；完整成绩沿用原评分规则。",
             "GPQA 32 题、MATH 100 题；未完成的任务只显示保存进度，不发布中途准确率。", "",
             "| " + " | ".join(report.HEADERS) + " |",
             "| " + " | ".join(["---"] * len(report.HEADERS)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    truncation_rows = truncation_report.make_rows()
    lines.extend(["", "### 非截断准确率与截断率简表", "",
                  "| " + " | ".join(truncation_report.HEADERS) + " |",
                  "| " + " | ".join(["---"] * len(truncation_report.HEADERS)) + " |"])
    lines.extend("| " + " | ".join(row) + " |" for row in truncation_rows)
    lines.extend(["", f"后台报告器每 {interval:g} 秒检查一次；题数、成绩或告警变化时刷新原有彩色 Excel、MD、CSV、TSV 和此区块。",
                  "非截断准确率只统计自然 EOS 题，截断率按计划题数计算；`未完成` 不作为正式成绩。",
                  "全部通过核验后报告器退出；GPU 调度与失败重试由原队列负责。报告器不自动提交或推送 Git。",
                  "报告器心跳和错误记录：`ARCQuant/runs/deepseek_tiny_sweep_20261106/results-watch.json`。"])
    if warnings:
        lines.extend(["", "当前告警：", "", *["- " + item for item in warnings]])
    lines.extend(["", END])
    block = "\n".join(lines)
    original = path.read_text(encoding="utf-8") if path.exists() else "# ARCQuant 工作记录与实验进度\n"
    if BEGIN in original or END in original:
        if original.count(BEGIN) != 1 or original.count(END) != 1:
            raise ValueError("Notes have ambiguous live-block markers; refusing to overwrite")
        start, stop = original.index(BEGIN), original.index(END) + len(END)
        if original.index(END) < start:
            raise ValueError("Notes live-block markers are reversed")
        updated = original[:start] + block + original[stop:]
    else:
        heading, _, body = original.partition("\n")
        updated = heading + "\n\n" + block + "\n\n" + body.lstrip("\n")
    # Refuse to clobber a concurrent edit noticed while preparing the block.
    if path.exists() and path.read_text(encoding="utf-8") != original:
        raise RuntimeError("Notes changed concurrently; will retry next poll")
    with report.atomic_target(path) as target:
        target.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, help="Optional shared work-record Markdown")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true", help="Refresh once without waiting")
    args = parser.parse_args()
    if not 5 <= args.interval <= 60:
        parser.error("--interval must be between 5 and 60 seconds")
    if not report.SWEEP.is_dir():
        parser.error(f"Sweep not present: {report.SWEEP}")
    with (report.SWEEP / "results-watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A report watcher already holds this sweep's lock")
        previous = None
        while True:
            state = dict(pid=os.getpid(), hostname=socket.gethostname(),
                         checked_at=datetime.now(timezone.utc).isoformat(),
                         interval_seconds=args.interval, phase="watching")
            try:
                rows = report.make_rows()
                warnings = warnings_for(rows)
                signature = json.dumps([rows, warnings], ensure_ascii=False)
                complete = completed_count(rows)
                if signature != previous:
                    report.write_text_tables(rows)
                    report.write_xlsx(rows)
                    truncation_report.write_all()
                    if args.notes:
                        update_notes(args.notes, rows, warnings, args.interval)
                    previous = signature
                    print(f"{state['checked_at']} refreshed: {complete}/20 verified; {warnings}", flush=True)
                state.update(completed=complete, total=20, warnings=warnings,
                             phase="completed" if complete == 20 else "watching")
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                # A JSONL append or summary replacement can overlap a read.
                # Keep the last verified report, record the problem, and retry.
                state.update(phase="read_error", error=f"{type(exc).__name__}: {exc}")
                print(json.dumps(state, ensure_ascii=False), flush=True)
            with report.atomic_target(report.SWEEP / "results-watch.json") as target:
                target.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            if args.once or state["phase"] == "completed":
                if args.once and state["phase"] == "read_error":
                    raise SystemExit(1)
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
