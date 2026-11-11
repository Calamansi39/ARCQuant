#!/usr/bin/env python3
"""Build the standalone DeepSeek five-sparsity tiny-results table."""
from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.queue_deepseek_tiny_sweep import summary_complete

SWEEP = ROOT / "runs/deepseek_tiny_sweep_20261106"
DOCS = ROOT / "docs"
OUTPUT_STEM = "deepseek_tiny_sparsity_results"
TABLE_TITLE = "DeepSeek 五种稀疏度 Tiny Bench 结果"
HEADERS = ["模型", "稀疏方法", "稀疏度", "量化方式", "GPQA", "MATH", "LongBench", "LiveCodeBench"]
METHODS = ("TEAL", "WINA")
RATES = (0.5, 0.6, 0.7, 0.8, 0.9)
BENCHMARKS = (("gpqa", "GPQA"), ("math500", "MATH"))
LIMITS = {"gpqa": 32, "math500": 100}


def read(path: Path):
    return json.loads(path.read_text())


@contextmanager
def atomic_target(path: Path):
    """Readers see either the previous complete report or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def result_cell(task_id: str, method: str, rate: float, benchmark: str) -> str:
    task = dict(id=task_id, method=method.lower(), sparsity=rate,
                benchmark=benchmark, limit=LIMITS[benchmark])
    output = SWEEP / task_id
    summaries = sorted(output.glob("arcquant-*/summary.json"))
    if len(summaries) > 1:
        raise ValueError(f"Multiple evaluator outputs for {task_id}: {summaries}")

    claim = SWEEP / "claims" / task_id
    state_path = claim / "state.json"
    state = read(state_path) if state_path.is_file() else {}
    if summaries:
        summary = read(summaries[0])
        if summary.get("complete"):
            if not summary_complete(output, task, max_new_tokens=8192):
                return "待核查"
            count = int(summary["correct_candidates"])
            total = LIMITS[benchmark]
            score = count / total * 100
            truncation = int(summary.get("truncated", 0))
            cell = f"{score:.1f}%（{count}/{total}）"
            if truncation:
                cell += f"；截断{truncation}"
            return cell
        done = int(summary.get("completed_candidates", 0))
        outputs = summaries[0].with_name("outputs.jsonl")
        if outputs.is_file():
            ids = {json.loads(line)["id"] for line in outputs.read_text().splitlines() if line.strip()}
            done = max(done, len(ids))
        return f"评测中（{done}/{LIMITS[benchmark]}）"

    phase = state.get("phase")
    if phase in ("running", "launching"):
        outputs = sorted(output.glob("arcquant-*/outputs.jsonl"))
        if outputs:
            ids = {json.loads(line)["id"] for line in outputs[0].read_text().splitlines() if line.strip()}
            if ids:
                return f"评测中（{len(ids)}/{LIMITS[benchmark]}）"
        return f"评测中（0/{LIMITS[benchmark]}）"
    if phase == "failed":
        return "待修复"
    return "评测中（排队）"


def make_rows():
    rows = []
    for rate in RATES:
        for method in METHODS:
            row = ["DeepSeek-V4-Flash-0731", f"{method}（greedy）", f"{rate:.0%}",
                   "HiF4 W4A4 + ARC补偿"]
            for benchmark, _ in BENCHMARKS:
                row.append(result_cell(f"{method.lower()}-{rate:g}-{benchmark}",
                                       method, rate, benchmark))
            row.extend(["无", "无"])
            rows.append(row)
    return rows


def write_text_tables(rows):
    DOCS.mkdir(parents=True, exist_ok=True)
    for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
        with atomic_target(DOCS / f"{OUTPUT_STEM}.{suffix}") as target:
            with target.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
                writer.writerow(HEADERS)
                writer.writerows(rows)

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [f"# {TABLE_TITLE}", "", f"截至：{timestamp}", "",
             "| " + " | ".join(HEADERS) + " |",
             "| " + " | ".join(["---"] * len(HEADERS)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", "GPQA 为 Diamond 固定前 32 题，MATH 为 MATH-500 固定前 100 题。",
                  "全部任务使用 ARC HiF4 W4A4、8192 token 上限和生成式评分；Tiny 结果不代表全量 benchmark。",
                  "完整成绩经题目覆盖、协议、ARC 身份和两个 rank 的 greedy 策略证据核验。截断题保留在分母内并按评测规则计分。",
                  "LongBench 与 LiveCodeBench 不属于本轮 Tiny 横评，记为“无”。“评测中”后为已完成题数/计划题数；中途分数不作为准确率展示。",
                  ""])
    with atomic_target(DOCS / f"{OUTPUT_STEM}.md") as target:
        target.write_text("\n".join(lines), encoding="utf-8")


def cell_xml(ref: str, value: str, style: int) -> str:
    return (f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">'
            f'{escape(str(value))}</t></is></c>')


def write_xlsx(rows):
    """Write a small styled XLSX with stdlib so color works in Excel/WPS."""
    path = DOCS / f"{OUTPUT_STEM}.xlsx"
    grid = [HEADERS, *rows]
    row_xml = []
    for row_index, row in enumerate(grid, 1):
        cells = []
        for col_index, value in enumerate(row, 1):
            style = 1 if row_index == 1 else 0
            if row_index > 1:
                if col_index == 2:
                    style = 4 if value.startswith("TEAL") else 5
                elif col_index in (5, 6):
                    style = 6 if value in ("待核查", "待修复") else (3 if value.startswith("评测中") else 2)
                elif value == "无":
                    style = 6
            col = ""
            n = col_index
            while n:
                n, rem = divmod(n - 1, 26)
                col = chr(65 + rem) + col
            cells.append(cell_xml(f"{col}{row_index}", value, style))
        row_xml.append(f'<row r="{row_index}" ht="30" customHeight="1">{"".join(cells)}</row>')

    widths = [24, 17, 11, 24, 28, 28, 18, 20]
    cols = "".join(f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>'
                   for i, width in enumerate(widths, 1))
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
             'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
             '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
             'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
             f'<cols>{cols}</cols><sheetData>{"".join(row_xml)}</sheetData>'
             f'<autoFilter ref="A1:H{len(grid)}"/><pageMargins left="0.25" right="0.25" '
             'top="0.5" bottom="0.5" header="0.2" footer="0.2"/></worksheet>')
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <fonts count="2"><font><sz val="10"/><name val="Arial"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Arial"/></font></fonts>
 <fills count="8"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FF17324D"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDCEFE3"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDDEBF7"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDDF2EE"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFE7E6E6"/><bgColor indexed="64"/></patternFill></fill></fills>
 <borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border>
  <border><left style="thin"><color rgb="FFD9E2EA"/></left><right style="thin"><color rgb="FFD9E2EA"/></right><top style="thin"><color rgb="FFD9E2EA"/></top><bottom style="thin"><color rgb="FFD9E2EA"/></bottom><diagonal/></border></borders>
 <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
 <cellXfs count="7">
  <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"><alignment vertical="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="7" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
 </cellXfs>
 <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    with atomic_target(path) as target, ZipFile(target, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
 <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
 <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>''')
        archive.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''')
        archive.writestr("xl/workbook.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets><sheet name="DeepSeek Tiny" sheetId="1" r:id="rId1"/></sheets>
</workbook>''')
        archive.writestr("xl/_rels/workbook.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''')
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def main():
    rows = make_rows()
    write_text_tables(rows)
    write_xlsx(rows)
    print(f"Wrote {len(rows)} sparse settings to {DOCS}")


if __name__ == "__main__":
    main()
