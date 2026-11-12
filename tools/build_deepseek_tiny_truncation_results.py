#!/usr/bin/env python3
"""Build the compact Tiny GPQA/MATH truncation table.

The table is intentionally separate from the ordinary all-question score
table.  Its accuracy denominator contains only records that ended with a
natural EOS; its truncation denominator is the planned question count.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_deepseek_tiny_results as base

SWEEP = base.SWEEP
DOCS = base.DOCS
METHODS = base.METHODS
RATES = base.RATES
BENCHMARKS = base.BENCHMARKS
LIMITS = base.LIMITS
HEADERS = [
    "模型", "稀疏方法", "稀疏度", "量化方式",
    "tinyGPQA 非截断准确率", "tinyGPQA 截断率",
    "tinyMATH 非截断准确率", "tinyMATH 截断率",
]
OUTPUT_STEM = "deepseek_tiny_truncation_results"
TABLE_TITLE = "DeepSeek Tiny GPQA/MATH 非截断准确率与截断率"


def _outputs(task_id: str):
    candidates = sorted((SWEEP / task_id).glob("arcquant-*/outputs.jsonl"))
    if len(candidates) > 1:
        raise ValueError(f"Multiple evaluator outputs for {task_id}: {candidates}")
    if not candidates:
        return []
    return [
        json.loads(line)
        for line in candidates[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary(task_id: str):
    candidates = sorted((SWEEP / task_id).glob("arcquant-*/summary.json"))
    if len(candidates) > 1:
        raise ValueError(f"Multiple evaluator summaries for {task_id}: {candidates}")
    if not candidates:
        return None
    return base.read(candidates[0])


def _task_status(task_id: str, benchmark: str, rows):
    """Return (complete, verified, planned, status_text)."""
    planned = LIMITS[benchmark]
    task = dict(id=task_id, method=task_id.split("-", 1)[0],
                sparsity=float(task_id.split("-")[1]), benchmark=benchmark,
                limit=planned)
    summary = _summary(task_id)
    if summary and summary.get("complete") is True:
        verified = base.summary_complete(
            SWEEP / task_id, task, max_new_tokens=8192)
        if verified:
            return True, True, planned, ""
        return False, False, planned, "待核查"
    if summary:
        return False, False, planned, f"未完成（{len(rows)}/{planned}）"
    state_path = SWEEP / "claims" / task_id / "state.json"
    state = base.read(state_path) if state_path.is_file() else {}
    phase = state.get("phase")
    if phase == "failed":
        return False, False, planned, f"失败（{len(rows)}/{planned}）"
    if rows or phase in ("running", "launching"):
        return False, False, planned, f"评测中（{len(rows)}/{planned}）"
    return False, False, planned, "待启动"


def _metric(task_id: str, benchmark: str):
    planned = LIMITS[benchmark]
    rows = _outputs(task_id)
    complete, _verified, planned, status = _task_status(task_id, benchmark, rows)
    if not complete:
        return status, status

    # The evaluator currently emits only `eos` and `length`; use the explicit
    # EOS marker so the first metric cannot accidentally include another stop
    # reason in a future evaluator version.
    eos_rows = [r for r in rows if r.get("generation", {}).get("finish_reason") == "eos"]
    truncated = sum(
        r.get("generation", {}).get("finish_reason") == "length" for r in rows
    )
    correct = sum(bool(r.get("score", {}).get("correct")) for r in eos_rows)
    if eos_rows:
        accuracy = f"{correct}/{len(eos_rows)}（{correct / len(eos_rows) * 100:.1f}%）"
    else:
        accuracy = "—（无自然EOS）"
    rate = f"{truncated}/{planned}（{truncated / planned * 100:.1f}%）"
    return accuracy, rate


def make_rows():
    rows = []
    for rate in RATES:
        for method in METHODS:
            metrics = []
            for benchmark, _ in BENCHMARKS:
                metrics.extend(_metric(f"{method.lower()}-{rate:g}-{benchmark}", benchmark))
            rows.append([
                "DeepSeek-V4-Flash-0731", f"{method}（greedy）", f"{rate:.0%}",
                "ARC HiF4 W4A4", metrics[0], metrics[1], metrics[2], metrics[3],
            ])
    return rows


def _write_csv_tsv(rows):
    for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
        with base.atomic_target(DOCS / f"{OUTPUT_STEM}.{suffix}") as target:
            with target.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
                writer.writerow(HEADERS)
                writer.writerows(rows)


def write_md(rows):
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [f"# {TABLE_TITLE}", "", f"截至：{timestamp}", "",
             "| " + " | ".join(HEADERS) + " |",
             "| " + " | ".join(["---"] * len(HEADERS)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend([
        "", "口径：非截断准确率只统计 `finish_reason=eos` 的题；截断率为 "
        "`finish_reason=length` 的题数/计划题数。自然 EOS 但未形成有效答案仍计入非截断准确率分母，"
        "并按评测器的正确性判定计分。`未完成` 行不是正式成绩。",
        "GPQA 为固定前 32 题，MATH 为固定前 100 题；两者均为 Tiny 筛查结果，不等于完整 benchmark。",
        "数据使用 ARC HiF4 W4A4、TEAL/WINA greedy 配置和 8192 token 生成上限。",
        "",
    ])
    with base.atomic_target(DOCS / f"{OUTPUT_STEM}.md") as target:
        target.write_text("\n".join(lines), encoding="utf-8")


def _cell_xml(ref: str, value: str, style: int) -> str:
    return (f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">'
            f'{escape(str(value))}</t></is></c>')


def write_xlsx(rows):
    """Write a colored XLSX that opens with formatting in Excel/WPS."""
    grid = [HEADERS, *rows]
    xml_rows = []
    for row_index, row in enumerate(grid, 1):
        cells = []
        for col_index, value in enumerate(row, 1):
            style = 1 if row_index == 1 else 0
            if row_index > 1:
                if col_index == 2:
                    style = 4 if value.startswith("TEAL") else 5
                elif col_index >= 5:
                    if value.startswith("未完成") or value.startswith("评测中") or value.startswith("失败"):
                        style = 6
                    elif value.startswith("待") or value.startswith("—"):
                        style = 7
                    else:
                        style = 2
            n, col = col_index, ""
            while n:
                n, rem = divmod(n - 1, 26)
                col = chr(65 + rem) + col
            cells.append(_cell_xml(f"{col}{row_index}", value, style))
        xml_rows.append(f'<row r="{row_index}" ht="32" customHeight="1">{"".join(cells)}</row>')

    widths = [27, 17, 10, 20, 27, 24, 27, 24]
    cols = "".join(f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>'
                   for i, width in enumerate(widths, 1))
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
             'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
             '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
             'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
             f'<cols>{cols}</cols><sheetData>{"".join(xml_rows)}</sheetData>'
             f'<autoFilter ref="A1:H{len(grid)}"/><pageMargins left="0.25" right="0.25" '
             'top="0.5" bottom="0.5" header="0.2" footer="0.2"/></worksheet>')
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <fonts count="2"><font><sz val="10"/><name val="Arial"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Arial"/></font></fonts>
 <fills count="9"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FF17324D"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDCEFE3"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDDEBF7"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFDDF2EE"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFE7E6E6"/><bgColor indexed="64"/></patternFill></fill>
  <fill><patternFill patternType="solid"><fgColor rgb="FFF4CCCC"/><bgColor indexed="64"/></patternFill></fill></fills>
 <borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border>
  <border><left style="thin"><color rgb="FFD9E2EA"/></left><right style="thin"><color rgb="FFD9E2EA"/></right><top style="thin"><color rgb="FFD9E2EA"/></top><bottom style="thin"><color rgb="FFD9E2EA"/></bottom><diagonal/></border></borders>
 <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
 <cellXfs count="8">
  <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"><alignment vertical="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="7" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
  <xf numFmtId="0" fontId="0" fillId="8" borderId="1" xfId="0"><alignment vertical="center" horizontal="center" wrapText="1"/></xf>
 </cellXfs>
 <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
    path = DOCS / f"{OUTPUT_STEM}.xlsx"
    with base.atomic_target(path) as target, ZipFile(target, "w", ZIP_DEFLATED) as archive:
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
 <sheets><sheet name="Tiny 截断统计" sheetId="1" r:id="rId1"/></sheets>
</workbook>''')
        archive.writestr("xl/_rels/workbook.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''')
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def write_all(rows=None):
    rows = make_rows() if rows is None else rows
    DOCS.mkdir(parents=True, exist_ok=True)
    _write_csv_tsv(rows)
    write_md(rows)
    write_xlsx(rows)
    return rows


def main():
    rows = write_all()
    print(f"Wrote {len(rows)} rows to {DOCS}")


if __name__ == "__main__":
    main()
