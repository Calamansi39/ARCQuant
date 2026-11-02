#!/usr/bin/env python3
"""Export the benchmark TSV as a colored, filterable Excel workbook.

This deliberately uses only the Python standard library.  The generated
``.xlsx`` file contains real Excel cell styles, so colors remain visible when
the file is opened in Excel, LibreOffice, or WPS; CSV/TSV cannot carry those
styles.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/completed_benchmarks.tsv"
DEFAULT_OUTPUT = ROOT / "docs/completed_benchmarks.xlsx"


def col_name(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def xml_text(value: object) -> str:
    return escape(str(value), {"\"": "&quot;", "'": "&apos;"})


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"empty result table: {path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def parse_number(value: str) -> int | float | None:
    if not value or not re.fullmatch(r"-?(?:\d+\.\d+|\d+)", value):
        return None
    return float(value) if "." in value else int(value)


def cell(ref: str, value: object, style: int, *, numeric: bool = False) -> str:
    if numeric:
        return f'<c r="{ref}" s="{style}"><v>{xml_text(value)}</v></c>'
    text = xml_text(value)
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def style_for_row(row: dict[str, str]) -> int:
    # 2 = base_eval blue, 3 = ARCQuant custom green.
    return 3 if row.get("evaluation_stack") == "ARCQuant custom" else 2


def style_for_cell(row: dict[str, str], field: str, default: int) -> int:
    value = row.get(field, "")
    if field == "precision":
        return {"BF16": 4, "NVFP4": 5, "NVFP4 W4A4": 5,
                "MXFP4": 6, "MXFP4 W4A4": 6,
                "HiFP4": 7, "HiFP4 W4A4": 7,
                "HiF4 W4A4": 7}.get(value, default)
    if field == "method":
        return {"TEAL": 8, "ARC + TEAL 50%": 8,
                "WINA": 9, "ARC + WINA 50%": 9}.get(value, default)
    if field == "status":
        return 10 if value == "completed" else 11
    return default


def worksheet_xml(headers: list[str], rows: list[dict[str, str]]) -> str:
    columns = len(headers)
    last_ref = f"{col_name(columns)}{len(rows) + 1}"
    body = []
    body.append("<row r=\"1\" ht=\"30\" customHeight=\"1\">")
    for index, header in enumerate(headers, 1):
        body.append(cell(f"{col_name(index)}1", header, 1))
    body.append("</row>")
    for row_number, row in enumerate(rows, 2):
        row_style = style_for_row(row)
        body.append(f'<row r="{row_number}">')
        for column_number, header in enumerate(headers, 1):
            value = row.get(header, "")
            style = style_for_cell(row, header, row_style)
            number = parse_number(value)
            body.append(cell(f"{col_name(column_number)}{row_number}",
                             number if number is not None else value, style,
                             numeric=number is not None))
        body.append("</row>")
    widths = {
        "model": 28, "benchmark": 23, "profile": 11, "variant": 18,
        "precision": 18, "method": 20, "evaluated_count": 15,
        "count_unit": 16, "metric": 18, "score": 12, "score_unit": 14,
        "protocol": 38, "evaluation_stack": 20, "status": 14,
        "source": 58, "notes": 42,
    }
    cols = []
    for index, header in enumerate(headers, 1):
        width = widths.get(header, max(14, min(45, len(header) + 3)))
        cols.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\""
            " xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            "<sheetViews><sheetView workbookViewId=\"0\"><pane ySplit=\"1\" topLeftCell=\"A2\""
            " activePane=\"bottomLeft\" state=\"frozen\"/><selection pane=\"bottomLeft\""
            " activeCell=\"A2\" sqref=\"A2\"/></sheetView></sheetViews>"
            f"<cols>{''.join(cols)}</cols><sheetData>{''.join(body)}</sheetData>"
            f'<autoFilter ref="A1:{last_ref}"/><pageMargins left="0.25" right="0.25"'
            ' top="0.5" bottom="0.5" header="0.3" footer="0.3"/></worksheet>')


def summary_xml() -> str:
    rows = [
        ("颜色", "含义"),
        ("蓝色行", "base_eval 独立量化基线"),
        ("绿色行", "ARCQuant custom 方法结果"),
        ("灰色精度", "BF16"),
        ("浅蓝精度", "NVFP4 / W4A4"),
        ("紫色精度", "MXFP4 / W4A4"),
        ("橙色精度", "HiFP4 / HiF4 W4A4"),
        ("黄色方法", "TEAL"),
        ("青色方法", "WINA"),
        ("绿色状态", "completed"),
    ]
    body = []
    for r, values in enumerate(rows, 1):
        body.append(f'<row r="{r}">')
        for c, value in enumerate(values, 1):
            style = 1 if r == 1 else (12 if c == 1 else 13)
            body.append(cell(f"{col_name(c)}{r}", value, style))
        body.append("</row>")
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
            f"<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews><sheetData>{''.join(body)}</sheetData>"
            "<cols><col min=\"1\" max=\"1\" width=\"18\" customWidth=\"1\"/>"
            "<col min=\"2\" max=\"2\" width=\"42\" customWidth=\"1\"/></cols>"
            "</worksheet>")


def styles_xml() -> str:
    fills = [
        '<fill><patternFill patternType="none"/></fill>',
        '<fill><patternFill patternType="gray125"/></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="DDEBF7"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="E2F0D9"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="E7E6E6"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="DDEBF7"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="E4DFEC"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="FCE4D6"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="FFF2CC"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="DDEBF7"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="C6E0B4"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="F4CCCC"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="FFF2CC"/><bgColor indexed="64"/></patternFill></fill>',
        '<fill><patternFill patternType="solid"><fgColor rgb="F2F2F2"/><bgColor indexed="64"/></patternFill></fill>',
    ]
    fonts = [
        '<font><sz val="11"/><name val="Calibri"/></font>',
        '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>',
    ]
    borders = '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
    xfs = ['<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>',
           '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" applyFont="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>']
    # 2-11: result rows and semantic cell overrides; 12-13: legend.
    for fill in range(2, 12):
        xfs.append(f'<xf numFmtId="0" fontId="0" fillId="{fill}" borderId="0" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>')
    for fill in (12, 13):
        xfs.append(f'<xf numFmtId="0" fontId="0" fillId="{fill}" borderId="0" applyFill="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>')
    # Keep score/count columns readable as numbers.
    xfs.append('<xf numFmtId="0" fontId="0" fillId="2" borderId="0" applyFill="1"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="0"/>'
            f'<fonts count="{len(fonts)}">{"".join(fonts)}</fonts>'
            f'<fills count="{len(fills)}">{"".join(fills)}</fills>{borders}'
            f'<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            f'<cellXfs count="{len(xfs)}">{"".join(xfs)}</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>')


def write_workbook(input_path: Path, output_path: Path) -> None:
    headers, rows = read_tsv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                     '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                     '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                     '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                     '</Types>')
    package_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                    '</Relationships>')
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="Results" sheetId="1" r:id="rId1"/><sheet name="Legend" sheetId="2" r:id="rId2"/></sheets>'
                '</workbook>')
    workbook_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                     '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                     '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
                     '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                     '</Relationships>')
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml(headers, rows))
        archive.writestr("xl/worksheets/sheet2.xml", summary_xml())
    print(f"wrote {output_path} ({len(rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_workbook(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
