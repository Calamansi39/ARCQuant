"""Compact result tables with separate generation and likelihood rows."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPARSITIES = (.5, .6, .7, .8, .9)
HEADERS = ['模型', 'Benchmark', '评测口径', '题数', '目标稀疏度 (%)',
           'BF16 (0%)', 'HiF4 基础量化 (0%)', 'HiF4 + ARC补偿 (0%)', 'TEAL (%)', 'WINA (%)']
NOTES = [
    '分数单位为 %。目标稀疏度只针对 TEAL/WINA；BF16、HiF4、ARC 对照均为 0%。重复的 MATH 基线复用同一份结果。',
    '两列均使用 HiF4 W4A4。HiF4 基础量化不含 ARC；HiF4 + ARC补偿额外启用校准通道重排和增强残差通道。hifp4/HiFP4 是外部评测器对 HiF4 的命名，不是另一种格式。',
    '当前生成式 TEAL/WINA 路径是 HiF4 基础量化 + 稀疏掩码，未启用 ARC 补偿；GPQA 标签似然诊断的 TEAL/WINA 则另含 ARC，WINA 为 mask-only。两种路径不能混报。',
    '50%–90% 是非结构化激活置零率目标，不是 2:4 或层覆盖率。未完成项目显示进度，不用部分题目计算最终分数。',
    'LiveCodeBench 为固定 test6 分片 175 题；128K 是 64K 到限题续写。生成式基线为本项目固定协议，不是官方榜单复现。',
    '蓝色＝完整生成式基线；紫色＝完整标签似然诊断；黄色＝待完成。其他 benchmark 暂列 50% 槽位，60%–90% 当前只登记 MATH 网格。',
]


def read(path):
    return json.loads(Path(path).read_text())


def source_label(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def math_progress(path, snapshot):
    """Publish a score only after complete coverage and protocol checks."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    summary = json.loads(raw)
    runtime = read(path.with_name('arcquant_sparse_runtime.json'))
    method, sparsity = runtime['method'], runtime['sparsity']
    if method not in ('teal', 'wina') or sparsity not in SPARSITIES:
        raise ValueError('Unsupported sparse method/rate in ' + str(path))
    if summary['benchmark'] != 'math500' or summary['quant'] != 'hifp4' or summary['questions'] != 500:
        raise ValueError('Expected HiF4 MATH-500 summary')
    completed = summary['completed_candidates']
    if not 0 <= completed <= 500:
        raise ValueError('Invalid completed question count')
    entry = dict(benchmark='MATH-500', method=method, sparsity=sparsity,
                 completed=completed, expected=500, score=None, status='未完成',
                 source=source_label(path), summary_sha256=hashlib.sha256(raw).hexdigest())
    if not summary['complete']:
        return entry
    if completed != 500 or not summary['is_full_dataset'] or summary['is_smoke_test']:
        raise ValueError('Complete flag disagrees with full coverage')
    protocol = read(path.with_name('protocol.json'))
    outputs = path.parent / summary.get('scored_outputs_path', 'outputs.jsonl')
    rows = [json.loads(line) for line in outputs.read_text().splitlines() if line.strip()]
    ids = [str(row['id']) for row in rows]
    if (len(ids) != 500 or len(set(ids)) != 500 or
            set(ids) != set(map(str, protocol['selected_ids'])) or
            protocol['dataset']['expected_count'] != 500):
        raise ValueError('Incomplete or duplicate full-run question coverage')
    if not all(row['sample_index'] == 0 and type(row['score']['correct']) is bool and
               row['score']['rules_version'] == summary['scoring_rules_version'] for row in rows):
        raise ValueError('Invalid or mixed per-question scoring')
    correct = sum(row['score']['correct'] for row in rows)
    if correct != summary['correct_candidates'] or not math.isclose(correct / 500, summary['pass_at_1']):
        raise ValueError('Aggregate score mismatch')
    baseline = next(r for r in snapshot['rows'] if r['group'] == 'baseline' and
                    r['benchmark'] == 'MATH-500' and r['variant'] == 'HiFP4')
    ids_hash = hashlib.sha256(json.dumps(sorted(ids), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (ids_hash != baseline['ids_sha256'] or
            summary['protocol_version'] + '; ' + summary['scoring_rules_version'] != baseline['protocol'] or
            protocol['dataset']['items_sha256'] != baseline['dataset_sha256'] or
            any(protocol['generation'][key] != value for key, value in baseline['generation'].items())):
        raise ValueError('Sparse run is not comparable with the displayed baseline')
    entry.update(score=correct / 500 * 100, status='完整已核查', correct=correct,
                 outputs_sha256=hashlib.sha256(outputs.read_bytes()).hexdigest(),
                 protocol_sha256=hashlib.sha256(path.with_name('protocol.json').read_bytes()).hexdigest())
    return entry


def refresh_progress(snapshot, math_summaries=(), sweep_statuses=()):
    entries = {(r['method'], r['sparsity']): r
               for r in snapshot.get('sparse_progress', {}).get('rows', [])}
    for path in sweep_statuses:
        state = read(path)
        if state.get('dry_run'):
            raise ValueError('CPU preflight is not evaluation progress')
        for job in state['jobs']:
            key = (job['method'], job['sparsity'])
            if entries.get(key, {}).get('status') == '完整已核查':
                # A queue refresh must not erase an already verified score.
                continue
            if job['status'] == 'pending':
                status = '等待 GPU' if state['state'].startswith('waiting_for_gpu') else '排队'
            else:
                status = {'running': '运行中', 'completed': '已完成，待核查',
                          'needs_debug': '待修复'}.get(job['status'], '未完成')
            entries[key] = dict(benchmark='MATH-500', method=key[0], sparsity=key[1],
                                status=status, completed=0, expected=500, score=None,
                                source=source_label(path))
    for path in math_summaries:
        entry = math_progress(path, snapshot)
        entries[(entry['method'], entry['sparsity'])] = entry
    snapshot['sparse_progress'] = dict(updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                      rows=list(entries.values()))


def build_matrix(snapshot):
    baseline = [r for r in snapshot['rows'] if r['group'] == 'baseline'
                and r['variant'] in ('BF16', 'HiFP4')]
    methods = [r for r in snapshot['rows'] if r['group'] == 'methods']
    progress = {(r['benchmark'], r['method'], r['sparsity']): r
                for r in snapshot.get('sparse_progress', {}).get('rows', [])}

    def sparse_value(bench, method, rate):
        entry = progress.get((bench, method, rate))
        if not entry:
            return '未完成'
        if entry.get('score') is not None and entry['status'] == '完整已核查':
            return entry['score']
        if entry.get('completed'):
            return f"已写入 {entry['completed']}/{entry['expected']}"
        return entry['status']

    matrix = []
    for bench in dict.fromkeys(r['benchmark'] for r in baseline):
        group = [r for r in baseline if r['benchmark'] == bench]
        reference = {r['variant']: r['score'] for r in group}
        for rate in SPARSITIES if bench == 'MATH-500' else (.5,):
            matrix.append([group[0]['model'], bench, '生成式 ' + group[0]['metric'],
                           group[0]['evaluated_count'], int(rate * 100),
                           reference.get('BF16', '未完成'), reference.get('HiFP4', '未完成'), '未完成',
                           sparse_value(bench, 'teal', rate), sparse_value(bench, 'wina', rate)])
    if methods:
        def score(method):
            return next((r['score'] for r in methods if r['variant'] == method), '未完成')
        teal = next((r for r in methods if 'TEAL' in r['method']), None)
        wina = next((r for r in methods if 'WINA' in r['method']), None)
        if any(r and r.get('sparsity') != .5 for r in (teal, wina)):
            raise ValueError('Likelihood results do not match the displayed 50% target')
        matrix.append([methods[0]['model'], 'GPQA Diamond', '标签似然诊断 Accuracy', 198, 50,
                       score('BF16'), score('HiF4 W4A4'), score('HiF4 + ARC'),
                       teal['score'] if teal else '未完成', wina['score'] if wina else '未完成'])
    return [[round(value, 2) if i >= 5 and isinstance(value, (float, int)) else value
             for i, value in enumerate(row)] for row in matrix]


def export(snapshot, out):
    from openpyxl import Workbook, load_workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(snapshot)
    timestamp = snapshot.get('sparse_progress', {}).get('updated_at', snapshot['audited_at'])
    navy, blue, purple, amber = '17365D', 'DDEBF7', 'E4DFEC', 'FFF2CC'
    wb = Workbook(); ws = wb.active; ws.title = '结果总表'
    ws.sheet_view.showGridLines = False
    ws.append(['DeepSeek 量化与稀疏评测结果']); ws.merge_cells('A1:J1')
    ws['A1'].font = Font(name='Microsoft YaHei', size=16, bold=True, color='FFFFFF')
    ws['A1'].fill = PatternFill('solid', fgColor=navy); ws.row_dimensions[1].height = 32
    ws.append(['进度快照：' + timestamp + '；紫色 GPQA 行为标签似然诊断，不能与生成式行混比。'])
    ws.merge_cells('A2:J2'); ws['A2'].alignment = Alignment(wrap_text=True, vertical='center')
    ws.row_dimensions[2].height = 32; ws.append(HEADERS)
    for c in ws[3]:
        c.font = Font(name='Microsoft YaHei', size=11, bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor=navy); c.alignment = Alignment(wrap_text=True, vertical='center')
    ws.row_dimensions[3].height = 34
    for values in matrix:
        ws.append(values); diagnostic = values[2].startswith('标签似然')
        for c in ws[ws.max_row]:
            c.font = Font(name='Microsoft YaHei', size=11, color='203040', bold=c.column >= 6 and isinstance(c.value, (int, float)))
            c.alignment = Alignment(vertical='center', wrap_text=True, horizontal='right' if isinstance(c.value, (int, float)) else 'left')
            c.border = Border(bottom=Side(style='hair', color='D8E2EC'))
            color = purple if diagnostic else blue if c.column in (6, 7) else 'FFFFFF'
            if c.column >= 8 and isinstance(c.value, str): color = amber
            c.fill = PatternFill('solid', fgColor=color)
            if isinstance(c.value, (int, float)):
                c.number_format = '0"%"' if c.column == 5 else '0.00"%"' if c.column >= 6 else '0'
        if diagnostic:
            ws.cell(ws.max_row, 9).comment = Comment('HiF4 W4A4 + ARC + TEAL，50% 非结构化掩码；198 题标签似然诊断。', 'ARCQuant')
            ws.cell(ws.max_row, 10).comment = Comment('HiF4 W4A4 + ARC + WINA mask-only，50% 非结构化掩码；无 optional transform。', 'ARCQuant')
        ws.row_dimensions[ws.max_row].height = 42
    ws.auto_filter.ref = f'A3:J{ws.max_row}'; ws.freeze_panes = 'D4'
    ws.print_title_rows = '1:3'; ws.page_setup.orientation = 'landscape'; ws.page_setup.fitToWidth = 1
    for col, width in enumerate([29, 34, 27, 9, 16, 17, 26, 28, 20, 23], 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    detail = wb.create_sheet('说明与来源'); detail.sheet_view.showGridLines = False
    detail.append(['项目', '说明 / 来源'])
    for i, note in enumerate(NOTES, 1): detail.append([f'说明 {i}', note])
    for r in snapshot['rows']:
        if r['group'] == 'methods' or (r['group'] == 'baseline' and r['variant'] in ('BF16', 'HiFP4')):
            detail.append([r['benchmark'] + ' / ' + r['variant'], r['source']])
    for r in snapshot.get('sparse_progress', {}).get('rows', []):
        detail.append([f"MATH-500 {r['method']} {r['sparsity']:.0%}", r['source']])
    detail.column_dimensions['A'].width = 51; detail.column_dimensions['B'].width = 112
    for cells in detail:
        for c in cells:
            c.font = Font(name='Microsoft YaHei', size=11)
            c.alignment = Alignment(wrap_text=True, vertical='center')
        detail.row_dimensions[cells[0].row].height = 36
    for c in detail[1]:
        c.font = Font(bold=True, color='FFFFFF'); c.fill = PatternFill('solid', fgColor=navy)
    wb.save(out / 'completed_benchmarks.xlsx')
    check = load_workbook(out / 'completed_benchmarks.xlsx', data_only=True)
    assert list(check['结果总表'].values)[3:] == [tuple(r) for r in matrix]
    check.close()
    for ext, separator in [('csv', ','), ('tsv', '\t')]:
        with (out / f'completed_benchmarks.{ext}').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=separator, lineterminator='\n')
            writer.writerow(HEADERS); writer.writerows(matrix)

    def display(value, index):
        if isinstance(value, (int, float)) and index >= 5: return f'{value:.2f}'
        if index == 4: return f'{value}%'
        return str(value)

    md = ['# DeepSeek 量化与稀疏评测结果', '', '模型：DeepSeek-V4-Flash-0731。分数单位为 %。进度快照：' + timestamp, '',
          '| ' + ' | '.join(HEADERS[1:]) + ' |', '| ' + ' | '.join(['---'] * 9) + ' |']
    for row in matrix: md.append('| ' + ' | '.join(display(v, i) for i, v in enumerate(row) if i) + ' |')
    md += ['', *['- ' + note for note in NOTES], '',
           '[彩色 Excel](completed_benchmarks.xlsx) · [CSV](completed_benchmarks.csv) · [TSV](completed_benchmarks.tsv) · [审计快照](results_audit.json)', '',
           '从仓库快照重新导出：`python benchmarks/present_results.py --snapshot-only`。',
           '有新进度时加 `--math-summary <完整的 summary.json 路径>` 和/或 `--sweep-status <队列 status.json 路径>`；两项都可重复指定。', '']
    (out / 'RESULTS_PRESENTATION.md').write_text('\n'.join(md))
    table = '<tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in HEADERS) + '</tr>'
    for row in matrix:
        kind = 'diagnostic' if row[2].startswith('标签似然') else 'baseline'
        table += f'<tr class="{kind}">' + ''.join('<td>' + html.escape(display(v, i)) + '</td>' for i, v in enumerate(row)) + '</tr>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DeepSeek 量化与稀疏评测结果</title>'
    page += '<style>body{font-family:system-ui;padding:24px}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #ccd6e0}th{background:#17365d;color:white}.baseline{background:#ddebf7}.diagnostic{background:#e4dfec}</style>'
    page += '<h1>DeepSeek 量化与稀疏评测结果</h1><p>快照：' + html.escape(timestamp) + '</p><table>' + table + '</table><ul>'
    page += ''.join('<li>' + html.escape(n) + '</li>' for n in NOTES) + '</ul></html>'
    (out / 'completed_benchmarks_colored.html').write_text(page)
