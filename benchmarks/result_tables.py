"""One configuration per row, with verified full-benchmark scores only."""
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
MODEL = 'DeepSeek-V4-Flash-0731'
HEADERS = ['模型', '稀疏方法', '稀疏度', '量化方式', 'GPQA', 'MATH', 'LongBench', 'LiveCodeBench']
# Choose one fixed protocol per column; never fill gaps with likelihood diagnostics
# or a longer-budget continuation of another run.
BENCHMARKS = (
    ('GPQA Diamond', 'generative_8k_v1; gpqa_final_letter_v4'),
    ('MATH-500', 'generative_8k_v1; math500_format_v3'),
    ('LongBench v2', 'longbench_explicit_answer_v2'),
    ('LiveCodeBench v6 64K', 'lcb-grading-v2; independent-64k'),
)
PENDING = '评测中'
NOTES = [
    '每行是一种模型设置，分数单位为 %。没有使用的配置填“无”；量化方式为“无”的一行是 BF16 参考。',
    'HiF4 W4A4 为基础量化；HiF4 W4A4 + ARC 补偿额外启用校准通道重排和增强残差通道。hifp4/HiFP4 是评测器对 HiF4 的命名。',
    'TEAL/WINA 行使用 HiF4 基础量化及非结构化激活稀疏，不含 ARC 补偿。50%–90% 是目标激活置零率，不是 2:4 或层覆盖率。',
    'GPQA 使用 Diamond 198 题生成式 Pass@1，MATH 使用 MATH-500 的 500 题 Pass@1，LongBench 使用 v2 的 503 题 Accuracy。',
    'LiveCodeBench 使用固定 v6 test6 分片 175 题的 64K 独立生成 Pass@1。128K 续写及 GPQA 标签似然诊断保留在审计快照，不填入主表。',
    '“评测中”统一表示尚无可呈现的完整成绩，包括未开始、等待 GPU、运行中及待修复；不表示每个单元格都已启动。当前稀疏队列仅覆盖 MATH。',
    '蓝色＝完整基线成绩；绿色＝完整稀疏成绩；黄色＝评测中。部分题目、smoke 和诊断分数不会用作完整评测成绩。',
    '生成式基线采用本项目固定协议，不声称官方榜单复现。',
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
    baseline = {(r['benchmark'], r['variant']): r for r in snapshot['rows']
                if r['group'] == 'baseline' and r['model'] == MODEL
                and r['variant'] in ('BF16', 'HiFP4')
                and r['status'] == '已核查完整'
                and (r['benchmark'], r['protocol']) in BENCHMARKS}
    progress = {(r['benchmark'], r['method'], r['sparsity']): r
                for r in snapshot.get('sparse_progress', {}).get('rows', [])
                if r.get('model', MODEL) == MODEL}

    def baseline_value(bench, variant):
        score = baseline.get((bench, variant), {}).get('score')
        return score if score is not None else PENDING

    def sparse_value(bench, method, rate):
        entry = progress.get((bench, method, rate))
        if (entry and entry.get('score') is not None and entry['status'] == '完整已核查'
                and entry.get('completed', 0) == entry.get('expected', -1)
                and entry.get('expected', 0) > 0):
            return entry['score']
        return PENDING

    matrix = [
        [MODEL, '无', '无', quant,
         *[baseline_value(bench, variant) for bench, _ in BENCHMARKS]]
        for variant, quant in [('BF16', '无'), ('HiFP4', 'HiF4 W4A4')]
    ]
    # ARC has only likelihood diagnostics so far, not full generation results.
    matrix.append([MODEL, '无', '无', 'HiF4 W4A4 + ARC 补偿', *[PENDING] * 4])
    for method in ('teal', 'wina'):
        for rate in SPARSITIES:
            matrix.append([MODEL, method.upper(), f'{rate:.0%}', 'HiF4 W4A4',
                           *[sparse_value(bench, method, rate) for bench, _ in BENCHMARKS]])
    return [[round(value, 2) if i >= 4 and isinstance(value, (float, int)) else value
             for i, value in enumerate(row)] for row in matrix]


def display(value, index):
    if isinstance(value, (int, float)) and index >= 4:
        return f'{value:.2f}%'
    return str(value)


def export(snapshot, out):
    from openpyxl import Workbook, load_workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(snapshot)
    timestamp = snapshot.get('sparse_progress', {}).get('updated_at', snapshot['audited_at'])
    navy, blue, green, amber = '17365D', 'DDEBF7', 'E2EFDA', 'FFF2CC'
    wb = Workbook(); ws = wb.active; ws.title = '结果总表'
    ws.sheet_view.showGridLines = False
    ws.append(HEADERS)
    for c in ws[1]:
        c.font = Font(name='Microsoft YaHei', size=11, bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor=navy); c.alignment = Alignment(wrap_text=True, vertical='center')
    ws.row_dimensions[1].height = 34
    for col, (bench, protocol) in enumerate(BENCHMARKS, 5):
        ws.cell(1, col).comment = Comment(bench + '\n' + protocol + '\n分数单位：%。', 'ARCQuant')
    for values in matrix:
        ws.append(values)
        for c in ws[ws.max_row]:
            c.font = Font(name='Microsoft YaHei', size=11, color='203040', bold=c.column >= 5 and isinstance(c.value, (int, float)))
            c.alignment = Alignment(vertical='center', wrap_text=True, horizontal='right' if isinstance(c.value, (int, float)) else 'left')
            c.border = Border(bottom=Side(style='hair', color='D8E2EC'))
            color = 'F2F5F9'
            if c.column >= 5:
                color = amber if c.value == PENDING else blue if values[1] == '无' else green
            c.fill = PatternFill('solid', fgColor=color)
            if c.column >= 5 and isinstance(c.value, (int, float)):
                c.number_format = '0.00"%"'
        ws.row_dimensions[ws.max_row].height = 42
    ws['D2'].comment = Comment('未增加量化，使用 BF16 参考结果。', 'ARCQuant')
    ws.auto_filter.ref = f'A1:H{ws.max_row}'; ws.freeze_panes = 'E2'
    ws.print_title_rows = '1:1'; ws.page_setup.orientation = 'landscape'; ws.page_setup.fitToWidth = 1
    for col, width in enumerate([31, 15, 12, 29, 15, 15, 17, 21], 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    detail = wb.create_sheet('说明与来源'); detail.sheet_view.showGridLines = False
    detail.append(['项目', '说明 / 来源'])
    detail.append(['进度快照', timestamp])
    for i, note in enumerate(NOTES, 1): detail.append([f'说明 {i}', note])
    for r in snapshot['rows']:
        if r['group'] == 'methods' or (r['group'] == 'baseline' and r['variant'] in ('BF16', 'HiFP4')):
            detail.append([r['benchmark'] + ' / ' + r['variant'], r['source']])
    for r in snapshot.get('sparse_progress', {}).get('rows', []):
        detail.append([f"{r['benchmark']} {r['method']} {r['sparsity']:.0%}", r['source']])
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
    assert list(check['结果总表'].values) == [tuple(HEADERS), *[tuple(r) for r in matrix]]
    check.close()
    for ext, separator in [('csv', ','), ('tsv', '\t')]:
        with (out / f'completed_benchmarks.{ext}').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f, delimiter=separator, lineterminator='\n')
            writer.writerow(HEADERS)
            writer.writerows([[display(v, i) for i, v in enumerate(row)] for row in matrix])

    md = ['| ' + ' | '.join(HEADERS) + ' |', '| ' + ' | '.join(['---'] * 4 + ['---:'] * 4) + ' |']
    for row in matrix: md.append('| ' + ' | '.join(display(v, i) for i, v in enumerate(row)) + ' |')
    md.append('')
    (out / 'RESULTS_PRESENTATION.md').write_text('\n'.join(md))
    table = '<tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in HEADERS) + '</tr>'
    for row in matrix:
        table += '<tr>'
        for i, value in enumerate(row):
            kind = 'config' if i < 4 else 'pending' if value == PENDING else 'baseline' if row[1] == '无' else 'sparse'
            table += f'<td class="{kind}">' + html.escape(display(value, i)) + '</td>'
        table += '</tr>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DeepSeek 量化与稀疏评测结果</title>'
    page += '<style>body{font-family:system-ui;padding:24px}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #ccd6e0}th{background:#17365d;color:white}.config{background:#f2f5f9}.baseline{background:#ddebf7}.sparse{background:#e2efda}.pending{background:#fff2cc}</style>'
    page += '<table>' + table + '</table></html>'
    (out / 'completed_benchmarks_colored.html').write_text(page)
