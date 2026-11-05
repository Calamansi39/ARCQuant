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
    'TEAL/WINA（greedy）行使用 HiF4 W4A4 + ARC 补偿及非结构化激活稀疏。50%–90% 是按校准实际执行的线性计算量加权的预算，各投影配额不同；不是实测全局零率或 2:4。WINA 未启用 transform。历史 uniform / 无 ARC 记录保留在审计快照。',
    'GPQA 使用 Diamond 198 题生成式 Pass@1，MATH 使用 MATH-500 的 500 题 Pass@1，LongBench 使用 v2 的 503 题 Accuracy。',
    'LiveCodeBench 使用固定 v6 test6 分片 175 题的 64K 独立生成 Pass@1。128K 续写及 GPQA 标签似然诊断保留在审计快照，不填入主表。',
    '“评测中”统一表示尚无可呈现的完整成绩，包括未开始、等待 GPU、运行中及待修复；不表示每个单元格都已启动。离线策略搜索进度单独记录，不计作答题成绩。',
    '蓝色＝完整基线成绩；绿色＝完整稀疏成绩；黄色＝评测中。部分题目、smoke 和诊断分数不会用作完整评测成绩。',
    '生成式基线采用本项目固定协议，不声称官方榜单复现。',
    'BF16 参考使用发布的混合 FP4/FP8 权重重构后作 BF16 计算，不是原始未量化母版权重。',
]


def read(path):
    return json.loads(Path(path).read_text())


def source_label(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def allocation(row):
    return 'none' if row['method'] == 'none' else row.get('allocation', 'unknown')


def progress_key(row):
    return (row['benchmark'], row.get('arc', False), allocation(row), row['method'], row['sparsity'])


def verify_greedy_evidence(directory, runtime):
    """Bind full scores to a complete strategy and its actual use on every rank."""
    from benchmarks.deepseek_greedy_strategy import load_strategy
    path = directory / 'greedy_strategy.json'
    data = read(path)
    calibration = runtime['calibration']
    data, rates, digest = load_strategy(
        path, method=runtime['method'], sparsity=runtime['sparsity'],
        source_signature=data['source_signature'], world_size=calibration['world_size'],
        quantization='arc' if runtime['arc'] else 'hif4', n_layers=43)
    expected = dict(strategy_sha256=digest,
                    calibration_manifest_sha256=calibration['manifest_sha256'],
                    budget_metric=data['budget_metric'], grouping=data['grouping'],
                    transform=False, method=runtime['method'],
                    calibration_samples=data['samples'], search_samples=data['search_samples'])
    if (runtime['strategy'] != expected or
            read(directory / 'protocol.json').get('arcquant') != runtime or
            data.get('calibration_rank_sha256') != [calibration['rank_sha256'][str(rank)]
                                                    for rank in range(calibration['world_size'])]):
        raise ValueError('Greedy strategy/calibration/protocol identity mismatch')
    for rank in range(calibration['world_size']):
        stats = read(directory / f'arcquant_runtime_rank{rank}.json')
        if (stats.get('allocation') != 'greedy' or stats.get('greedy_strategy') != expected
                or stats.get('per_projection_sparsity') != rates):
            raise ValueError(f'Greedy execution evidence mismatch on rank {rank}')
    return expected


def math_progress(path, snapshot):
    """Publish a score only after complete coverage and protocol checks."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    summary = json.loads(raw)
    runtime = read(path.with_name('arcquant_sparse_runtime.json'))
    method, sparsity = runtime['method'], runtime['sparsity']
    arc = runtime.get('arc', False)
    if not ((method in ('teal', 'wina') and sparsity in SPARSITIES)
            or (arc and method == 'none' and sparsity == 0)):
        raise ValueError('Unsupported sparse method/rate in ' + str(path))
    if summary['benchmark'] != 'math500' or summary['quant'] != 'hifp4' or summary['questions'] != 500:
        raise ValueError('Expected HiF4 MATH-500 summary')
    completed = summary['completed_candidates']
    if not 0 <= completed <= 500:
        raise ValueError('Invalid completed question count')
    entry = dict(benchmark='MATH-500', method=method, sparsity=sparsity,
                 arc=arc, allocation=allocation(runtime),
                 completed=completed, expected=500, score=None, status='未完成',
                 source=source_label(path), summary_sha256=hashlib.sha256(raw).hexdigest())
    if not summary['complete']:
        return entry
    if completed != 500 or not summary['is_full_dataset'] or summary['is_smoke_test']:
        raise ValueError('Complete flag disagrees with full coverage')
    if allocation(runtime) == 'greedy':
        entry['strategy'] = verify_greedy_evidence(path.parent, runtime)
    if arc:
        from benchmarks.queue_sparse_baselines import arc_health
        ok, detail = arc_health(path.parent, require_ragged=True)
        if not ok:
            raise ValueError(detail)
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
    entries = {progress_key(r): r
               for r in snapshot.get('sparse_progress', {}).get('rows', [])}
    for path in sweep_statuses:
        state = read(path)
        if state.get('dry_run'):
            raise ValueError('CPU preflight is not evaluation progress')
        for job in state['jobs']:
            identity = dict(benchmark=job.get('benchmark', 'MATH-500'),
                            arc=job.get('arc', state.get('arc', False)), method=job['method'],
                            sparsity=job['sparsity'], allocation=job.get('allocation', state.get('allocation', 'unknown')))
            identity['allocation'] = allocation(identity)
            key = progress_key(identity)
            if entries.get(key, {}).get('status') == '完整已核查':
                # A queue refresh must not erase an already verified score.
                continue
            if state['state'] in ('blocked', 'failed', 'stopped', 'interrupted', 'cancelled') and job['status'] != 'completed':
                status = '待修复' if state['state'] in ('blocked', 'failed') else '已停止'
            elif job['status'] == 'pending':
                status = '等待 GPU' if state['state'].startswith('waiting_for_gpu') else '排队'
            else:
                status = {'running': '运行中', 'completed': '已完成，待核查',
                          'needs_debug': '待修复'}.get(job['status'], '未完成')
            entries[key] = dict(**identity,
                                status=status, completed=0, expected=500, score=None,
                                source=source_label(path), queue_state=state['state'],
                                error=state.get('error'), source_updated_at=state.get('updated_at'))
    for path in math_summaries:
        entry = math_progress(path, snapshot)
        entries[progress_key(entry)] = entry
    snapshot['sparse_progress'] = dict(updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                      rows=list(entries.values()))


def refresh_preparation(snapshot, search_roots):
    """Read portable search artifacts; never interpret layers as answered questions."""
    entries = {r['source']: r for r in snapshot.get('preparation_progress', {}).get('rows', [])}
    for root in map(Path, search_roots):
        manifest = read(root / 'search_manifest.json')
        status = read(root / 'status.json')
        for method in manifest['methods']:
            path = root / method / 'strategy.json'
            if not path.exists():
                continue
            raw = path.read_bytes()
            data = json.loads(raw)
            expected = {str(i) for i in range(data['n_layers'])}
            coverage = {str(p['target']): len(set(p['layers']) & expected) for p in data['profiles']}
            complete = bool(data['complete']) and all(coverage.get(str(t)) == data['n_layers'] for t in manifest['targets'])
            label = source_label(path)
            entries[label] = dict(source=label, method=method, allocation='greedy',
                quantization=manifest['quantization'], complete=complete,
                layers_per_target=coverage, expected_layers=data['n_layers'],
                calibration_samples=manifest['samples'], calibration_length=manifest['length'],
                search_samples=manifest['search_samples'], budget_metric=manifest['budget_metric'],
                strategy_sha256=hashlib.sha256(raw).hexdigest(),
                source_phase=status['phase'], source_updated_at=status.get('updated_at'))
    snapshot['preparation_progress'] = dict(updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                           rows=list(entries.values()))


def build_matrix(snapshot):
    baseline = {(r['benchmark'], r['variant']): r for r in snapshot['rows']
                if r['group'] == 'baseline' and r['model'] == MODEL
                and r['variant'] in ('BF16', 'HiFP4')
                and r['status'] == '已核查完整'
                and (r['benchmark'], r['protocol']) in BENCHMARKS}
    progress = {progress_key(r): r
                for r in snapshot.get('sparse_progress', {}).get('rows', [])
                if r.get('model', MODEL) == MODEL}

    def baseline_value(bench, variant):
        score = baseline.get((bench, variant), {}).get('score')
        return score if score is not None else PENDING

    def sparse_value(bench, method, rate):
        entry = progress.get((bench, True, 'none' if method == 'none' else 'greedy', method, rate))
        if (entry and entry.get('score') is not None and entry['status'] == '完整已核查'
                and entry.get('completed', 0) == entry.get('expected', -1)
                and entry.get('expected', 0) > 0
                and (method == 'none' or entry.get('strategy', {}).get('strategy_sha256'))):
            return entry['score']
        return PENDING

    matrix = [
        [MODEL, '无', '无', quant,
         *[baseline_value(bench, variant) for bench, _ in BENCHMARKS]]
        for variant, quant in [('BF16', '无'), ('HiFP4', 'HiF4 W4A4')]
    ]
    matrix.append([MODEL, '无', '无', 'HiF4 W4A4 + ARC 补偿',
                   *[sparse_value(bench, 'none', 0.) for bench, _ in BENCHMARKS]])
    for method in ('teal', 'wina'):
        for rate in SPARSITIES:
            matrix.append([MODEL, method.upper() + '（greedy）', f'{rate:.0%}', 'HiF4 W4A4 + ARC 补偿',
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
    timestamp = max(snapshot['audited_at'], *[snapshot.get(key, {}).get('updated_at', '')
                    for key in ('sparse_progress', 'preparation_progress')])
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
    ws['D2'].comment = Comment(NOTES[-1], 'ARCQuant')
    ws.auto_filter.ref = f'A1:H{ws.max_row}'; ws.freeze_panes = 'E2'
    ws.print_title_rows = '1:1'; ws.page_setup.orientation = 'landscape'; ws.page_setup.fitToWidth = 1
    for col, width in enumerate([31, 23, 12, 29, 15, 15, 17, 21], 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    detail = wb.create_sheet('说明与来源'); detail.sheet_view.showGridLines = False
    detail.append(['项目', '说明 / 来源'])
    detail.append(['进度快照', timestamp])
    detail.append(['完整结果核查时间', snapshot['audited_at']])
    for i, note in enumerate(NOTES, 1): detail.append([f'说明 {i}', note])
    for note in snapshot.get('audit_notes', []): detail.append(['本次核查', note])
    for r in snapshot.get('preparation_progress', {}).get('rows', []):
        coverage = '；'.join(f'{float(target):.0%}: {layers}/{r["expected_layers"]} 层'
                            for target, layers in r['layers_per_target'].items())
        detail.append([f'{r["method"].upper()} greedy 离线搜索（非准确率）',
                       coverage + '\n' + ('策略文件已完成，仍需完整评测' if r['complete'] else '策略文件未完成') + '\n' + r['source']])
    for r in snapshot['rows']:
        if r['group'] == 'methods' or (r['group'] == 'baseline' and r['variant'] in ('BF16', 'HiFP4')):
            detail.append([r['benchmark'] + ' / ' + r['variant'], r['source']])
    for r in snapshot.get('sparse_progress', {}).get('rows', []):
        detail.append([f"{r['benchmark']} {r['method']} {r['sparsity']:.0%} ARC={r.get('arc', False)} / {allocation(r)}",
                       f"{r['status']}；{r['completed']}/{r['expected']} 题\n{r['source']}" +
                       (f"\n{r['error']}" if r.get('error') else '')])
    detail.column_dimensions['A'].width = 51; detail.column_dimensions['B'].width = 112
    for cells in detail:
        for c in cells:
            c.font = Font(name='Microsoft YaHei', size=11)
            c.alignment = Alignment(wrap_text=True, vertical='center')
        lines = sum(max(1, math.ceil(sum(2 if ord(c) > 255 else 1 for c in line) / 105))
                    for line in str(cells[1].value).split('\n'))
        detail.row_dimensions[cells[0].row].height = max(36, 16 * lines + 6)
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
