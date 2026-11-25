"""Publish audited Excel, CSV and TSV views from the same existing results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'reports/verified'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evidence(paths, base):
    result = []
    for path in dict.fromkeys(Path(p).resolve() for p in paths):
        if path.is_relative_to(ROOT):
            label = 'repo/' + str(path.relative_to(ROOT))
        elif path.is_relative_to(base):
            label = 'base_eval/' + str(path.relative_to(base))
        else:
            label = str(path)
        result.append(dict(path=label, sha256=digest(path), bytes=path.stat().st_size))
    return result


def validate_answers(directory, expected):
    s, p = read(directory / 'summary.json'), read(directory / 'protocol.json')
    raw = directory / s.get('scored_outputs_path', 'outputs.jsonl')
    rows = jsonl(raw)
    require(s['complete'] and s['questions'] == s['completed_candidates'] == len(rows) == expected,
            f'Incomplete answer coverage: {directory}')
    ids = p['selected_ids']
    require(len(ids) == len(set(ids)) == expected and {r['id'] for r in rows} == set(ids),
            f'Duplicate or missing IDs: {directory}')
    require(all(r['sample_index'] == 0 and type(r['score']['correct']) is bool
                and r['score']['rules_version'] == s['scoring_rules_version'] for r in rows),
            f'Mixed scoring or samples: {directory}')
    correct = sum(r['score']['correct'] for r in rows)
    require(correct == s['correct_candidates'] and math.isclose(correct / expected, s['pass_at_1']),
            f'Incorrect aggregate: {directory}')
    require(all(r['generation']['finish_reason'] in ('length', 'eos') for r in rows),
            f'Unknown finish reason: {directory}')
    require(sum(r['generation']['finish_reason'] == 'length' for r in rows) == s['truncated'],
            f'Truncation mismatch: {directory}')
    return s, p, rows, raw


def comparable_subset(full_protocol, score_manifest, full_rows, tiny_protocol, tiny_rows, benchmark):
    """Extract the same cases only after checking generation, prompts and grading.

    Old generation protocols can name an older scorer. A verified rescore
    manifest must bind the current scoring policy before their answers are used.
    """
    for key in ('schema', 'fewshot', 'thinking', 'reasoning_effort', 'generation', 'checkpoint_hashes'):
        require(full_protocol.get(key) == tiny_protocol.get(key), f'Incompatible subset {key}')
    require(score_manifest['scoring'] == tiny_protocol['scoring'][benchmark], 'Subset scoring differs')
    require(score_manifest['generation'] == full_protocol['generation'], 'Rescore changed generation')
    indexed = {r['id']: r for r in full_rows}
    reference = {r['id']: r for r in tiny_rows}
    require(len(indexed) == len(full_rows), 'Duplicate full-set IDs')
    selected = []
    for identity in tiny_protocol['selected_ids']:
        require(identity in indexed and identity in reference, 'Missing subset answer')
        row, ref = indexed[identity], reference[identity]
        for key in ('id', 'seed', 'question', 'gold_answer', 'messages', 'prompt_sha256'):
            require(row.get(key) == ref.get(key), f'Subset {identity}: different {key}')
        require(row['score']['rules_version'] == ref['score']['rules_version'], 'Different answer scoring')
        selected.append(row)
    return selected


def answer_metrics(rows):
    return dict(n=len(rows), correct=sum(r['score']['correct'] for r in rows),
                eos=sum(r['generation']['finish_reason'] == 'eos' for r in rows),
                truncated=sum(r['generation']['finish_reason'] == 'length' for r in rows))


def collect_text(base):
    from benchmarks import present_results as legacy
    from benchmarks.queue_deepseek_tiny_sweep import summary_complete
    from benchmarks.queue_sparse_baselines import arc_health

    # Audit only the requested BF16/HiF4 precisions. This only reads files.
    old = legacy.QUANTS, legacy.LABELS
    try:
        legacy.QUANTS, legacy.LABELS = ('bf16', 'hifp4'), ('BF16', 'HiFP4')
        source_audit = legacy.audit(base)
    finally:
        legacy.QUANTS, legacy.LABELS = old
    full = [r for r in source_audit['rows'] if r['group'] == 'baseline']
    require(len(full) == 10, f'Expected 8 full results and 2 LCB continuation results: {source_audit["rejected"]}')
    tiny, references = [], {}
    for method, rate in [('none', 0.), *[(m, r / 100) for r in (40, 50, 60, 70, 80, 90) for m in ('teal', 'wina')]]:
        sweep = ROOT / 'runs' / ('deepseek_arc0_tiny_20261118' if rate == 0 else
                'deepseek_40_tiny_sweep_20261111' if rate == .4 else 'deepseek_tiny_sweep_20261106')
        for benchmark, n in [('gpqa', 32), ('math500', 100)]:
            task = dict(id=f'{method}-{rate:g}-{benchmark}', method=method, sparsity=rate,
                        benchmark=benchmark, limit=n)
            paths = list((sweep / task['id']).glob('arcquant-*/summary.json'))
            require(len(paths) == 1, f'Ambiguous result: {task["id"]}')
            directory = paths[0].parent
            if rate:
                require(summary_complete(directory, task, max_new_tokens=8192), f'Invalid sparse evidence: {directory}')
            s, p, rows, raw = validate_answers(directory, n)
            runtime = read(directory / 'arcquant_sparse_runtime.json')
            require(runtime['method'] == method and runtime['sparsity'] == rate
                    and runtime['arc'] is True and runtime['quantization'] == 'arc', 'Wrong ARC identity')
            healthy, detail = arc_health(directory)
            require(healthy, detail)
            require(p['generation']['max_new_tokens'] == 8192, 'Wrong token budget')
            if benchmark in references:
                ref_protocol, ref_rows = references[benchmark]
                comparable_subset(p, dict(generation=p['generation'], scoring=p['scoring'][benchmark]),
                                  rows, ref_protocol, ref_rows, benchmark)
            else:
                references[benchmark] = (p, rows)
            files = [paths[0], directory / 'protocol.json', raw, directory / 'arcquant_sparse_runtime.json']
            files += sorted(directory.glob('arcquant_runtime_rank*.json'))
            if rate:
                files.append(directory / 'greedy_strategy.json')
            tiny.append(dict(model='DS', method=method, sparsity=rate, quantization='HiF4 W4A4 + ARC',
                             arc=True, benchmark=benchmark, **answer_metrics(rows),
                             origin='evaluated', selected_ids=p['selected_ids'], evidence=evidence(files, base)))
            if rate == 0:
                for quant, label in [('bf16', 'BF16'), ('hifp4', 'HiF4 W4A4')]:
                    source = base / f'results/DeepSeek-V4-Flash-0731/{benchmark}_generative/{quant}'
                    fs, fp, fr, fr_path = validate_answers(source, 198 if benchmark == 'gpqa' else 500)
                    manifest = source / fs['scoring_manifest_path']
                    selected = comparable_subset(fp, read(manifest), fr, p, rows, benchmark)
                    tiny.append(dict(model='DS', method='none', sparsity=0., quantization=label, arc=False,
                                     benchmark=benchmark, **answer_metrics(selected),
                                     origin='same-cases-from-completed-full-run', selected_ids=p['selected_ids'],
                                     evidence=evidence([source / 'summary.json', source / 'protocol.json',
                                                        fr_path, manifest, directory / 'protocol.json'], base)))
    # No mixing prompt sets or runtime generation settings across the sparse sweep.
    for bench in ('gpqa', 'math500'):
        cells = [r for r in tiny if r['benchmark'] == bench]
        require(all(r['selected_ids'] == cells[0]['selected_ids'] for r in cells), 'Different Tiny cases')
    return full, tiny, source_audit


def audit_magi(base, variant, verify_bytes):
    sys.path.insert(0, str(base))
    from vbench_generation_cache import generation_identity, cache_video_path
    directory = base / 'results/MAGI-2-preview/vbench_standard' / variant / 'tiny'
    cfg = read(directory / 'generation_config.json')
    summary = read(directory / 'vbench_summary.json')
    scoring = read(directory / 'scoring_status.json')
    jobs = jsonl(directory / 'generation_jobs.jsonl')
    require(len(jobs) == len({(j['prompt'], j['sample']) for j in jobs}) == 740, 'Incomplete video job list')
    require(cfg['protocol'] == 'tiny16' and cfg['preview_steps'] == 50 and cfg['refiner_steps'] == 5,
            'Different video generation protocol')
    cache = Path(cfg['cache_dir'])
    require(read(cache / 'generation_identity.json') == generation_identity(cfg), 'Video cache identity mismatch')
    records, known = [], {}
    for job in jobs:
        path = cache_video_path(cache, job['prompt'], job['sample'])
        record = read(path.with_suffix('.mp4.json'))
        require(all(record[k] == job[k] for k in ('prompt', 'sample', 'seed')), 'Video job identity mismatch')
        require(path.is_file() and path.stat().st_size == record['size'] > 0, f'Missing video: {path}')
        require(record['video'] == dict(width=1920, height=1080, fps=25., num_frames=249), 'Wrong video dimensions')
        if verify_bytes:
            require(digest(path) == record['sha256'], f'Video hash mismatch: {path}')
        known[path.resolve()] = record['sha256']
        records.append(dict(name=path.name, sha256=record['sha256'], bytes=record['size']))
    require(scoring['state'] == 'completed' and not scoring['failed_dimensions'], 'Incomplete video scoring')
    require(set(scoring['completed_dimensions']) == set(summary['scores']) and len(summary['scores']) == 16,
            'Missing VBench dimensions')
    coverage = summary['temporal_flickering_coverage']
    require(coverage['has_complete_static_coverage'] and coverage['passed_videos'] == coverage['target_videos'] == 50,
            'Incomplete static coverage')
    require(summary['subset_manifest_sha256'] == digest(base / 'vbench_tiny_manifest.json'), 'Wrong video subset')
    files = [directory / f for f in ('generation_config.json', 'gen_summary.json', 'generation_jobs.jsonl',
                                    'evaluation_config.json', 'scoring_status.json', 'vbench_summary.json')]
    files += [base / 'vbench_tiny_manifest.json', cache / 'generation_identity.json']
    for dimension, source in summary['source_files'].items():
        path = base / source.split('/base_eval/', 1)[1]
        value, details = read(path)[dimension]
        require(details and math.isclose(value, summary['scores'][dimension]), 'Dimension result differs')
        binding = path.parent / 'scoring_config.json'
        require(binding.exists(), 'Unbound video scoring')
        inputs = read(binding)['inputs']
        expected = {(directory / item['path']).resolve() for item in inputs}
        actual = [Path(item['video_path']).resolve() for item in details]
        require(len(actual) == len(set(actual)) and set(actual) <= expected, 'Unexpected dimension video')
        require(all(math.isfinite(item['video_results']) for item in details), 'Nonfinite video score')
        if dimension != 'color':
            require(set(actual) == expected, 'Incomplete dimension scoring')
        for item in inputs:
            video = directory / item['path']
            require(video.is_file() and video.stat().st_size == item['size'], 'Scored video missing')
            if verify_bytes:
                actual = known.get(video.resolve())
                if actual is None:
                    actual = digest(video)
                    known[video.resolve()] = actual
                require(actual == item['sha256'], 'Scoring input video hash differs')
        files += [path, binding]
    scripts = base.parent / 'data/vbench/VBench/scripts'
    sys.path.insert(0, str(scripts))
    import cal_final_score as aggregate
    normalized = aggregate.get_nomalized_score({key.replace('_', ' '): value for key, value in summary['scores'].items()})
    quality, semantic = aggregate.get_quality_score(normalized), aggregate.get_semantic_score(normalized)
    scores = dict(quality=quality, semantic=semantic, total=aggregate.get_final_score(quality, semantic))
    require(all(math.isclose(value, summary['tiny_scores'][key]) for key, value in scores.items()), 'Video total differs')
    # These exact quantizer files implement format-only fake quantization, no ARC residual channels.
    for name in ('magi2_quant.py', 'arcquant_fakequant.py'):
        require(digest(base / name) == cfg['source_sha256'][name], 'Quantization source changed; review ARC labeling')
        files.append(base / name)
    return dict(model='MAGI-2', method='none', sparsity=0., arc=False,
                quantization='BF16' if cfg['quant'] == 'bf16' else 'HiF4 W4A4',
                benchmark='VBench Tiny', n=740, prompts=108, dimensions=16, **scores,
                video_bytes_verified=verify_bytes, video_records=records, evidence=evidence(files, base))


GAPS = [
    ['DS', 'HiF4 W4A4 + ARC，0%', 'GPQA 198 / MATH 500 / LongBench 503 / LiveCodeBench v6 test6 175',
     '未完成；仅 GPQA 32 / MATH 100 的 Tiny 对照完成', '正式稀疏全量比较前补齐；先选定稀疏度'],
    ['MAGI-2', 'HiF4 W4A4 + ARC，0%', 'VBench Tiny 740 视频',
     '尚无 ARC 对照；79.1241% 是普通 HiF4，无 ARC', '先完成 ARC 适配；与方法使用同一量化基座'],
    ['MiniMax-H3', 'BF16 / HiF4 W4A4，0%', 'VBench Tiny',
     '本机只有历史评分文件，缺生成视频/配置；BF16 静态覆盖仅 46/50', '从 A800 同步原始证据、补齐 BF16 覆盖，再确认是否需要重跑'],
    ['MiniMax-H3', 'HiF4 W4A4 + ARC，0%', 'VBench Tiny 740 视频',
     '尚无可核验的 ARC 对照', '完成适配后补齐'],
    ['DS（可选）', 'dense', 'Terminal-Bench 86 任务',
     '尚无结果，代理和容器评测协议尚待接入', '仅在保留此 benchmark 时安排'],
]


def collect(base, verify_bytes=False):
    print('[audit] Checking DS full, Tiny and same-case dense controls', flush=True)
    full, tiny, legacy = collect_text(base)
    videos = []
    for variant in ('bf16-50p5', 'hifp4-w4a4-50p5'):
        print(f'[audit] Checking MAGI {variant}: 740 videos and 16 dimensions', flush=True)
        videos.append(audit_magi(base, variant, verify_bytes))
    return dict(schema='arcquant-verified-results-v1', audited_at=datetime.now(timezone.utc).isoformat(),
                full=full, tiny=tiny, videos=videos, dense_gaps=GAPS,
                exclusions=[dict(model='MiniMax-H3', reason='Missing local generation files; BF16 static coverage 46/50'),
                            dict(model='DS', reason='30% search in progress; no completed benchmark scores'),
                            dict(model='DS', reason='Old likelihood, uniform masks, NVFP4/MXFP4 outside current presentation protocol')],
                legacy_audit=legacy)


def refresh_dense(snapshot, run_dir):
    """Publish completed ARC full scores; retain partial counts as progress only."""
    from benchmarks.queue_sparse_baselines import arc_health
    from benchmarks.reuse_dense_prefix import compatible, digest as protocol_digest
    plan = read(run_dir / 'plan.json')
    require(plan['schema'] == 'deepseek-arc-dense-full-v1', 'Unknown dense plan')
    snapshot['full'] = [r for r in snapshot['full'] if r['variant'] != 'HiF4 W4A4 + ARC']
    progress = []
    for job in plan['jobs']:
        paths = list(Path(job['output']).glob('arcquant-*/summary.json'))
        require(len(paths) == 1, 'Missing/ambiguous dense summary')
        directory, summary = paths[0].parent, read(paths[0])
        progress.append(dict(benchmark=job['benchmark'], expected=job['expected'],
                             completed=summary['completed_candidates'], complete=summary['complete']))
        if not summary['complete']:
            continue
        s, protocol, rows, raw = validate_answers(directory, job['expected'])
        require(s['is_full_dataset'] and not s['is_smoke_test'], 'Dense result is not full')
        runtime = protocol['arcquant']
        require(runtime['method'] == 'none' and runtime['sparsity'] == 0 and arc_health(directory)[0],
                'Missing dense ARC execution evidence')
        files = [paths[0], raw, directory / 'protocol.json', directory / 'arcquant_sparse_runtime.json',
                 *sorted(directory.glob('arcquant_runtime_rank*.json'))]
        reused_path = directory / 'reused_prefix.json'
        if reused_path.exists():
            reused = read(reused_path)
            source = Path(reused['source'])
            sp = read(source / 'protocol.json')
            compatible(sp, protocol)
            require(reused['source_protocol_sha256'] == protocol_digest(sp)
                    and reused['target_protocol_sha256'] == protocol_digest(protocol), 'Changed reuse identity')
            indexed = {r['id']: r for r in rows}
            original = {r['id']: r for r in jsonl(source / 'outputs.jsonl')}
            for binding in reused['records']:
                new, old = indexed[binding['id']], original[binding['id']]
                require(old['record_sha256'] == binding['source_record_sha256']
                        and new['record_sha256'] == binding['imported_record_sha256'], 'Changed imported answer')
                ignore = {'record_sha256', 'protocol_sha256'}
                require({k:v for k,v in new.items() if k not in ignore} ==
                        {k:v for k,v in old.items() if k not in ignore}, 'Reused answer content changed')
            files += [reused_path, source / 'protocol.json', source / 'outputs.jsonl']
        snapshot['full'].append(dict(model='DS', group='baseline', variant='HiF4 W4A4 + ARC',
            benchmark={'gpqa': 'GPQA Diamond', 'math500': 'MATH-500'}[job['benchmark']],
            evaluated_count=s['questions'], correct=s['correct_candidates'], score=s['pass_at_1']*100,
            protocol=protocol['schema'], source='repo/' + str(directory.relative_to(ROOT)),
            evidence=evidence(files, ROOT)))
    snapshot['dense_progress'] = progress
    snapshot['dense_progress_at'] = datetime.now(timezone.utc).isoformat()
    snapshot['dense_gaps'][0][3] = '；'.join(
        f"{r['benchmark']} {r['completed']}/{r['expected']}" for r in progress) + '；LongBench/LiveCodeBench 待适配核验'
    snapshot['dense_gaps'][0][4] = (
        'GPQA/MATH 全量已完成；下一步核验 LongBench/LiveCodeBench 的 ARC 接入'
        if len(progress) == 2 and all(r['complete'] for r in progress)
        else '优先续跑 GPQA/MATH 全量；复用同协议 Tiny 已保存答案')


def matrices(snapshot):
    headers = ['模型', '稀疏方法', '稀疏度', '量化方式', 'GPQA', 'GPQA 未截断率', 'MATH', 'MATH 未截断率', 'LongBench', 'LiveCodeBench']
    configs = sorted({(r['method'], r['sparsity'], r['quantization']) for r in snapshot['tiny']},
                     key=lambda v: (v[1], {'BF16': 0, 'HiF4 W4A4': 1, 'HiF4 W4A4 + ARC': 2}[v[2]], v[0]))
    rows = []
    for method, rate, quant in configs:
        row = ['DS', '无' if method == 'none' else method.upper(), rate, quant]
        for bench in ('gpqa', 'math500'):
            found = [r for r in snapshot['tiny'] if (r['method'], r['sparsity'], r['quantization'], r['benchmark']) == (method, rate, quant, bench)]
            require(len(found) == 1, 'Ambiguous Tiny cell')
            r = found[0]
            row.extend([r['correct'] / r['n'], r['eos'] / r['n']])
        row.extend(['评测中', '评测中'])
        rows.append(row)
    full_headers = ['模型', '稀疏方法', '稀疏度', '量化方式', 'GPQA', 'MATH', 'LongBench', 'LiveCodeBench 64K', 'LiveCodeBench 128K续写']
    full_rows = []
    full_variants = [('BF16', 'BF16'), ('HiFP4', 'HiF4 W4A4')]
    if snapshot.get('dense_progress'):
        full_variants.append(('HiF4 W4A4 + ARC', 'HiF4 W4A4 + ARC'))
    for q, label in full_variants:
        row = ['DS', '无', 0., label]
        for bench in ('GPQA Diamond', 'MATH-500', 'LongBench v2', 'LiveCodeBench v6 64K', 'LiveCodeBench v6 128K 续写'):
            matches = [r for r in snapshot['full'] if r['variant'] == q and r['benchmark'] == bench]
            require(len(matches) <= 1 and (len(matches) or q == 'HiF4 W4A4 + ARC'), 'Ambiguous full result')
            row.append(matches[0]['score'] / 100 if matches else '评测中')
        full_rows.append(row)
    video_headers = ['模型', '稀疏方法', '稀疏度', '量化方式', 'Total', 'Quality', 'Semantic', '视频数', '评分维度']
    video_rows = [[r['model'], '无', 0., r['quantization'], r['total'], r['quality'], r['semantic'], r['n'], r['dimensions']] for r in snapshot['videos']]
    return [('DS_Tiny', headers, rows, (5, 6, 7, 8)), ('DS_全量dense', full_headers, full_rows, (5, 6, 7, 8, 9)),
            ('MAGI_Tiny', video_headers, video_rows, (5, 6, 7))]


def display_row(title, row, metrics):
    """Keep percentages and missing-result labels identical in all text views."""
    return [f'{v:.0%}' if i == 2 and isinstance(v, (int, float))
            else f'{(Decimal(str(v)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)}%'
            if i + 1 in metrics and title == 'DS_Tiny' and isinstance(v, (int, float))
            else f'{v:.4%}' if i + 1 in metrics and title == 'MAGI_Tiny' and isinstance(v, (int, float))
            else f'{v:.2%}' if i + 1 in metrics and isinstance(v, (int, float))
            else str(v) for i, v in enumerate(row)]


def render(snapshot, output):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.comments import Comment
    from openpyxl.utils import get_column_letter
    output.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    md = ['# 统一实验结果', '', f'核查时间：{snapshot["audited_at"]}', '',
          '**本目录是唯一结果发布入口。[彩色 Excel](实验结果.xlsx)、CSV 和 TSV 使用同一份核验数据，同步更新。**', '',
          '| 评测范围 | CSV | TSV |', '| --- | --- | --- |',
          '| DS 稀疏度 Tiny 横评 | [DS_Tiny.csv](DS_Tiny.csv) | [DS_Tiny.tsv](DS_Tiny.tsv) |',
          '| DS 全量 dense 对照 | [DS_全量dense.csv](DS_全量dense.csv) | [DS_全量dense.tsv](DS_全量dense.tsv) |',
          '| MAGI Tiny | [MAGI_Tiny.csv](MAGI_Tiny.csv) | [MAGI_Tiny.tsv](MAGI_Tiny.tsv) |', '',
          '## DS Tiny', '', 'GPQA 固定 32 题、MATH 固定 100 题，生成上限 8192 token。0% 的 BF16 和普通 HiF4 从已完成全量逐题记录中提取；题目、提示词、seed 和当前评分规则已逐项核对，没有重新推理。', '',
          '这些是固定测试集前缀上的初筛结果，不能替代全量结论；“评测中”表示该范围尚无结果，不表示有进程正在运行。', '',
          '未截断率是自然 EOS 比例，不等于有效回答比例；90% 稀疏的提前 EOS 虽然不截断，准确率仍为零。', '']
    for title, headers, rows, metrics in matrices(snapshot):
        formatted_rows = [display_row(title, row, metrics) for row in rows]
        for suffix, delimiter, encoding in (('csv', ',', 'utf-8-sig'), ('tsv', '\t', 'utf-8')):
            destination = output / f'{title}.{suffix}'
            temp_text = destination.with_suffix(destination.suffix + '.tmp')
            with temp_text.open('w', encoding=encoding, newline='') as stream:
                writer = csv.writer(stream, delimiter=delimiter, lineterminator='\n')
                writer.writerow(headers)
                writer.writerows(formatted_rows)
            temp_text.replace(destination)
        ws = wb.create_sheet(title)
        ws.append(headers)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = 'E2'
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = False
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='17365D')
            cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.fill = PatternFill('solid', fgColor='EAF2F8' if row[1].value == '无' else 'F2F5F8')
                cell.alignment = Alignment(horizontal='center', vertical='center')
                if cell.column == 3:
                    cell.number_format = '0%'
                elif cell.column in metrics:
                    cell.number_format = '0.0%' if title == 'DS_Tiny' else '0.00%' if title == 'DS_全量dense' else '0.0000%'
                    val = cell.value
                    cell.fill = PatternFill('solid', fgColor='FFF2CC' if not isinstance(val, (int, float))
                        else 'E2F0D9' if val >= .8 else 'FFF2CC' if val >= .4 else 'FCE4D6')
                elif cell.value == '评测中':
                    cell.fill = PatternFill('solid', fgColor='FFF2CC')
        for i, heading in enumerate(headers, 1):
            ws.column_dimensions[get_column_letter(i)].width = 27 if i == 4 else 24 if '128K' in heading else 20 if '未截断' in heading else 17
        for col in metrics:
            ws.conditional_formatting.add(f'{get_column_letter(col)}2:{get_column_letter(col)}{ws.max_row}',
                ColorScaleRule(start_type='num', start_value=0, start_color='FCE4D6', mid_type='num', mid_value=.5,
                               mid_color='FFF2CC', end_type='num', end_value=1, end_color='E2F0D9'))
        if title == 'DS_Tiny':
            for index in (2, 3):
                ws.cell(index, 4).comment = Comment('从已有全量结果提取相同题目，未新增 GPU 推理；协议核验见 audit.json。', 'ARCQuant')
            ws['I1'].comment = Comment('本轮未纳入 LongBench/LiveCodeBench；评测中是缺失结果占位，不表示正在运行。', 'ARCQuant')
        if title == 'DS_全量dense':
            md += ['## DS 已完成全量 dense', '', 'GPQA 198、MATH 500、LongBench 503；LiveCodeBench 是 release-v6 的 test6 分片 175 题。64K 为独立生成，128K 为到限题续写，两者分别列出。普通 HiF4 不含 ARC。', '']
            md += ['**此表是成绩表，不是实时运行队列。ARC 行的 LongBench / LiveCodeBench 缺失格继续沿用“评测中”占位；它们的实际启动状态请看 [DS 调度记录](../../docs/DEEPSEEK_DENSE_PRIORITY.md)。**', '']
            if snapshot.get('dense_progress'):
                md += [f'ARC dense 续跑进度更新：{snapshot["dense_progress_at"]}。已复用同协议 Tiny 的 132 道答案；未完成项不填中途准确率。[调度与续跑说明](../../docs/DEEPSEEK_DENSE_PRIORITY.md)。', '']
        if title == 'MAGI_Tiny':
            md += ['## MAGI 已完成 Tiny dense', '', '108 个唯一 prompt、740 个视频、16 维。两组都是稠密模型；HiF4 采用 ARCQuant 来源的量化格式，但**没有 ARC 重排/残差补偿**。先前将 79.1241% 标作 ARC 成绩不准确，现已更正。', '']
        md += ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        for formatted in formatted_rows:
            md.append('| ' + ' | '.join(formatted) + ' |')
        md.append('')
    temp = output / '实验结果.tmp.xlsx'
    wb.save(temp)
    check = load_workbook(temp, data_only=True)
    require(check.sheetnames == ['DS_Tiny', 'DS_全量dense', 'MAGI_Tiny'], 'Workbook verification failed')
    check.close()
    temp.replace(output / '实验结果.xlsx')
    md += ['## Dense 还缺什么', '', '| 模型 | 对照 | 范围 | 状态 | 后续 |', '| --- | --- | --- | --- | --- |']
    md += ['| ' + ' | '.join(row) + ' |' for row in snapshot['dense_gaps']]
    md += ['', 'DS 的 BF16/普通 HiF4 全量基线、三种 dense 的同题 Tiny、MAGI BF16/普通 HiF4 Tiny 已有结果，无需重复运行。视频官方全集不属于当前首轮 Tiny 选点的必做项。', '',
           '## 核验与更新', '',
           '- `audit.json` 保存精确计数、题目 ID、来源路径、文件 SHA256 和纳入/排除原因。',
           '- MAGI 生成身份、740 个视频的记录与文件、16 个评分结果、评分输入绑定、官方加权总分均已核验。是否重算视频字节哈希记录在 `video_bytes_verified`。',
           '- 原始逐题输出、视频、模型、日志与搜索策略保留在原运行目录；删除的是过时展示表。',
           '- MiniMax 历史评分缺少本机生成证据，BF16 静态覆盖不足，不进入可信主表。30% 的 8/16 题小探针与主表的 32/100 题范围不同，不混报。',
           '- GPQA 使用生成式得分；旧标签条件似然成绩不混入。当前没有 acc_norm 指标可替换这些生成式得分。', '',
           '在仓库根目录刷新（不使用 GPU）：', '', '```bash',
           'python -m pip install -r requirements-report.txt',
           'python tools/build_verified_results.py --base-eval /path/to/base_eval --verify-video-bytes',
           '# 其他机器只查看/重建已核验快照：', 'python tools/build_verified_results.py --snapshot-only', '```', '']
    (output / 'README.md').write_text('\n'.join(md))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-eval', type=Path)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--snapshot-only', action='store_true')
    p.add_argument('--verify-video-bytes', action='store_true')
    p.add_argument('--refresh-dense-run', type=Path, help='Merge completed ARC dense scores and partial counts from this run')
    args = p.parse_args(argv)
    try:
        import openpyxl  # Fail before spending time auditing large video files.
    except ImportError:
        p.error('Install report dependencies: python -m pip install -r requirements-report.txt')
    if args.output_dir.resolve() == (ROOT / 'docs').resolve():
        p.error('Use reports/verified; scattered tables under docs have been retired')
    if args.snapshot_only:
        snapshot = read(args.output_dir / 'audit.json')
    else:
        if not args.base_eval:
            p.error('--base-eval is required for a fresh audit')
        snapshot = collect(args.base_eval.resolve(), args.verify_video_bytes)
    if args.refresh_dense_run:
        refresh_dense(snapshot, args.refresh_dense_run.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(snapshot, args.output_dir)
    (args.output_dir / 'audit.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output_dir), tiny_cells=len(snapshot['tiny']),
                         full_cells=len(snapshot['full']), video_configs=len(snapshot['videos'])), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
