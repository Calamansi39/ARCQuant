"""Audit existing experiment artifacts and export a shareable, colored workbook.

No model inference or re-grading is performed. Rebuild from the committed audit
snapshot with --snapshot-only; a fresh audit requires the original run artifacts.
Only the presentation step needs openpyxl (pip install -r requirements-report.txt).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'DeepSeek-V4-Flash-0731'
QUANTS = ('bf16', 'nvfp4', 'mxfp4', 'hifp4')
LABELS = ('BF16', 'NVFP4', 'MXFP4', 'HiFP4')
# QUANTS above enumerates existing files for provenance checks; it never launches
# evaluations. Presentation follows the requested format for each model.
TARGET_QUANT = {MODEL: 'hifp4', 'MiniMax-H3': 'hifp4', 'MAGI-2-preview': 'hifp4'}


def requested_result(row):
    precision = row.get('precision', row['variant']).lower().split()[0].split('-')[0]
    precision = 'hifp4' if precision == 'hif4' else precision
    return precision in ('bf16', TARGET_QUANT[row['model']])


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonl(path, keys):
    # LongBench contexts are large: do not retain question/context/answer text.
    with Path(path).open() as f:
        return [{k: r[k] for k in keys if k in r}
                for line in f if line.strip() for r in [json.loads(line)]]


def check_ids(rows, key, expected):
    ids = [str(r[key]) for r in rows]
    require(len(ids) == len(set(ids)), f'duplicate {key}')
    require(set(ids) == {str(x) for x in expected}, f'{key} coverage mismatch')
    return canonical_hash(sorted(ids))


def audit(base):
    rows, videos, rejected = [], [], []
    root = base / 'results' / MODEL

    def evidence(paths):
        out = []
        for p in paths:
            label = ('base_eval/' + str(p.relative_to(base))) if p.is_relative_to(base) else str(p.relative_to(ROOT))
            out.append({'path': label, 'sha256': digest(p), 'bytes': p.stat().st_size})
        return out

    def result(group, benchmark, variant, method, n, correct, protocol, paths, **extra):
        sources = evidence(paths)
        rows.append(dict(result_id=f'R{len(rows)+1:02}', model=MODEL, group=group,
                         benchmark=benchmark, profile='full', variant=variant,
                         precision='BF16 reference' if variant.lower() == 'bf16' else 'HiF4 W4A4' if group == 'methods' else variant+' W4A4',
                         method=method, evaluated_count=n, correct=correct,
                         metric='Accuracy' if benchmark == 'LongBench v2' or group == 'methods' else 'Pass@1',
                         score=correct/n*100, score_unit='percent', protocol=protocol,
                         status='已核查完整', source=sources[0]['path'],
                         evidence=sources, **extra))

    def guarded(label, fn):
        try:
            fn()
        except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
            rejected.append({'item': label, 'reason': str(exc)})

    for sub, bench, n in [('gpqa_generative', 'GPQA Diamond', 198), ('math500_generative', 'MATH-500', 500)]:
        for q, label in zip(QUANTS, LABELS):
            def run():
                p = root/sub/q
                s, pr = read(p/'summary.json'), read(p/'protocol.json')
                raw = p/s['scored_outputs_path']
                rs = jsonl(raw, ('id', 'sample_index', 'score', 'generation', 'protocol_sha256'))
                ids_hash = check_ids(rs, 'id', pr['selected_ids'])
                cases = read(p/'cases.json')
                check_ids(cases, 'id', pr['selected_ids'])
                require(len(rs) == n == s['completed_candidates'] == s['questions'] == pr['dataset']['expected_count'], 'count mismatch')
                require(s['complete'] and s['is_full_dataset'] and not s['is_smoke_test'], 'incomplete run')
                require(all(r['sample_index'] == 0 and type(r['score']['correct']) is bool for r in rs), 'invalid candidate/score')
                require(all(r['score']['rules_version'] == s['scoring_rules_version'] for r in rs), 'mixed scoring versions')
                correct = sum(r['score']['correct'] for r in rs)
                truncated = sum(r['generation']['finish_reason'] == 'length' for r in rs)
                require(correct == s['correct_candidates'] and math.isclose(correct/n, s['pass_at_1']), 'score mismatch')
                require(truncated == s['truncated'], 'truncation count mismatch')
                require(pr['fewshot'] == 0 and pr['generation']['n'] == 1, 'unexpected generation protocol')
                result('baseline', bench, label, 'none', n, correct,
                       s['protocol_version']+'; '+s['scoring_rules_version'],
                       [p/'summary.json', raw, p/'protocol.json', p/'cases.json', p/s['scoring_manifest_path']],
                       truncated=truncated, ids_sha256=ids_hash, dataset_sha256=pr['dataset']['items_sha256'],
                       notes='0-shot thinking；8192 生成 tokens；截断计错并保留分母；非官方榜单复现',
                       generation={k:pr['generation'][k] for k in ('max_new_tokens','temperature','top_p','top_k','seed')})
            guarded(bench+' '+label, run)

    for q, label in zip(QUANTS, LABELS):
        def run():
            p = root/'longbench_v2'
            name = 'bf16' if q == 'bf16' else q+'-w4a4'
            raw = p/(name+'.jsonl')
            sp = p/(name+'.jsonl.summary.json')
            s = read(sp)
            scored = p/s['scored_outputs_path']
            rs = jsonl(raw, ('_id','generation'))
            scores = jsonl(scored, ('_id','judge'))
            require(len(rs) == len(scores) == s['evaluated'] == s['num'] == 503, 'count mismatch')
            ids_hash = check_ids(scores, '_id', [r['_id'] for r in rs])
            require(len({r['_id'] for r in rs}) == 503, 'duplicate raw ids')
            require(digest(raw) == s['source_jsonl_sha256'], 'scored source hash mismatch')
            require(all(type(r['judge']) is bool for r in scores), 'invalid judge')
            correct = sum(r['judge'] for r in scores)
            require(abs(correct/503*100-s['overall']) < .0051, 'score mismatch')
            config = p/(name+'.jsonl.config.json')
            cfg = read(config)
            require(cfg['limit'] == -1, 'subset run')
            result('baseline', 'LongBench v2', label, 'none', 503, correct, s['scoring_version'],
                   [sp, raw, scored, config], ids_sha256=ids_hash,
                   truncated=sum(r['generation'].get('finish_reason') == 'length' for r in rs),
                   notes='503 题；输入上限 131072 tokens，中间截断；输出上限 8192；chat 模式',
                   generation={k:cfg[k] for k in ('max_input_tokens','max_new_tokens','temperature','top_p','top_k','input_truncation')})
        guarded('LongBench '+label, run)

    for sub, bench in [('livecodebench_v6_64k_n1_batch16','LiveCodeBench v6 64K'),
                       ('livecodebench_v6_128k_n1_continued_batch8','LiveCodeBench v6 128K 续写')]:
        for q, label in zip(QUANTS, LABELS):
            def run():
                name = q if '128k' in sub or q == 'bf16' else q+'-w4a4'
                raw = root/sub/(name+'.jsonl')
                sp = root/sub/(name+'.jsonl.summary.json')
                d = read(sp); s, pr = d['summary'], d['protocol']
                rs = jsonl(raw, ('question_id','generation','protocol_hash'))
                ids_hash = check_ids(rs, 'question_id', pr['data']['question_ids'])
                require(len(rs) == s['evaluated'] == pr['data']['count'] == 175 and s['n'] == 1, 'count mismatch')
                require(all(r['protocol_hash'] == s['protocol_hash'] for r in rs), 'mixed protocols')
                detail, tests = d['per_problem'][0]['detail']['pass@1'], d['per_problem'][1]
                require(set(detail) == set(tests) == {str(i) for i in range(175)}, 'incomplete execution grading')
                outcomes = {}
                for i, v in tests.items():
                    require(len(v) == 1 and len(v[0]) > 0, 'missing execution result')
                    outcomes[i] = int(all(x is True or x == 1 for x in v[0]))
                    require(outcomes[i] == detail[i], 'per-problem score mismatch')
                correct = sum(outcomes.values())
                require(math.isclose(correct/175, s['pass@1']), 'aggregate score mismatch')
                if 'source_sha256' in d:
                    require(digest(raw) == d['source_sha256'], 'scored source hash mismatch')
                continued = 'continuation' in pr
                if continued:
                    require(pr['continuation']['kind'] == 'natural-prefix-continuation', 'unknown continuation')
                result('baseline', bench, label, 'none', 175, correct,
                       s['scorer']+('; natural-prefix-continuation' if continued else '; independent-64k'),
                       [sp, raw, root/sub/(name+'.jsonl.config.json')], ids_sha256=ids_hash,
                       truncated=s['generation_diagnostics']['length'],
                       notes='固定 release-v6 test6 分片 175 题；public+private 执行；'+('64K 到限题继续生成，已结束答案复用；不是独立 128K 采样' if continued else '独立采样；生成预算 65536 tokens'),
                       dataset_revision=pr['data']['revision'], dataset_files=pr['data']['files'],
                       generation={k:pr['generation'][k] for k in ('n','max_new_tokens','temperature','top_p','top_k')})
            guarded(bench+' '+label, run)

    for variant, label, method in [('bf16','BF16','none'),('hif4','HiF4 W4A4','none'),('arc','HiF4 + ARC','ARC'),
                                   ('teal','HiF4 + ARC + TEAL 50%','ARC + TEAL 50%'),
                                   ('wina','HiF4 + ARC + WINA mask-only 50%','ARC + WINA mask-only 50%')]:
        def run():
            sp = ROOT/'runs/deepseek_formal'/f'gpqa_{variant}_full.json'
            d = read(sp); rs = d['rows']
            ids_hash = check_ids(rs, 'id', range(198))
            require(len(rs) == d['completed_questions'] == d['requested_questions'] == 198, 'count mismatch')
            require(d['status'] == 'completed' and d['full_dataset'], 'incomplete run')
            for r in rs:
                require(len(r['scores']) == 4 and all(math.isfinite(x) for x in r['scores']), 'invalid likelihoods')
                require(max(range(4), key=lambda i:r['scores'][i]) == r['predicted'], 'prediction mismatch')
                require((r['target'] == r['predicted']) == r['correct'], 'grading mismatch')
            correct = sum(r['correct'] for r in rs)
            require(correct == d['correct'] and math.isclose(correct/198, d['accuracy']), 'score mismatch')
            result('methods', 'GPQA Diamond（条件似然）', label, method, 198, correct, d['protocol'], [sp],
                   ids_sha256=ids_hash, dataset_sha256=d['dataset_sha256'],
                   question_set_sha256=canonical_hash(sorted((r['id'],r['question_sha256'],r['permutation']) for r in rs)),
                   runtime_revision=d['arcquant_revision'], implementation_sha256=d['implementation_sha256'],
                   sparsity=d['sparsity'], notes='历史实现快照；dense GEMM 精度实验；条件似然非生成式；'+('WINA mask-only，无 optional transform' if variant=='wina' else '非 2:4 结构化稀疏成绩'))
        guarded('GPQA likelihood '+label, run)

    # Never promote video Tiny/reference summaries into the full-benchmark table.
    for mod, pre in [('MiniMax-H3',x) for x in ('bf16','nvfp4-w4a4','mxfp4-w4a4','hifp4-w4a4')] + [('MAGI-2-preview','bf16-50p5')]:
        def run():
            p = base/'results'/mod/'vbench_standard'/pre/'tiny'
            sp = p/'scoring/static-coverage-v2/vbench_summary.json' if mod == 'MiniMax-H3' else p/'vbench_summary.json'
            d = read(sp); c = d['temporal_flickering_coverage']
            require(len(d['scores']) == 16 and not d['is_full_prompt_suite'], 'unexpected video protocol')
            dimensions = []; paths = [sp, base/'vbench_tiny_manifest.json']
            manifest = read(paths[1])
            require(d['subset_manifest_sha256'] == digest(paths[1]), 'subset manifest hash mismatch')
            for dim, path in d['source_files'].items():
                # MiniMax results were copied from a different machine; relocate only this known root.
                original = Path(path)
                rel = str(original).split('/base_eval/',1)[1]
                source = base/rel
                value, records = read(source)[dim]
                require(math.isclose(value, d['scores'][dim]), 'dimension summary mismatch')
                require(math.isfinite(value) and len(records) > 0, 'invalid dimension score')
                dimensions.append(dict(dimension=dim, score=value, scored_records=len(records)))
                paths.append(source)
            total = d['tiny_scores']['total']*100
            require(math.isfinite(total) and 0 <= total <= 100, 'invalid video total')
            videos.append(dict(result_id=f'V{len(videos)+1:02}', model=mod, variant=pre,
                               benchmark='VBench Tiny', profile='tiny', total=total,
                               quality=d['tiny_scores']['quality']*100, semantic=d['tiny_scores']['semantic']*100,
                               prompts=manifest['unique_prompts'], planned_videos=manifest['videos_per_precision'],
                               flicker_scored=c['passed_videos'], flicker_expected=c['target_videos'],
                               status='仅 Tiny 参考；非全量' if c['has_complete_static_coverage'] else 'Tiny 覆盖不足；仅参考',
                               dimensions=dimensions, subset_manifest_sha256=d['subset_manifest_sha256'],
                               notes='16 维报告已核对；视频生成文件未逐个验证；color 维度可能按检测结果筛选；不声称完整官方 benchmark',
                               source=evidence([sp])[0]['path'], evidence=evidence(paths)))
        guarded(mod+' '+pre+' Tiny', run)

    # Pairwise comparisons require identical datasets/protocols, not just equal counts.
    for bench in {r['benchmark'] for r in rows}:
        group = [r for r in rows if r['benchmark'] == bench]
        require(len({r['ids_sha256'] for r in group}) == 1, bench+' cross-variant ID mismatch')
        require(len({r['protocol'] for r in group}) == 1, bench+' cross-variant protocol mismatch')
        for key in ('question_set_sha256', 'dataset_sha256', 'generation'):
            require(len({canonical_hash(r[key]) for r in group if key in r}) <= 1, bench+' cross-variant '+key+' mismatch')
        ref = next((r for r in group if r['variant'] == 'BF16'), None)
        for r in group:
            r['delta_bf16_pp'] = r['score']-ref['score'] if ref else None
    return dict(schema_version=1, audited_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                scope='Existing complete text evaluations; video Tiny reference appendix; no new inference or re-grading',
                rows=rows, video_reference=videos, rejected=rejected)


GAPS = [
    ['DeepSeek', 'GPQA 生成式', 'TEAL / WINA', '未完成', '目前只有条件似然历史对照；需相同生成式协议的全量结果'],
    ['DeepSeek', 'MATH-500', 'TEAL / WINA', '未完成', '此前短测失败；修复后的任务已建队列，不能用 smoke 分数填表'],
    ['DeepSeek', 'LongBench v2 / LiveCodeBench', 'TEAL / WINA', '未完成', '需完成整套生成、逐题评分和覆盖核查'],
    ['MiniMax-H3', 'VBench', '全部', '仅 Tiny 参考', 'BF16 闪烁覆盖 46/50；无全量官方 suite 和稀疏方法成绩'],
    ['MAGI-2-preview', 'VBench', '全部', '仅 BF16 Tiny 参考', '缺其他精度及 TEAL/WINA 的完整结果'],
    ['适用模型', 'Terminal-Bench', '全部', '未完成', '缺完整 agent / 环境 / 任务评分报告'],
    ['目标模型', '全部', 'HiF4 + 2:4', '未完成', '当前结构与 smoke 检查不构成完整 benchmark 成绩'],
]


def export(snapshot, out):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in snapshot['rows'] if requested_result(r)]
    videos = [v for v in snapshot['video_reference'] if requested_result(v)]
    blue, green, red, amber, purple = '17365D', 'E2F0D9', 'FCE4D6', 'FFF2CC', 'E4DFEC'
    wb = Workbook(); wb.remove(wb.active)

    def sheet(name, headers, data, widths=None, note=None):
        ws = wb.create_sheet(name); ws.sheet_view.showGridLines = False
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = 'landscape'; ws.page_setup.paperSize = ws.PAPERSIZE_A3
        ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0
        ws.append([name]); ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(headers))
        ws['A1'].font=Font(name='Microsoft YaHei',size=18,bold=True,color='FFFFFF'); ws['A1'].fill=PatternFill('solid',fgColor=blue)
        ws.row_dimensions[1].height=35
        ws.append([note or '核查日期：'+snapshot['audited_at']]); ws.merge_cells(start_row=2,start_column=1,end_row=2,end_column=len(headers))
        ws['A2'].font=Font(name='Microsoft YaHei',size=11,color='666666'); ws['A2'].alignment=Alignment(wrap_text=True,vertical='center'); ws.row_dimensions[2].height=48
        ws.append(headers)
        for c in ws[3]:
            c.fill=PatternFill('solid',fgColor=blue); c.font=Font(name='Microsoft YaHei',bold=True,color='FFFFFF'); c.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[3].height=32
        for values in data:
            ws.append(values)
            for c in ws[ws.max_row]:
                c.font=Font(name='Microsoft YaHei',size=11,color='203040'); c.alignment=Alignment(wrap_text=True,vertical='center')
                c.fill=PatternFill('solid',fgColor='F0F5FA' if c.row%2==0 else 'FFFFFF')
                c.border=Border(bottom=Side(style='hair',color='D8E2EC'))
                if isinstance(c.value, float): c.number_format='0.00'
                if isinstance(c.value,str) and any(k in c.value for k in ('未完成','覆盖不足','仅 Tiny','仅 BF16 Tiny')):
                    c.fill=PatternFill('solid',fgColor=amber)
            ws.row_dimensions[ws.max_row].height=48
        ws.freeze_panes='C4'; ws.auto_filter.ref=f'A3:{get_column_letter(len(headers))}{ws.max_row}'
        ws.print_title_rows='1:3'; ws.print_options.horizontalCentered=True
        for i in range(1,len(headers)+1): ws.column_dimensions[get_column_letter(i)].width=(widths[i-1] if widths and i<=len(widths) else 24)
        if data:
            table=Table(displayName=f'Table{len(wb.worksheets)}',ref=ws.auto_filter.ref)
            table.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=False)
            ws.add_table(table)
        return ws

    info=[
        ['主表范围',f'{len(rows)} 项完整文本评测；视频 {len(videos)} 项仅作 Tiny 附录；成绩均由原始文件核查'],
        ['目标格式','按当前任务指定的 HiF4 W4A4 展示，保留 BF16 计算参考。NVFP4/MXFP4 历史结果只保存在审计快照，不列为本任务成果。'],
        ['历史结果来源','外部 base_eval 的量化基线为已有记录。本次只读取、核对与汇总，没有为制作表格重新跑模型。'],
        ['阅读顺序','先看文本基线，再看 GPQA 方法对照；两者协议不同，不能混比'],
        ['完整的含义','固定题集全部有输出及评分；保留答错、截断和执行失败；不等于方法达标'],
        ['BF16 的含义','DeepSeek 发布的是混合 FP8/FP4 权重；这里 BF16 指反量化后计算参考，并非原始全 BF16 训练权重'],
        ['FP4 的含义','W4A4 数值模拟/重建后 dense GEMM；量化范围见运行协议；本表没有速度或硬件 FP4 性能结论'],
        ['历史方法结果','GPQA 方法对照来自记录内代码快照；TEAL/WINA 50% 为非结构化输入掩码；WINA 为 mask-only'],
        ['128K 续写','LiveCodeBench 保留 64K 已结束答案，只续写到限题；不能当作独立 128K 采样'],
        ['Tiny 的含义','108 个 prompts、计划 740 次生成、16 维评分；不等于 VBench 全量'],
        ['颜色含义','蓝色＝量化基线；紫色＝方法对照；黄色＝参考/待完成；差值绿＝上升、红＝下降'],
        ['分享方式','GitHub 文档可直接分享；下载本 XLSX 导入腾讯文档/飞书/Excel Online 可共同编辑，颜色已写在文件里'],
        ['证据边界','重新汇总现有评分，未重跑模型或执行判题；来源哈希可核对，无题目正文、模型输出或模型权重上传'],
        ['核查时间',snapshot['audited_at']],
    ]
    sheet('阅读说明',['项目','说明'],info,[24,112])
    baselines=[r for r in rows if r['group']=='baseline']
    benchmarks=list(dict.fromkeys(r['benchmark'] for r in baselines))
    matrix=[]
    display_labels=('BF16', next(label for q,label in zip(QUANTS,LABELS) if q == TARGET_QUANT[MODEL]))
    for bench in benchmarks:
        group=[r for r in baselines if r['benchmark']==bench]
        matrix.append([MODEL,bench,group[0]['evaluated_count'],group[0]['metric']]+[next((r['score'] for r in group if r['variant']==q),'未完成') for q in display_labels]+[group[0]['notes']])
    ws=sheet('文本全量基线',['模型','Benchmark','题数','指标','BF16 参考 (%)','HiF4 W4A4 目标 (%)','协议说明'],matrix,[30,33,10,15,20,23,66],
             '仅展示本任务指定的 HiF4 和 BF16 参考。成绩复用服务器已有结果，本次整理没有重新跑模型。每行分数越高越好。')
    for row in ws.iter_rows(min_row=4,min_col=5,max_col=6):
        for c in row: c.fill=PatternFill('solid',fgColor='DDEBF7'); c.font=Font(bold=True,size=13,color=blue)
    methods=[r for r in rows if r['group']=='methods']
    ws=sheet('GPQA方法对照',['方法配置','正确题数','总题数','准确率 (%)','较 BF16 (pp)','状态','解释'],
             [[r['variant'],r['correct'],r['evaluated_count'],r['score'],r['delta_bf16_pp'],r['status'],r['notes']] for r in methods],
             [37,14,12,18,20,20,85], '198 题完整条件似然评测；不是生成式 Pass@1。历史实现结果，不代表当前修复版本已重跑。')
    for row in ws.iter_rows(min_row=4):
        row[0].fill=PatternFill('solid',fgColor=purple)
        if isinstance(row[4].value,(float,int)):
            row[4].fill=PatternFill('solid',fgColor=green if row[4].value>=0 else red); row[4].number_format='+0.00;-0.00;0.00'
    headers=['编号','模型','Benchmark','配置','精度','方法','完成题数','正确题数','分数 (%)','较 BF16 (pp)','截断数','协议','来源','说明']
    values=[[r['result_id'],r['model'],r['benchmark'],r['variant'],r['precision'],r['method'],r['evaluated_count'],r['correct'],r['score'],r['delta_bf16_pp'],r.get('truncated','不适用'),r['protocol'],r['source'],r['notes']] for r in rows]
    sheet('完整结果明细',headers,values,[10,30,32,37,22,31,13,13,16,18,12,58,70,85])
    sheet('视频Tiny仅供参考',['编号','模型','配置','Tiny 总分','Quality','Semantic','Prompt 数','计划生成数','闪烁评分数','闪烁目标数','状态','说明'],
          [[v['result_id'],v['model'],v['variant'],v['total'],v['quality'],v['semantic'],v['prompts'],v['planned_videos'],v['flicker_scored'],v['flicker_expected'],v['status'],v['notes']] for v in videos],
          [10,25,24,16,16,16,15,15,16,16,30,75], '不纳入完整 benchmark 主表。计划生成数不代表已逐个核验的视频数；MAGI 使用官方聚合总分，非维度简单平均。')
    sheet('视频维度明细',['编号','模型','配置','维度','维度原始分','评分记录数'],
          [[v['result_id'],v['model'],v['variant'],d['dimension'],d['score'],d['scored_records']] for v in videos for d in v['dimensions']],
          [10,25,24,32,20,20], '原始维度分数不能直接与归一化聚合总分比较；color 输出数可能受检测筛选影响。')
    sheet('待补齐与未纳入',['模型','Benchmark','方法','状态','下一步或原因'],GAPS+[['审计未通过',r['item'],'','未纳入',r['reason']] for r in snapshot['rejected']], [24,39,24,25,90])
    sheet('证据索引',['结果编号','文件路径（相对根目录）','SHA256','字节数'],
          [[r['result_id'],e['path'],e['sha256'],e['bytes']] for r in rows+videos for e in r['evidence']], [14,105,72,18],
          'base_eval/ 相对外部评测器根目录；runs/ 相对 ARCQuant。哈希锁定本次核查内容，原始逐题大文件不随表公开。')
    for name in ('完整结果明细','视频维度明细','待补齐与未纳入','证据索引'):
        wb[name].sheet_state='hidden'
    wb.active=wb.sheetnames.index('文本全量基线')
    workbook=out/'completed_benchmarks.xlsx'; wb.save(workbook)
    check=load_workbook(workbook,data_only=True)
    require(check['完整结果明细'].max_row == len(rows)+3, 'XLSX row count mismatch')
    require(check['文本全量基线']['E4'].fill.fgColor.rgb.endswith('DDEBF7'), 'XLSX colors missing')
    check.close()

    fields=['result_id','model','benchmark','profile','variant','precision','method','evaluated_count','correct','metric','score','score_unit','delta_bf16_pp','protocol','status','source','notes']
    for ext, sep in [('csv',','),('tsv','\t')]:
        with (out/f'completed_benchmarks.{ext}').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,delimiter=sep,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (out/'video_tiny_reference.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['result_id','model','variant','benchmark','profile','total','quality','semantic','prompts','planned_videos','flicker_scored','flicker_expected','status','source','notes']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(videos)

    md=['# 实验结果展示：已核查的完整评测','',f'核查时间：{snapshot["audited_at"]}。本页由原始结果导出；本次整理没有新增模型评测。','',
        '[下载彩色 Excel](completed_benchmarks.xlsx?raw=true) · [CSV](completed_benchmarks.csv) · [TSV](completed_benchmarks.tsv) · [核查快照与证据哈希](results_audit.json)','',
        f'**按目标格式筛选：完整文本成绩 {len(rows)} 项；视频 Tiny 参考 {len(videos)} 项。** 保留 BF16 参考和指定的 HiF4 W4A4。“完整”表示固定题集有完整输出与评分，不表示稀疏方法已达到最终目标。','',
        'NVFP4/MXFP4 来自服务器已有的历史结果；本次整理没有为它们新增推理任务。旧版把这些额外格式也放入了主表，现已移除。原始核查快照保留历史记录供追查，不计入本任务展示数量。','',
        '## 1. DeepSeek 文本基线','',
        '模型为 `DeepSeek-V4-Flash-0731`。只看 BF16 参考与指定的 HiF4 W4A4，后者在外部结果文件中命名为 HiFP4。下面尚未加入 TEAL/WINA；分数单位为 %，越高越好。','',
        '| Benchmark / 协议 | 题数 | BF16 参考 | HiF4 W4A4（目标） |',
        '|---|---:|---:|---:|']
    for m in matrix:
        md.append('| '+' | '.join([m[1],str(m[2])]+[f'{v:.2f}' if isinstance(v,(float,int)) else v for v in m[4:6]])+' |')
    md+=['','- GPQA / MATH：0-shot thinking，8192 生成 tokens；分别用 `gpqa_final_letter_v4` / `math500_format_v3` 判分。',
         '- LongBench：503 题，chat 模式，输入上限 131072 tokens，中间截断；输出上限 8192。',
         '- LiveCodeBench：固定 release-v6 `test6` 分片 175 题，public + private 测试执行；不泛指所有版本题库。128K 为 64K 到限题续写，保留此前已结束的答案。',
         '- BF16 是已发布混合 FP8/FP4 checkpoint 反量化后的计算参考；不是恢复出原始 BF16 训练权重。FP4 是指定投影的 W4A4 数值模拟/重建后 GEMM，不能据此声称硬件 FP4 或稀疏加速。',
         '- 截断、答错、空代码及执行失败均留在完整分母中，未为提高分数删题。每项截断数见 Excel 明细。','',
         '## 2. GPQA 完整方法对照（独立协议）','',
         '**这是四选项条件似然准确率，不能与上面的 GPQA 生成式 Pass@1 横向比较。** 全部为 198 题。','',
         '| 方法 | 正确题数 | 准确率 | 较同协议 BF16 |','|---|---:|---:|---:|']
    for r in methods:
        delta='—' if r['delta_bf16_pp'] is None else f'{r["delta_bf16_pp"]:+.2f} pp'
        md.append(f'| {r["variant"]} | {r["correct"]}/198 | {r["score"]:.2f}% | {delta} |')
    md+=['','这组结果来自记录内的历史代码快照（版本和实现哈希见核查文件），不能解释为后续修复代码已完成重跑。TEAL/WINA 的 50% 是非结构化输入掩码；WINA 为 mask-only，没有 optional transform；当前使用 dense GEMM。它们是精度对照证据，不是 2:4 结构化稀疏或端到端加速成果。','',
         '## 3. 视频 Tiny 附录（不计入完整 benchmark 主表）','',
         '固定 Tiny manifest：108 个 prompts、计划 740 次生成、16 个维度。以下来自已保存的评分报告；本次未逐个验证全部视频文件。**没有完整 VBench 官方 suite 成绩。**','',
         '| 模型 | 配置 | Tiny 聚合总分 ×100 | 闪烁覆盖 | 状态 |','|---|---|---:|---:|---|']
    for v in videos:
        md.append(f'| {v["model"]} | {v["variant"]} | {v["total"]:.2f} | {v["flicker_scored"]}/{v["flicker_expected"]} | {v["status"]} |')
    md+=['','总分采用报告中的 `tiny_scores.total × 100`。MAGI 旧表的 `0.6372` 是维度简单平均，已纠正为该报告的聚合总分；MiniMax BF16 闪烁维度不足，不用于完整同条件差值。','',
         '## 4. 离最终方法结果表还缺什么','',
         '| 模型 | Benchmark | 方法 | 状态 / 待补全 |','|---|---|---|---|']
    for g in GAPS: md.append('| '+' | '.join([g[0],g[1],g[2],g[3]+'：'+g[4]])+' |')
    md+=['','## 5. 核查与分享说明','',
         '- 文本：核对唯一题号、数量、逐题评分和总分；LCB 从已保存的执行结果重新汇总；条件似然检查有限数值、argmax 和对错标签。各精度间核对题集与协议。没有重新执行模型或判题器。',
         '- 视频：核对 16 维报告与 summary、Tiny manifest 哈希和闪烁覆盖；仅作参考。',
         '- 修正旧表：MATH-500 的来源改为 `math500_generative/...`；旧路径 `math500/...` 属于另一套 4-shot 协议，不能给 0-shot 分数作证。',
         '- Excel 默认显示阅读说明和三张成绩页；四张核查/待补全附页默认隐藏，可在表格软件中取消隐藏。CSV/TSV 与主展示范围一致，只承载数据，没有颜色。',
         '- 可直接分享本 GitHub 页面。需要多人编辑时，下载 XLSX，在腾讯文档、飞书或 Excel Online 中选择导入，再设置平台内分享权限。本次交付未创建这些平台的在线文档。',
         '- 只发布汇总指标和相对来源路径/哈希；权重、题目正文、逐题生成内容及本机私人文件不包含在共享包内。','',
         '### 更新方法','',
         '```bash','pip install -r requirements-report.txt',
         '# 有原始结果的机器：重新核查并导出',
         'python benchmarks/present_results.py --base-eval /path/to/base_eval',
         '# 其他机器：从仓库已核查快照重新导出，不需要模型/GPU/原始题目',
         'python benchmarks/present_results.py --snapshot-only','```','',
         '核查快照锁定本次来源文件的 SHA256。原始结果变化后应重新核查；有缺失或分数不一致的条目会进入拒绝清单，不能自动填入主表。','']
    if snapshot['rejected']:
        md += ['### 未通过核查的候选','']+[f'- {x["item"]}：{x["reason"]}' for x in snapshot['rejected']]+['']
    (out/'RESULTS_PRESENTATION.md').write_text('\n'.join(md))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-eval',type=Path,help='Root containing external results/ and vbench_tiny_manifest.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'docs')
    p.add_argument('--snapshot-only',action='store_true')
    args=p.parse_args()
    dest=args.output_dir/'results_audit.json'
    if args.snapshot_only:
        snapshot=read(dest)
    else:
        if not args.base_eval: p.error('--base-eval is required for a fresh audit')
        snapshot=audit(args.base_eval.resolve())
        args.output_dir.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    export(snapshot,args.output_dir)
    print(json.dumps({'audited_complete_results':len(snapshot['rows']),
                      'display_complete_results':sum(requested_result(r) for r in snapshot['rows']),
                      'display_video_reference':sum(requested_result(v) for v in snapshot['video_reference']),
                      'rejected':snapshot['rejected'],'workbook':str(args.output_dir/'completed_benchmarks.xlsx')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
