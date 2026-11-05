"""Audit existing experiment artifacts and export a shareable, colored workbook.

No model inference or re-grading is performed. Rebuild from the committed audit
snapshot with --snapshot-only; a fresh audit requires the original run artifacts.
Only the presentation step needs openpyxl (pip install -r requirements-report.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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


def export(snapshot, out):
    from benchmarks.result_tables import export as export_tables
    export_tables(snapshot, out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-eval',type=Path,help='Root containing external results/ and vbench_tiny_manifest.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'docs')
    p.add_argument('--snapshot-only',action='store_true')
    p.add_argument('--math-summary', type=Path, action='append', default=[],
                   help='Refresh a sparse MATH-500 summary; repeatable, no inference')
    p.add_argument('--sweep-status', type=Path, action='append', default=[],
                   help='Refresh a sparse sweep status snapshot; repeatable')
    args=p.parse_args()
    dest=args.output_dir/'results_audit.json'
    if args.snapshot_only:
        snapshot=read(dest)
    else:
        if not args.base_eval: p.error('--base-eval is required for a fresh audit')
        snapshot=audit(args.base_eval.resolve())
        args.output_dir.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    if args.math_summary or args.sweep_status:
        from benchmarks.result_tables import refresh_progress
        refresh_progress(snapshot, args.math_summary, args.sweep_status)
        dest.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    export(snapshot,args.output_dir)
    print(json.dumps({'audited_complete_results':len(snapshot['rows']),
                      'display_complete_results':sum(requested_result(r) for r in snapshot['rows']),
                      'archived_video_reference':sum(requested_result(v) for v in snapshot['video_reference']),
                      'rejected':snapshot['rejected'],'workbook':str(args.output_dir/'completed_benchmarks.xlsx')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
