"""Publish only complete ARC extended results with per-rank execution evidence."""
from pathlib import Path
import math
import os
import sys

from benchmarks.verified_results import read, jsonl, require, digest, evidence


def runtime_evidence(folder, identity, ragged):
    files = []
    require(identity['arc'] and identity['sparsity'] == 0 and identity['method'] == 'none', 'Not dense ARC')
    for rank in range(2):
        path = folder/f'arcquant_runtime_rank{rank}.json'
        stats = read(path)
        require(stats['runtime_sha256'] == identity['runtime_sha256'] and stats['run_succeeded'], 'Failed/mixed ARC run')
        require(stats['arc'] and stats['sparse_method'] == 'none' and stats['sparse_sparsity'] == 0, 'Wrong runtime mode')
        require(stats['arc_finite_logits_checks'] > 0 and stats['arc_residual_calls'] > 0, 'No ARC execution')
        require(set(stats['arc_executed_families']) == {'attention','grouped_wo_a','routed_expert','shared_expert'}, 'Incomplete ARC families')
        require(stats['arc_attention_paths'].get('standard', 0) > 0, 'No ARC prefill')
        if ragged:
            require(stats['arc_attention_paths'].get('ragged', 0) > 0, 'No ARC batch decode')
        files.append(path)
    return files


def audit(directory, stage, base):
    os.environ.setdefault('HF_HOME', str(base.parent/'data/hf_home'))
    sys.path.insert(0, str(base))
    from lcb_protocol import digest as protocol_hash
    continuation = stage == 'lcb128'
    raw = directory/('hifp4.jsonl' if continuation else 'hifp4-w4a4.jsonl')
    cfg, rows, report = read(str(raw)+'.config.json'), jsonl(raw), read(str(raw)+'.summary.json')
    files = [raw, Path(str(raw)+'.config.json'), Path(str(raw)+'.summary.json')]
    if continuation:
        run = directory/'runs/hifp4'
        manifest = read(run/'run.json')
        source = Path(manifest['assets']['source_dir'])
        source_config = read(source/'hifp4-w4a4.jsonl.config.json')
        identity = source_config['runtime_signature']['arcquant']
        require(cfg['protocol']['runtime']['arcquant_extension/benchmarks/run_deepseek_arc_extended.py'] ==
                identity['implementation_sha256']['benchmarks/run_deepseek_arc_extended.py'], 'Unbound ARC continuation code')
        # A zero-continuation run legitimately reuses all source EOS answers.
        if any(r['continuation']['kind'] != 'source-eos-reuse' for r in rows):
            files += runtime_evidence(run, identity, True)
        original = Path(manifest['assets']['original_source_dir'])
        files += runtime_evidence(original, identity, True)
        files.append(run/'run.json')
        for name, expected in manifest['protocol']['continuation']['input_hashes'].items():
            require(digest(source/name) == expected, 'Source continuation snapshot changed')
            files.append(source/name)
    else:
        identity = cfg['runtime_signature']['arcquant']
        require(read(directory/'arcquant_sparse_runtime.json') == identity, 'Unbound ARC metadata')
        files += runtime_evidence(directory, identity, stage == 'lcb64')
        files.append(directory/'arcquant_sparse_runtime.json')
        for name, expected in identity['implementation_sha256'].items():
            path = directory/'arcquant_sources'/name
            require(digest(path) == expected, 'ARC source snapshot changed')
            files.append(path)
    if stage == 'longbench':
        import longbench_policy
        from datasets import load_dataset
        items = list(load_dataset('THUDM/LongBench-v2', split='train'))
        selected = {item['_id']:item for item in items}
        require(len(rows) == len(selected) == 503 and {r['_id'] for r in rows} == set(selected), 'LongBench coverage mismatch')
        require(all(r['answer'] == selected[r['_id']]['answer'] for r in rows), 'LongBench answers changed')
        for key, expected in dict(limit=-1, quant='hifp4', mode='w4a4', max_input_tokens=131072,
            max_new_tokens=8192, thinking_mode='chat', temperature=.1, top_p=.95, top_k=20).items():
            require(cfg[key] == expected, 'LongBench protocol mismatch: '+key)
        scores = jsonl(directory/report['scored_outputs_path'])
        require(report['scoring_version'] == longbench_policy.SCORING_VERSION and
                report['source_jsonl_sha256'] == digest(raw) and
                report['scoring_source_sha256'] == digest(longbench_policy.__file__), 'Unbound LongBench scores')
        actual = {r['_id']:longbench_policy.score_record(r) for r in rows}
        require(len(scores) == 503 and {r['_id'] for r in scores} == set(actual) and
                all(all(s[k] == actual[s['_id']][k] for k in ('pred','judge')) for s in scores), 'LongBench grading mismatch')
        correct = sum(s['judge'] for s in scores)
        require(abs(correct/503*100-report['overall']) < .0051, 'LongBench aggregate mismatch')
        n, label = 503, 'LongBench v2'
        files.append(directory/report['scored_outputs_path'])
    else:
        protocol, summary = report['protocol'], report['summary']
        require(cfg['protocol'] == protocol and protocol_hash(protocol) == summary['protocol_hash'], 'LCB protocol mismatch')
        require(len(rows) == len({r['question_id'] for r in rows}) == summary['evaluated'] == 175 and
                {str(r['question_id']) for r in rows} == set(protocol['data']['question_ids']), 'LCB coverage mismatch')
        require(all(r['protocol_hash'] == summary['protocol_hash'] for r in rows), 'Mixed LCB answers')
        require(protocol['generation']['max_new_tokens'] == (131072 if continuation else 65536) and
                summary['n'] == 1, 'LCB budget/sampling mismatch')
        detail, tests = report['per_problem'][0]['detail']['pass@1'], report['per_problem'][1]
        require(set(detail) == set(tests) == {str(i) for i in range(175)}, 'LCB missing test results')
        correct = 0
        for i, test in tests.items():
            require(len(test) == 1 and len(test[0]) > 0, 'LCB empty test result')
            hit = int(all(x is True or x == 1 for x in test[0]))
            require(hit == detail[i], 'LCB execution score mismatch')
            correct += hit
        require(math.isclose(correct/175, summary['pass@1']), 'LCB aggregate mismatch')
        # Bind scoring to the exported rows even if the legacy scorer omitted
        # a raw-source hash from its own summary format.
        binding = read(directory/'arcquant_score_binding.json')
        require(binding['raw_sha256'] == digest(raw) and binding['summary_sha256'] == digest(str(raw)+'.summary.json'),
                'LCB scored outputs changed')
        files.append(directory/'arcquant_score_binding.json')
        n, label = 175, 'LiveCodeBench v6 128K 续写' if continuation else 'LiveCodeBench v6 64K'
    return dict(model='DS', group='baseline', variant='HiF4 W4A4 + ARC', benchmark=label,
        evaluated_count=n, correct=correct, score=100*correct/n, source=str(directory), evidence=evidence(files, base))


def refresh(snapshot, run, base):
    labels = ('LongBench v2', 'LiveCodeBench v6 64K', 'LiveCodeBench v6 128K 续写')
    snapshot['full'] = [r for r in snapshot['full'] if not (r['variant'] == 'HiF4 W4A4 + ARC' and r['benchmark'] in labels)]
    progress = []
    for stage in ('longbench','lcb64','lcb128'):
        path = run/stage
        name = 'hifp4' if stage == 'lcb128' else 'hifp4-w4a4'
        if not (path/(name+'.jsonl.summary.json')).exists():
            progress.append(dict(stage=stage, status='pending'))
            continue
        result = audit(path, stage, base)
        snapshot['full'].append(result)
        progress.append(dict(stage=stage, status='complete', n=result['evaluated_count']))
    snapshot['extended_progress'] = progress
    snapshot['dense_gaps'][0][3] = 'GPQA 198/198；MATH 500/500；'+'；'.join(p['stage']+' '+p['status'] for p in progress)
    snapshot['dense_gaps'][0][4] = 'H20-2 GPU 2、3：当前 MAGI 搜索完成后接续 DS；以 handoff.json 为准'
