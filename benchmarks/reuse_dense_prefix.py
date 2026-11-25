"""Import identical ARC dense answers into a larger native evaluation protocol.

Only dataset selection may change. Original runs remain immutable; imported rows
keep answer/score/seed/timing intact and record both old and new checksums.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def compatible(source, target, *, greedy=False):
    ignored = {'selected_ids', 'prompts'}
    if {k: v for k, v in source.items() if k not in ignored} != {
            k: v for k, v in target.items() if k not in ignored}:
        changed = [k for k in source.keys() | target.keys()
                   if k not in ignored and source.get(k) != target.get(k)]
        raise ValueError(f'Cannot reuse changed model/method/protocol: {changed}')
    arc = source['arcquant']
    eligible = (arc['method'] in ('teal', 'wina') and arc.get('allocation') == 'greedy'
                and 0 < arc['sparsity'] < 1) if greedy else (arc['method'] == 'none' and arc['sparsity'] == 0)
    if not (arc['arc'] and arc['quantization'] == 'arc' and eligible
            and not source['diagnostic']['smoke_steps']):
        raise ValueError('Only matching non-diagnostic ARC answers are eligible')
    for key in source['selected_ids']:
        if key not in target['selected_ids'] or source['prompts'][key] != target['prompts'][key]:
            raise ValueError(f'Changed or missing prompt: {key}')


def import_prefix(source, target, *, greedy=False):
    from benchmarks.queue_sparse_baselines import arc_health
    from benchmarks.verified_results import validate_answers
    source, target = Path(source), Path(target)
    sp, tp = read(source / 'protocol.json'), read(target / 'protocol.json')
    compatible(sp, tp, greedy=greedy)
    if greedy:
        from benchmarks.result_tables import verify_greedy_evidence
        verify_greedy_evidence(source, sp['arcquant'])
    healthy, detail = arc_health(source)
    if not healthy:
        raise ValueError(detail)
    _, _, rows, _ = validate_answers(source, len(sp['selected_ids']))
    with (target / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output, manifest = target / 'outputs.jsonl', target / 'reused_prefix.json'
        transformed, mapping = [], []
        for row in rows:
            original = row['record_sha256']
            if row['protocol_sha256'] != digest(sp) or original != digest({
                    k: v for k, v in row.items() if k != 'record_sha256'}):
                raise ValueError('Source answer checksum or protocol differs')
            row = copy.deepcopy(row)
            row['protocol_sha256'] = digest(tp)
            row['record_sha256'] = digest({k: v for k, v in row.items() if k != 'record_sha256'})
            transformed.append(row)
            mapping.append(dict(id=row['id'], source_record_sha256=original,
                                imported_record_sha256=row['record_sha256']))
        provenance = dict(schema='arc-greedy-prefix-reuse-v1' if greedy else 'arc-dense-prefix-reuse-v1', source=str(source.resolve()),
                          source_protocol_sha256=digest(sp), target_protocol_sha256=digest(tp),
                          reused=len(rows), records=mapping,
                          changed_fields=['protocol_sha256', 'record_sha256'])
        if output.exists():
            existing = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
            indexed = {r['id']: r for r in existing}
            if len(indexed) != len(existing) or any(indexed.get(r['id']) != r for r in transformed):
                raise ValueError('Existing answers differ; refusing to overwrite any progress')
        if manifest.exists() and read(manifest) != provenance:
            raise ValueError('Existing reuse provenance differs')
        manifest.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
        if not output.exists():
            tmp = output.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                                   for r in transformed))
            tmp.replace(output)
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    args = p.parse_args()
    result = import_prefix(args.source, args.target)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
