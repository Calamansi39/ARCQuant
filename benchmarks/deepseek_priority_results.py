"""Merge full DS40 results without replacing Tiny or confusing sparse/dense."""
from datetime import datetime
from pathlib import Path
from benchmarks.verified_results import read,validate_answers,evidence,require


def refresh(snapshot,plan_path,base):
    from benchmarks.queue_sparse_baselines import arc_health
    from benchmarks.result_tables import verify_greedy_evidence
    from benchmarks.deepseek_arc_extended_audit import audit
    plan=read(plan_path)
    progress=[]
    for job in plan.get('report_jobs',plan['jobs']):
        if not job['gpu'] or job.get('validation')=='magi-arc':continue
        directory=Path(job['output'])
        expected=job.get('expected',job.get('question_count'))
        row=dict(id=job['id'],method=job['method'],sparsity=job['sparsity'],expected=expected,
                 state='pending',complete=False,output=str(directory))
        if job.get('stage'):
            raw=directory/'hifp4-w4a4.jsonl'
            if raw.exists():row['completed']=len(raw.read_text().splitlines())
            marker=directory/('hifp4-w4a4.jsonl.summary.json' if job['stage']=='longbench' else 'arcquant_score_binding.json')
            if (marker.exists() and row.get('completed')==expected
                    and all((directory/f'arcquant_runtime_rank{r}.json').exists() for r in range(2))
                    and (job['stage']!='longbench' or read(marker).get('scoring_version')=='longbench_explicit_answer_v2')):
                result=audit(directory,job['stage'],base)
                row.update(complete=True,state='completed')
            else:
                progress.append(row);continue
        else:
            summary=read(directory/'summary.json')
            row['completed']=summary['completed_candidates']
            if not summary['complete']:
                progress.append(row);continue
            if not all((directory/f'arcquant_runtime_rank{r}.json').exists() for r in range(2)):
                row['state']='awaiting_runtime_evidence';progress.append(row);continue
            summary,protocol,answers,raw=validate_answers(directory,expected)
            require(summary['is_full_dataset'] and not summary['is_smoke_test'],'Not full benchmark')
            require(arc_health(directory)[0],'ARC execution incomplete')
            verify_greedy_evidence(directory,protocol['arcquant'])
            result=dict(model='DS',group='sparse',method=job['method'],sparsity=job['sparsity'],
                variant='HiF4 W4A4 + ARC',benchmark='GPQA Diamond' if summary['benchmark']=='gpqa' else 'MATH-500',
                evaluated_count=expected,correct=summary['correct_candidates'],score=summary['pass_at_1']*100,
                source=str(directory),evidence=evidence([directory/'summary.json',directory/'protocol.json',raw,
                    directory/'greedy_strategy.json',*[directory/f'arcquant_runtime_rank{r}.json' for r in range(2)]],base))
            row.update(complete=True,state='completed')
        key=lambda r:(r['variant'],r['benchmark'],r.get('method','none'),r.get('sparsity',0.))
        snapshot['full']=[r for r in snapshot['full'] if key(r)!=key(result)]+[result]
        progress.append(row)
    snapshot['priority_progress']=progress
    snapshot['priority_progress_at']=datetime.now().isoformat(timespec='seconds')
