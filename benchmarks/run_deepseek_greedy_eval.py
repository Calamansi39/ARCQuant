"""Generate MATH/GPQA with a completed DeepSeek greedy budget artifact.

Uses the existing ARC generation entry and scorers. Checked AST insertion adds
strategy identity before protocol hashing, avoiding collisions with uniform
results. No source files in the running uniform evaluator are modified.
"""
from __future__ import annotations
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def compile_strategy_entry(entry, provenance, fingerprint, prepare):
    tree=ast.parse(textwrap.dedent(inspect.getsource(entry.main)))
    body=tree.body[0].body
    positions=[i for i,node in enumerate(body) if isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='identity' for t in node.targets)]
    if len(positions)!=1:
        raise RuntimeError('Generation entry changed; strategy metadata patch refused')
    body[positions[0]:positions[0]]=ast.parse("metadata.update(allocation='greedy', strategy=budget_provenance)").body
    ast.fix_missing_locations(tree)
    namespace={**entry.__dict__,'budget_provenance':provenance,'patched_prepare':prepare,
               'runtime_fingerprint':fingerprint}
    exec(compile(tree,'<deepseek-greedy-generation>','exec'),namespace)
    return namespace[tree.body[0].name]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--strategy',type=Path,required=True)
    p.add_argument('--sparse-method',choices=('wina','teal'),required=True)
    p.add_argument('--sparsity',type=float,required=True)
    p.add_argument('--quantization',choices=('arc','hif4'),default='arc')
    p.add_argument('--base-eval-root',type=Path,required=True)
    p.add_argument('--source-model',type=Path,required=True)
    p.add_argument('--converted-model',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    a,remaining=p.parse_known_args(argv)
    sys.path.insert(0,str(a.base_eval_root.resolve()))
    from deepseek_v4_support import source_signature,reference_config
    from benchmarks.deepseek_greedy_strategy import load_strategy
    from benchmarks import run_deepseek_sparse_math as entry
    from model import deepseek_generation_runtime as generation
    from model.deepseek_budget_inference import DeepSeekBudgetInferenceMixin
    conversion=json.loads((a.converted_model/'conversion.json').read_text())
    world=int(os.environ.get('WORLD_SIZE',conversion['world_size']))
    config=reference_config(a.source_model)
    data,rates,strategy_hash=load_strategy(a.strategy,method=a.sparse_method,sparsity=a.sparsity,
        source_signature=source_signature(a.source_model),world_size=world,
        quantization=a.quantization,n_layers=config['n_layers'])
    calibration=json.loads((a.calibration/'manifest.json').read_text())
    for key in ('source_signature','world_size','quantization','samples','length','data_sha256','implementation_sha256','calibration_rank_sha256'):
        if calibration.get(key)!=data.get(key):
            raise ValueError(f'Greedy calibration/strategy {key} mismatch')
    if not calibration.get('complete'):
        raise ValueError('Greedy evaluation requires complete calibrated projection files')
    hashes = calibration.get('calibration_rank_sha256')
    if not isinstance(hashes,list) or len(hashes)!=world:
        raise ValueError('Greedy strategy must bind every calibration rank tensor')
    rank = int(os.environ.get('RANK','0'))
    if entry.digest(a.calibration/f'rank{rank}.pt') != hashes[rank]:
        raise ValueError('Calibration tensors differ from the ones used in greedy search')
    provenance=dict(strategy_sha256=strategy_hash,calibration_manifest_sha256=entry.digest(a.calibration/'manifest.json'),
                    budget_metric=data['budget_metric'],grouping=data['grouping'],transform=False,method=a.sparse_method,
                    calibration_samples=data['samples'],search_samples=data['search_samples'])
    names=('DeepSeekGenerationWinaMixin','DeepSeekGenerationTealMixin')
    original={name:getattr(generation,name) for name in names}
    for name,base in original.items():
        class BudgetMixin(DeepSeekBudgetInferenceMixin,base):
            def configure_sparse(self,ds,model,*args,**kwargs):
                super().configure_sparse(ds,model,*args,**kwargs)
                self.greedy_row_parallel_keys = {name+'.weight' for name,module in model.named_modules()
                    if name.startswith('layers.') and isinstance(module,ds.RowParallelLinear)}
                self.configure_budget(rates,provenance)
        setattr(generation,name,BudgetMixin)
    original_fingerprint=entry.runtime_fingerprint
    def fingerprint():
        result=original_fingerprint()
        for name in ('benchmarks/run_deepseek_greedy_eval.py','benchmarks/deepseek_greedy_strategy.py',
                     'model/deepseek_budget_inference.py'):
            result[name]=entry.digest(ROOT/name)
        return result
    entry.runtime_fingerprint=fingerprint
    original_prepare=entry.patched_prepare
    def prepare(original_fn,metadata):
        function=original_prepare(original_fn,metadata)
        def wrapped(args,*extra,**kw):
            result=function(args,*extra,**kw)
            if int(os.environ.get('RANK','0'))==0:
                destination=Path(args.out_dir)/'greedy_strategy.json'
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(a.strategy.read_bytes())
            return result
        return wrapped
    entry_main=compile_strategy_entry(entry,provenance,fingerprint,prepare)
    args=['--sparse-method',a.sparse_method,'--sparsity',str(a.sparsity),'--quantization',a.quantization,
          '--base-eval-root',str(a.base_eval_root),'--source-model',str(a.source_model),
          '--converted-model',str(a.converted_model),'--calibration',str(a.calibration),*remaining]
    try:
        return entry_main(args)
    finally:
        entry.runtime_fingerprint=original_fingerprint
        for name,base in original.items():setattr(generation,name,base)


if __name__=='__main__':raise SystemExit(main())
