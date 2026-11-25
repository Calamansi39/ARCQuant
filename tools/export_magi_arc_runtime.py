#!/usr/bin/env python3
"""Export only explicit evaluator code/metadata; no credentials or model weights.

The archived base_eval files are private runtime dependencies, not published
experiment scores. New ARC method code is obtained by cloning this repository.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-eval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.base_eval.resolve()
    if a.output.exists():p.error('Archive exists; refusing to overwrite')
    todo=['gen_vbench_magi2.py','magi2_runtime.py','magi2_quant.py','magi2_fast_quant.py','magi2_batched_moe.py',
          'magi2_vbench_worker.py','magi2_conditioning.py','arcquant_fakequant.py','deepseek_v4_fast_quant.py',
          'vbench_generation_cache.py','vbench_protocol.py','eval_vbench.py','scripts/magi_memory_guard.py']
    todo += [str(path.relative_to(base)) for path in (base/'sparse_eval/magi').glob('*.py')]
    found=set()
    while todo:
        name=todo.pop()
        if name in found:continue
        found.add(name)
        for node in ast.walk(ast.parse((base/name).read_text())):
            imports=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            for dep in imports:
                path=base/(dep.replace('.','/')+'.py')
                if path.is_file():todo.append(str(path.relative_to(base)))
    # Package initializers and the exact seed/prompt/model metadata used by prepare.
    for name in list(found):
        for parent in Path(name).parents:
            if (base/parent/'__init__.py').is_file():found.add(str(parent/'__init__.py'))
    found.update(('magi2_download_manifest.json','vbench_tiny_manifest.json','sparse_eval/magi/prompts.json'))
    # Legacy teacher identities include all package JSON/shell files.
    found.update(str(path.relative_to(base)) for path in (base/'sparse_eval/magi').rglob('*')
                 if path.suffix in ('.json','.sh'))
    files={name:dict(sha256=hashlib.sha256((base/name).read_bytes()).hexdigest(),bytes=(base/name).stat().st_size)
           for name in sorted(found)}
    manifest=dict(schema='magi-arc-runtime-bundle-v1',files=files,
        excluded=['weights','datasets','videos','credentials','environments'],
        note='Extract under runtime; use --base-eval-root runtime/base_eval; original teacher paths may need relocation or fresh capture')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.output,'w:gz') as archive:
        for name in files:archive.add(base/name,arcname='base_eval/'+name,recursive=False)
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('runtime_manifest.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    with tarfile.open(a.output) as archive:
        for name,record in files.items():
            if hashlib.sha256(archive.extractfile('base_eval/'+name).read()).hexdigest()!=record['sha256']:
                raise ValueError('Archive verification failed')
    a.output.with_suffix('.json').write_text(json.dumps(dict(archive=a.output.name,
        sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),bytes=a.output.stat().st_size,**manifest),indent=2)+'\n')
    print(json.dumps(dict(archive=str(a.output),files=len(files),bytes=a.output.stat().st_size)))


if __name__=='__main__':main()
