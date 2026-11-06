"""Run greedy preparation as the next job on an idle GPU pair, without kills."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import fcntl
import time
import hashlib
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json
from benchmarks.gpu_wait import idle_gpus


def snapshot_source(out):
    """Freeze runnable Python dependencies before waiting for the GPU lease."""
    source = out / 'source'
    paths = list(ROOT.glob('*.py'))
    for folder in ('benchmarks', 'model', 'kernels'):
        paths.extend((ROOT / folder).rglob('*.py'))
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    atomic_json(out / 'source_manifest.json', hashes)
    return source


@contextmanager
def priority_pair(ids, min_free_mib, save):
    """Join blocking lock waiters before the current owner's short job ends.

    Coordinates this UID's ARCQuant queues; no CUDA reservation or process kill.
    Check physical GPU idleness after obtaining both locks.
    """
    root=Path('/tmp')/f'arcquant-gpu-locks-{os.getuid()}'
    root.mkdir(exist_ok=True,mode=0o700)
    handles=[]
    try:
        for gpu in sorted(ids,key=int):
            handle=(root/f'gpu{gpu}.lock').open('a');handles.append(handle)
            save('waiting_for_current_gpu_lease')
            fcntl.flock(handle,fcntl.LOCK_EX)
        save('waiting_for_gpu_memory')
        while set(idle_gpus(min_free_mib,ids))!=set(ids):time.sleep(2)
        time.sleep(2)
        while set(idle_gpus(min_free_mib,ids))!=set(ids):time.sleep(2)
        yield ','.join(ids)
    finally:
        for handle in reversed(handles):handle.close()


@contextmanager
def search_lease(path, save):
    """Serialize writers that resume the same search directory.

    A resumed search updates layer files, calibration shards, and the
    strategy atomically only at the file level.  Two torchrun jobs can still
    interleave different layers and leave a mixed artifact, so the directory
    itself needs a process-level lease.  Waiting for this lease happens before
    claiming GPUs so a duplicate queue does not reserve cards while blocked.
    """
    if path is None:
        yield
        return
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.arcquant-search.lock').open('a')
    try:
        save('waiting_for_search_lease')
        fcntl.flock(lock, fcntl.LOCK_EX)
        save('search_lease_acquired')
        yield
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-ids',default='0,1')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--pilot-only',action='store_true')
    p.add_argument('--resume-search',type=Path,help='continue a compatible formal-size search after obtaining the GPU lease')
    p.add_argument('--min-free-mib',type=int,default=138000)
    p.add_argument('--methods',default='wina,teal',help='comma-separated native methods for a fresh search')
    p.add_argument('--targets',default='0.5,0.6,0.7,0.8,0.9',help='comma-separated sparsity targets for a fresh search')
    for name in ('base-eval-root','source-model','converted-model','calibration-data'):
        p.add_argument('--'+name,type=Path,required=True)
    return p


def run(a):
    p=parser()
    ids=a.gpu_ids.split(',')
    if len(ids)!=2 or len(set(ids))!=2 or not all(v.isdecimal() for v in ids):
        p.error('exactly two distinct GPU indices required')
    out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'queue_status.json').exists():p.error('use a fresh queue output directory')
    methods=','.join(v.strip() for v in a.methods.split(',') if v.strip())
    targets=','.join(v.strip() for v in a.targets.split(',') if v.strip())
    if not methods or any(v not in ('wina','teal') for v in methods.split(',')):
        p.error('--methods must contain only wina and/or teal')
    try:
        parsed_targets=[float(v) for v in targets.split(',')]
    except ValueError:
        p.error('--targets must be comma-separated numbers')
    if not parsed_targets or any(not 0 < v < 1 for v in parsed_targets):
        p.error('--targets must be between 0 and 1')
    source = snapshot_source(out)
    state=dict(pid=os.getpid(),hostname=os.uname().nodename,jobs=[],state='waiting_for_gpu',gpu_ids=ids)
    def save(phase=None):
        if phase:state['state']=phase
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(out/'queue_status.json',state)
    base=[a.python,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
          str(source/'benchmarks/deepseek_greedy_search.py'),'--layer-streaming']
    for name in ('base_eval_root','source_model','converted_model','calibration_data'):
        base+=['--'+name.replace('_','-'),str(getattr(a,name).resolve())]
    stages=[('pilot',['--samples','2','--search-samples','2','--length','128','--layers','1',
                     '--max-steps','1','--methods',methods,'--targets',str(parsed_targets[0])])]
    if not a.pilot_only:
        stages.append(('full',['--samples','128','--search-samples','20','--length','512',
                               '--methods',methods,'--targets',targets]))
    if a.resume_search:
        if a.pilot_only:p.error('--resume-search and --pilot-only cannot be combined')
        manifest=json.loads((a.resume_search/'search_manifest.json').read_text())
        if manifest['samples'] < 128 or manifest['length'] < 512:
            p.error('formal continuation requires at least 128x512 calibration')
        for name,expected in manifest['implementation_sha256'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
                p.error(f'resume source mismatch: {name}; use the original frozen source')
        options=['--resume','--methods',','.join(manifest['methods']),
                 '--targets',','.join(map(str,manifest['targets'])),'--quantization',manifest['quantization']]
        for key in ('samples','search_samples','search_batch_size','length','step','last_fraction','histogram_sample_limit','seed'):
            options+=['--'+key.replace('_','-'),str(manifest[key])]
        if not manifest.get('layer_streaming'):
            base.remove('--layer-streaming')
        stages=[('resume',options)]
    save()
    child=None
    def stop(sig,_):
        nonlocal child
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        state['signal']=sig;save('interrupted');raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lease = search_lease(a.resume_search, save) if a.resume_search else nullcontext()
    with lease:
        with priority_pair(ids,a.min_free_mib,save) as visible:
            for name,options in stages:
                destination=a.resume_search.resolve() if name=='resume' else out/name
                command=base+['--out-dir',str(destination),*options]
                job=dict(name=name,state='running',command=command)
                state['jobs'].append(job);save(name)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':visible,'HF_HUB_OFFLINE':'1',
                     'HF_DATASETS_OFFLINE':'1','OMP_NUM_THREADS':'4',
                     'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
                with (out/f'{name}.log').open('w') as log:
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    job['pid']=child.pid;save()
                    rc=child.wait();child=None
                job.update(exit_code=rc,state='completed' if rc==0 else 'failed');save()
                if rc:
                    save('failed');return rc
                status=json.loads((destination/'status.json').read_text())
                if name=='pilot' and status['phase']!='pilot_paused':
                    save('failed_pilot_validation');return 1
                if name=='pilot':
                    for method in ('wina','teal'):
                        result=json.loads((out/name/method/'layer-0.json').read_text())
                        if result['iterations'] < 1 or result['candidate_forwards'] < len(result['weights']):
                            save('failed_pilot_validation');return 1
    save('completed');return 0


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
