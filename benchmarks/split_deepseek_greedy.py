"""Hand off TEAL to a second host after a GPU lease and a real pilot pass.

The original dual-method job continues while the remote host waits for GPUs.
Only after remote validation are consistent method checkpoints forked, the
identified original queue stopped, and the two independent searches resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json
from benchmarks.fork_deepseek_greedy import fork_checkpoint, digest
from benchmarks.queue_deepseek_greedy import priority_pair, snapshot_source


def process_identity(pid):
    path = Path('/proc') / str(pid)
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=stat[19],
                command=(path / 'cmdline').read_bytes().split(b'\0')[:-1])


def process_alive(pid, start_ticks):
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return stat[0] != 'Z' and stat[19] == start_ticks
    except FileNotFoundError:
        return False


def prepare(parent_queue, out, remote_host, remote_hostname, remote_gpus, python):
    parent_queue, out = Path(parent_queue).resolve(), Path(out).resolve()
    state = json.loads((parent_queue / 'queue_status.json').read_text())
    if state['hostname'] != socket.gethostname() or state['state'] != 'full':
        raise ValueError('Prepare on the host of the running full search')
    original = process_identity(state['pid'])
    command = [v.decode() for v in original.pop('command')]
    if str(parent_queue) not in command or not any(Path(v).name == 'queue_deepseek_greedy.py' for v in command):
        raise ValueError('PID is not the requested original queue')
    jobs = [j for j in state['jobs'] if j['name'] == 'full' and j['state'] == 'running']
    if len(jobs) != 1:
        raise ValueError('Expected one running full-search child')
    args = jobs[0]['command']
    child = process_identity(jobs[0]['pid'])
    child.pop('command')
    manifest = json.loads((parent_queue / 'full/search_manifest.json').read_text())
    if set(manifest['methods']) != {'wina', 'teal'} or manifest['world_size'] != 2:
        raise ValueError('Split requires an existing TP2 dual-method run')
    if remote_hostname == socket.gethostname() or len(set(remote_gpus)) != 2:
        raise ValueError('Use a second host and two distinct remote GPUs')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = snapshot_source(out)
    for name, expected in manifest['implementation_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError(f'Active search source differs: {name}')
    assets = {name: args[args.index('--' + name) + 1] for name in
              ('base-eval-root', 'source-model', 'converted-model', 'calibration-data')}
    plan = dict(schema='deepseek-method-split-v1', out=str(out), source=str(source),
                python=python, parent_queue=str(parent_queue), parent_search=str(parent_queue / 'full'),
                priority_pointer=str(ROOT / 'runs/greedy_priority_current.json'),
                local_hostname=socket.gethostname(), remote_hostname=remote_hostname,
                remote_host=remote_host, local_gpus=state['gpu_ids'], remote_gpus=remote_gpus,
                original_queue=original, original_child=child, assets=assets,
                min_free_mib=138000, created_at=datetime.now(timezone.utc).isoformat())
    atomic_json(out / 'plan.json', plan)
    return plan


def search_command(plan, method, *, pilot=False, manifest=None):
    out = Path(plan['out']) / ('remote-pilot' if pilot else f'{method}-search')
    command = [plan['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
               str(Path(plan['source']) / 'benchmarks/deepseek_greedy_search.py'), '--out-dir', str(out)]
    for name, path in plan['assets'].items():
        command += ['--' + name, path]
    if pilot:
        command += ['--layer-streaming', '--samples', '2', '--search-samples', '2',
                    '--length', '128', '--layers', '1', '--max-steps', '1',
                    '--methods', method, '--targets', '0.5']
    else:
        m = manifest or json.loads((out / 'search_manifest.json').read_text())
        if m['methods'] != [method]:
            raise ValueError('Unexpected fork method')
        command += ['--resume', '--methods', method, '--targets', ','.join(map(str, m['targets'])),
                    '--quantization', m['quantization']]
        for key in ('samples', 'search_samples', 'search_batch_size', 'length', 'step',
                    'last_fraction', 'histogram_sample_limit', 'seed'):
            command += ['--' + key.replace('_', '-'), str(m[key])]
        if m['layer_streaming']:
            command.append('--layer-streaming')
    return command


def validate_runtime(plan, method):
    """Compare the full CPU preflight to the frozen parent before any stop."""
    expected = json.loads((Path(plan['parent_search']) / 'search_manifest.json').read_text())
    expected['methods'] = [method]
    distributed = search_command(plan, method, manifest=expected)
    command = [distributed[0], *distributed[5:], '--dry-run']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    actual = json.loads(result.stdout)
    if actual.pop('status') != 'cpu_preflight_passed' or actual.pop('gpu_execution'):
        raise ValueError('CPU preflight unexpectedly executed GPUs')
    if actual != expected:
        different = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError(f'Runtime differs from parent search: {different}')


class Runner:
    def __init__(self, plan, role):
        self.plan, self.role = plan, role
        self.out = Path(plan['out'])
        self.child = None
        self.state = dict(pid=os.getpid(), hostname=socket.gethostname(), role=role)

    def save(self, phase=None, **details):
        if phase:
            self.state['phase'] = phase
        self.state.update(details, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(self.out / f'{self.role}_status.json', self.state)

    def child_run(self, command, visible, name):
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': visible, 'HF_HUB_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
                       'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                       'TRITON_CACHE_DIR': f'/tmp/arcquant-greedy-{os.getuid()}-{self.role}'}
        with (self.out / f'{name}.log').open('w') as log:
            self.child = subprocess.Popen(command, env=environment, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            self.save(name, child_pid=self.child.pid, command=command)
            rc = self.child.wait()
            self.child = None
        if rc:
            raise RuntimeError(f'{name} failed, exit={rc}; inspect {name}.log')

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()

    def wait_commit(self):
        deadline = time.monotonic() + 900
        while not (self.out / 'handoff_commit.json').exists():
            local_path = self.out / 'local_status.json'
            if local_path.exists() and json.loads(local_path.read_text()).get('phase') == 'failed':
                raise RuntimeError('Local handoff failed; releasing remote GPUs')
            if time.monotonic() > deadline:
                raise TimeoutError('Local handoff did not commit within 15 minutes')
            time.sleep(2)

    def remote(self):
        if socket.gethostname() != self.plan['remote_hostname']:
            raise ValueError('Remote host identity mismatch')
        validate_runtime(self.plan, 'teal')
        with priority_pair(self.plan['remote_gpus'], self.plan['min_free_mib'], self.save) as visible:
            self.child_run(search_command(self.plan, 'teal', pilot=True), visible, 'remote-pilot')
            pilot = json.loads((self.out / 'remote-pilot/teal/layer-0.json').read_text())
            replay = json.loads((self.out / 'remote-pilot/teal/layer-0.replay.json').read_text())
            if pilot['iterations'] != 1 or replay['zero_sparsity_error'] != 0:
                raise ValueError('Remote pilot or zero-mask equivalence failed')
            self.save('ready_for_handoff', pilot_passed=True)
            self.wait_commit()
            self.child_run(search_command(self.plan, 'teal'), visible, 'teal-full')
            if json.loads((self.out / 'teal-search/status.json').read_text())['phase'] != 'completed':
                raise ValueError('Remote search did not complete all layers')
        self.save('completed')

    def launch_remote(self):
        command = [self.plan['python'], str(Path(self.plan['source']) / 'benchmarks/split_deepseek_greedy.py'),
                   '--plan', str(self.out / 'plan.json'), '--role', 'remote']
        code = ('import subprocess; '
                f'log=open({str(self.out / "remote-driver.log")!r},"a"); '
                f'p=subprocess.Popen({command!r},stdin=subprocess.DEVNULL,stdout=log,'
                'stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)')
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.plan['remote_host'],
               shlex.join([self.plan['python'], '-c', code])]
        result = subprocess.run(ssh, check=True, capture_output=True, text=True, timeout=30)
        self.save('waiting_for_remote_gpu_and_pilot', remote_driver_pid=int(result.stdout.strip()))

    def local(self):
        if socket.gethostname() != self.plan['local_hostname']:
            raise ValueError('Local host identity mismatch')
        self.launch_remote()
        while True:
            remote_path = self.out / 'remote_status.json'
            if remote_path.exists():
                remote = json.loads(remote_path.read_text())
                if remote['phase'] == 'failed':
                    raise RuntimeError(f'Remote failed before handoff: {remote.get("error")}')
                if remote['phase'] == 'ready_for_handoff':
                    break
            self.save()
            time.sleep(5)
        for method in ('teal', 'wina'):
            self.save('copying_checkpoint', method=method)
            for attempt in range(6):
                try:
                    fork_checkpoint(self.plan['parent_search'], self.out / f'{method}-search',
                                    method, self.plan['source'])
                    break
                except RuntimeError:
                    if attempt == 5:
                        raise
                    time.sleep(2)
        validate_runtime(self.plan, 'wina')
        remote = json.loads((self.out / 'remote_status.json').read_text())
        if remote['phase'] != 'ready_for_handoff':
            raise RuntimeError('Remote readiness changed; original queue remains running')
        original = self.plan['original_queue']
        child = self.plan['original_child']
        if not process_alive(original['pid'], original['start_ticks']):
            raise RuntimeError('Original queue exited or PID changed before handoff; no process stopped')
        self.save('stopping_original_after_snapshot')
        os.kill(original['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 90
        while process_alive(original['pid'], original['start_ticks']) or process_alive(child['pid'], child['start_ticks']):
            if time.monotonic() > deadline:
                raise TimeoutError('Original queue has not exited; copied checkpoints retained')
            time.sleep(1)
        atomic_json(self.out / 'handoff_commit.json', dict(original_queue=original,
                    committed_at=datetime.now(timezone.utc).isoformat(),
                    wina=str(self.out / 'wina-search'), teal=str(self.out / 'teal-search')))
        atomic_json(Path(self.plan['priority_pointer']), dict(pid=os.getpid(), out_dir=str(self.out),
                    mode='independent-method-split', plan=str(self.out / 'plan.json'),
                    previous_queue=self.plan['parent_queue'],
                    wina=dict(host=self.plan['local_hostname'], gpus=self.plan['local_gpus'],
                              output=str(self.out / 'wina-search')),
                    teal=dict(host=self.plan['remote_hostname'], gpus=self.plan['remote_gpus'],
                              output=str(self.out / 'teal-search'))))
        with priority_pair(self.plan['local_gpus'], self.plan['min_free_mib'], self.save) as visible:
            self.child_run(search_command(self.plan, 'wina'), visible, 'wina-full')
            if json.loads((self.out / 'wina-search/status.json').read_text())['phase'] != 'completed':
                raise ValueError('Local search did not complete all layers')
        self.save('completed')

    def run(self):
        lock = (self.out / f'{self.role}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            self.save('starting')
            getattr(self, self.role)()
        except BaseException as error:
            self.stop_child()
            self.save('failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            lock.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path)
    p.add_argument('--role', choices=('local', 'remote'))
    p.add_argument('--parent-queue', type=Path)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--remote-host')
    p.add_argument('--remote-hostname')
    p.add_argument('--remote-gpus', default='5,6')
    p.add_argument('--python', default=sys.executable)
    a = p.parse_args(argv)
    if a.plan:
        if not a.role:
            p.error('--plan requires --role')
        Runner(json.loads(a.plan.read_text()), a.role).run()
    else:
        if not all((a.parent_queue, a.out_dir, a.remote_host, a.remote_hostname)):
            p.error('Preparing a plan requires parent, output and remote host')
        plan = prepare(a.parent_queue, a.out_dir, a.remote_host, a.remote_hostname,
                       a.remote_gpus.split(','), a.python)
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
