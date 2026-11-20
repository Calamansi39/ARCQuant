"""Run the existing TP2 layer search with both ranks on one physical GPU.

Only the search transport changes: synchronous CUDA SUM and all-gather are
staged through CPU/Gloo. The weight shards, calibration and greedy remain TP2.
This is slower than NCCL on two GPUs and is NOT a single-GPU generation path.
Execution provenance is saved next to the unmodified search manifest.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.deepseek_greedy_strategy import atomic_json


def setup_colocated():
    import torch
    import torch.distributed as dist

    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    if world != 2 or torch.cuda.device_count() != 1:
        raise ValueError('Requires torchrun with two ranks and exactly one visible GPU')
    torch.cuda.set_device(0)
    cache = Path(tempfile.gettempdir()) / f'arcquant-colocated-{os.getuid()}' / f'rank{rank}'
    for variable, name in [('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        directory = cache / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    original = dist.all_reduce
    original_gather = dist.all_gather

    def staged_sum(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if not tensor.is_cuda:
            return original(tensor, op=op, group=group, async_op=async_op)
        if async_op or op != dist.ReduceOp.SUM:
            raise ValueError('Colocated search supports synchronous CUDA SUM only')
        host = tensor.detach().to('cpu')
        original(host, op=op, group=group)
        tensor.copy_(host)
        return None

    dist.all_reduce = staged_sum

    def staged_gather(tensor_list, tensor, group=None, async_op=False):
        if not tensor.is_cuda:
            return original_gather(tensor_list, tensor, group=group, async_op=async_op)
        if async_op or not all(item.is_cuda for item in tensor_list):
            raise ValueError('Colocated search supports synchronous CUDA all-gather only')
        host = tensor.detach().to('cpu')
        outputs = [torch.empty_like(item, device='cpu') for item in tensor_list]
        original_gather(outputs, host, group=group)
        for destination, source in zip(tensor_list, outputs):
            destination.copy_(source)
        return None

    dist.all_gather = staged_gather
    # Check actual GPU -> CPU -> Gloo -> GPU communication before loading weights.
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        base = torch.arange(8192, dtype=torch.float32).remainder(997).mul_(.0151)
        left, right = base.to(dtype), (base + 1).to(dtype)
        value = (left if rank == 0 else right).cuda()
        expected = (left + right).cuda()
        dist.all_reduce(value)
        if not torch.equal(value, expected):
            raise ValueError(f'Colocated transport reduction mismatch: {dtype}')
        value = (left if rank == 0 else right).cuda()
        gathered = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(gathered, value)
        if not all(torch.equal(actual.cpu(), expected) for actual, expected in zip(gathered, [left, right])):
            raise ValueError(f'Colocated transport all-gather mismatch: {dtype}')
    ranks = [None, None]
    dist.all_gather_object(ranks, rank)
    if ranks != [0, 1]:
        raise ValueError('Colocated rank identity mismatch')
    dist.barrier()
    if rank == 0:
        print('[colocated] TP2 SUM, all-gather and object communication passed; both ranks use GPU 0', flush=True)
    return world, rank, 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv == ['--transport-check']:
        setup_colocated()
        import torch.distributed as dist
        dist.destroy_process_group()
        return 0
    from benchmarks import deepseek_greedy_search as search
    args = search.parser().parse_args(argv)
    if not args.layer_streaming:
        raise ValueError('Colocated ranks require --layer-streaming; full model does not fit')
    search.validate(args)
    sys.path.insert(0, str(args.base_eval_root.resolve()))
    import run_deepseek_v4_baseline as baseline
    baseline.setup_distributed = setup_colocated
    rank = int(os.environ.get('RANK', '0'))
    record = dict(schema='deepseek-colocated-search-v1', tp_world_size=2, physical_gpu_count=1,
                  backend='gloo-cpu-staged-synchronous-sum-and-all-gather',
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  hostname=os.uname().nodename, started_at=time.time(),
                  wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  command=argv, phase='starting',
                  note='Search only; generation still uses two physical GPUs and NCCL.')
    path = args.out_dir / 'colocated_execution.json'
    if rank == 0:
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['wrapper_sha256'] != record['wrapper_sha256']:
                raise ValueError('Colocated transport implementation changed; resume refused')
        atomic_json(path, record)
        snapshot = args.out_dir / 'sources/benchmarks/colocated_deepseek_search.py'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(__file__).read_bytes())
    try:
        result = search.main(argv)
    except BaseException as exc:
        if rank == 0:
            atomic_json(path, dict(record, phase='failed', error=repr(exc), finished_at=time.time()))
        raise
    if rank == 0:
        atomic_json(path, dict(record, phase='finished', finished_at=time.time(), returncode=result))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
