#!/usr/bin/env python3
"""Resource-only wrapper: keep the evaluated greedy numerical files unchanged."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--guard-dir', type=Path, required=True)
    opt, rest = p.parse_known_args()
    base = rest[rest.index('--base-eval-root') + 1]
    sys.path.insert(0, base)
    import torch
    from scripts.magi_memory_guard import CacheGuard
    from benchmarks.run_deepseek_arc_extended import release_previous, hold_for_successor
    from benchmarks.run_deepseek_greedy_eval import main as evaluate
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    release_previous()
    class Guard(CacheGuard):
        def after_forward(self, module, inputs, output):
            if type(module).__name__ == 'Transformer':
                self.armed = True
                self.refill('successful DS forward')
    guard = Guard(torch, headroom_mib=4096, status_dir=opt.guard_dir).install()
    try:
        code = evaluate(rest)
        if code:
            raise RuntimeError(f'Greedy worker exit {code}')
        hold_for_successor()
    finally:
        guard.close()


if __name__ == '__main__':
    main()
