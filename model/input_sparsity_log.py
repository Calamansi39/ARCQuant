import json
import os
import re
from collections import defaultdict

import torch


_ACTIVE_INPUT_SPARSITY_LOGGER = None


def _tag_sort_key(tag):
    match = re.match(r"layer_(\d+)\.(\w+)", tag)
    if match:
        return (0, int(match.group(1)), match.group(2))
    match = re.match(r"avg\.(\w+)", tag)
    if match:
        return (1, 0, match.group(1))
    return (2, 0, tag)


class InputSparsityLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.summary_path = os.path.splitext(log_path)[0] + ".json"
        self._stats = {}

    @torch.no_grad()
    def observe(self, tag, x):
        if not torch.is_tensor(x) or x.numel() == 0:
            return

        x = x.detach()
        entry = self._stats.setdefault(
            tag,
            {
                "calls": 0,
                "zero_count": 0,
                "total_count": 0,
                "two_zero_group_count": 0,
                "group_count": 0,
            },
        )
        entry["calls"] += 1

        zero_count = int((x == 0).sum().item())
        total_count = int(x.numel())
        entry["zero_count"] += zero_count
        entry["total_count"] += total_count

        if x.shape[-1] % 4 == 0:
            groups = x.reshape(-1, x.shape[-1] // 4, 4)
            zero_groups = (groups == 0).sum(dim=-1)
            entry["two_zero_group_count"] += int((zero_groups >= 2).sum().item())
            entry["group_count"] += int(zero_groups.numel())

    def _build_summary(self):
        per_tag = {}
        proj_acc = defaultdict(
            lambda: {
                "calls": 0,
                "zero_count": 0,
                "total_count": 0,
                "two_zero_group_count": 0,
                "group_count": 0,
            }
        )

        for tag, stats in self._stats.items():
            zero_ratio = stats["zero_count"] / stats["total_count"] if stats["total_count"] else 0.0
            two_zeros_ratio = (
                stats["two_zero_group_count"] / stats["group_count"] if stats["group_count"] else None
            )
            per_tag[tag] = {
                "calls": stats["calls"],
                "zero_ratio": zero_ratio,
                "two_zeros_ratio": two_zeros_ratio,
            }
            proj_name = tag.split(".")[-1]
            acc = proj_acc[proj_name]
            for key in ("calls", "zero_count", "total_count", "two_zero_group_count", "group_count"):
                acc[key] += stats[key]

        avg = {}
        for proj_name, stats in proj_acc.items():
            avg[f"avg.{proj_name}"] = {
                "calls": stats["calls"],
                "zero_ratio": stats["zero_count"] / stats["total_count"] if stats["total_count"] else 0.0,
                "two_zeros_ratio": (
                    stats["two_zero_group_count"] / stats["group_count"] if stats["group_count"] else None
                ),
            }

        return {
            "per_tag": dict(sorted(per_tag.items(), key=lambda item: _tag_sort_key(item[0]))),
            "avg": dict(sorted(avg.items(), key=lambda item: _tag_sort_key(item[0]))),
        }

    def dump(self):
        summary = self._build_summary()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        with open(self.log_path, "w", encoding="utf-8") as f:
            for tag, stats in summary["per_tag"].items():
                two_zeros_ratio = stats["two_zeros_ratio"]
                if two_zeros_ratio is None:
                    f.write(
                        f"[{tag}] calls={stats['calls']} zero_ratio={stats['zero_ratio']:.6f} "
                        "two_zeros_ratio=N/A\n"
                    )
                else:
                    f.write(
                        f"[{tag}] calls={stats['calls']} zero_ratio={stats['zero_ratio']:.6f} "
                        f"two_zeros_ratio={two_zeros_ratio:.6f}\n"
                    )
            f.write("\n")
            for tag, stats in summary["avg"].items():
                two_zeros_ratio = stats["two_zeros_ratio"]
                if two_zeros_ratio is None:
                    f.write(
                        f"[{tag}] calls={stats['calls']} zero_ratio={stats['zero_ratio']:.6f} "
                        "two_zeros_ratio=N/A\n"
                    )
                else:
                    f.write(
                        f"[{tag}] calls={stats['calls']} zero_ratio={stats['zero_ratio']:.6f} "
                        f"two_zeros_ratio={two_zeros_ratio:.6f}\n"
                    )

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        return self.log_path, self.summary_path


def set_input_sparsity_logger(logger):
    global _ACTIVE_INPUT_SPARSITY_LOGGER
    _ACTIVE_INPUT_SPARSITY_LOGGER = logger


def clear_input_sparsity_logger():
    global _ACTIVE_INPUT_SPARSITY_LOGGER
    _ACTIVE_INPUT_SPARSITY_LOGGER = None


@torch.no_grad()
def record_input_sparsity(tag, x):
    if _ACTIVE_INPUT_SPARSITY_LOGGER is not None:
        _ACTIVE_INPUT_SPARSITY_LOGGER.observe(tag, x)


def dump_input_sparsity_logger():
    if _ACTIVE_INPUT_SPARSITY_LOGGER is None:
        return None
    return _ACTIVE_INPUT_SPARSITY_LOGGER.dump()
