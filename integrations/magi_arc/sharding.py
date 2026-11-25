"""Stable work partitions; partitioning never changes numerical identities."""
import argparse


def parse_shard(value):
    try:
        index, count = map(int, value.split('/'))
        if not 0 <= index < count:
            raise ValueError()
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError('Use a zero-based shard i/n, with 0 <= i < n') from None
    return index, count


def partition(items, shard=None):
    items = list(items)
    return items if shard is None else items[shard[0]::shard[1]]


def shard_name(shard):
    return 'all' if shard is None else f'shard-{shard[0]}-of-{shard[1]}'
