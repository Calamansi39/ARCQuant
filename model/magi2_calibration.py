"""Bounded signed-activation calibration for MAGI's real denoising forwards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def signed_histogram(values, bins=10_000):
    values = values.float().cpu().sort().values
    if not values.numel() or not torch.isfinite(values).all():
        raise ValueError("Calibration samples must be nonempty and finite")
    if values[0] == values[-1]:
        return values[:1], torch.tensor([values.numel()], dtype=torch.float32)
    tail = max(1, int(values.numel() * .01))
    lower, upper = values[min(values.numel() - 1, tail)], values[max(0, values.numel() - tail)]
    if lower >= upper:
        lower, upper = values[0], values[-1]
    first = torch.nextafter(values[0], torch.tensor(float("-inf")))
    last = torch.nextafter(values[-1], torch.tensor(float("inf")))
    edges = torch.cat((first.view(1), torch.linspace(lower.item(), upper.item(), bins - 1), last.view(1)))
    if not torch.all(edges[1:] > edges[:-1]):
        edges = torch.linspace(first.item(), last.item(), bins + 1)
    return (edges[:-1] + edges[1:]) / 2, torch.histogram(values, bins=edges).hist


def histogram_quantile(centers, counts, quantile):
    cumulative = counts.cumsum(0)
    target = quantile * cumulative[-1]
    index = int(torch.searchsorted(cumulative, target).clamp(max=len(centers) - 1))
    if index == 0:
        return float(centers[0])
    fraction = (target - cumulative[index - 1]) / (cumulative[index] - cumulative[index - 1])
    return float(centers[index - 1] + fraction * (centers[index] - centers[index - 1]))


class ActivationRecorder:
    def __init__(self, per_key=16384, per_event=512, seed=314159):
        if per_key < 1 or per_event < 1:
            raise ValueError("Calibration budgets must be positive")
        self.per_key, self.per_event = per_key, per_event
        self.seed = seed
        self.samples, self.priorities, self.observations = {}, {}, {}
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def observe(self, stage, name, bucket, activation, valid_rows=None):
        key = f"{stage}/{name}/bucket{bucket}"
        count = min(self.per_event, activation.numel())
        if not count:
            return
        index = torch.randint(activation.numel(), (count,), generator=self.generator).to(activation.device)
        sample = (activation.view(-1)[index] if activation.is_contiguous()
                  else activation[torch.unravel_index(index, activation.shape)])
        if valid_rows is not None:
            sample = sample[valid_rows.reshape(-1)[index // activation.shape[-1]]]
        sample = sample.float().cpu()
        if not torch.isfinite(sample).all():
            raise ValueError(f"Nonfinite calibration activation: {key}")
        if not sample.numel():
            return
        self.observations[key] = self.observations.get(key, 0) + sample.numel()
        priorities = torch.rand(sample.numel(), generator=self.generator)
        if key in self.samples:
            sample = torch.cat((self.samples[key], sample))
            priorities = torch.cat((self.priorities[key], priorities))
        retained = priorities.topk(min(self.per_key, priorities.numel()), sorted=False).indices
        self.samples[key], self.priorities[key] = sample[retained], priorities[retained]

    def state(self):
        return dict(samples=self.samples, observations=self.observations,
                    per_key=self.per_key, per_event=self.per_event, seed=self.seed,
                    sampling="seeded event-stratified reservoir; padding excluded; not a full-population histogram")


class VideoCalibration:
    def __init__(self, artifact):
        if artifact.get("schema") != "magi-teal-signed-v1" or not artifact.get("complete"):
            raise ValueError("Incomplete or incompatible MAGI TEAL calibration")
        self.artifact = artifact
        self.cache = {}

    @classmethod
    def load(cls, path, expected=None):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        calibration = cls(artifact)
        if expected is not None and artifact["identity"] != expected:
            raise ValueError("MAGI calibration model/protocol/runtime identity mismatch")
        return calibration

    def threshold(self, stage, name, bucket, sparsity):
        key = f"{stage}/{name}/bucket{bucket}"
        cache_key = (key, sparsity)
        if cache_key not in self.cache:
            if key not in self.artifact["histograms"]:
                raise ValueError(f"Uncalibrated MAGI projection/time bucket: {key}")
            histogram = self.artifact["histograms"][key]
            self.cache[cache_key] = histogram_quantile(histogram["centers"], histogram["counts"], .5 + sparsity / 2)
        return self.cache[cache_key]


def merge_calibration(folder, destination, ranks=2):
    folder, destination = Path(folder), Path(destination)
    shards = []
    for rank in range(ranks):
        shard = torch.load(folder / f"calibration-rank{rank}.pt", map_location="cpu", weights_only=True)
        if not shard.get("complete"):
            raise ValueError(f"Calibration rank {rank} did not finish")
        shards.append(shard)
    identity = shards[0]["identity"]
    if any(shard["identity"] != identity for shard in shards):
        raise ValueError("Calibration rank identities differ")
    keys = set().union(*(shard["samples"] for shard in shards))
    if not all(any(key.startswith(stage + "/") for key in keys) for stage in ("preview", "refiner")):
        raise ValueError("Both preview and refiner calibration are required")
    histograms = {}
    for key in sorted(keys):
        values = torch.cat([shard["samples"][key] for shard in shards if key in shard["samples"]])
        centers, counts = signed_histogram(values)
        histograms[key] = dict(centers=centers, counts=counts, samples=values.numel())
    artifact = dict(schema="magi-teal-signed-v1", complete=True, identity=identity,
                    histograms=histograms, ranks=ranks,
                    threshold_rule="abs(x) > signed_histogram.icdf(0.5 + sparsity/2)",
                    time_buckets="DiT forward invocation // 10; includes CFG; resets each video",
                    routed_policy="per-layer projection pooled over local experts; merge CP/EP ranks",
                    sampling=shards[0]["sampling"], per_key=shards[0]["per_key"],
                    per_event=shards[0]["per_event"], seed=shards[0]["seed"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(artifact, temporary)
    temporary.replace(destination)
    return dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                histograms=len(histograms), identity=identity)
