"""TEAL/WINA greedy search order, generalized to named DeepSeek projections.

Algorithm reference: FasterDecoding/TEAL fb7373c, teal/greedyopt.py;
Microsoft/WINA 04ac879, wina/greedyopt.py. The reference chooses the candidate
with smallest block output error, using inverse-size rate increments. Here
model forwards are supplied by an adapter. Target clipping, finite checks and
restart checkpoints are explicit additions; no gradient-based optimization.
"""
from __future__ import annotations

import math
import time

from benchmarks.deepseek_greedy_strategy import effective_sparsity


def search(weights, evaluate, *, targets=(.5, .6, .7, .8, .9), step=.025,
           reference_weight=None, checkpoint=None, resume=None, max_steps=None):
    """One monotone search curve, with exact snapshots at every requested rate.

    evaluate(rates) returns the same dense-reference layer error for all
    candidates. All distributed ranks must call this routine in the same order.
    max_steps is a diagnostic limit, never marks an unfinished curve complete.
    """
    groups = list(weights)
    zero = dict.fromkeys(groups, 0.)
    effective_sparsity(zero, weights)
    targets = sorted(set(float(v) for v in targets))
    if not targets or any(not math.isfinite(v) or not 0 < v < 1 for v in targets):
        raise ValueError("Targets must be finite and in (0, 1)")
    reference_weight = reference_weight or max(weights.values())
    if not 0 < step < 1 or not math.isfinite(reference_weight) or reference_weight <= 0:
        raise ValueError("Invalid greedy step/reference weight")
    if resume:
        if resume.get("weights") != weights or resume.get("targets") != targets or resume.get("step") != step:
            raise ValueError("Greedy restart budget mismatch")
        if resume.get("reference_weight") != reference_weight:
            raise ValueError("Greedy restart step unit mismatch")
    state = resume or dict(weights=weights, targets=targets, step=step,
                           reference_weight=reference_weight, rates=zero,
                           iterations=0, candidate_forwards=0, profiles=[], curve=[],
                           elapsed_seconds=0., complete=False)
    rates = state["rates"].copy()
    effective_sparsity(rates, weights)
    started = time.monotonic()
    previous_time = state["elapsed_seconds"]
    initial_iteration = state["iterations"]
    total = sum(weights.values())
    for target in targets:
        if any(abs(p["target"] - target) < 1e-9 for p in state["profiles"]):
            continue
        while effective_sparsity(rates, weights) < target - 1e-9:
            if max_steps is not None and state["iterations"] - initial_iteration >= max_steps:
                state["elapsed_seconds"] = previous_time + time.monotonic() - started
                if checkpoint:
                    checkpoint(state)
                return state
            best = None
            candidates = []
            remaining = target - effective_sparsity(rates, weights)
            for group in groups:
                if weights[group] == 0:
                    continue
                delta = min(step * reference_weight / weights[group],
                            remaining * total / weights[group], 1 - 1e-6 - rates[group])
                if delta <= 1e-10:
                    continue
                candidate = rates.copy()
                candidate[group] += delta
                error = float(evaluate(candidate))
                if not math.isfinite(error) or error < 0:
                    raise ValueError(f"Non-finite/negative candidate error for {group}")
                state["candidate_forwards"] += 1
                candidates.append(dict(group=group, error=error, delta=delta,
                    effective_sparsity=effective_sparsity(candidate, weights)))
                # Native loop resolves equal errors by projection iteration order.
                if best is None or error < best[0]:
                    best = error, group, candidate
            if best is None:
                raise ValueError("No remaining candidate can satisfy the requested budget")
            error, group, rates = best
            state.update(rates=rates.copy(), iterations=state["iterations"] + 1,
                         elapsed_seconds=previous_time + time.monotonic() - started)
            state["curve"].append(dict(iteration=state["iterations"], chosen=group,
                effective_sparsity=effective_sparsity(rates, weights), error=error,
                rates=rates.copy(), candidates=candidates, elapsed_seconds=state["elapsed_seconds"]))
            if checkpoint:
                checkpoint(state)
        uniform_error = float(evaluate({g: target if weights[g] else 0. for g in groups}))
        if not math.isfinite(uniform_error):
            raise ValueError("Non-finite uniform reference error")
        state["profiles"].append(dict(target=target, rates=rates.copy(), weights=weights,
            error=state["curve"][-1]["error"], uniform_error=uniform_error))
        if checkpoint:
            checkpoint(state)
    state.update(complete=True, elapsed_seconds=previous_time + time.monotonic() - started)
    if checkpoint:
        checkpoint(state)
    return state


def native_activation_error(target, actual, last_fraction=.25):
    """Exact upstream norm axis: sequence dimension, then mean other axes."""
    import torch
    if target.shape != actual.shape or actual.ndim < 3 or not 0 < last_fraction <= 1:
        raise ValueError("Invalid block outputs/error window")
    start = int(actual.shape[1] * (1 - last_fraction))
    return torch.norm(target[:, start:].float() - actual[:, start:].float(), dim=1).mean()
