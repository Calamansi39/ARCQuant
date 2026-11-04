"""Merge per-rank 2:4 diagnostics, without promoting them to benchmark scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def merge_reports(reports):
    if not reports:
        raise ValueError("No rank reports")
    first = reports[0]
    base = first["runtime"]["structured_sparsity"]
    ranks, counts = set(), {}
    for report in reports:
        current = report["runtime"]["structured_sparsity"]
        if (report["configuration"] != first["configuration"] or
                report["world_size"] != first["world_size"] or
                current["inventory"] != base["inventory"] or
                current["selected_names"] != base["selected_names"] or
                current["counter_columns"] != base["counter_columns"]):
            raise ValueError("Incompatible rank reports")
        rank = report["rank"]
        if rank in ranks or not 0 <= rank < first["world_size"]:
            raise ValueError("Duplicate or invalid rank")
        ranks.add(rank)
        for name, values in current["counters_on_this_rank"].items():
            if name not in base["selected_names"] or len(values) != 7 or any(v < 0 for v in values):
                raise ValueError("Invalid structured counters")
            old = counts.setdefault(name, [0] * 7)
            counts[name] = [a + b for a, b in zip(old, values)]
    total = [sum(values[i] for values in counts.values()) for i in range(7)]
    observed = {name for name, values in counts.items() if values[0] > 0}
    selected = set(base["selected_names"])
    missing_ranks = sorted(set(range(first["world_size"])) - ranks)
    structurally_valid = (total[0] > 0 and total[1] * 2 == total[0] == total[6]
                          and total[3] == total[4] == 0)
    return {
        "configuration": first["configuration"], "coverage_unit": base["coverage_unit"],
        "total_projections": base["total_projections"],
        "configured_all_coverage": base["configured_all_coverage"],
        "configured_scope_coverage": base["configured_scope_coverage"],
        "by_area": base["by_area"], "missing_ranks": missing_ranks,
        "observed_selected_projections": len(observed),
        "observed_all_coverage_lower_bound": len(observed) / base["total_projections"],
        "unobserved_selected": sorted(selected - observed),
        "counter_columns": base["counter_columns"], "totals": total,
        "mask_zero_fraction": total[1] / total[0] if total[0] else None,
        "post_quant_zero_fraction": total[5] / total[6] if total[6] else None,
        "observed_structure_valid": structurally_valid,
        "all_rank_evaluators_returned_successfully": not missing_ranks and
            all(r["evaluation_returned_successfully"] for r in reports),
        "acceptance_status": "not_evaluated",
        "note": "A projection observed on one shard is not proof all its TP shards were exercised. "
                "No task quality, full dataset completeness, share-index or hardware speed is certified.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    result = merge_reports([json.loads(path.read_text()) for path in args.reports])
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
