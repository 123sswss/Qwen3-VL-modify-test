#!/usr/bin/env python3
"""Cluster-paired fixed Validation comparison: normalized V1 5 epochs versus 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics.compare_pathvqa_conditioning_mismatches import (
    clustered_paired_bootstrap, indexed, load_json, paired_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-epoch-eval", type=Path, required=True)
    parser.add_argument("--five-epoch-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="pathvqa_v1_norm_fixed_5ep_seed44")
    parser.add_argument(
        "--baseline", default="pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44"
    )
    args = parser.parse_args()
    baseline = load_json(args.three_epoch_eval / "pathvqa_comparisons.json")
    variant = load_json(args.five_epoch_eval / "pathvqa_comparisons.json")
    summary = load_json(args.five_epoch_eval / "pathvqa_summary.json")
    if len(baseline) != 6259 or len(variant) != 6259:
        raise ValueError(f"Complete PathVQA Validation required: 3ep={len(baseline)} 5ep={len(variant)}")
    baseline_by_id, variant_by_id = indexed(baseline), indexed(variant)
    if set(baseline_by_id) != set(variant_by_id):
        raise ValueError("3ep/5ep question ID sets differ")
    paired = [(row, variant_by_id[str(row["question_id"])]) for row in baseline]
    for old, new in paired:
        for key in ("image_id", "answer_type", "question_type", "reference"):
            if str(old.get(key)) != str(new.get(key)):
                raise ValueError(f"3ep/5ep metadata differ at {old['question_id']}: {key}")
    if len({str(old["image_id"]) for old, _ in paired}) != 832:
        raise ValueError("Expected 832 image clusters")
    groups = {
        "overall": paired,
        "yes_no": [(a, b) for a, b in paired if str(a["answer_type"]) == "yes/no"],
        "free_form": [(a, b) for a, b in paired if str(a["answer_type"]) != "yes/no"],
        "what": [(a, b) for a, b in paired if str(a["question_type"]) == "what"],
        "where": [(a, b) for a, b in paired if str(a["question_type"]) == "where"],
    }
    results = {}
    for name, rows in groups.items():
        old, new = [a for a, _ in rows], [b for _, b in rows]
        metrics = paired_metrics(old, new)
        metrics["clustered_paired_delta_ci"] = clustered_paired_bootstrap(
            old, new, iterations=10000, seed=42,
        )
        metrics["question_count"] = len(rows)
        metrics["image_clusters"] = len({str(a["image_id"]) for a in old})
        metrics["exploratory"] = name != "overall"
        results[name] = metrics
        print(
            f"[V1_5EP_VS_3EP] group={name} n={len(rows)} clusters={metrics['image_clusters']} "
            f"three_epoch={metrics['baseline_accuracy']:.4f} "
            f"five_epoch={metrics['variant_accuracy']:.4f} delta={metrics['delta']:+.4f} "
            f"five_only={metrics['variant_only_correct']} "
            f"three_only={metrics['baseline_only_correct']} "
            f"ci={metrics['clustered_paired_delta_ci']}", flush=True,
        )
    payload = {
        "experiment": args.experiment,
        "baseline": args.baseline,
        "bootstrap": {"unit": "image_id", "iterations": 10000, "seed": 42},
        "validation_summary": summary, "groups": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
