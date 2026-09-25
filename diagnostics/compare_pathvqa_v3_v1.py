#!/usr/bin/env python3
"""Cluster-paired fixed epoch3 Validation comparison: V3 versus normalized V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics.compare_pathvqa_conditioning_mismatches import (
    clustered_paired_bootstrap, indexed, load_json, paired_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-eval", type=Path, required=True)
    parser.add_argument("--v3-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_json(args.v1_eval / "pathvqa_comparisons.json")
    variant = load_json(args.v3_eval / "pathvqa_comparisons.json")
    summary = load_json(args.v3_eval / "pathvqa_summary.json")
    if len(baseline) != 6259 or len(variant) != 6259:
        raise ValueError(f"Complete PathVQA Validation required: V1={len(baseline)} V3={len(variant)}")
    v1_by_id, v3_by_id = indexed(baseline), indexed(variant)
    if set(v1_by_id) != set(v3_by_id):
        raise ValueError("V1/V3 question ID sets differ")
    paired = [(row, v3_by_id[str(row["question_id"])]) for row in baseline]
    for old, new in paired:
        for key in ("image_id", "answer_type", "question_type", "reference"):
            if str(old.get(key)) != str(new.get(key)):
                raise ValueError(f"V1/V3 paired metadata differ at {old['question_id']}: {key}")
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
        print(f"[V3_V1_PAIRED] group={name} n={len(rows)} clusters={metrics['image_clusters']} "
              f"v1={metrics['baseline_accuracy']:.4f} v3={metrics['variant_accuracy']:.4f} "
              f"delta={metrics['delta']:+.4f} v3_only={metrics['variant_only_correct']} "
              f"v1_only={metrics['baseline_only_correct']} "
              f"ci={metrics['clustered_paired_delta_ci']}", flush=True)
    payload = {
        "experiment": "pathvqa_v3_postvisual_prefix_p20_norm_fixed_seed44",
        "baseline": "pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44",
        "bootstrap": {"unit": "image_id", "iterations": 10000, "seed": 42},
        "validation_summary": summary, "groups": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
