#!/usr/bin/env python3
"""Pair one complete V2 Validation against fixed normalized V1 seed44."""

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
    parser.add_argument("--v2-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v1 = load_json(args.v1_eval / "pathvqa_comparisons.json")
    v2 = load_json(args.v2_eval / "pathvqa_comparisons.json")
    v2_summary = load_json(args.v2_eval / "pathvqa_summary.json")
    if len(v1) != 6259 or len(v2) != 6259:
        raise ValueError(f"Complete PathVQA Validation required: V1={len(v1)} V2={len(v2)}")
    first, second = indexed(v1), indexed(v2)
    if set(first) != set(second):
        raise ValueError("V1/V2 question ID sets differ")
    ordered = [(row, second[str(row["question_id"])]) for row in v1]
    for a, b in ordered:
        for key in ("image_id", "answer_type", "question_type", "reference"):
            if str(a.get(key)) != str(b.get(key)):
                raise ValueError(f"V1/V2 paired metadata differ at {a['question_id']}: {key}")
    if len({str(a["image_id"]) for a, _ in ordered}) != 832:
        raise ValueError("Expected 832 image clusters")
    groups = {
        "overall": ordered,
        "yes_no": [(a, b) for a, b in ordered if str(a["answer_type"]) == "yes/no"],
        "free_form": [(a, b) for a, b in ordered if str(a["answer_type"]) != "yes/no"],
        "what": [(a, b) for a, b in ordered if str(a["question_type"]) == "what"],
        "where": [(a, b) for a, b in ordered if str(a["question_type"]) == "where"],
    }
    results = {}
    for name, pairs in groups.items():
        baseline = [a for a, _ in pairs]
        variant = [b for _, b in pairs]
        metrics = paired_metrics(baseline, variant)
        metrics["clustered_paired_delta_ci"] = clustered_paired_bootstrap(
            baseline, variant, iterations=10000, seed=42,
        )
        metrics["question_count"] = len(pairs)
        metrics["image_clusters"] = len({str(a["image_id"]) for a in baseline})
        metrics["exploratory"] = name != "overall"
        results[name] = metrics
        print(f"[V2_V1_PAIRED] group={name} n={len(pairs)} clusters={metrics['image_clusters']} "
              f"v1={metrics['baseline_accuracy']:.4f} v2={metrics['variant_accuracy']:.4f} "
              f"delta={metrics['delta']:+.4f} v2_only={metrics['variant_only_correct']} "
              f"v1_only={metrics['baseline_only_correct']} "
              f"ci={metrics['clustered_paired_delta_ci']}", flush=True)
    payload = {
        "experiment": "pathvqa_v2_layer_mix_prefix_p20_norm_fixed_seed44",
        "baseline": "pathvqa_v1_visual_selection_prefix_p20_norm_fixed_seed44",
        "bootstrap": {"unit": "image_id", "iterations": 10000, "seed": 42},
        "validation_summary": v2_summary,
        "groups": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
