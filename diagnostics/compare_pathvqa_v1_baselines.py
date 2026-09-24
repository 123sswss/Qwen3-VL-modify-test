#!/usr/bin/env python3
"""Pair V1 Validation with available fixed seed44 baseline predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics.compare_pathvqa_conditioning_mismatches import compare_variant, load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-eval", type=Path, required=True)
    parser.add_argument("--original-v1-eval", type=Path)
    parser.add_argument("--v0-eval", type=Path, required=True)
    parser.add_argument("--cocoop-eval", type=Path, required=True)
    parser.add_argument("--static-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variant_path = args.v1_eval / "pathvqa_comparisons.json"
    summary_path = args.v1_eval / "pathvqa_summary.json"
    if not variant_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("V1 complete Validation comparisons and summary are required")
    variant_rows = load_json(variant_path)
    summary = load_json(summary_path)
    results = {}
    baselines = []
    if args.original_v1_eval is not None:
        baselines.append(("original_v1_normalization_deviation_seed44", args.original_v1_eval))
    baselines.extend((
        ("repaired_v0", args.v0_eval),
        ("cocoop_style_seed44", args.cocoop_eval),
        ("static_p20_seed44", args.static_eval),
    ))
    for name, baseline in baselines:
        path = baseline / "pathvqa_comparisons.json"
        if not path.is_file():
            results[name] = {"status": "predictions_missing", "path": str(path)}
            print(f"[V1_PAIRED_MISSING] {name} path={path}")
            continue
        try:
            results[name] = compare_variant(
                name, load_json(path), variant_rows, summary,
                iterations=10000, seed=42,
            )
            row = results[name]["overall"]
            print(
                f"[V1_PAIRED] {name} baseline={row['baseline_accuracy']:.4f} "
                f"v1={row['variant_accuracy']:.4f} delta={row['delta']:+.4f} "
                f"v1_only={row['variant_only_correct']} baseline_only={row['baseline_only_correct']} "
                f"clustered_ci={row['clustered_paired_delta_ci']}"
            )
        except (ValueError, KeyError) as exc:
            results[name] = {"status": "not_pairable", "path": str(path), "reason": str(exc)}
            print(f"[V1_PAIRED_UNAVAILABLE] {name}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
