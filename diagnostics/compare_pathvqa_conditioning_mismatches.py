#!/usr/bin/env python3
"""Compare PathVQA conditioning mismatches against one fixed checkpoint run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def exact_mcnemar_p(new_only: int, baseline_only: int) -> float:
    discordant = new_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = min(new_only, baseline_only)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def indexed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in result:
            raise ValueError(f"Duplicate PathVQA question_id={question_id}")
        result[question_id] = row
    return result


def paired_metrics(
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(baseline_rows) != len(variant_rows) or not baseline_rows:
        raise ValueError("Paired PathVQA rows must be non-empty and equal length")
    baseline_correct = [bool(row["correct"]) for row in baseline_rows]
    variant_correct = [bool(row["correct"]) for row in variant_rows]
    new_only = sum(
        variant and not baseline
        for baseline, variant in zip(baseline_correct, variant_correct)
    )
    baseline_only = sum(
        baseline and not variant
        for baseline, variant in zip(baseline_correct, variant_correct)
    )
    count = len(baseline_rows)
    baseline_accuracy = 100.0 * sum(baseline_correct) / count
    variant_accuracy = 100.0 * sum(variant_correct) / count
    return {
        "count": count,
        "baseline_accuracy": baseline_accuracy,
        "variant_accuracy": variant_accuracy,
        "delta": variant_accuracy - baseline_accuracy,
        "variant_only_correct": new_only,
        "baseline_only_correct": baseline_only,
        "mcnemar_exact_p": exact_mcnemar_p(new_only, baseline_only),
    }


def clustered_paired_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(baseline_rows):
        clusters[str(row.get("image_id") or row["question_id"])].append(index)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        delta_sum = 0
        count = 0
        for _ in cluster_ids:
            indices = clusters[rng.choice(cluster_ids)]
            delta_sum += sum(
                int(bool(variant_rows[index]["correct"]))
                - int(bool(baseline_rows[index]["correct"]))
                for index in indices
            )
            count += len(indices)
        estimates.append(100.0 * delta_sum / count)
    return {
        "confidence_level": 0.95,
        "lower": percentile(estimates, 0.025),
        "upper": percentile(estimates, 0.975),
        "iterations": iterations,
        "seed": seed,
        "image_clusters": len(cluster_ids),
    }


def selected_rows(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    value: str,
) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row[key]) == value]


def compare_variant(
    name: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    baseline = indexed(baseline_rows)
    variant = indexed(variant_rows)
    if set(baseline) != set(variant):
        raise ValueError(f"Question IDs do not match for {name}")
    ordered_ids = [str(row["question_id"]) for row in baseline_rows]
    ordered_baseline = [baseline[question_id] for question_id in ordered_ids]
    ordered_variant = [variant[question_id] for question_id in ordered_ids]
    for question_id, base_row, variant_row in zip(
        ordered_ids,
        ordered_baseline,
        ordered_variant,
    ):
        for key in ("image_id", "answer_type", "question_type", "reference"):
            if str(base_row.get(key)) != str(variant_row.get(key)):
                raise ValueError(
                    f"Paired metadata differs for {question_id}: {key}"
                )

    answer_types = sorted({str(row["answer_type"]) for row in ordered_baseline})
    question_types = sorted(
        {str(row["question_type"]) for row in ordered_baseline}
    )
    overall = paired_metrics(ordered_baseline, ordered_variant)
    overall["clustered_paired_delta_ci"] = clustered_paired_bootstrap(
        ordered_baseline,
        ordered_variant,
        iterations=iterations,
        seed=seed,
    )
    return {
        "variant": name,
        "overall": overall,
        "per_answer_type": {
            value: paired_metrics(
                selected_rows(ordered_baseline, "answer_type", value),
                selected_rows(ordered_variant, "answer_type", value),
            )
            for value in answer_types
        },
        "per_question_type": {
            value: paired_metrics(
                selected_rows(ordered_baseline, "question_type", value),
                selected_rows(ordered_variant, "question_type", value),
            )
            for value in question_types
        },
        "intervention": summary.get("dynamic_prompt_intervention", {}),
        "audit": summary.get("directional_conditioning_audit", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention-root", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be positive")
    baseline_rows = load_json(args.baseline / "pathvqa_comparisons.json")
    results = []
    for comparison_path in sorted(
        args.intervention_root.glob("*/pathvqa_comparisons.json")
    ):
        variant_dir = comparison_path.parent
        results.append(
            compare_variant(
                variant_dir.name,
                baseline_rows,
                load_json(comparison_path),
                load_json(variant_dir / "pathvqa_summary.json"),
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            )
        )
    if not results:
        raise FileNotFoundError(
            f"No PathVQA mismatch results found under {args.intervention_root}"
        )

    json_path = args.intervention_root / "conditioning_mismatch_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    columns = (
        "variant",
        "overall",
        "delta",
        "variant_only",
        "baseline_only",
        "mcnemar_exact_p",
        "paired_ci_lower",
        "paired_ci_upper",
        "yes_no",
        "yes_no_delta",
        "free_form",
        "free_form_delta",
    )
    tsv_path = args.intervention_root / "conditioning_mismatch_comparison.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for result in results:
            overall = result["overall"]
            ci = overall["clustered_paired_delta_ci"]
            yes_no = result["per_answer_type"]["yes/no"]
            free_form = result["per_answer_type"]["free-form"]
            writer.writerow(
                {
                    "variant": result["variant"],
                    "overall": overall["variant_accuracy"],
                    "delta": overall["delta"],
                    "variant_only": overall["variant_only_correct"],
                    "baseline_only": overall["baseline_only_correct"],
                    "mcnemar_exact_p": overall["mcnemar_exact_p"],
                    "paired_ci_lower": ci["lower"],
                    "paired_ci_upper": ci["upper"],
                    "yes_no": yes_no["variant_accuracy"],
                    "yes_no_delta": yes_no["delta"],
                    "free_form": free_form["variant_accuracy"],
                    "free_form_delta": free_form["delta"],
                }
            )

    print("\t".join(columns))
    with tsv_path.open("r", encoding="utf-8") as handle:
        for line in list(handle)[1:]:
            print(line.rstrip("\n"))
    for result in results:
        deltas = {
            key: value["delta"]
            for key, value in result["per_question_type"].items()
        }
        print(
            "[PATHVQA_MISMATCH_QUESTION_TYPES] "
            f"variant={result['variant']} deltas={json.dumps(deltas, sort_keys=True)}"
        )
    print(f"[PATHVQA_MISMATCH_COMPARISON] json={json_path} tsv={tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
