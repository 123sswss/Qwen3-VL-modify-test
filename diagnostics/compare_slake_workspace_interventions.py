#!/usr/bin/env python3
"""Compare SLAKE Workspace path interventions with one fixed baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


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


def indexed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in rows}


def accuracy(rows: list[dict[str, Any]]) -> float:
    return 100.0 * sum(bool(row["correct"]) for row in rows) / len(rows)


def grouped_accuracy(
    rows: list[dict[str, Any]], field: str, value: str
) -> float:
    selected = [row for row in rows if row[field] == value]
    return accuracy(selected)


def compare_variant(
    name: str,
    baseline_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    baseline = indexed(baseline_rows)
    variant = indexed(variant_rows)
    if set(baseline) != set(variant):
        raise ValueError(f"Question IDs do not match for {name}")
    new_only = sum(
        bool(variant[qid]["correct"]) and not bool(baseline[qid]["correct"])
        for qid in baseline
    )
    baseline_only = sum(
        bool(baseline[qid]["correct"]) and not bool(variant[qid]["correct"])
        for qid in baseline
    )
    baseline_overall = accuracy(baseline_rows)
    variant_overall = accuracy(variant_rows)
    return {
        "variant": name,
        "overall": variant_overall,
        "delta": variant_overall - baseline_overall,
        "new_only": new_only,
        "baseline_only": baseline_only,
        "mcnemar_exact_p": exact_mcnemar_p(new_only, baseline_only),
        "closed": grouped_accuracy(variant_rows, "answer_type", "CLOSED"),
        "open": grouped_accuracy(variant_rows, "answer_type", "OPEN"),
        "kvqa": grouped_accuracy(variant_rows, "question_type", "kvqa"),
        "vqa": grouped_accuracy(variant_rows, "question_type", "vqa"),
        "en": grouped_accuracy(variant_rows, "language", "en"),
        "zh": grouped_accuracy(variant_rows, "language", "zh"),
        "intervention": summary.get("dynamic_prompt_intervention", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_rows = load_json(args.baseline / "slake_comparisons.json")
    results = []
    for comparison_path in sorted(
        args.intervention_root.glob("*/slake_comparisons.json")
    ):
        variant_dir = comparison_path.parent
        results.append(
            compare_variant(
                variant_dir.name,
                baseline_rows,
                load_json(comparison_path),
                load_json(variant_dir / "slake_summary.json"),
            )
        )
    if not results:
        raise FileNotFoundError(
            f"No intervention results found under {args.intervention_root}"
        )

    json_path = args.intervention_root / "workspace_path_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    columns = (
        "variant",
        "overall",
        "delta",
        "new_only",
        "baseline_only",
        "mcnemar_exact_p",
        "closed",
        "open",
        "kvqa",
        "vqa",
        "en",
        "zh",
    )
    tsv_path = args.intervention_root / "workspace_path_comparison.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in columns})

    print("\t".join(columns))
    for result in results:
        print("\t".join(str(result[key]) for key in columns))
    print(f"[SLAKE_WORKSPACE_COMPARISON] json={json_path} tsv={tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
