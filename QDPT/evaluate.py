#!/usr/bin/env python3
"""Compute PathVQA normalized exact-match metrics."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from data import PathVQAParquetStore


YES_NO_ANSWERS = frozenset(("yes", "no"))


def normalize_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.rstrip(" .")


def evaluate_records(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    prediction_map = {
        str(prediction["question_id"]): prediction for prediction in predictions
    }
    grouped: dict[str, list[bool]] = defaultdict(list)
    correct = []
    for record in records:
        reference = normalize_answer(record["answer"])
        prediction = normalize_answer(
            prediction_map[str(record["question_id"])]["answer"]
        )
        answer_type = "yes/no" if reference in YES_NO_ANSWERS else "free-form"
        match = prediction == reference
        correct.append(match)
        grouped[answer_type].append(match)

    accuracy = lambda values: round(100.0 * sum(values) / len(values), 4)
    return {
        "count": len(correct),
        "overall_accuracy": accuracy(correct),
        "yes_no_accuracy": accuracy(grouped["yes/no"]),
        "free_form_accuracy": accuracy(grouped["free-form"]),
    }


def evaluate_file(
    data_root: Path,
    split: str,
    predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    store = PathVQAParquetStore(data_root, split)
    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    summary = evaluate_records(store.samples, predictions)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PathVQA predictions")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test"
    )
    args = parser.parse_args()
    summary = evaluate_file(
        args.data_root, args.split, args.predictions, args.output
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

