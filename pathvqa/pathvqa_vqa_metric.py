"""PathVQA normalized exact-match metrics and clustered confidence intervals."""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


YES_NO_ANSWERS = frozenset(("yes", "no"))
QUESTION_PREFIXES = frozenset(("what", "where", "when", "why", "who", "which", "how"))


def normalize_pathvqa_answer(value: Any) -> str:
    """Match the public PathVQA cleaning while tolerating generation punctuation."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.rstrip(" .")


def answer_type(reference_answer: Any) -> str:
    return (
        "yes/no"
        if normalize_pathvqa_answer(reference_answer) in YES_NO_ANSWERS
        else "free-form"
    )


def question_type(question: Any, reference_answer: Any = "") -> str:
    if answer_type(reference_answer) == "yes/no":
        return "yes/no"
    match = re.search(r"[a-z]+", normalize_pathvqa_answer(question))
    prefix = match.group(0) if match else ""
    return prefix if prefix in QUESTION_PREFIXES else "other"


def _accuracy(rows: Iterable[Mapping[str, Any]]) -> float:
    values = [bool(row["correct"]) for row in rows]
    return 100.0 * sum(values) / len(values) if values else 0.0


def _group_accuracy(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, float]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: round(_accuracy(items), 4)
        for name, items in sorted(grouped.items())
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def clustered_bootstrap_accuracy(
    rows: Sequence[Mapping[str, Any]],
    iterations: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap QA accuracy by image cluster rather than by correlated QA row."""

    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = str(row.get("image_id") or row.get("question_id"))
        grouped[image_id].append(row)
    cluster_ids = sorted(grouped)
    if not cluster_ids:
        raise ValueError("cannot bootstrap an empty PathVQA evaluation")

    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        correct = 0
        total = 0
        for _ in cluster_ids:
            sampled = grouped[rng.choice(cluster_ids)]
            correct += sum(bool(row["correct"]) for row in sampled)
            total += len(sampled)
        estimates.append(100.0 * correct / total)

    return {
        "point_estimate": round(_accuracy(rows), 4),
        "confidence_level": 0.95,
        "lower": round(_percentile(estimates, 0.025), 4),
        "upper": round(_percentile(estimates, 0.975), 4),
        "iterations": iterations,
        "seed": seed,
        "image_clusters": len(cluster_ids),
    }


def evaluate_pathvqa_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 42,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prediction_map = {}
    for row in predictions:
        key = str(row["question_id"])
        if key in prediction_map:
            raise ValueError(f"Duplicate PathVQA prediction ID: {key}")
        prediction_map[key] = row

    comparisons = []
    for record in records:
        question_id = str(record["question_id"])
        if question_id not in prediction_map:
            raise ValueError(f"Missing PathVQA prediction ID: {question_id}")
        prediction = prediction_map[question_id]
        reference = normalize_pathvqa_answer(record["answer"])
        predicted = normalize_pathvqa_answer(prediction.get("answer", ""))
        row_answer_type = answer_type(reference)
        row_question_type = question_type(record["question"], reference)
        comparisons.append(
            {
                "question_id": question_id,
                "image_id": str(
                    prediction.get("image_id")
                    or record.get("image_id")
                    or question_id
                ),
                "question": str(record["question"]),
                "reference": reference,
                "prediction": predicted,
                "answer_type": row_answer_type,
                "question_type": row_question_type,
                "correct": predicted == reference,
            }
        )

    if len(prediction_map) != len(comparisons):
        expected_ids = {row["question_id"] for row in comparisons}
        extras = sorted(set(prediction_map) - expected_ids)
        raise ValueError(f"Unexpected PathVQA prediction IDs: {extras[:5]}")

    per_answer_type = _group_accuracy(comparisons, "answer_type")
    per_question_type = _group_accuracy(comparisons, "question_type")
    free_form_question_scores = [
        score
        for name, score in per_question_type.items()
        if name != "yes/no"
    ]
    summary = {
        "count": len(comparisons),
        "overall_accuracy": round(_accuracy(comparisons), 4),
        "yes_no_accuracy": per_answer_type.get("yes/no", 0.0),
        "free_form_accuracy": per_answer_type.get("free-form", 0.0),
        "per_answer_type_accuracy": per_answer_type,
        "per_question_type_accuracy": per_question_type,
        "free_form_question_type_macro_accuracy": round(
            sum(free_form_question_scores) / len(free_form_question_scores),
            4,
        )
        if free_form_question_scores
        else 0.0,
        "clustered_bootstrap": clustered_bootstrap_accuracy(
            comparisons,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
    }
    return summary, comparisons
