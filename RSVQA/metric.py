"""RSVQA-LR answer normalization and official OA/AA metrics."""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple


QUESTION_TYPES = ("rural_urban", "presence", "count", "comp")
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
ANSWER_PREFIX = re.compile(
    r"^(?:answer\s*:\s*|the answer is\s+|there (?:is|are)\s+)", re.IGNORECASE
)
INTEGER = re.compile(r"^[+-]?\d+(?:\.0+)?$")


def _clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value if value is not None else ""))
    text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    text = ANSWER_PREFIX.sub("", text.strip()).strip(" \t\r\n\"'`.,;:!?()[]{}")
    text = " ".join(text.lower().split())
    return NUMBER_WORDS.get(text, text)


def range_number(value: int) -> str:
    """Reproduce official ``VocabEncoder(range_numbers=True)`` for LR counts."""

    if value == 0:
        return "0"
    if 0 < value <= 10:
        return "between 0 and 10"
    if 10 < value <= 100:
        return "between 10 and 100"
    if 100 < value <= 1000:
        return "between 100 and 1000"
    if value > 1000:
        return "more than 1000"
    return str(value)


def normalize_rsvqa_answer(value: Any, question_type: str) -> str:
    text = _clean_text(value)
    if question_type != "count":
        return text
    aliases = {
        "between 0-10": "between 0 and 10",
        "between 10-100": "between 10 and 100",
        "between 100-1000": "between 100 and 1000",
        ">1000": "more than 1000",
        "over 1000": "more than 1000",
    }
    text = aliases.get(text, text)
    if INTEGER.fullmatch(text):
        return range_number(int(float(text)))
    return text


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def image_clustered_bootstrap(
    comparisons: Sequence[Mapping[str, Any]], iterations: int = 10000, seed: int = 42,
) -> Dict[str, Any]:
    clusters: Dict[str, List[int]] = defaultdict(list)
    for row in comparisons:
        clusters[str(row["image_id"])].append(int(row["correct"]))
    keys = sorted(clusters)
    if not keys or iterations <= 0:
        return {"iterations": iterations, "seed": seed, "clusters": len(keys), "lower": None, "upper": None}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        correct = total = 0
        for _ in keys:
            values = clusters[rng.choice(keys)]
            correct += sum(values)
            total += len(values)
        samples.append(100.0 * correct / total)
    return {
        "iterations": iterations,
        "seed": seed,
        "clusters": len(keys),
        "lower": round(_percentile(samples, 0.025), 4),
        "upper": round(_percentile(samples, 0.975), 4),
    }


def evaluate_rsvqa_predictions(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prediction_by_id: Dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        key = str(prediction.get("question_id"))
        if key in prediction_by_id:
            raise ValueError(f"Duplicate RSVQA prediction question_id={key}")
        prediction_by_id[key] = prediction

    comparisons: List[Dict[str, Any]] = []
    seen = set()
    for reference in references:
        key = str(reference["question_id"])
        if key in seen:
            raise ValueError(f"Duplicate RSVQA reference question_id={key}")
        seen.add(key)
        qtype = str(reference["question_type"])
        prediction = prediction_by_id.get(key)
        raw_prediction = "" if prediction is None else str(prediction.get("answer", ""))
        normalized_prediction = normalize_rsvqa_answer(raw_prediction, qtype)
        normalized_reference = normalize_rsvqa_answer(reference["answer"], qtype)
        comparisons.append(
            {
                "question_id": reference["question_id"],
                "image_id": reference["image_id"],
                "question_type": qtype,
                "question": reference["question"],
                "ground_truth_answer": reference["answer"],
                "normalized_ground_truth_answer": normalized_reference,
                "predicted_answer": raw_prediction,
                "normalized_predicted_answer": normalized_prediction,
                "correct": int(prediction is not None and normalized_prediction == normalized_reference),
                "missing_prediction": prediction is None,
            }
        )

    per_type: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for qtype in QUESTION_TYPES:
        rows = [row for row in comparisons if row["question_type"] == qtype]
        if rows:
            counts[qtype] = len(rows)
            per_type[qtype] = round(100.0 * sum(row["correct"] for row in rows) / len(rows), 4)
    total = len(comparisons)
    overall = 100.0 * sum(row["correct"] for row in comparisons) / total if total else 0.0
    summary = {
        "dataset": "RSVQA-LR",
        "metric": "official_range_normalized_exact_match",
        "normalization": {
            "count": "official VocabEncoder range_numbers=True: 0,1-10,11-100,101-1000,>1000",
            "other_types": "NFKC + lowercase + whitespace + short-answer wrapper/punctuation stripping",
        },
        "count": total,
        "correct": sum(row["correct"] for row in comparisons),
        "overall_accuracy": round(overall, 4),
        "average_accuracy": round(sum(per_type.values()) / len(per_type), 4) if per_type else 0.0,
        "per_question_type_accuracy": per_type,
        "per_question_type_count": counts,
        "missing_predictions": sum(row["missing_prediction"] for row in comparisons),
        "extra_prediction_ids": sorted(set(prediction_by_id) - seen),
        "image_clustered_95_ci": image_clustered_bootstrap(
            comparisons, iterations=bootstrap_iterations, seed=bootstrap_seed
        ),
    }
    return summary, comparisons
