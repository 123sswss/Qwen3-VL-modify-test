#!/usr/bin/env python3
"""Analyze sample-level PathVQA disagreement across existing QDPT seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathvqa.pathvqa_vqa_metric import normalize_pathvqa_answer


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Expected non-empty NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def resolve_eval_dir(path: Path) -> Path:
    path = path.resolve()
    candidates = (
        path,
        path / "eval_validation" / "epoch_3",
    )
    for candidate in candidates:
        if (
            (candidate / "pathvqa_comparisons.json").is_file()
            and (candidate / "pathvqa_summary.json").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find PathVQA comparison and summary files under "
        f"{path} or its eval_validation/epoch_3 directory"
    )


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in result:
            raise ValueError(f"Duplicate PathVQA question_id={question_id}")
        result[question_id] = row
    if not result:
        raise ValueError("PathVQA comparison file is empty")
    return result


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def accuracy(rows: Iterable[Mapping[str, Any]]) -> float:
    values = [bool(row["correct"]) for row in rows]
    return 100.0 * sum(values) / len(values) if values else 0.0


def validate_and_order(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[str], list[str], dict[str, dict[str, Mapping[str, Any]]]]:
    names = list(runs)
    if len(names) < 2:
        raise ValueError("At least two seed runs are required")
    indexed = {name: index_rows(runs[name]) for name in names}
    reference_ids = set(indexed[names[0]])
    for name in names[1:]:
        if set(indexed[name]) != reference_ids:
            missing = sorted(reference_ids - set(indexed[name]))[:5]
            extra = sorted(set(indexed[name]) - reference_ids)[:5]
            raise ValueError(
                f"Question IDs differ for {name}: missing={missing} extra={extra}"
            )
    ordered_ids = list(indexed[names[0]])
    metadata_keys = (
        "image_id",
        "question",
        "reference",
        "answer_type",
        "question_type",
    )
    for question_id in ordered_ids:
        reference = indexed[names[0]][question_id]
        for name in names[1:]:
            candidate = indexed[name][question_id]
            for key in metadata_keys:
                if str(reference.get(key)) != str(candidate.get(key)):
                    raise ValueError(
                        f"Paired metadata differs for {question_id}: run={name} key={key}"
                    )
    return names, ordered_ids, indexed


def group_summary(
    question_ids: Sequence[str],
    names: Sequence[str],
    indexed: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    count = len(question_ids)
    run_accuracies = {
        name: accuracy(indexed[name][question_id] for question_id in question_ids)
        for name in names
    }
    all_correct = 0
    all_wrong = 0
    prediction_equal = 0
    correct_histogram = {str(index): 0 for index in range(len(names) + 1)}
    for question_id in question_ids:
        correctness = [
            bool(indexed[name][question_id]["correct"]) for name in names
        ]
        predictions = [
            str(indexed[name][question_id]["prediction"]) for name in names
        ]
        correct_count = sum(correctness)
        correct_histogram[str(correct_count)] += 1
        all_correct += correct_count == len(names)
        all_wrong += correct_count == 0
        prediction_equal += len(set(predictions)) == 1
    return {
        "count": count,
        "run_accuracies": run_accuracies,
        "accuracy_range": (
            max(run_accuracies.values()) - min(run_accuracies.values())
            if run_accuracies
            else 0.0
        ),
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "partial_correct": count - all_correct - all_wrong,
        "correct_count_histogram": correct_histogram,
        "prediction_all_equal": prediction_equal,
        "prediction_all_equal_rate": 100.0 * prediction_equal / count if count else 0.0,
    }


def analyze_runs(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    details: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    names, ordered_ids, indexed = validate_and_order(runs)
    indexed_details = (
        {name: index_rows(details[name]) for name in names}
        if details is not None
        else None
    )
    if indexed_details is not None:
        for name in names:
            if set(indexed_details[name]) != set(ordered_ids):
                raise ValueError(f"Detail question IDs differ for {name}")
    sample_rows: list[dict[str, Any]] = []
    for question_id in ordered_ids:
        reference = indexed[names[0]][question_id]
        correctness = [bool(indexed[name][question_id]["correct"]) for name in names]
        predictions = [str(indexed[name][question_id]["prediction"]) for name in names]
        correct_count = sum(correctness)
        if correct_count == len(names):
            category = "all_correct"
        elif correct_count == 0:
            category = "all_wrong"
        else:
            category = f"exactly_{correct_count}_correct"
        row: dict[str, Any] = {
            "question_id": question_id,
            "image_id": str(reference["image_id"]),
            "question": str(reference["question"]),
            "reference": str(reference["reference"]),
            "answer_type": str(reference["answer_type"]),
            "question_type": str(reference["question_type"]),
            "correct_count": correct_count,
            "category": category,
            "unique_prediction_count": len(set(predictions)),
            "all_predictions_equal": len(set(predictions)) == 1,
        }
        for name in names:
            run_row = indexed[name][question_id]
            row[f"{name}_prediction"] = str(run_row["prediction"])
            row[f"{name}_correct"] = bool(run_row["correct"])
            if indexed_details is not None:
                row[f"{name}_raw_answer"] = str(
                    indexed_details[name][question_id].get("answer", "")
                )
        if indexed_details is not None:
            raw_answers = [row[f"{name}_raw_answer"] for name in names]
            raw_normalized = [normalize_pathvqa_answer(value) for value in raw_answers]
            if len(set(raw_answers)) == 1:
                disagreement_kind = "raw_equal"
            elif len(set(raw_normalized)) == 1:
                disagreement_kind = "formatting_only"
            else:
                disagreement_kind = "normalized_answer_disagreement"
            row["disagreement_kind"] = disagreement_kind
        sample_rows.append(row)

    pairwise: list[dict[str, Any]] = []
    for left, right in combinations(names, 2):
        left_only = 0
        right_only = 0
        same_correctness = 0
        same_prediction = 0
        for question_id in ordered_ids:
            left_row = indexed[left][question_id]
            right_row = indexed[right][question_id]
            left_correct = bool(left_row["correct"])
            right_correct = bool(right_row["correct"])
            left_only += left_correct and not right_correct
            right_only += right_correct and not left_correct
            same_correctness += left_correct == right_correct
            same_prediction += str(left_row["prediction"]) == str(
                right_row["prediction"]
            )
        count = len(ordered_ids)
        left_accuracy = accuracy(indexed[left][qid] for qid in ordered_ids)
        right_accuracy = accuracy(indexed[right][qid] for qid in ordered_ids)
        pairwise.append(
            {
                "left": left,
                "right": right,
                "count": count,
                "left_accuracy": left_accuracy,
                "right_accuracy": right_accuracy,
                "right_minus_left": right_accuracy - left_accuracy,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
                "correctness_agreement": same_correctness,
                "correctness_agreement_rate": 100.0 * same_correctness / count,
                "prediction_agreement": same_prediction,
                "prediction_agreement_rate": 100.0 * same_prediction / count,
            }
        )

    answer_types = sorted(
        {str(indexed[names[0]][qid]["answer_type"]) for qid in ordered_ids}
    )
    question_types = sorted(
        {str(indexed[names[0]][qid]["question_type"]) for qid in ordered_ids}
    )
    image_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        image_groups[str(row["image_id"])].append(row)
    image_cluster_rows = []
    for image_id, rows_for_image in sorted(image_groups.items()):
        unstable = sum(
            row["category"] not in {"all_correct", "all_wrong"}
            for row in rows_for_image
        )
        answer_disagreement = sum(
            int(row["unique_prediction_count"]) > 1 for row in rows_for_image
        )
        image_cluster_rows.append(
            {
                "image_id": image_id,
                "question_count": len(rows_for_image),
                "unstable_question_count": unstable,
                "unstable_question_rate": 100.0 * unstable / len(rows_for_image),
                "answer_disagreement_count": answer_disagreement,
            }
        )

    summary = {
        "runs": list(names),
        "comparison_space": "PathVQA-normalized exact-match predictions",
        "overall": group_summary(ordered_ids, names, indexed),
        "per_answer_type": {
            value: group_summary(
                [
                    qid
                    for qid in ordered_ids
                    if str(indexed[names[0]][qid]["answer_type"]) == value
                ],
                names,
                indexed,
            )
            for value in answer_types
        },
        "per_question_type": {
            value: group_summary(
                [
                    qid
                    for qid in ordered_ids
                    if str(indexed[names[0]][qid]["question_type"]) == value
                ],
                names,
                indexed,
            )
            for value in question_types
        },
        "pairwise": pairwise,
        "image_cluster_summary": {
            "count": len(image_cluster_rows),
            "clusters_with_unstable_questions": sum(
                row["unstable_question_count"] > 0 for row in image_cluster_rows
            ),
            "clusters_with_answer_disagreement": sum(
                row["answer_disagreement_count"] > 0
                for row in image_cluster_rows
            ),
        },
        "disagreement_taxonomy": (
            {
                kind: sum(
                    row.get("disagreement_kind") == kind for row in sample_rows
                )
                for kind in (
                    "raw_equal",
                    "formatting_only",
                    "normalized_answer_disagreement",
                )
            }
            if indexed_details is not None
            else None
        ),
    }
    summary["image_cluster_rows"] = image_cluster_rows
    return summary, sample_rows, pairwise


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty TSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read existing QDPT PathVQA runs and report seed disagreement."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    named_paths = dict(args.run)
    if len(named_paths) != len(args.run):
        raise ValueError("Duplicate --run names are not allowed")
    resolved = {name: resolve_eval_dir(path) for name, path in args.run}
    rows = {
        name: load_json(path / "pathvqa_comparisons.json")
        for name, path in resolved.items()
    }
    detail_paths = {
        name: path / "pathvqa_details.json" for name, path in resolved.items()
    }
    details = (
        {name: load_json(path) for name, path in detail_paths.items()}
        if all(path.is_file() for path in detail_paths.values())
        else None
    )
    summary, sample_rows, pairwise = analyze_runs(rows, details=details)
    summary["sources"] = {
        name: {
            "eval_dir": str(path),
            "recorded_summary": load_json(path / "pathvqa_summary.json"),
        }
        for name, path in resolved.items()
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "seed_stability_samples.tsv", sample_rows)
    write_tsv(output_dir / "seed_stability_pairwise.tsv", pairwise)
    image_cluster_rows = summary.pop("image_cluster_rows")
    write_tsv(output_dir / "seed_stability_image_clusters.tsv", image_cluster_rows)
    with (output_dir / "seed_stability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    overall = summary["overall"]
    print(
        "[QDPT_SEED_STABILITY] "
        f"runs={summary['runs']} count={overall['count']} "
        f"all_correct={overall['all_correct']} all_wrong={overall['all_wrong']} "
        f"partial={overall['partial_correct']} "
        f"prediction_agreement={overall['prediction_all_equal_rate']:.4f} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
