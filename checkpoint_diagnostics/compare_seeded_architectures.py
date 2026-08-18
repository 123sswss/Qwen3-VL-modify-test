#!/usr/bin/env python3
"""Compare two architectures across matched SLAKE training seeds.

The analysis is read-only and uses the per-question JSON emitted by
``slake_official_eval.py``. Matching seeds are compared question by question,
while cross-seed summaries expose whether a gain is stable or just moves errors
between samples.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ROOT = Path("/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl")
DEFAULT_LEFT_PREFIX = "slake_mmrl_shared_direct_relation0050_repro3"
DEFAULT_RIGHT_PREFIX = "slake_mmrl_layer_mlp_full_ca_relation0050_repro3"
GROUP_FIELDS = ("answer_type", "question_type", "language")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def resolve_latest_experiment(root: Path, prefix: str, seed: int) -> Path:
    candidates = [
        path
        for path in root.glob(f"{prefix}_seed{seed}_*")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No experiment matches {prefix}_seed{seed}_* under {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_comparisons(experiment: Path) -> Path:
    candidates = (
        experiment / "eval" / "slake_comparisons.json",
        experiment / "slake_comparisons.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No final slake_comparisons.json found under {experiment}"
    )


def resolve_latest_evaluated_experiment(
    root: Path,
    prefix: str,
    seed: int,
) -> tuple[Path, Path]:
    candidates = [
        path
        for path in root.glob(f"{prefix}_seed{seed}_*")
        if path.is_dir()
    ]
    evaluated = []
    for experiment in candidates:
        try:
            comparisons = resolve_comparisons(experiment)
        except FileNotFoundError:
            continue
        evaluated.append((comparisons.stat().st_mtime, experiment, comparisons))
    if not evaluated:
        candidate_names = [path.name for path in sorted(candidates)]
        raise FileNotFoundError(
            "No evaluated experiment matches "
            f"{prefix}_seed{seed}_* under {root}; candidates={candidate_names}"
        )
    _, experiment, comparisons = max(evaluated, key=lambda item: item[0])
    return experiment, comparisons


def load_rows(experiment: Path) -> dict[str, dict[str, Any]]:
    path = resolve_comparisons(experiment)
    rows = load_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty comparison list: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row["question_id"])
        if qid in indexed:
            raise ValueError(f"Duplicate question_id={qid} in {path}")
        indexed[qid] = dict(row)
    return indexed


def accuracy(rows: Iterable[Mapping[str, Any]]) -> float:
    values = [int(row["correct"]) for row in rows]
    return 100.0 * sum(values) / max(len(values), 1)


def group_key(row: Mapping[str, Any], field: str) -> str:
    return str(row.get(field) or "unknown")


def checkpoint_metrics(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    metrics = {"overall": accuracy(rows.values())}
    for field in GROUP_FIELDS:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows.values():
            grouped[group_key(row, field)].append(row)
        for value, group_rows in sorted(grouped.items()):
            metrics[f"{field}:{value}"] = accuracy(group_rows)
    return metrics


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, index)
        for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def validate_alignment(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:10]
        missing_right = sorted(set(left) - set(right))[:10]
        raise ValueError(
            "Question IDs differ between paired checkpoints: "
            f"missing_left={missing_left} missing_right={missing_right}"
        )
    qids = sorted(left)
    for qid in qids:
        for field in GROUP_FIELDS:
            if group_key(left[qid], field) != group_key(right[qid], field):
                raise ValueError(
                    f"Metadata mismatch for question_id={qid} field={field}"
                )
    return qids


def paired_counts(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    qids: Sequence[str],
) -> dict[str, Any]:
    counts = Counter()
    for qid in qids:
        left_correct = bool(left[qid]["correct"])
        right_correct = bool(right[qid]["correct"])
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif left_correct:
            counts["left_only"] += 1
        elif right_correct:
            counts["right_only"] += 1
        else:
            counts["both_wrong"] += 1
    total = len(qids)
    return {
        "total": total,
        "both_correct": counts["both_correct"],
        "left_only": counts["left_only"],
        "right_only": counts["right_only"],
        "both_wrong": counts["both_wrong"],
        "right_minus_left_pp": (
            100.0 * (counts["right_only"] - counts["left_only"]) / total
        ),
        "mcnemar_exact_p": exact_mcnemar_p(
            counts["left_only"], counts["right_only"]
        ),
    }


def paired_group_counts(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    qids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in qids:
        for field in GROUP_FIELDS:
            groups[f"{field}:{group_key(left[qid], field)}"].append(qid)
    return {
        name: paired_counts(left, right, group_qids)
        for name, group_qids in sorted(groups.items())
    }


def architecture_stability(
    seed_rows: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    seeds = sorted(seed_rows)
    qids = sorted(next(iter(seed_rows.values())))
    counts = Counter()
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for qid in qids:
        correctness = [bool(seed_rows[seed][qid]["correct"]) for seed in seeds]
        answers = {
            str(seed_rows[seed][qid].get("normalized_predicted_answer", ""))
            for seed in seeds
        }
        correct_count = sum(correctness)
        if correct_count == len(seeds):
            status = "unanimous_correct"
        elif correct_count == 0:
            status = "unanimous_wrong"
        else:
            status = "correctness_unstable"
        counts[status] += 1
        if len(answers) > 1:
            counts["answer_unstable"] += 1
        reference = seed_rows[seeds[0]][qid]
        for field in GROUP_FIELDS:
            group_counts[f"{field}:{group_key(reference, field)}"][status] += 1

    total = len(qids)
    return {
        "total": total,
        "unanimous_correct": counts["unanimous_correct"],
        "unanimous_wrong": counts["unanimous_wrong"],
        "correctness_unstable": counts["correctness_unstable"],
        "correctness_unstable_ratio": counts["correctness_unstable"] / total,
        "answer_unstable": counts["answer_unstable"],
        "answer_unstable_ratio": counts["answer_unstable"] / total,
        "groups": {
            name: {
                **dict(values),
                "total": sum(values.values()),
                "correctness_unstable_ratio": (
                    values["correctness_unstable"] / max(sum(values.values()), 1)
                ),
            }
            for name, values in sorted(group_counts.items())
        },
    }


def cross_seed_architecture_effect(
    left_rows: Mapping[int, Mapping[str, Mapping[str, Any]]],
    right_rows: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    seeds = sorted(left_rows)
    qids = sorted(next(iter(left_rows.values())))
    counts = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid in qids:
        left_correct = sum(bool(left_rows[seed][qid]["correct"]) for seed in seeds)
        right_correct = sum(bool(right_rows[seed][qid]["correct"]) for seed in seeds)
        if right_correct > left_correct:
            status = "right_more_often_correct"
        elif left_correct > right_correct:
            status = "left_more_often_correct"
        else:
            status = "tied"
        counts[status] += 1
        if right_correct == len(seeds) and left_correct <= 1:
            counts["stable_right_rescue"] += 1
        if left_correct == len(seeds) and right_correct <= 1:
            counts["stable_right_harm"] += 1
        reference = left_rows[seeds[0]][qid]
        for field in GROUP_FIELDS:
            groups[f"{field}:{group_key(reference, field)}"][status] += 1
        if status != "tied" and len(examples[status]) < 20:
            examples[status].append({
                "question_id": reference["question_id"],
                "question": reference.get("question", ""),
                "ground_truth_answers": reference.get("ground_truth_answers", []),
                "left_correct_seeds": left_correct,
                "right_correct_seeds": right_correct,
                "answer_type": reference.get("answer_type", "unknown"),
                "question_type": reference.get("question_type", "unknown"),
                "language": reference.get("language", "unknown"),
            })
    return {
        **dict(counts),
        "total": len(qids),
        "groups": {
            name: {**dict(values), "total": sum(values.values())}
            for name, values in sorted(groups.items())
        },
        "examples": dict(examples),
    }


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return output


def build_report(result: Mapping[str, Any]) -> str:
    left_name = result["left_name"]
    right_name = result["right_name"]
    lines = ["# Seeded Architecture Comparison", ""]
    lines.extend(markdown_table(
        ("Seed", left_name, right_name, f"{right_name}-{left_name}", "McNemar p"),
        [
            (
                seed,
                f"{row['left_accuracy']:.2f}",
                f"{row['right_accuracy']:.2f}",
                f"{row['paired']['right_minus_left_pp']:+.2f}",
                f"{row['paired']['mcnemar_exact_p']:.4g}",
            )
            for seed, row in result["seeds"].items()
        ],
    ))
    lines.extend(("", "## Architecture Mean", ""))
    lines.extend(markdown_table(
        ("Architecture", "Accuracy mean", "Sample std", "Correctness unstable", "Answer unstable"),
        [
            (
                name,
                f"{summary['accuracy_mean']:.2f}",
                f"{summary['accuracy_std']:.2f}",
                f"{100.0 * summary['stability']['correctness_unstable_ratio']:.2f}%",
                f"{100.0 * summary['stability']['answer_unstable_ratio']:.2f}%",
            )
            for name, summary in result["architectures"].items()
        ],
    ))
    effect = result["cross_seed_effect"]
    lines.extend(("", "## Cross-Seed Question Effect", ""))
    lines.extend(markdown_table(
        (f"{right_name} more often correct", f"{left_name} more often correct", "Tied", f"Stable {right_name} rescue", f"Stable {right_name} harm"),
        [(
            effect.get("right_more_often_correct", 0),
            effect.get("left_more_often_correct", 0),
            effect.get("tied", 0),
            effect.get("stable_right_rescue", 0),
            effect.get("stable_right_harm", 0),
        )],
    ))
    lines.extend(("", "## Paired Group Deltas", ""))
    group_names = sorted(next(iter(result["seeds"].values()))["groups"])
    group_rows = []
    for group_name in group_names:
        deltas = [
            result["seeds"][str(seed)]["groups"][group_name]["right_minus_left_pp"]
            for seed in result["seed_values"]
        ]
        mean, std = mean_std(deltas)
        group_rows.append((group_name, f"{mean:+.2f}", f"{std:.2f}", ", ".join(f"{value:+.2f}" for value in deltas)))
    lines.extend(markdown_table(
        ("Group", "Mean delta (pp)", "Std", "Per-seed deltas"),
        group_rows,
    ))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline paired comparison of two SLAKE architectures."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--left-prefix", default=DEFAULT_LEFT_PREFIX)
    parser.add_argument("--right-prefix", default=DEFAULT_RIGHT_PREFIX)
    parser.add_argument("--left-name", default="shared_direct")
    parser.add_argument("--right-name", default="layer_mlp")
    parser.add_argument("--seeds", type=int, nargs="+", default=(44, 45, 46))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = sorted(set(args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two matched seeds are required")
    root = args.root.expanduser().resolve()
    left_rows: dict[int, dict[str, dict[str, Any]]] = {}
    right_rows: dict[int, dict[str, dict[str, Any]]] = {}
    seed_results: dict[str, Any] = {}

    for seed in seeds:
        left_experiment, left_comparisons = resolve_latest_evaluated_experiment(
            root, args.left_prefix, seed
        )
        right_experiment, right_comparisons = resolve_latest_evaluated_experiment(
            root, args.right_prefix, seed
        )
        print(
            f"[SELECT] seed={seed} {args.left_name}={left_experiment.name} "
            f"comparisons={left_comparisons}"
        )
        print(
            f"[SELECT] seed={seed} {args.right_name}={right_experiment.name} "
            f"comparisons={right_comparisons}"
        )
        left_rows[seed] = load_rows(left_experiment)
        right_rows[seed] = load_rows(right_experiment)
        qids = validate_alignment(left_rows[seed], right_rows[seed])
        left_metrics = checkpoint_metrics(left_rows[seed])
        right_metrics = checkpoint_metrics(right_rows[seed])
        seed_results[str(seed)] = {
            "left_experiment": str(left_experiment),
            "right_experiment": str(right_experiment),
            "left_accuracy": left_metrics["overall"],
            "right_accuracy": right_metrics["overall"],
            "left_metrics": left_metrics,
            "right_metrics": right_metrics,
            "paired": paired_counts(left_rows[seed], right_rows[seed], qids),
            "groups": paired_group_counts(left_rows[seed], right_rows[seed], qids),
        }

    reference_qids = set(next(iter(left_rows.values())))
    for architecture, runs in ((args.left_name, left_rows), (args.right_name, right_rows)):
        for seed, rows in runs.items():
            if set(rows) != reference_qids:
                raise ValueError(
                    f"Question IDs differ across {architecture} seeds; seed={seed}"
                )

    left_accuracies = [seed_results[str(seed)]["left_accuracy"] for seed in seeds]
    right_accuracies = [seed_results[str(seed)]["right_accuracy"] for seed in seeds]
    left_mean, left_std = mean_std(left_accuracies)
    right_mean, right_std = mean_std(right_accuracies)
    result = {
        "root": str(root),
        "seed_values": seeds,
        "left_name": args.left_name,
        "right_name": args.right_name,
        "seeds": seed_results,
        "architectures": {
            args.left_name: {
                "accuracy_mean": left_mean,
                "accuracy_std": left_std,
                "stability": architecture_stability(left_rows),
            },
            args.right_name: {
                "accuracy_mean": right_mean,
                "accuracy_std": right_std,
                "stability": architecture_stability(right_rows),
            },
        },
        "cross_seed_effect": cross_seed_architecture_effect(left_rows, right_rows),
    }
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "outputs" / f"seeded_architecture_{timestamp}"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", result)
    report = build_report(result)
    (output_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {output_dir / 'comparison.md'}")
    print(f"JSON: {output_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
