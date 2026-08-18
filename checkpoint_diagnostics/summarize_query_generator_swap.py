"""Summarize seed44/45 MMRL query-generator component swaps."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROWS = (
    ("baseline_44", "Q44 + Rest44", "baseline_44"),
    ("baseline_45", "Q45 + Rest45", "baseline_45"),
    ("query44_rest45", "Q44 + Rest45", "baseline_45"),
    ("query45_rest44", "Q45 + Rest44", "baseline_44"),
)


def load_summary(root: Path, tag: str) -> dict[str, Any] | None:
    path = root / tag / "slake_summary.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group(summary: dict[str, Any], family: str, key: str) -> float | None:
    value = summary.get(family, {}).get(key)
    return None if value is None else float(value)


def fmt(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_query_generator_swap.py OUTPUT_ROOT")
    root = Path(sys.argv[1]).expanduser().resolve()
    summaries = {tag: load_summary(root, tag) for tag, _, _ in ROWS}
    rows = []
    for tag, label, baseline_tag in ROWS:
        summary = summaries[tag]
        baseline_summary = summaries[baseline_tag]
        if summary is None:
            rows.append({"tag": tag, "label": label, "status": "missing"})
            continue
        overall = float(summary["overall_accuracy"])
        baseline = (
            None
            if baseline_summary is None
            else float(baseline_summary["overall_accuracy"])
        )
        rows.append({
            "tag": tag,
            "label": label,
            "status": "ok",
            "overall": overall,
            "recipient_delta": None if baseline is None else overall - baseline,
            "kvqa": group(summary, "per_question_type_accuracy", "kvqa"),
            "vqa": group(summary, "per_question_type_accuracy", "vqa"),
            "closed": group(summary, "per_answer_type_accuracy", "CLOSED"),
            "open": group(summary, "per_answer_type_accuracy", "OPEN"),
            "en": group(summary, "per_language_accuracy", "en"),
            "zh": group(summary, "per_language_accuracy", "zh"),
        })

    json_path = root / "query_generator_swap_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"output_root": str(root), "results": rows},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    lines = [
        "# MMRL Query-Generator Swap Ablation",
        "",
        "| Composition | Overall | Recipient delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(
                f"| {row['label']} | missing | -- | -- | -- | -- | -- | -- | -- |"
            )
            continue
        delta = (
            "--"
            if row["recipient_delta"] is None
            else f"{row['recipient_delta']:+.2f}"
        )
        lines.append(
            f"| {row['label']} | {row['overall']:.2f} | {delta} | "
            f"{fmt(row['kvqa'])} | {fmt(row['vqa'])} | "
            f"{fmt(row['closed'])} | {fmt(row['open'])} | "
            f"{fmt(row['en'])} | {fmt(row['zh'])} |"
        )
    markdown_path = root / "query_generator_swap_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
