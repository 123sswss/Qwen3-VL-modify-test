"""Summarize inference-only MMRL Cross-Attention delta scaling."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


SCALES = (("scale_0", 0.0), ("scale_0p5", 0.5), ("scale_1", 1.0))


def load_summary(root: Path, tag: str) -> dict[str, Any] | None:
    path = root / tag / "slake_summary.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_value(summary: dict[str, Any], group: str, key: str) -> float | None:
    value = summary.get(group, {}).get(key)
    return None if value is None else float(value)


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_cross_delta_scale.py OUTPUT_ROOT")
    root = Path(sys.argv[1]).expanduser().resolve()
    rows = []
    baseline = None
    for tag, scale in SCALES:
        summary = load_summary(root, tag)
        if summary is None:
            rows.append({"tag": tag, "scale": scale, "status": "missing"})
            continue
        row = {
            "tag": tag,
            "scale": scale,
            "status": "ok",
            "overall": float(summary["overall_accuracy"]),
            "kvqa": group_value(summary, "per_question_type_accuracy", "kvqa"),
            "vqa": group_value(summary, "per_question_type_accuracy", "vqa"),
            "closed": group_value(summary, "per_answer_type_accuracy", "CLOSED"),
            "open": group_value(summary, "per_answer_type_accuracy", "OPEN"),
            "en": group_value(summary, "per_language_accuracy", "en"),
            "zh": group_value(summary, "per_language_accuracy", "zh"),
            "delta": None,
        }
        rows.append(row)
        if scale == 1.0:
            baseline = row["overall"]

    if baseline is not None:
        for row in rows:
            if row["status"] == "ok":
                row["delta"] = row["overall"] - baseline

    payload = {
        "output_root": str(root),
        "baseline_accuracy": baseline,
        "results": rows,
    }
    json_path = root / "cross_delta_scale_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "# MMRL Cross-Attention Delta Scale Ablation",
        "",
        "| Scale | Overall | Delta | KVQA | VQA | CLOSED | OPEN | EN | ZH |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['scale']:.1f} | missing | -- | -- | -- | -- | -- | -- | -- |")
            continue
        delta = "--" if row["delta"] is None else f"{row['delta']:+.2f}"
        lines.append(
            f"| {row['scale']:.1f} | {row['overall']:.2f} | {delta} | "
            f"{format_number(row['kvqa'])} | {format_number(row['vqa'])} | "
            f"{format_number(row['closed'])} | {format_number(row['open'])} | "
            f"{format_number(row['en'])} | {format_number(row['zh'])} |"
        )
    markdown_path = root / "cross_delta_scale_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
