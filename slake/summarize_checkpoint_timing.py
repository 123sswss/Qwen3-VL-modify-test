#!/usr/bin/env python3
"""Aggregate SLAKE accuracy and timing summaries from existing checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


METHOD_ORDER = (
    "base_untrained",
    "lora_visual_all_attention_r32",
    "lora_visual_all_attention_r64",
    "lora_visual_last8_attention_r32",
    "dora_visual_all_attention_r32",
    "lora_full_model_attention_r16",
)


def nested(mapping: Dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def load_rows(result_root: Path) -> List[Dict[str, Any]]:
    rows = []
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    for summary_path in result_root.glob("*/slake_summary.json"):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        method = summary_path.parent.name
        timing = summary.get("timing", {})
        rows.append({
            "method": method,
            "accuracy": summary.get("overall_accuracy"),
            "ttft_mean_seconds": nested(timing, "ttft_seconds", "mean"),
            "ttft_p50_seconds": nested(timing, "ttft_seconds", "p50"),
            "ttft_p95_seconds": nested(timing, "ttft_seconds", "p95"),
            "tpot_seconds_per_token": timing.get("tpot_weighted_seconds"),
            "decode_tokens_per_second": timing.get("decode_tokens_per_second"),
            "request_mean_seconds": nested(timing, "request_seconds", "mean"),
            "model_tokens_per_second": timing.get(
                "model_generated_tokens_per_second"
            ),
            "end_to_end_tokens_per_second": timing.get(
                "end_to_end_generated_tokens_per_second"
            ),
            "timed_requests": timing.get("model_timed_requests"),
            "generated_tokens": timing.get("generated_tokens"),
            "summary_path": str(summary_path),
        })
    rows.sort(key=lambda row: (order.get(row["method"], len(order)), row["method"]))
    return rows


def display(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_outputs(result_root: Path, rows: List[Dict[str, Any]]) -> None:
    json_path = result_root / "comparison_summary.json"
    csv_path = result_root / "comparison_summary.csv"
    markdown_path = result_root / "comparison_summary.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0]) if rows else ["method"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Method | Accuracy | TTFT mean (s) | TPOT (s/token) | Decode token/s | Request mean (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {ttft} | {tpot} | {decode} | {request} |".format(
                method=row["method"],
                accuracy=display(row["accuracy"], 2),
                ttft=display(row["ttft_mean_seconds"]),
                tpot=display(row["tpot_seconds_per_token"]),
                decode=display(row["decode_tokens_per_second"], 2),
                request=display(row["request_mean_seconds"]),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"[SUMMARY] json={json_path}")
    print(f"[SUMMARY] csv={csv_path}")
    print(f"[SUMMARY] markdown={markdown_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.expanduser().resolve()
    rows = load_rows(result_root)
    if not rows:
        raise RuntimeError(f"No slake_summary.json files found under {result_root}")
    write_outputs(result_root, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
