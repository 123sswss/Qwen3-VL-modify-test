"""Summarize the four inference-only MMRL memory-collapse evaluations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


MODES = ("none", "text", "visual", "both")


def load_summary(root: Path, mode: str) -> dict[str, Any] | None:
    path = root / mode / "slake_summary.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_memory_collapse.py OUTPUT_ROOT")
    root = Path(sys.argv[1]).expanduser().resolve()
    rows = []
    baseline = None
    for mode in MODES:
        summary = load_summary(root, mode)
        if summary is None:
            rows.append({"mode": mode, "status": "missing"})
            continue
        accuracy = float(summary["overall_accuracy"])
        if mode == "none":
            baseline = accuracy
        timing = summary.get("timing", {})
        rows.append(
            {
                "mode": mode,
                "status": "ok",
                "accuracy": accuracy,
                "accuracy_delta": None,
                "ttft_mean_seconds": timing.get("ttft_seconds", {}).get("mean"),
                "tpot_seconds": timing.get("tpot_weighted_seconds"),
                "decode_tokens_per_second": timing.get("decode_tokens_per_second"),
                "request_mean_seconds": timing.get("request_seconds", {}).get("mean"),
            }
        )

    if baseline is not None:
        for row in rows:
            if row.get("status") == "ok":
                row["accuracy_delta"] = round(row["accuracy"] - baseline, 4)

    output = {
        "output_root": str(root),
        "baseline_accuracy": baseline,
        "results": rows,
    }
    json_path = root / "memory_collapse_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "# MMRL Memory Collapse Ablation",
        "",
        "| Mode | Accuracy | Delta | TTFT mean (s) | TPOT (s/token) | Decode token/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def format_number(value: Any, digits: int) -> str:
        if value is None:
            return "--"
        return f"{float(value):.{digits}f}"

    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| `{row['mode']}` | missing | -- | -- | -- | -- |")
            continue
        delta_text = (
            "--"
            if row["accuracy_delta"] is None
            else f"{row['accuracy_delta']:+.2f}"
        )
        lines.append(
            f"| `{row['mode']}` | {row['accuracy']:.2f} | "
            f"{delta_text} | "
            f"{format_number(row['ttft_mean_seconds'], 6)} | "
            f"{format_number(row['tpot_seconds'], 6)} | "
            f"{format_number(row['decode_tokens_per_second'], 3)} |"
        )
    markdown_path = root / "memory_collapse_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
