#!/usr/bin/env python3
"""Select the best PathVQA validation epoch without reading test results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_best_epoch(root: Path, epochs: int) -> tuple[int, float]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    rows = []
    for epoch in range(1, epochs + 1):
        summary_path = (
            root
            / "eval_validation"
            / f"epoch_{epoch}"
            / "pathvqa_summary.json"
        )
        with summary_path.open("r", encoding="utf-8") as handle:
            score = float(json.load(handle)["overall_accuracy"])
        rows.append((epoch, score))
    return max(rows, key=lambda item: (item[1], -item[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    epoch, score = select_best_epoch(args.root, args.epochs)
    print(f"{epoch}\t{score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
