#!/usr/bin/env python3
"""Deterministically rank PathVQA images that have diverse question types."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def question_type(question: str) -> str:
    first = str(question).strip().lower().split(maxsplit=1)[0].rstrip("?,.:;")
    return first if first in {"how", "what", "when", "where", "why"} else "yes/no"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/root/autodl-tmp/dataset/pathVQA")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    from pathvqa.data_pipeline import PathVQAParquetStore
    from pathvqa.pathvqa_official_eval import image_fingerprint

    args = parse_args()
    if args.top < 1:
        raise ValueError("--top must be positive")
    store = PathVQAParquetStore(
        args.data_root,
        args.split,
        cache_dir=args.cache_dir,
    )
    grouped = defaultdict(list)
    for position, record in enumerate(store.samples, 1):
        image = store.load_image(record)
        try:
            fingerprint = image_fingerprint(image)
        finally:
            image.close()
        grouped[fingerprint].append(
            {
                "row_index": int(record["row_index"]),
                "question": str(record["question"]),
                "answer": str(record["answer"]),
                "question_type": question_type(record["question"]),
            }
        )
        if position % 500 == 0:
            print(f"[ATTENTION_CASE_SCAN] {position}/{len(store.samples)}")
    candidates = []
    for fingerprint, records in grouped.items():
        types = sorted({record["question_type"] for record in records})
        if len(records) < 2 or len(types) < 2:
            continue
        candidates.append(
            {
                "image_sha256": fingerprint,
                "question_count": len(records),
                "distinct_question_types": len(types),
                "question_types": types,
                "records": records,
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["distinct_question_types"],
            -row["question_count"],
            row["image_sha256"],
        )
    )
    output = {
        "selection_rule": (
            "descending distinct question-type count, then descending question "
            "count, then ascending image SHA-256"
        ),
        "dataset": f"PathVQA {args.split}",
        "candidates": candidates[: args.top],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
