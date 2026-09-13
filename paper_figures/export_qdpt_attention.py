#!/usr/bin/env python3
"""Export full QDPT visual attention for multiple questions sharing one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathvqa.data_pipeline import PathVQAParquetStore
from pathvqa.model_interfaces import load_pathvqa_model_interface
from pathvqa.pathvqa_official_eval import build_prompt, image_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="/root/autodl-tmp/model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/root/autodl-tmp/dataset/pathVQA")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument(
        "--row-index",
        action="append",
        type=int,
        required=True,
        help="Repeat for at least two records that use the same image.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--instruction", default="Answer with only a short answer.")
    return parser.parse_args()


def _record_by_row(store: PathVQAParquetStore, row_index: int) -> dict[str, Any]:
    matches = [row for row in store.samples if int(row["row_index"]) == row_index]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {store.split} record for row_index={row_index}; "
            f"found={len(matches)}"
        )
    return dict(matches[0])


def main() -> int:
    args = parse_args()
    if len(args.row_index) < 2:
        raise ValueError("At least two --row-index values are required")
    if len(set(args.row_index)) != len(args.row_index):
        raise ValueError("Repeated --row-index values are not allowed")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    store = PathVQAParquetStore(
        args.data_root,
        args.split,
        cache_dir=args.cache_dir,
    )
    records = [_record_by_row(store, index) for index in args.row_index]
    questions = [record["question"] for record in records]
    if len(set(questions)) != len(questions):
        raise ValueError("Attention comparison requires distinct questions")

    images = [store.load_image(record) for record in records]
    try:
        fingerprints = [image_fingerprint(image) for image in images]
        if len(set(fingerprints)) != 1:
            raise ValueError(
                "Selected PathVQA rows do not share the same decoded image: "
                f"rows={args.row_index}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        image_name = "source_image.png"
        images[0].save(args.output_dir / image_name)

        model = load_pathvqa_model_interface(
            "dynamic-prompt",
            base_model_path=args.base_model,
            checkpoint_path=args.checkpoint,
        )
        model.configure_attention_capture(True)
        entries = []
        for item_index, (record, image) in enumerate(zip(records, images), 1):
            if hasattr(model, "reset_inference_state"):
                model.reset_inference_state()
            model.clear_attention_captures()
            prediction = model.infer(
                image,
                build_prompt(record["question"], args.instruction),
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            captures = model.attention_captures()
            if len(captures) != 1:
                raise RuntimeError(
                    "Expected one attention capture for one image; "
                    f"found={len(captures)}"
                )
            capture = captures[0]
            archive_name = f"question_{item_index:02d}_attention.npz"
            arrays = {
                key: value.numpy()
                for key, value in capture.items()
                if hasattr(value, "numpy")
            }
            np.savez_compressed(args.output_dir / archive_name, **arrays)
            entries.append(
                {
                    "row_index": int(record["row_index"]),
                    "question_id": str(record["question_id"]),
                    "question": str(record["question"]),
                    "reference_answer": str(record["answer"]),
                    "prediction": prediction,
                    "attention": archive_name,
                    "cross_attention_shape": list(arrays["cross_attention"].shape),
                    "grid_thw": arrays["grid_thw"].astype(int).tolist(),
                }
            )
            print(
                "[QDPT_ATTENTION_EXPORT] "
                f"row={record['row_index']} question={record['question']!r} "
                f"prediction={prediction!r} shape={arrays['cross_attention'].shape} "
                f"grid={arrays['grid_thw'].tolist()}"
            )
    finally:
        for image in images:
            image.close()

    manifest = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "base_model": str(args.base_model),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "split": args.split,
        "image": image_name,
        "image_sha256": fingerprints[0],
        "aggregation": {
            "all_queries": "mean over 16 heads, then mean over 10 queries",
            "focused_queries": "three queries with the lowest normalized visual-attention entropy",
            "temporal": "mean over temporal grid dimension before spatial rendering",
        },
        "entries": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
