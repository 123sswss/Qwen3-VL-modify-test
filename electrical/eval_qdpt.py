#!/usr/bin/env python3
"""Evaluate one QDPT checkpoint on the private electrical held-out sets."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from loraTest.data_protocol import (
    EVAL_IMAGE_DIRS,
    EVAL_JSON_PATHS,
    EVAL_MAX_NEW_TOKENS,
)
from slake.dynamic_prompt_tuning_interface import (
    DynamicPromptTuningModelInterface,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator() -> Any:
    source = PROJECT_ROOT / "test" / "test.py"
    spec = importlib.util.spec_from_file_location(
        "electrical_private_evaluator",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load electrical evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final QDPT on private electrical data"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/root/autodl-tmp/model"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--json-paths",
        type=Path,
        nargs="+",
        default=[Path(path) for path in EVAL_JSON_PATHS],
    )
    parser.add_argument(
        "--image-dirs",
        type=Path,
        nargs="+",
        default=[Path(path) for path in EVAL_IMAGE_DIRS],
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    evaluator = _load_evaluator()
    json_paths = [str(path) for path in args.json_paths]
    image_dirs = [str(path) for path in args.image_dirs]
    preflight = evaluator.validate_evaluation_data(json_paths, image_dirs)
    if preflight["valid"] == 0:
        raise RuntimeError("Electrical evaluation contains no valid samples")

    model = DynamicPromptTuningModelInterface(
        str(args.checkpoint),
        str(args.base_model),
    )
    summary = evaluator.run_evaluation(
        json_paths,
        model,
        image_dirs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    summary.update(
        {
            "dataset": "private_electrical",
            "checkpoint": str(args.checkpoint.resolve()),
            "base_model": str(args.base_model.resolve()),
            "preflight": preflight,
            "decoding": "greedy",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "electrical_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        "[ELECTRICAL_QDPT_RESULT] "
        f"score={summary['score']} evaluated={summary['evaluated']} "
        f"skipped={summary['skipped']} saved={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
