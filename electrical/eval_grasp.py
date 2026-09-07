#!/usr/bin/env python3
"""Evaluate GRASP on the existing private electrical holdout protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from loraTest.data_protocol import EVAL_IMAGE_DIRS, EVAL_JSON_PATHS, EVAL_MAX_NEW_TOKENS
from slake.grasp_prompt_tuning_interface import GRASPModelInterface


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GRASP on private electrical data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    args = parser.parse_args()
    source = PROJECT_ROOT / "test" / "test.py"
    spec = importlib.util.spec_from_file_location("electrical_grasp_evaluator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load electrical evaluator: {source}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    json_paths = list(EVAL_JSON_PATHS)
    image_dirs = list(EVAL_IMAGE_DIRS)
    preflight = evaluator.validate_evaluation_data(json_paths, image_dirs)
    if preflight["valid"] == 0:
        raise RuntimeError("Electrical evaluation contains no valid samples")
    model = GRASPModelInterface(str(args.checkpoint), str(args.base_model))
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
            "method": "grasp_reimplementation",
            "checkpoint": str(args.checkpoint.resolve()),
            "preflight": preflight,
            "decoding": "greedy",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "electrical_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        "[ELECTRICAL_GRASP_RESULT] "
        f"score={summary['score']} evaluated={summary['evaluated']} "
        f"skipped={summary['skipped']} saved={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
