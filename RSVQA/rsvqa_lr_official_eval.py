#!/usr/bin/env python3
"""Full-split generative inference and official-style evaluation for RSVQA-LR."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .data import load_rsvqa_lr_split
from .metric import evaluate_rsvqa_predictions
from .model_interfaces import BACKEND_SPECS, load_rsvqa_model_interface
from .prompts import build_prompt


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def extract_answer(raw_output: Any) -> str:
    text = str(raw_output if raw_output is not None else "").strip()
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _load_progress(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["question_id"])
            if key in rows:
                raise ValueError(f"Duplicate progress question_id={key} at line {line_number}")
            rows[key] = row
    return rows


def _append_progress(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def run_inference(
    records: Sequence[Mapping[str, Any]], model: Any, output_dir: Path,
    max_new_tokens: int, resume: bool, overwrite: bool,
) -> list[Dict[str, Any]]:
    from PIL import Image

    progress_path = output_dir / "rsvqa_progress.jsonl"
    if progress_path.exists() and overwrite:
        progress_path.unlink()
    elif progress_path.exists() and not resume:
        raise FileExistsError(f"Progress already exists: {progress_path}; use --resume or --overwrite")
    completed = _load_progress(progress_path) if resume else {}
    predictions: list[Dict[str, Any]] = []
    started = time.time()
    for index, record in enumerate(records, start=1):
        key = str(record["question_id"])
        if key in completed:
            predictions.append(completed[key])
            continue
        prompt = build_prompt(record)
        with Image.open(record["image_path"]) as image_file:
            image = image_file.convert("RGB")
        if hasattr(model, "last_generation_timing"):
            model.last_generation_timing = None
        request_started = time.perf_counter()
        raw_question_kwargs = (
            {"question": str(record["question"])}
            if getattr(model, "requires_raw_question", False)
            else {}
        )
        raw_output = model.infer(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            **raw_question_kwargs,
        )
        request_seconds = time.perf_counter() - request_started
        row = {
            "question_id": record["question_id"],
            "image_id": record["image_id"],
            "question_type": record["question_type"],
            "question": record["question"],
            "answer": extract_answer(raw_output),
            "raw_output": str(raw_output),
            "prompt": prompt,
            "status": "ok",
            "request_seconds": request_seconds,
        }
        timing = getattr(model, "last_generation_timing", None)
        if isinstance(timing, Mapping):
            row["model_timing"] = dict(timing)
        predictions.append(row)
        _append_progress(progress_path, row)
        print(
            f"[RSVQA_EVAL {index}/{len(records)}] id={key} "
            f"type={record['question_type']} answer={row['answer']!r} "
            f"elapsed={time.time() - started:.1f}s"
        )
    return predictions


def _timing_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(row["request_seconds"]) for row in rows if row.get("request_seconds") is not None]
    values.sort()
    if not values:
        return {"count": 0}
    def percentile(q: float) -> float:
        return values[round((len(values) - 1) * q)]
    return {
        "count": len(values), "mean": round(sum(values) / len(values), 6),
        "p50": round(percentile(0.5), 6), "p95": round(percentile(0.95), 6),
        "min": round(values[0], 6), "max": round(values[-1], 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-VL backend on official RSVQA-LR splits")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--backend", choices=sorted(BACKEND_SPECS), default="base")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-enforce-official-counts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.max_new_tokens < 1 or args.bootstrap_iterations < 0:
        raise ValueError("Generation length must be positive and bootstrap iterations non-negative")
    records, manifest = load_rsvqa_lr_split(
        args.data_root, args.split,
        enforce_official_counts=not args.no_enforce_official_counts,
    )
    print(
        "[RSVQA_PREFLIGHT] "
        f"split={args.split} questions={len(records)} images={manifest['active_images']} "
        f"types={manifest['question_type_counts']}"
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
        print(f"[RSVQA_PARTIAL_EVAL] selected={len(records)} official_result=false")
    if args.audit_only:
        print("[RSVQA_AUDIT_ONLY] data audit passed; model was not loaded")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_rsvqa_model_interface(
        args.backend, base_model_path=args.base_model, checkpoint_path=args.checkpoint
    )
    prediction_rows = run_inference(
        records, model, output_dir, args.max_new_tokens, args.resume, args.overwrite
    )
    official_predictions = [
        {"question_id": row["question_id"], "answer": row["answer"]}
        for row in prediction_rows
    ]
    summary, comparisons = evaluate_rsvqa_predictions(
        records, official_predictions,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "backend": args.backend, "base_model": args.base_model,
            "checkpoint": args.checkpoint, "split": args.split,
            "partial_evaluation": args.limit is not None,
            "max_new_tokens": args.max_new_tokens,
            "prompt_policy": "question_type_specific_short_answer_v1",
            "timing": _timing_summary(prediction_rows),
            "data_manifest": manifest,
        }
    )
    write_json(output_dir / "rsvqa_predictions.json", prediction_rows)
    write_json(output_dir / "rsvqa_comparisons.json", comparisons)
    write_json(output_dir / "rsvqa_summary.json", summary)
    print("\n========== RSVQA-LR Evaluation ==========")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Average Accuracy: {summary['average_accuracy']:.4f}")
    print(f"Per Question Type: {summary['per_question_type_accuracy']}")
    ci = summary["image_clustered_95_ci"]
    print(f"Image-clustered 95% CI: [{ci['lower']}, {ci['upper']}] clusters={ci['clusters']}")
    print(f"Predictions: {output_dir / 'rsvqa_predictions.json'}")
    print(f"Summary: {output_dir / 'rsvqa_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
