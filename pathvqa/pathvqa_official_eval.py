#!/usr/bin/env python3
"""Run reproducible PathVQA inference and normalized exact-match evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathvqa.data_pipeline import PathVQAParquetStore
from pathvqa.model_interfaces import BACKEND_SPECS, load_pathvqa_model_interface
from pathvqa.pathvqa_vqa_metric import evaluate_pathvqa_predictions
from slake.slake_official_eval import (
    _normalized_timing,
    append_progress_row,
    extract_generated_answer,
    load_resume_rows,
    summarize_generation_timings,
    write_json,
)


DEFAULT_INSTRUCTION = "Answer with only a short answer."


def build_prompt(question: str, instruction: str | None) -> str:
    question = str(question).strip()
    suffix = DEFAULT_INSTRUCTION if instruction is None else instruction.strip()
    return question if not suffix else f"{question}\n{suffix}"


def image_fingerprint(image) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.mode).encode("ascii", errors="replace"))
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def run_timing_warmup(
    store: PathVQAParquetStore,
    records: Sequence[Mapping[str, Any]],
    model: Any,
    runs: int,
    max_new_tokens: int,
    temperature: float,
    instruction: str | None,
) -> None:
    if runs <= 0:
        return
    record = records[0]
    prompt = build_prompt(str(record["question"]), instruction)
    for index in range(1, runs + 1):
        image = store.load_image(dict(record))
        try:
            model.infer(
                image,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        finally:
            image.close()
        print(f"[PATHVQA_TIMING_WARMUP {index}/{runs}] complete")


def run_inference(
    store: PathVQAParquetStore,
    records: Sequence[Dict[str, Any]],
    model: Any,
    output_dir: Path,
    max_new_tokens: int,
    temperature: float,
    instruction: str | None,
    answer_mode: str,
    resume: bool,
    overwrite: bool,
    continue_on_error: bool,
) -> List[Dict[str, Any]]:
    progress_path = output_dir / "pathvqa_progress.jsonl"
    if progress_path.exists() and overwrite:
        progress_path.unlink()
    elif progress_path.exists() and not resume:
        raise FileExistsError(
            f"Progress file already exists: {progress_path}. "
            "Use --resume or --overwrite."
        )

    completed = load_resume_rows(progress_path) if resume else {}
    predictions = []
    started_at = time.time()
    for index, record in enumerate(records, start=1):
        question_id = str(record["question_id"])
        existing = completed.get(question_id)
        if existing is not None:
            predictions.append(existing)
            continue

        prompt = build_prompt(str(record["question"]), instruction)
        image = None
        try:
            image = store.load_image(record)
            image_id = image_fingerprint(image)
            if hasattr(model, "last_generation_timing"):
                model.last_generation_timing = None
            request_started_at = time.perf_counter()
            raw_output = model.infer(
                image,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            request_seconds = time.perf_counter() - request_started_at
            row = {
                "question_id": question_id,
                "image_id": image_id,
                "answer": extract_generated_answer(raw_output, answer_mode),
                "raw_output": raw_output,
                "question": record["question"],
                "status": "ok",
                "timing": _normalized_timing(
                    getattr(model, "last_generation_timing", None),
                    request_seconds,
                ),
            }
        except Exception as exc:
            if not continue_on_error:
                raise
            row = {
                "question_id": question_id,
                "image_id": question_id,
                "answer": "",
                "raw_output": "",
                "question": record["question"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if image is not None:
                image.close()

        predictions.append(row)
        append_progress_row(progress_path, row)
        elapsed = time.time() - started_at
        print(
            f"[PATHVQA_EVAL {index}/{len(records)}] "
            f"id={question_id} answer={row['answer']!r} "
            f"status={row['status']} elapsed={elapsed:.1f}s"
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base, MMRL, or PEFT Qwen3-VL models on PathVQA."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--backend", choices=sorted(BACKEND_SPECS), default="base")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--instruction")
    parser.add_argument("--no-instruction", action="store_true")
    parser.add_argument("--answer-mode", choices=("raw", "first-line"), default="raw")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timing-warmup-runs", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.timing_warmup_runs < 0:
        raise ValueError("--timing-warmup-runs must be non-negative")
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    store = PathVQAParquetStore(
        args.data_root,
        args.split,
        cache_dir=args.cache_dir,
    )
    records = list(store.samples)
    if args.limit is not None:
        records = records[: args.limit]
        print(f"[PATHVQA_PARTIAL_EVAL] limit={args.limit} selected={len(records)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_pathvqa_model_interface(
        args.backend,
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
    )
    instruction = "" if args.no_instruction else args.instruction
    run_timing_warmup(
        store,
        records,
        model,
        runs=args.timing_warmup_runs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        instruction=instruction,
    )
    prediction_rows = run_inference(
        store,
        records,
        model,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        instruction=instruction,
        answer_mode=args.answer_mode,
        resume=args.resume,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    summary, comparisons = evaluate_pathvqa_predictions(
        records,
        prediction_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    timing_summary = summarize_generation_timings(
        prediction_rows,
        warmup_runs=args.timing_warmup_runs,
    )
    summary.update(
        {
            "backend": args.backend,
            "base_model": args.base_model,
            "checkpoint": args.checkpoint,
            "data_root": str(store.data_root),
            "split": args.split,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "answer_mode": args.answer_mode,
            "instruction": (
                None
                if args.no_instruction
                else args.instruction or "short-answer"
            ),
            "partial_evaluation": args.limit is not None,
            "timing": timing_summary,
        }
    )

    official_predictions = [
        {"question_id": row["question_id"], "answer": row["answer"]}
        for row in prediction_rows
    ]
    write_json(output_dir / "pathvqa_predictions.json", official_predictions)
    write_json(output_dir / "pathvqa_details.json", prediction_rows)
    write_json(output_dir / "pathvqa_comparisons.json", comparisons)
    write_json(output_dir / "pathvqa_summary.json", summary)

    bootstrap = summary["clustered_bootstrap"]
    print("\n========== PathVQA Evaluation ==========")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}")
    print(f"Yes/No Accuracy: {summary['yes_no_accuracy']:.2f}")
    print(f"Free-form Accuracy: {summary['free_form_accuracy']:.2f}")
    print(f"Per Question Type: {summary['per_question_type_accuracy']}")
    print(
        "Image-clustered 95% CI: "
        f"[{bootstrap['lower']:.2f}, {bootstrap['upper']:.2f}] "
        f"clusters={bootstrap['image_clusters']}"
    )
    print(f"TTFT: {timing_summary['ttft_seconds']}")
    print(
        "TPOT: "
        f"{timing_summary['tpot_weighted_seconds']} s/token, "
        f"{timing_summary['decode_tokens_per_second']} token/s"
    )
    print(f"Request Latency: {timing_summary['request_seconds']}")
    print(f"Predictions: {output_dir / 'pathvqa_predictions.json'}")
    print(f"Summary: {output_dir / 'pathvqa_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
