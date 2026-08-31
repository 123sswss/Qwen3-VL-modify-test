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
from slake.directional_concat_workspace import (
    DIRECTIONAL_QUESTION_QUERY_INTERVENTIONS,
    DIRECTIONAL_VISUAL_MEMORY_INTERVENTIONS,
)
from slake.dynamic_prompt_tuning import DYNAMIC_PROMPT_INTERVENTIONS
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


def _distinct_priming_record(
    store: PathVQAParquetStore,
    first_record: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str]:
    first_image = store.load_image(dict(first_record))
    try:
        first_image_id = image_fingerprint(first_image)
    finally:
        first_image.close()
    first_question = str(first_record["question"]).strip()

    for candidate in reversed(store.samples):
        if str(candidate["question"]).strip() == first_question:
            continue
        candidate_image = store.load_image(dict(candidate))
        try:
            candidate_image_id = image_fingerprint(candidate_image)
        finally:
            candidate_image.close()
        if candidate_image_id != first_image_id:
            return candidate, first_image_id, candidate_image_id
    raise RuntimeError(
        "Could not find a PathVQA priming record with both a distinct question "
        "and a distinct image"
    )


def prime_directional_intervention(
    store: PathVQAParquetStore,
    records: Sequence[Mapping[str, Any]],
    model: Any,
    max_new_tokens: int,
    temperature: float,
    instruction: str | None,
    question_query_mode: str,
    visual_memory_mode: str,
) -> Dict[str, Any] | None:
    if question_query_mode == "normal" and visual_memory_mode == "normal":
        return None
    if not records:
        raise ValueError("Directional intervention priming requires scored records")

    record, first_image_id, priming_image_id = _distinct_priming_record(
        store,
        records[0],
    )
    prompt = build_prompt(str(record["question"]), instruction)
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
    if hasattr(model, "reset_inference_measurements"):
        model.reset_inference_measurements()

    metadata = {
        "excluded_from_scoring": True,
        "question_id": str(record["question_id"]),
        "question_distinct_from_first_scored": (
            str(record["question"]).strip()
            != str(records[0]["question"]).strip()
        ),
        "image_distinct_from_first_scored": priming_image_id != first_image_id,
        "question_query_mode": question_query_mode,
        "visual_memory_mode": visual_memory_mode,
    }
    if not metadata["question_distinct_from_first_scored"]:
        raise RuntimeError("Directional priming question is not distinct")
    if not metadata["image_distinct_from_first_scored"]:
        raise RuntimeError("Directional priming image is not distinct")
    print(
        "[PATHVQA_DIRECTIONAL_INTERVENTION_PRIME] "
        f"question_id={metadata['question_id']} "
        "excluded_from_scoring=True question_distinct=True image_distinct=True"
    )
    return metadata


def audit_directional_intervention(
    intervention: Mapping[str, Any],
    scored_count: int,
    priming: Mapping[str, Any] | None,
    question_query_mode: str,
    visual_memory_mode: str,
) -> Dict[str, Any] | None:
    if question_query_mode == "normal" and visual_memory_mode == "normal":
        return None
    workspace = intervention.get("workspace")
    if not isinstance(workspace, Mapping):
        raise RuntimeError("Directional intervention summary is unavailable")
    if priming is None:
        raise RuntimeError("Directional intervention was not primed")
    for key in (
        "excluded_from_scoring",
        "question_distinct_from_first_scored",
        "image_distinct_from_first_scored",
    ):
        if priming.get(key) is not True:
            raise RuntimeError(
                f"Directional intervention priming audit failed: {key}="
                f"{priming.get(key)!r}"
            )

    common_expected = {
        "original_image_input_unchanged": True,
        "original_question_input_unchanged": True,
        "anchors_preserved": True,
        "token_count_preserved": True,
    }
    for key, expected in common_expected.items():
        if workspace.get(key) is not expected:
            raise RuntimeError(
                f"Directional intervention audit failed: {key}={workspace.get(key)!r}"
            )
    if workspace.get("question_query_mode") != question_query_mode:
        raise RuntimeError("Directional question-Q mode propagation audit failed")
    if workspace.get("visual_memory_mode") != visual_memory_mode:
        raise RuntimeError("Directional visual-K/V mode propagation audit failed")

    audit: Dict[str, Any] = {
        **common_expected,
        "scored_samples": int(scored_count),
        "excluded_priming_samples": 1,
        "question_query_mode": question_query_mode,
        "visual_memory_mode": visual_memory_mode,
        "mode_propagation_audit_pass": True,
    }
    if question_query_mode != "normal":
        expected_seen = scored_count + 1
        actual = {
            "seen": int(workspace.get("question_query_units_seen", -1)),
            "mismatched": int(
                workspace.get("question_query_units_mismatched", -1)
            ),
            "natural": int(workspace.get("question_query_units_natural", -1)),
        }
        expected = {"seen": expected_seen, "mismatched": scored_count, "natural": 1}
        if actual != expected:
            raise RuntimeError(
                "Directional question-Q mismatch audit failed: "
                f"expected={expected} actual={actual}"
            )
        audit["question_query_counts"] = actual
        audit["question_query_mismatch_audit_pass"] = True
    if visual_memory_mode != "normal":
        expected_seen = scored_count + 1
        actual = {
            "seen": int(workspace.get("visual_memory_images_seen", -1)),
            "mismatched": int(
                workspace.get("visual_memory_images_mismatched", -1)
            ),
            "natural": int(workspace.get("visual_memory_images_natural", -1)),
        }
        expected = {"seen": expected_seen, "mismatched": scored_count, "natural": 1}
        if actual != expected:
            raise RuntimeError(
                "Directional visual-K/V mismatch audit failed: "
                f"expected={expected} actual={actual}"
            )
        audit["visual_memory_counts"] = actual
        audit["visual_memory_mismatch_audit_pass"] = True
    print(
        "[PATHVQA_DIRECTIONAL_INTERVENTION_AUDIT_PASS] "
        f"scored={scored_count} excluded_priming=1 "
        f"question_query_mode={question_query_mode} "
        f"visual_memory_mode={visual_memory_mode}"
    )
    return audit


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
    parser.add_argument(
        "--dynamic-prompt-intervention",
        choices=DYNAMIC_PROMPT_INTERVENTIONS,
        default="normal",
    )
    parser.add_argument("--dynamic-prompt-memory-lag", type=int, default=32)
    parser.add_argument(
        "--directional-question-query-mode",
        choices=DIRECTIONAL_QUESTION_QUERY_INTERVENTIONS,
        default="normal",
    )
    parser.add_argument(
        "--directional-visual-memory-mode",
        choices=DIRECTIONAL_VISUAL_MEMORY_INTERVENTIONS,
        default="normal",
    )
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
    if args.dynamic_prompt_memory_lag < 1:
        raise ValueError("--dynamic-prompt-memory-lag must be positive")
    dynamic_intervention_requested = (
        args.dynamic_prompt_intervention != "normal"
        or args.directional_question_query_mode != "normal"
        or args.directional_visual_memory_mode != "normal"
    )
    if args.backend != "dynamic-prompt" and dynamic_intervention_requested:
        raise ValueError(
            "Dynamic Prompt interventions require --backend dynamic-prompt"
        )
    stateful_intervention_requested = (
        args.dynamic_prompt_intervention in {"mean-residual", "lagged-memory"}
        or args.directional_question_query_mode != "normal"
        or args.directional_visual_memory_mode != "normal"
    )
    if args.resume and stateful_intervention_requested:
        raise ValueError(
            "Stateful Dynamic Prompt interventions do not support --resume; "
            "rerun with --overwrite"
        )
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
    interface_kwargs = (
        {
            "intervention": args.dynamic_prompt_intervention,
            "memory_lag": args.dynamic_prompt_memory_lag,
            "directional_question_query_mode": (
                args.directional_question_query_mode
            ),
            "directional_visual_memory_mode": (
                args.directional_visual_memory_mode
            ),
        }
        if args.backend == "dynamic-prompt"
        else None
    )
    model = load_pathvqa_model_interface(
        args.backend,
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        interface_kwargs=interface_kwargs,
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
    if hasattr(model, "reset_inference_state"):
        model.reset_inference_state()
    priming = prime_directional_intervention(
        store,
        records,
        model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        instruction=instruction,
        question_query_mode=args.directional_question_query_mode,
        visual_memory_mode=args.directional_visual_memory_mode,
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
    if hasattr(model, "inference_intervention_summary"):
        intervention_summary = model.inference_intervention_summary()
        summary["dynamic_prompt_intervention"] = intervention_summary
        directional_audit = audit_directional_intervention(
            intervention_summary,
            scored_count=len(prediction_rows),
            priming=priming,
            question_query_mode=args.directional_question_query_mode,
            visual_memory_mode=args.directional_visual_memory_mode,
        )
        if directional_audit is not None:
            summary["directional_conditioning_audit"] = directional_audit
            summary["directional_conditioning_priming"] = priming

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
