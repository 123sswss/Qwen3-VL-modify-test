#!/usr/bin/env python3
"""Run reproducible SLAKE inference and normalized exact-match evaluation.

This script is intentionally isolated from the project's existing evaluators.
It consumes an official SLAKE split JSON, calls an existing project ``infer``
interface, writes standard ``question_id``/``answer`` predictions, and reports
the VQA-style metric commonly used by SLAKE implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from slake_model_interfaces import BACKEND_SPECS, load_slake_model_interface
from slake_vqa_metric import evaluate_slake_predictions, record_id


DEFAULT_ENGLISH_INSTRUCTION = "Answer with only a short answer."
DEFAULT_CHINESE_INSTRUCTION = "只输出简短答案，不要解释。"
QUESTION_KEYS = ("question", "query", "text")
IMAGE_KEYS = ("img_name", "image_name", "image", "img")
LANGUAGE_KEYS = ("q_lang", "language", "lang")
SPLIT_KEYS = ("split", "subset", "set")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def unwrap_question_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("questions", "annotations", "data", "records"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            raise ValueError(
                "SLAKE question JSON must be a list or contain a list under "
                "questions/annotations/data/records"
            )
    else:
        raise ValueError(f"Unsupported SLAKE question JSON type: {type(payload).__name__}")

    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"SLAKE record index {index} is not an object")
        normalized.append(dict(record))
    return normalized


def first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalized_language(record: Mapping[str, Any]) -> str:
    language = first_text(record, LANGUAGE_KEYS).lower()
    aliases = {
        "english": "en",
        "eng": "en",
        "chinese": "zh",
        "cn": "zh",
        "zh-cn": "zh",
    }
    return aliases.get(language, language)


def normalized_split(record: Mapping[str, Any]) -> str:
    return first_text(record, SPLIT_KEYS).lower()


def select_records(
    records: Sequence[Dict[str, Any]],
    language: str,
    base_types: Sequence[str],
    expected_split: str | None,
) -> List[Dict[str, Any]]:
    selected = []
    allowed_base_types = {item.strip().lower() for item in base_types if item.strip()}
    seen_ids = set()

    for index, record in enumerate(records):
        qid = record_id(record, fallback=index)
        qid_key = str(qid)
        if qid_key in seen_ids:
            raise ValueError(f"Duplicate SLAKE question ID: {qid_key}")
        seen_ids.add(qid_key)
        record.setdefault("question_id", qid)

        question = first_text(record, QUESTION_KEYS)
        answer = record.get("answer", record.get("answers"))
        image_name = first_text(record, IMAGE_KEYS)
        if not question or answer is None or not image_name:
            raise ValueError(
                f"Incomplete SLAKE record question_id={qid!r}: "
                f"question={bool(question)} answer={answer is not None} "
                f"image={bool(image_name)}"
            )

        record_language = normalized_language(record)
        if language != "all" and record_language and record_language != language:
            continue

        base_type = str(record.get("base_type", "")).strip().lower()
        if allowed_base_types and base_type not in allowed_base_types:
            continue

        record_split = normalized_split(record)
        if expected_split and record_split and record_split != expected_split:
            continue

        record["_slake_question"] = question
        record["_slake_image_name"] = image_name
        record["_slake_language"] = record_language or language
        selected.append(record)

    if not selected:
        raise ValueError(
            "No SLAKE records remain after filtering. Check --language, "
            "--base-type, and --expected-split."
        )
    return selected


class ImageResolver:
    def __init__(self, image_root: Path):
        self.image_root = image_root.expanduser().resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"SLAKE image root not found: {self.image_root}")
        self._basename_map: Dict[str, List[Path]] | None = None

    def _build_basename_map(self) -> Dict[str, List[Path]]:
        if self._basename_map is None:
            mapping: Dict[str, List[Path]] = {}
            for path in self.image_root.rglob("*"):
                if path.is_file():
                    mapping.setdefault(path.name, []).append(path.resolve())
            self._basename_map = mapping
        return self._basename_map

    def resolve(self, raw_image_name: str) -> Path | None:
        normalized_name = raw_image_name.replace("\\", "/").lstrip("/")
        relative_candidates = [normalized_name]
        if normalized_name.startswith("imgs/"):
            relative_candidates.append(normalized_name[len("imgs/"):])

        for relative_name in relative_candidates:
            candidate = (self.image_root / Path(relative_name)).resolve()
            try:
                if os.path.commonpath((self.image_root, candidate)) != str(self.image_root):
                    continue
            except ValueError:
                continue
            if candidate.is_file():
                return candidate

        basename_matches = self._build_basename_map().get(Path(normalized_name).name, [])
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise ValueError(
                f"Ambiguous SLAKE image basename {Path(normalized_name).name!r}: "
                f"{[str(path) for path in basename_matches[:5]]}"
            )
        return None


def audit_samples(
    records: Sequence[Dict[str, Any]],
    resolver: ImageResolver,
    allow_missing_images: bool,
) -> List[Dict[str, Any]]:
    audited = []
    missing_images = []
    for record in records:
        image_path = resolver.resolve(record["_slake_image_name"])
        if image_path is None:
            missing_images.append(
                {
                    "question_id": record["question_id"],
                    "image": record["_slake_image_name"],
                }
            )
            if allow_missing_images:
                continue
        else:
            record["_slake_image_path"] = str(image_path)
            audited.append(record)

    if missing_images and not allow_missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} SLAKE images are missing; examples={missing_images[:5]}"
        )

    language_counts = Counter(record["_slake_language"] or "unknown" for record in audited)
    answer_type_counts = Counter(
        str(record.get("answer_type") or "unknown")
        for record in audited
    )
    question_type_counts = Counter(
        str(record.get("question_type", record.get("base_type")) or "unknown")
        for record in audited
    )
    print(
        "[SLAKE_PREFLIGHT] "
        f"selected={len(records)} valid={len(audited)} missing={len(missing_images)} "
        f"languages={dict(sorted(language_counts.items()))} "
        f"answer_types={dict(sorted(answer_type_counts.items()))} "
        f"question_types={dict(sorted(question_type_counts.items()))}"
    )
    return audited


def build_prompt(record: Mapping[str, Any], instruction: str | None) -> str:
    question = str(record["_slake_question"]).strip()
    if instruction is not None:
        suffix = instruction.strip()
    elif record.get("_slake_language") == "zh":
        suffix = DEFAULT_CHINESE_INSTRUCTION
    else:
        suffix = DEFAULT_ENGLISH_INSTRUCTION
    return question if not suffix else f"{question}\n{suffix}"


def extract_generated_answer(raw_output: str, mode: str) -> str:
    output = str(raw_output).strip()
    if mode == "raw":
        return output
    if mode == "first-line":
        for line in output.splitlines():
            if line.strip():
                return line.strip()
        return ""
    raise ValueError(f"Unsupported answer extraction mode: {mode}")


def load_resume_rows(progress_path: Path) -> Dict[str, Dict[str, Any]]:
    completed = {}
    if not progress_path.is_file():
        return completed
    with progress_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(record_id(row))
            if qid in completed:
                raise ValueError(
                    f"Duplicate question ID in {progress_path} line {line_number}: {qid}"
                )
            completed[qid] = row
    return completed


def append_progress_row(progress_path: Path, row: Mapping[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_distribution(values: Sequence[float]) -> Dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": round(sum(finite) / len(finite), 6),
        "p50": round(_percentile(finite, 0.50), 6),
        "p95": round(_percentile(finite, 0.95), 6),
        "min": round(min(finite), 6),
        "max": round(max(finite), 6),
    }


def _normalized_timing(raw_timing: Any, request_total_seconds: float) -> Dict[str, Any]:
    timing = {"request_total_seconds": float(request_total_seconds)}
    if not isinstance(raw_timing, Mapping):
        return timing
    for key in (
        "first_token_latency_seconds",
        "subsequent_token_mean_seconds",
        "generation_total_seconds",
        "generated_token_count",
        "measured_token_steps",
        "timing_method",
    ):
        value = raw_timing.get(key)
        if value is not None:
            timing[key] = value
    return timing


def summarize_generation_timings(
    prediction_rows: Sequence[Mapping[str, Any]],
    warmup_runs: int,
) -> Dict[str, Any]:
    timings = [
        row["timing"]
        for row in prediction_rows
        if row.get("status") == "ok" and isinstance(row.get("timing"), Mapping)
    ]
    request_times = [
        float(timing["request_total_seconds"])
        for timing in timings
        if timing.get("request_total_seconds") is not None
    ]
    model_timings = [
        timing
        for timing in timings
        if timing.get("generated_token_count") is not None
        and timing.get("generation_total_seconds") is not None
    ]
    first_token_times = [
        float(timing["first_token_latency_seconds"])
        for timing in model_timings
        if timing.get("first_token_latency_seconds") is not None
    ]
    per_request_tpot = [
        float(timing["subsequent_token_mean_seconds"])
        for timing in model_timings
        if timing.get("subsequent_token_mean_seconds") is not None
    ]

    subsequent_token_count = 0
    subsequent_seconds = 0.0
    generated_token_count = 0
    generation_seconds = 0.0
    request_seconds_for_timed_generations = 0.0
    methods = Counter()
    for timing in model_timings:
        generated_tokens = max(int(timing["generated_token_count"]), 0)
        generated_token_count += generated_tokens
        generation_seconds += float(timing["generation_total_seconds"])
        if timing.get("request_total_seconds") is not None:
            request_seconds_for_timed_generations += float(
                timing["request_total_seconds"]
            )
        method = timing.get("timing_method")
        if method:
            methods[str(method)] += 1
        interval_count = max(int(timing.get("measured_token_steps", 0)) - 1, 0)
        interval_mean = timing.get("subsequent_token_mean_seconds")
        if interval_count and interval_mean is not None:
            subsequent_token_count += interval_count
            subsequent_seconds += float(interval_mean) * interval_count

    weighted_tpot = (
        subsequent_seconds / subsequent_token_count
        if subsequent_token_count > 0
        else None
    )
    return {
        "methodology": {
            "warmup_runs_excluded": warmup_runs,
            "ttft": "generate start to first generated-token logits ready",
            "tpot": "token-count-weighted interval between later token logits",
            "request": "model interface call including preprocessing and decoding",
            "timing_methods": dict(sorted(methods.items())),
        },
        "successful_requests": len(timings),
        "model_timed_requests": len(model_timings),
        "generated_tokens": generated_token_count,
        "subsequent_tokens": subsequent_token_count,
        "ttft_seconds": _timing_distribution(first_token_times),
        "tpot_per_request_seconds": _timing_distribution(per_request_tpot),
        "tpot_weighted_seconds": (
            round(weighted_tpot, 6) if weighted_tpot is not None else None
        ),
        "decode_tokens_per_second": (
            round(1.0 / weighted_tpot, 3)
            if weighted_tpot is not None and weighted_tpot > 0
            else None
        ),
        "generation_seconds": _timing_distribution([
            float(timing["generation_total_seconds"])
            for timing in model_timings
        ]),
        "request_seconds": _timing_distribution(request_times),
        "model_generated_tokens_per_second": (
            round(generated_token_count / generation_seconds, 3)
            if generation_seconds > 0
            else None
        ),
        "end_to_end_generated_tokens_per_second": (
            round(generated_token_count / request_seconds_for_timed_generations, 3)
            if request_seconds_for_timed_generations > 0
            else None
        ),
    }


def run_timing_warmup(
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
    prompt = build_prompt(record, instruction)
    for index in range(1, runs + 1):
        with Image.open(record["_slake_image_path"]) as image_file:
            image = image_file.convert("RGB")
        model.infer(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(f"[SLAKE_TIMING_WARMUP {index}/{runs}] complete")


def run_inference(
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
    progress_path = output_dir / "slake_progress.jsonl"
    if progress_path.exists() and overwrite:
        progress_path.unlink()
    elif progress_path.exists() and not resume:
        raise FileExistsError(
            f"Progress file already exists: {progress_path}. "
            "Use --resume or --overwrite."
        )

    completed = load_resume_rows(progress_path) if resume else {}
    predictions = []
    total = len(records)
    start_time = time.time()

    for index, record in enumerate(records, start=1):
        qid = record["question_id"]
        existing = completed.get(str(qid))
        if existing is not None:
            predictions.append(existing)
            continue

        prompt = build_prompt(record, instruction)
        try:
            with Image.open(record["_slake_image_path"]) as image_file:
                image = image_file.convert("RGB")
            if hasattr(model, "last_generation_timing"):
                model.last_generation_timing = None
            request_started_at = time.perf_counter()
            raw_output = model.infer(
                image,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            request_total_seconds = time.perf_counter() - request_started_at
            answer = extract_generated_answer(raw_output, answer_mode)
            row = {
                "question_id": qid,
                "answer": answer,
                "raw_output": raw_output,
                "question": record["_slake_question"],
                "image": record["_slake_image_name"],
                "status": "ok",
                "timing": _normalized_timing(
                    getattr(model, "last_generation_timing", None),
                    request_total_seconds,
                ),
            }
        except Exception as exc:
            if not continue_on_error:
                raise
            row = {
                "question_id": qid,
                "answer": "",
                "raw_output": "",
                "question": record["_slake_question"],
                "image": record["_slake_image_name"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        predictions.append(row)
        append_progress_row(progress_path, row)
        elapsed = time.time() - start_time
        print(
            f"[SLAKE_EVAL {index}/{total}] "
            f"id={qid} answer={row['answer']!r} status={row['status']} "
            f"elapsed={elapsed:.1f}s"
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base, MMRL, or PEFT Qwen3-VL models on SLAKE."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--backend", choices=sorted(BACKEND_SPECS), default="base")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--workspace-disable-visual-write-layer",
        action="append",
        type=int,
        default=[],
        help="Inference-only: zero the Workspace visual residual at this 0-based anchor.",
    )
    parser.add_argument(
        "--workspace-bypass-update-layer",
        action="append",
        type=int,
        default=[],
        help="Inference-only: bypass the Workspace CA/SA/FFN update at this anchor.",
    )
    parser.add_argument(
        "--workspace-disable-visual-rep-write-layer",
        action="append",
        type=int,
        default=[],
        help=(
            "Inference-only: compute Workspace and Rep diagnostics but execute "
            "this visual block without inserting any Rep tokens."
        ),
    )
    parser.add_argument(
        "--workspace-disable-text-write",
        action="store_true",
        help="Inference-only: compute but zero the Workspace Text residual.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "zh", "all"), default="en")
    parser.add_argument("--base-type", action="append", default=[])
    parser.add_argument("--expected-split", default="test")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--instruction",
        help="Custom answer instruction. Defaults to a language-aware short-answer prompt.",
    )
    parser.add_argument(
        "--no-instruction",
        action="store_true",
        help="Send the original SLAKE question without an answer-format instruction.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=("raw", "first-line"),
        default="raw",
        help="Raw is the strict, publication-comparable default.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--timing-warmup-runs",
        type=int,
        default=3,
        help="Warmup inference requests excluded from accuracy and timing summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.timing_warmup_runs < 0:
        raise ValueError("--timing-warmup-runs must be non-negative")
    workspace_intervention_requested = bool(
        args.workspace_disable_visual_write_layer
        or args.workspace_disable_visual_rep_write_layer
        or args.workspace_bypass_update_layer
        or args.workspace_disable_text_write
    )
    if workspace_intervention_requested and args.backend != "dynamic-prompt":
        raise ValueError(
            "Workspace interventions require --backend dynamic-prompt"
        )

    question_path = args.questions.expanduser().resolve()
    if not question_path.is_file():
        raise FileNotFoundError(f"SLAKE question file not found: {question_path}")

    all_records = unwrap_question_records(load_json(question_path))
    selected_records = select_records(
        all_records,
        language=args.language,
        base_types=args.base_type,
        expected_split=args.expected_split.strip().lower() or None,
    )
    resolver = ImageResolver(args.image_root)
    records = audit_samples(
        selected_records,
        resolver,
        allow_missing_images=args.allow_missing_images,
    )
    if args.limit is not None:
        records = records[: args.limit]
        print(f"[SLAKE_PARTIAL_EVAL] limit={args.limit} selected={len(records)}")

    if args.audit_only:
        print("[SLAKE_AUDIT_ONLY] dataset validation passed; model was not loaded")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    interface_kwargs = (
        {
            "workspace_visual_write_disabled_layers": tuple(
                args.workspace_disable_visual_write_layer
            ),
            "workspace_visual_rep_write_disabled_layers": tuple(
                args.workspace_disable_visual_rep_write_layer
            ),
            "workspace_update_bypassed_layers": tuple(
                args.workspace_bypass_update_layer
            ),
            "workspace_text_write_disabled": args.workspace_disable_text_write,
        }
        if args.backend == "dynamic-prompt"
        else None
    )
    model = load_slake_model_interface(
        args.backend,
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        interface_kwargs=interface_kwargs,
    )
    instruction = "" if args.no_instruction else args.instruction
    run_timing_warmup(
        records,
        model,
        runs=args.timing_warmup_runs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        instruction=instruction,
    )
    prediction_rows = run_inference(
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

    official_predictions = [
        {"question_id": row["question_id"], "answer": row["answer"]}
        for row in prediction_rows
    ]
    summary, comparisons = evaluate_slake_predictions(records, official_predictions)
    timing_summary = summarize_generation_timings(
        prediction_rows,
        warmup_runs=args.timing_warmup_runs,
    )
    summary.update(
        {
            "backend": args.backend,
            "base_model": args.base_model,
            "checkpoint": args.checkpoint,
            "questions": str(question_path),
            "image_root": str(resolver.image_root),
            "language": args.language,
            "base_types": list(args.base_type),
            "expected_split": args.expected_split,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "answer_mode": args.answer_mode,
            "instruction": (
                None
                if args.no_instruction
                else args.instruction or "language-aware-short-answer"
            ),
            "partial_evaluation": args.limit is not None,
            "timing": timing_summary,
        }
    )
    if hasattr(model, "inference_intervention_summary"):
        summary["dynamic_prompt_intervention"] = (
            model.inference_intervention_summary()
        )

    write_json(output_dir / "slake_predictions.json", official_predictions)
    write_json(output_dir / "slake_details.json", prediction_rows)
    write_json(output_dir / "slake_comparisons.json", comparisons)
    write_json(output_dir / "slake_summary.json", summary)

    print("\n========== SLAKE Evaluation ==========")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}")
    print(f"Per Answer Type: {summary['per_answer_type_accuracy']}")
    print(f"Per Question Type: {summary['per_question_type_accuracy']}")
    print(f"Per Language: {summary['per_language_accuracy']}")
    print(f"TTFT: {timing_summary['ttft_seconds']}")
    print(
        "TPOT: "
        f"{timing_summary['tpot_weighted_seconds']} s/token, "
        f"{timing_summary['decode_tokens_per_second']} token/s"
    )
    print(f"Request Latency: {timing_summary['request_seconds']}")
    print(f"Predictions: {output_dir / 'slake_predictions.json'}")
    print(f"Summary: {output_dir / 'slake_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
