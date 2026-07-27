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
            raw_output = model.infer(
                image,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            answer = extract_generated_answer(raw_output, answer_mode)
            row = {
                "question_id": qid,
                "answer": answer,
                "raw_output": raw_output,
                "question": record["_slake_question"],
                "image": record["_slake_image_name"],
                "status": "ok",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

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
    model = load_slake_model_interface(
        args.backend,
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
    )
    instruction = "" if args.no_instruction else args.instruction
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
        }
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
    print(f"Predictions: {output_dir / 'slake_predictions.json'}")
    print(f"Summary: {output_dir / 'slake_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
