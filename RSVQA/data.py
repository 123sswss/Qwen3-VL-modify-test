"""Strict loader for the official RSVQA-LR split JSON files.

The released split files retain inactive records from the other splits.  This
module therefore filters ``active == true`` independently in the image,
question, and answer files before joining them.  The resulting list-of-dicts
contract is deliberately model agnostic so the same records can later be used
by training code.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


QUESTION_TYPES = ("rural_urban", "presence", "count", "comp")
OFFICIAL_SPLIT_COUNTS = {
    "train": (572, 57223),
    "val": (100, 10005),
    "test": (100, 10004),
}
SPLIT_ALIASES = {"validation": "val", "valid": "val"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_split(split: str) -> str:
    value = SPLIT_ALIASES.get(split.strip().lower(), split.strip().lower())
    if value not in OFFICIAL_SPLIT_COUNTS:
        raise ValueError(f"Unsupported RSVQA-LR split {split!r}; use train/val/test")
    return value


def _discover_file(data_root: Path, split: str, kind: str) -> Path:
    candidates: List[Path] = []
    for path in data_root.rglob("*.json"):
        name = path.name.lower()
        if f"split_{split}_" in name and kind in name:
            candidates.append(path.resolve())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *split_{split}_*{kind}*.json under "
            f"{data_root}, found {len(candidates)}: {[str(x) for x in candidates]}"
        )
    return candidates[0]


def _records(payload: Any, key: str, path: Path) -> List[Dict[str, Any]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get(key), list):
        raise ValueError(f"{path} must contain a top-level {key!r} list")
    result = []
    for index, row in enumerate(payload[key]):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{key}[{index}] is not an object")
        result.append(dict(row))
    return result


def _active(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("active") is True]


def _index_unique(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("id") is None:
            raise ValueError(f"Active RSVQA {label} record has no id: {row}")
        key = str(row["id"])
        if key in indexed:
            raise ValueError(f"Duplicate active RSVQA {label} id: {key}")
        indexed[key] = row
    return indexed


def _image_root(data_root: Path) -> Path:
    direct = data_root / "Images_LR"
    if direct.is_dir():
        return direct.resolve()
    matches = [p.resolve() for p in data_root.rglob("Images_LR") if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Images_LR directory under {data_root}; "
            f"found {len(matches)}: {[str(x) for x in matches]}"
        )
    return matches[0]


def load_rsvqa_lr_split(
    data_root: Path | str,
    split: str,
    *,
    enforce_official_counts: bool = True,
    require_images: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load one official split and return normalized records plus a manifest."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RSVQA-LR data root not found: {root}")
    split = _canonical_split(split)
    paths = {
        kind: _discover_file(root, split, kind)
        for kind in ("images", "questions", "answers")
    }
    image_dir = _image_root(root)

    active_images = _active(_records(_load_json(paths["images"]), "images", paths["images"]))
    active_questions = _active(
        _records(_load_json(paths["questions"]), "questions", paths["questions"])
    )
    active_answers = _active(
        _records(_load_json(paths["answers"]), "answers", paths["answers"])
    )
    images = _index_unique(active_images, "image")
    questions = _index_unique(active_questions, "question")
    answers = _index_unique(active_answers, "answer")

    records: List[Dict[str, Any]] = []
    missing_images = []
    referenced_answers = set()
    for question_key, question in questions.items():
        image_key = str(question.get("img_id"))
        if image_key not in images:
            raise ValueError(
                f"Question {question_key} references inactive/missing image {image_key}"
            )
        answer_ids = question.get("answers_ids")
        if not isinstance(answer_ids, list) or len(answer_ids) != 1:
            raise ValueError(
                f"Question {question_key} must reference exactly one answer; got {answer_ids!r}"
            )
        answer_key = str(answer_ids[0])
        if answer_key not in answers:
            raise ValueError(
                f"Question {question_key} references inactive/missing answer {answer_key}"
            )
        answer = answers[answer_key]
        if str(answer.get("question_id")) != question_key:
            raise ValueError(
                f"Answer {answer_key} reverse question_id={answer.get('question_id')!r} "
                f"does not match question {question_key}"
            )
        qtype = str(question.get("type", "")).strip().lower()
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"Question {question_key} has unsupported type {qtype!r}")
        image_question_ids = {str(value) for value in images[image_key].get("questions_ids", [])}
        if question_key not in image_question_ids:
            raise ValueError(
                f"Active image {image_key} does not list active question {question_key}"
            )
        image_path = image_dir / f"{image_key}.tif"
        if not image_path.is_file():
            missing_images.append(str(image_path))
        referenced_answers.add(answer_key)
        records.append(
            {
                "question_id": question["id"],
                "image_id": images[image_key]["id"],
                "answer_id": answer["id"],
                "question": str(question.get("question", "")).strip(),
                "answer": str(answer.get("answer", "")).strip(),
                "question_type": qtype,
                "image_path": str(image_path.resolve()),
                "split": split,
            }
        )

    if referenced_answers != set(answers):
        unused = sorted(set(answers) - referenced_answers)
        raise ValueError(f"Active answer records not referenced exactly once; examples={unused[:10]}")
    if require_images and missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} active RSVQA-LR images are missing; "
            f"examples={missing_images[:5]}"
        )
    records.sort(key=lambda row: int(row["question_id"]))

    expected_images, expected_questions = OFFICIAL_SPLIT_COUNTS[split]
    if enforce_official_counts and (
        len(images) != expected_images or len(records) != expected_questions
    ):
        raise ValueError(
            f"Official RSVQA-LR {split} count mismatch: "
            f"images={len(images)} expected={expected_images}, "
            f"questions={len(records)} expected={expected_questions}"
        )
    type_counts = Counter(row["question_type"] for row in records)
    manifest = {
        "dataset": "RSVQA-LR",
        "split": split,
        "data_root": str(root),
        "image_root": str(image_dir),
        "source_files": {key: str(value) for key, value in paths.items()},
        "active_images": len(images),
        "active_questions": len(records),
        "active_answers": len(answers),
        "missing_images": len(missing_images),
        "question_type_counts": dict(sorted(type_counts.items())),
        "official_counts_enforced": enforce_official_counts,
    }
    return records, manifest
