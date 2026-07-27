"""SLAKE/VQA exact-match metric compatible with common Med-VQA evaluators.

The normalization follows the VQA evaluation code used by M2I2's
``vqaSlakeEval.py``: punctuation cleanup, lower-casing, English article
removal, number-word mapping, and contraction normalization.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id've": "i'd've",
    "i'dve": "i'd've",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = {"a", "an", "the"}
PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_vqa_answer(value: Any) -> str:
    """Normalize one answer exactly as the common SLAKE VQA evaluator does."""

    text = str(value if value is not None else "")
    text = text.replace("\n", " ").replace("\t", " ").strip()

    normalized = text
    for punctuation in PUNCTUATION:
        if (
            f"{punctuation} " in text
            or f" {punctuation}" in text
            or COMMA_STRIP.search(text) is not None
        ):
            normalized = normalized.replace(punctuation, "")
        else:
            normalized = normalized.replace(punctuation, " ")
    normalized = PERIOD_STRIP.sub("", normalized)

    words = []
    for raw_word in normalized.lower().split():
        word = NUMBER_WORDS.get(raw_word, raw_word)
        if word not in ARTICLES:
            words.append(CONTRACTIONS.get(word, word))
    return " ".join(words)


def record_id(record: Mapping[str, Any], fallback: Any = None) -> Any:
    for key in ("question_id", "qid", "id"):
        if record.get(key) is not None:
            return record[key]
    if fallback is not None:
        return fallback
    raise ValueError("SLAKE record has no question_id, qid, or id")


def _reference_answers(record: Mapping[str, Any]) -> List[str]:
    raw_answer = record.get("answer", record.get("answers"))
    if isinstance(raw_answer, list):
        answers = []
        for item in raw_answer:
            if isinstance(item, Mapping):
                item = item.get("answer", "")
            answers.append(str(item))
        return answers
    return [str(raw_answer if raw_answer is not None else "")]


def _group_accuracy(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(int(row["correct"]))
    return {
        name: round(100.0 * sum(values) / len(values), 2)
        for name, values in sorted(grouped.items())
    }


def evaluate_slake_predictions(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Score predictions and return a summary plus per-question comparisons."""

    prediction_by_id: Dict[str, Mapping[str, Any]] = {}
    duplicate_prediction_ids = []
    for prediction in predictions:
        qid = str(record_id(prediction))
        if qid in prediction_by_id:
            duplicate_prediction_ids.append(qid)
        prediction_by_id[qid] = prediction
    if duplicate_prediction_ids:
        raise ValueError(
            "Duplicate prediction question IDs: "
            f"{sorted(set(duplicate_prediction_ids))[:10]}"
        )

    seen_reference_ids = set()
    comparisons: List[Dict[str, Any]] = []
    for index, reference in enumerate(references):
        original_id = record_id(reference, fallback=index)
        qid = str(original_id)
        if qid in seen_reference_ids:
            raise ValueError(f"Duplicate reference question ID: {qid}")
        seen_reference_ids.add(qid)

        prediction = prediction_by_id.get(qid)
        predicted_answer = "" if prediction is None else str(prediction.get("answer", ""))
        normalized_prediction = normalize_vqa_answer(predicted_answer)
        ground_truth_answers = _reference_answers(reference)
        normalized_ground_truths = [
            normalize_vqa_answer(answer)
            for answer in ground_truth_answers
        ]
        correct = int(
            prediction is not None
            and normalized_prediction in normalized_ground_truths
        )
        comparisons.append(
            {
                "question_id": original_id,
                "question": str(reference.get("question", "")),
                "ground_truth_answers": ground_truth_answers,
                "normalized_ground_truth_answers": normalized_ground_truths,
                "predicted_answer": predicted_answer,
                "normalized_predicted_answer": normalized_prediction,
                "correct": correct,
                "answer_type": str(reference.get("answer_type", "unknown")),
                "question_type": str(
                    reference.get("question_type", reference.get("base_type", "unknown"))
                ),
                "language": str(reference.get("q_lang", reference.get("language", "unknown"))),
                "missing_prediction": prediction is None,
            }
        )

    correct_count = sum(row["correct"] for row in comparisons)
    missing_count = sum(row["missing_prediction"] for row in comparisons)
    extra_prediction_ids = sorted(set(prediction_by_id) - seen_reference_ids)
    total = len(comparisons)
    summary = {
        "metric": "slake_vqa_normalized_exact_match",
        "total": total,
        "correct": correct_count,
        "wrong": total - correct_count,
        "missing_predictions": missing_count,
        "extra_predictions": len(extra_prediction_ids),
        "extra_prediction_ids": extra_prediction_ids[:20],
        "overall_accuracy": round(100.0 * correct_count / max(total, 1), 2),
        "per_answer_type_accuracy": _group_accuracy(comparisons, "answer_type"),
        "per_question_type_accuracy": _group_accuracy(comparisons, "question_type"),
        "per_language_accuracy": _group_accuracy(comparisons, "language"),
    }
    return summary, comparisons
