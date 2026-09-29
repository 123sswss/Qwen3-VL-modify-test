"""Shared RSVQA-LR prompt policy for training and generation."""

from __future__ import annotations

from typing import Any, Mapping


DEFAULT_INSTRUCTIONS = {
    "count": "Answer with only an integer, without explanation.",
    "presence": "Answer only yes or no.",
    "rural_urban": "Answer only rural or urban.",
    "comp": "Answer with only the final short answer, without explanation.",
}


def build_prompt(record: Mapping[str, Any]) -> str:
    question_type = str(record["question_type"])
    return f"{record['question']}\n{DEFAULT_INSTRUCTIONS[question_type]}"
