"""Small generation-timing helper shared by baseline evaluation entry points."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import torch
from transformers import LogitsProcessor


class _TokenTimer(LogitsProcessor):
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.token_ready_at = []

    def __call__(self, input_ids, scores):
        if scores.device.type == "cuda":
            torch.cuda.synchronize(scores.device)
        self.token_ready_at.append(time.perf_counter())
        return scores


def generate_with_timing(
    model: Any,
    inputs: Dict[str, Any],
    generate_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run generate and return approximate TTFT/inter-token timing."""
    timer = _TokenTimer()
    kwargs = dict(generate_kwargs)
    existing_processors = list(kwargs.pop("logits_processor", []) or [])
    output_ids = model.generate(
        **inputs,
        **kwargs,
        logits_processor=existing_processors + [timer],
    )
    output_device = output_ids.device
    if output_device.type == "cuda":
        torch.cuda.synchronize(output_device)
    finished_at = time.perf_counter()
    input_length = inputs["input_ids"].shape[-1]
    generated_token_count = int(output_ids.shape[-1] - input_length)
    intervals = [
        current - previous
        for previous, current in zip(timer.token_ready_at, timer.token_ready_at[1:])
    ]
    timing = {
        "first_token_latency_seconds": (
            timer.token_ready_at[0] - timer.started_at if timer.token_ready_at else None
        ),
        "subsequent_token_mean_seconds": (
            sum(intervals) / len(intervals) if intervals else None
        ),
        "generation_total_seconds": finished_at - timer.started_at,
        "generated_token_count": generated_token_count,
        "measured_token_steps": len(timer.token_ready_at),
    }
    return output_ids, timing
