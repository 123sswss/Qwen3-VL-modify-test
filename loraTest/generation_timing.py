"""Small generation-timing helper shared by baseline evaluation entry points."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import torch
from transformers import LogitsProcessor


class _TokenTimer(LogitsProcessor):
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.uses_cuda_events = device.type == "cuda"
        self.started_at = time.perf_counter()
        self.token_ready_at = []
        self.start_event = None
        self.token_ready_events = []

        if self.uses_cuda_events:
            torch.cuda.synchronize(device)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record(torch.cuda.current_stream(device))
            self.started_at = time.perf_counter()

    def __call__(self, input_ids, scores):
        if self.uses_cuda_events:
            if scores.device != self.device:
                raise RuntimeError(
                    "Generation timing device changed during decoding: "
                    f"start={self.device} scores={scores.device}"
                )
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(scores.device))
            self.token_ready_events.append(event)
        else:
            self.token_ready_at.append(time.perf_counter())
        return scores

    def token_times(self) -> list[float]:
        if not self.uses_cuda_events:
            return [timestamp - self.started_at for timestamp in self.token_ready_at]
        if self.start_event is None:
            return []
        return [
            self.start_event.elapsed_time(event) / 1000.0
            for event in self.token_ready_events
        ]


def _timing_device(model: Any, inputs: Dict[str, Any]) -> torch.device:
    for key in ("input_ids", "inputs_embeds", "pixel_values"):
        value = inputs.get(key)
        if torch.is_tensor(value):
            return value.device
    return next(model.parameters()).device


def generate_with_timing(
    model: Any,
    inputs: Dict[str, Any],
    generate_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run generate and return synchronized TTFT/inter-token timing."""
    timer = _TokenTimer(_timing_device(model, inputs))
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
    token_ready_at = timer.token_times()
    intervals = [
        current - previous
        for previous, current in zip(token_ready_at, token_ready_at[1:])
    ]
    timing = {
        "first_token_latency_seconds": (
            token_ready_at[0] if token_ready_at else None
        ),
        "subsequent_token_mean_seconds": (
            sum(intervals) / len(intervals) if intervals else None
        ),
        "generation_total_seconds": finished_at - timer.started_at,
        "generated_token_count": generated_token_count,
        "measured_token_steps": len(token_ready_at),
        "timing_method": (
            "cuda-events-logits-ready-v2"
            if timer.uses_cuda_events
            else "perf-counter-logits-ready-v2"
        ),
    }
    return output_ids, timing
