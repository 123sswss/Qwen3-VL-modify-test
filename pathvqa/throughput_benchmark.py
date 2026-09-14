"""Shared GPU-only training throughput instrumentation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainerCallback


class TrainingWorkTracker:
    def __init__(self, spatial_merge_size: int = 1) -> None:
        if spatial_merge_size < 1:
            raise ValueError("spatial_merge_size must be positive")
        self.spatial_merge_area = spatial_merge_size**2
        self.samples = 0
        self._visual_tokens: Optional[torch.Tensor] = None

    def observe(self, inputs: Dict[str, Any]) -> None:
        input_ids = inputs.get("input_ids")
        if torch.is_tensor(input_ids):
            self.samples += int(input_ids.shape[0])
        grid = inputs.get("image_grid_thw")
        if not torch.is_tensor(grid) or grid.numel() == 0:
            return
        token_count = (
            grid.detach().to(dtype=torch.int64).prod(dim=-1).sum()
            // self.spatial_merge_area
        )
        if self._visual_tokens is None:
            self._visual_tokens = token_count.clone()
        else:
            self._visual_tokens.add_(token_count)

    def snapshot(self) -> Dict[str, int]:
        visual_tokens = (
            int(self._visual_tokens.item())
            if self._visual_tokens is not None
            else 0
        )
        return {"samples": self.samples, "visual_tokens": visual_tokens}


class ThroughputTrainer(Trainer):
    def __init__(self, *args, work_tracker: TrainingWorkTracker, **kwargs) -> None:
        self.work_tracker = work_tracker
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, *args, **kwargs):
        self.work_tracker.observe(inputs)
        return super().training_step(model, inputs, *args, **kwargs)


class ThroughputBenchmarkCallback(TrainerCallback):
    def __init__(
        self,
        *,
        output_path: Path,
        warmup_steps: int,
        timed_steps: int,
        effective_batch_size: int,
        work_tracker: TrainingWorkTracker,
        estimated_total_optimizer_steps: int,
        method: str,
    ) -> None:
        if warmup_steps < 0 or timed_steps < 1:
            raise ValueError("warmup_steps must be non-negative and timed_steps positive")
        self.output_path = Path(output_path)
        self.warmup_steps = int(warmup_steps)
        self.timed_steps = int(timed_steps)
        self.target_step = self.warmup_steps + self.timed_steps
        self.effective_batch_size = int(effective_batch_size)
        self.work_tracker = work_tracker
        self.estimated_total_optimizer_steps = int(estimated_total_optimizer_steps)
        self.method = str(method)
        self.started_at: Optional[float] = None
        self.start_work: Optional[Dict[str, int]] = None
        self.report: Optional[Dict[str, Any]] = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("Throughput benchmark requires CUDA; CPU fallback is forbidden")
        if int(args.max_steps) != self.target_step:
            raise RuntimeError(
                f"Benchmark max_steps must equal {self.target_step}, got {args.max_steps}"
            )
        if self.warmup_steps == 0:
            self._start()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if self.started_at is None and int(state.global_step) == self.warmup_steps:
            self._start()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.report is None and int(state.global_step) >= self.target_step:
            self._finish(int(state.global_step))
            control.should_training_stop = True
        return control

    def _start(self) -> None:
        torch.cuda.synchronize()
        self.start_work = self.work_tracker.snapshot()
        torch.cuda.reset_peak_memory_stats()
        self.started_at = time.perf_counter()

    def _finish(self, global_step: int) -> None:
        if self.started_at is None or self.start_work is None:
            raise RuntimeError("Benchmark timing window was never started")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started_at
        end_work = self.work_tracker.snapshot()
        samples = end_work["samples"] - self.start_work["samples"]
        visual_tokens = end_work["visual_tokens"] - self.start_work["visual_tokens"]
        seconds_per_step = elapsed / self.timed_steps
        self.report = {
            "method": self.method,
            "status": "complete",
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "warmup_optimizer_steps": self.warmup_steps,
            "timed_optimizer_steps": self.timed_steps,
            "end_global_step": global_step,
            "effective_batch_size": self.effective_batch_size,
            "timed_samples": samples,
            "timed_visual_tokens": visual_tokens,
            "elapsed_seconds": elapsed,
            "seconds_per_optimizer_step": seconds_per_step,
            "samples_per_second": samples / elapsed,
            "visual_tokens_per_second": visual_tokens / elapsed,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "estimated_total_optimizer_steps": self.estimated_total_optimizer_steps,
            "estimated_pure_training_seconds": (
                seconds_per_step * self.estimated_total_optimizer_steps
            ),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.report, handle, ensure_ascii=False, indent=2)
        print("[THROUGHPUT_BENCHMARK] " + json.dumps(self.report, ensure_ascii=False))


def estimated_optimizer_steps(
    dataset_size: int,
    effective_batch_size: int,
    epochs: int,
) -> int:
    if min(dataset_size, effective_batch_size, epochs) < 1:
        raise ValueError("dataset size, effective batch size, and epochs must be positive")
    return math.ceil(dataset_size / effective_batch_size) * epochs


def spatial_merge_size(model: Any) -> int:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    return int(getattr(vision_config, "spatial_merge_size", 1))
