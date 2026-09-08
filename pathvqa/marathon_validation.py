"""In-process PathVQA validation for long-running QDPT training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from loraTest.generation_timing import generate_with_timing
from pathvqa.data_pipeline import PathVQAParquetStore
from pathvqa.marathon_progress import (
    MarathonProgressLog,
    scheduled_validation_epochs,
)
from pathvqa.pathvqa_official_eval import run_inference
from pathvqa.pathvqa_vqa_metric import evaluate_pathvqa_predictions
from slake.slake_official_eval import summarize_generation_timings, write_json


def _move_inputs(
    inputs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            moved[key] = value.to(device=device)
    return moved


class InMemoryDynamicPromptInterface:
    """Use the training model for generation without loading a second base model."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor
        self.device = next(model.base_model.parameters()).device
        self.last_generation_timing = None

    def reset_inference_state(self) -> None:
        self.model.reset_inference_intervention_state()
        self.last_generation_timing = None

    def infer(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = _move_inputs(dict(inputs), self.device)
        original_length = int(inputs["input_ids"].shape[-1])
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": True,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids, self.last_generation_timing = generate_with_timing(
                self.model,
                inputs,
                generate_kwargs,
            )
        generated = output_ids[:, original_length + self.model.prompt_length :]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()


class PathVQAMarathonValidator:
    def __init__(
        self,
        processor: Any,
        data_root: Path,
        output_dir: Path,
        start_epoch: int,
        total_epochs: int,
        cache_dir: Path | None = None,
        bootstrap_iterations: int = 2000,
        bootstrap_seed: int = 42,
    ) -> None:
        self.processor = processor
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.output_dir = Path(output_dir)
        self.validation_epochs = frozenset(
            scheduled_validation_epochs(total_epochs, start_epoch)
        )
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.bootstrap_seed = int(bootstrap_seed)
        self.progress = MarathonProgressLog(
            self.output_dir / "marathon_progress.tsv"
        )
        self.started_at = time.monotonic()
        self._store: PathVQAParquetStore | None = None

    def should_validate(self, epoch: int) -> bool:
        return int(epoch) in self.validation_epochs

    def _validation_data(
        self,
    ) -> tuple[PathVQAParquetStore, Sequence[Dict[str, Any]]]:
        if self._store is None:
            self._store = PathVQAParquetStore(
                self.data_root,
                "validation",
                cache_dir=self.cache_dir,
            )
        return self._store, list(self._store.samples)

    def evaluate(
        self,
        model: Any,
        epoch: int,
        global_step: int,
        checkpoint: Path,
        latest_train_loss: float | None,
    ) -> Dict[str, Any] | None:
        epoch = int(epoch)
        if not self.should_validate(epoch):
            return None

        eval_dir = self.output_dir / "eval_validation" / f"epoch_{epoch}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        evaluation_started = time.monotonic()
        was_training = bool(model.training)
        original_use_cache = bool(model.base_model.config.use_cache)
        try:
            store, records = self._validation_data()
            interface = InMemoryDynamicPromptInterface(model, self.processor)
            model.eval()
            model.base_model.config.use_cache = True
            interface.reset_inference_state()
            predictions = run_inference(
                store,
                records,
                model=interface,
                output_dir=eval_dir,
                max_new_tokens=32,
                temperature=0.0,
                instruction=None,
                answer_mode="raw",
                resume=False,
                overwrite=True,
                continue_on_error=False,
            )
            summary, comparisons = evaluate_pathvqa_predictions(
                records,
                predictions,
                bootstrap_iterations=self.bootstrap_iterations,
                bootstrap_seed=self.bootstrap_seed,
            )
            summary.update(
                {
                    "backend": "dynamic-prompt-in-memory",
                    "checkpoint": str(checkpoint),
                    "split": "validation",
                    "epoch": epoch,
                    "global_step": int(global_step),
                    "partial_evaluation": False,
                    "timing": summarize_generation_timings(
                        predictions,
                        warmup_runs=0,
                    ),
                }
            )
            write_json(
                eval_dir / "pathvqa_predictions.json",
                [
                    {"question_id": row["question_id"], "answer": row["answer"]}
                    for row in predictions
                ],
            )
            write_json(eval_dir / "pathvqa_details.json", predictions)
            write_json(eval_dir / "pathvqa_comparisons.json", comparisons)
            write_json(eval_dir / "pathvqa_summary.json", summary)
            evaluation_minutes = (time.monotonic() - evaluation_started) / 60.0
            self.progress.append(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "epoch": epoch,
                    "global_step": int(global_step),
                    "overall": f"{summary['overall_accuracy']:.4f}",
                    "yes_no": f"{summary['yes_no_accuracy']:.4f}",
                    "free_form": f"{summary['free_form_accuracy']:.4f}",
                    "latest_train_loss": (
                        "" if latest_train_loss is None else f"{latest_train_loss:.6f}"
                    ),
                    "elapsed_hours": f"{(time.monotonic() - self.started_at) / 3600.0:.4f}",
                    "evaluation_minutes": f"{evaluation_minutes:.2f}",
                    "checkpoint": str(checkpoint),
                    "status": "complete",
                    "error": "",
                }
            )
            print(
                "[PATHVQA_QDPT_MARATHON_VALIDATION] "
                f"epoch={epoch} global_step={int(global_step)} "
                f"overall={summary['overall_accuracy']:.4f} "
                f"yes_no={summary['yes_no_accuracy']:.4f} "
                f"free_form={summary['free_form_accuracy']:.4f} "
                f"minutes={evaluation_minutes:.2f} status=complete"
            )
            return summary
        except Exception as exc:
            self.progress.append(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "epoch": epoch,
                    "global_step": int(global_step),
                    "latest_train_loss": (
                        "" if latest_train_loss is None else f"{latest_train_loss:.6f}"
                    ),
                    "elapsed_hours": f"{(time.monotonic() - self.started_at) / 3600.0:.4f}",
                    "evaluation_minutes": f"{(time.monotonic() - evaluation_started) / 60.0:.2f}",
                    "checkpoint": str(checkpoint),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            model.base_model.config.use_cache = original_use_cache
            model.reset_inference_intervention_state()
            if was_training:
                model.train()
