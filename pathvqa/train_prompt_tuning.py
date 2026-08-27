#!/usr/bin/env python3
"""Train the classic static Prompt Tuning baseline on PathVQA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from pathvqa.data_pipeline import PathVQADataCollator, PathVQADataset
from slake.prompt_tuning import StaticPromptTuningModel


MODEL_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "labels",
)


class PromptTuningCollator:
    def __init__(self, processor: Any) -> None:
        self.base = PathVQADataCollator(processor)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base(features)
        return {
            key: batch[key]
            for key in MODEL_BATCH_KEYS
            if key in batch and batch[key] is not None
        }


class EpochPromptCheckpointCallback(TrainerCallback):
    def __init__(self, processor: Any, output_dir: Path) -> None:
        self.processor = processor
        self.output_dir = output_dir
        self.completed_epochs = set()

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        epoch_id = max(1, int(round(float(state.epoch or 0.0))))
        if epoch_id in self.completed_epochs:
            return control
        self.completed_epochs.add(epoch_id)
        checkpoint_dir = self.output_dir / "checkpoints" / f"epoch_{epoch_id}"
        kwargs["model"].save_prompt(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
        print(
            f"[PATHVQA_PROMPT_EPOCH_CHECKPOINT] epoch={epoch_id} "
            f"global_step={int(state.global_step)} saved={checkpoint_dir}"
        )
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static Prompt Tuning for Qwen3-VL on PathVQA"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/model"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/pathVQA"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default="pathvqa_prompt_tuning_len20")
    parser.add_argument("--prompt-length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.prompt_length < 1 or args.epochs < 1:
        parser.error("--prompt-length and --epochs must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    model = StaticPromptTuningModel(base_model, args.prompt_length, args.seed)
    dataset = PathVQADataset(
        processor=processor,
        data_root=args.data_root,
        split="train",
        cache_dir=args.cache_dir,
        ce_enabled=True,
        seed=args.data_seed,
        deterministic_sampling=True,
        max_length=args.max_length,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = args.prompt_length * model.soft_prompt.shape[-1]
    if trainable != expected:
        raise RuntimeError(
            f"Trainable parameter audit failed: expected={expected} actual={trainable}"
        )
    print(
        "[PATHVQA_PROMPT_TUNING_CONFIG] "
        f"prompt_length={args.prompt_length} "
        f"hidden_size={model.soft_prompt.shape[-1]} trainable={trainable} "
        f"epochs={args.epochs} lr={args.learning_rate} "
        f"seed={args.seed} data_seed={args.data_seed}"
    )

    callback = EpochPromptCheckpointCallback(processor, args.output_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=0.0,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            gradient_checkpointing=False,
            dataloader_num_workers=args.dataloader_workers,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            data_seed=args.data_seed,
        ),
        train_dataset=dataset,
        data_collator=PromptTuningCollator(processor),
        processing_class=processor,
        callbacks=[callback],
    )
    result = trainer.train()
    final_dir = args.output_dir / "final"
    model.save_prompt(final_dir)
    processor.save_pretrained(final_dir)
    report = {
        "method": "static_prompt_tuning",
        "experiment": args.experiment_name,
        "dataset": "PathVQA",
        "prompt_length": args.prompt_length,
        "trainable_parameters": trainable,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "train_metrics": result.metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[PATHVQA_PROMPT_TUNING_PASS] checkpoint={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
