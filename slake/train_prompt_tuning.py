#!/usr/bin/env python3
"""Train the classic static Prompt Tuning baseline on SLAKE."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slake.data_pipeline import SLAKEDataCollator, SLAKEDataset
from slake.prompt_tuning import StaticPromptTuningModel


MODEL_BATCH_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels")


class PromptTuningCollator:
    def __init__(self, processor: Any) -> None:
        self.base = SLAKEDataCollator(processor)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base(features)
        return {key: batch[key] for key in MODEL_BATCH_KEYS if batch.get(key) is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static Prompt Tuning for Qwen3-VL")
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/dataset/slake"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_model.config.use_cache = False
    model = StaticPromptTuningModel(base_model, args.prompt_length, args.seed)
    dataset = SLAKEDataset(
        processor=processor,
        questions_path=str(args.data_root / "train.json"),
        image_root=str(args.data_root / "imgs"),
        splits=("train",),
        ce_enabled=True,
        seed=args.data_seed,
        deterministic_sampling=True,
        max_length=2048,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = args.prompt_length * model.soft_prompt.shape[-1]
    if trainable != expected:
        raise RuntimeError(f"Trainable parameter audit failed: expected={expected} actual={trainable}")
    print(
        "[SLAKE_PROMPT_TUNING_CONFIG] "
        f"prompt_length={args.prompt_length} hidden_size={model.soft_prompt.shape[-1]} "
        f"trainable={trainable} epochs={args.epochs} lr={args.learning_rate} "
        f"seed={args.seed} data_seed={args.data_seed}"
    )

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
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            data_seed=args.data_seed,
        ),
        train_dataset=dataset,
        data_collator=PromptTuningCollator(processor),
        processing_class=processor,
    )
    result = trainer.train()
    final_dir = args.output_dir / "final"
    model.save_prompt(final_dir)
    processor.save_pretrained(final_dir)
    report = {
        "method": "static_prompt_tuning",
        "prompt_length": args.prompt_length,
        "trainable_parameters": trainable,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "train_metrics": result.metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[SLAKE_PROMPT_TUNING_PASS] checkpoint={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
