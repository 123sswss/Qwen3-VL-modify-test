#!/usr/bin/env python3
"""Train the final QDPT adapter on PathVQA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from config import (
    DATA_SEED,
    DATALOADER_WORKERS,
    EPOCHS,
    EXPECTED_TRAINABLE_PARAMETERS,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    MICRO_BATCH_SIZE,
    MODEL_NAME,
    PROMPT_LEARNING_RATE,
    QDPT_LEARNING_RATE,
    VISUAL_PROMPT_LEARNING_RATE,
    WARMUP_RATIO,
)
from data import PathVQACollator, PathVQATrainDataset
from model import QDPTModel, trainable_parameter_count


class QDPTTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.model.parameter_groups()
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": groups["language_prompt"],
                    "lr": PROMPT_LEARNING_RATE,
                    "weight_decay": 0.0,
                },
                {
                    "params": groups["visual_prompt"],
                    "lr": VISUAL_PROMPT_LEARNING_RATE,
                    "weight_decay": 0.0,
                },
                {
                    "params": groups["qdpt"],
                    "lr": QDPT_LEARNING_RATE,
                    "weight_decay": 0.0,
                },
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return self.optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QDPT on PathVQA")
    parser.add_argument("--model-path", default=MODEL_NAME)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    model = QDPTModel(base_model, processor.tokenizer, seed=args.seed)

    parameter_groups = model.parameter_groups()
    grouped = [parameter for values in parameter_groups.values() for parameter in values]
    active = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in active}:
        raise RuntimeError("QDPT optimizer groups do not cover all trainable parameters")
    if len(grouped) != len({id(parameter) for parameter in grouped}):
        raise RuntimeError("A QDPT parameter occurs in more than one optimizer group")
    trainable = trainable_parameter_count(model)
    if trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINABLE_PARAMETERS} trainable parameters, got {trainable}"
        )

    dataset = PathVQATrainDataset(processor, args.data_root)
    trainer = QDPTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=PROMPT_LEARNING_RATE,
            weight_decay=0.0,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="linear",
            max_grad_norm=MAX_GRAD_NORM,
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            gradient_checkpointing=False,
            dataloader_num_workers=DATALOADER_WORKERS,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            data_seed=DATA_SEED,
        ),
        train_dataset=dataset,
        data_collator=PathVQACollator(processor),
        processing_class=processor,
    )
    result = trainer.train()

    checkpoint_dir = args.output_dir / "checkpoints" / "epoch_3"
    model.save_adapter(checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "qdpt_dense_d768_sandwich",
                "dataset": "PathVQA",
                "seed": args.seed,
                "data_seed": DATA_SEED,
                "trainable_parameters": trainable,
                "train_metrics": result.metrics,
                "checkpoint": str(checkpoint_dir),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved QDPT adapter to {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

