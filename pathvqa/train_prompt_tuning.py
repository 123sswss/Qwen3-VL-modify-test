#!/usr/bin/env python3
"""Train the classic static Prompt Tuning baseline."""

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
from pathvqa.train_dynamic_prompt import (
    DynamicPromptCollator,
    _build_train_dataset,
    _dataset_display_name,
    _normalize_dataset_name,
)
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


class DualRatePromptTrainer(Trainer):
    def __init__(self, *args, text_prompt_lr: float, visual_prompt_lr: float, **kwargs):
        self.text_prompt_lr = float(text_prompt_lr)
        self.visual_prompt_lr = float(visual_prompt_lr)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.model.trainable_parameter_groups()
        optimizer_groups = []
        if groups["text_prompt"]:
            optimizer_groups.append({
                "params": groups["text_prompt"],
                "lr": self.text_prompt_lr,
                "weight_decay": 0.0,
                "group_name": "text_prompt",
            })
        if groups["visual_prompt"]:
            optimizer_groups.append({
                "params": groups["visual_prompt"],
                "lr": self.visual_prompt_lr,
                "weight_decay": 0.0,
                "group_name": "visual_prompt",
            })
        grouped = [parameter for group in optimizer_groups for parameter in group["params"]]
        active = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if {id(parameter) for parameter in grouped} != {id(parameter) for parameter in active}:
            raise RuntimeError("Static Prompt parameters must belong to one optimizer group")
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, betas=(0.9, 0.999), eps=1e-8
        )
        print(
            "[STATIC_PROMPT_OPTIMIZER] "
            f"text_lr={self.text_prompt_lr} text_tensors={len(groups['text_prompt'])} "
            f"visual_lr={self.visual_prompt_lr} visual_tensors={len(groups['visual_prompt'])}"
        )
        return self.optimizer


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
    parser.add_argument("--visual-prompt-length", type=int, default=0)
    parser.add_argument("--visual-anchor-layer", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional short-run override for smoke tests; -1 keeps epoch training.",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--visual-learning-rate", type=float, default=1e-4)
    parser.add_argument("--expected-trainable-parameters", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.prompt_length < 0 or args.visual_prompt_length < 0 or args.epochs < 1:
        parser.error("Prompt lengths must be non-negative and --epochs positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    if args.prompt_length == 0 and args.visual_prompt_length == 0:
        parser.error("At least one text or visual Prompt is required")
    if args.visual_anchor_layer < 0:
        parser.error("--visual-anchor-layer must be non-negative")
    if args.learning_rate <= 0.0 or args.visual_learning_rate <= 0.0:
        parser.error("Prompt learning rates must be positive")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    return args


def main(dataset_name: str = "pathvqa") -> int:
    dataset_name = _normalize_dataset_name(dataset_name)
    display_name = _dataset_display_name(dataset_name)
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
    model = StaticPromptTuningModel(
        base_model,
        args.prompt_length,
        args.seed,
        visual_prompt_length=args.visual_prompt_length,
        visual_anchor_layers=(args.visual_anchor_layer,),
    )
    dataset = _build_train_dataset(dataset_name, args, processor)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = (
        args.prompt_length * model.get_input_embeddings().weight.shape[-1]
        + args.visual_prompt_length
        * (
            model.static_visual_prompt.visual_dim
            if model.static_visual_prompt is not None
            else 0
        )
    )
    if args.expected_trainable_parameters is not None:
        expected = args.expected_trainable_parameters
    if trainable != expected:
        raise RuntimeError(
            f"Trainable parameter audit failed: expected={expected} actual={trainable}"
        )
    print(
        f"[{display_name.upper()}_PROMPT_TUNING_CONFIG] "
        f"prompt_length={args.prompt_length} "
        f"hidden_size={model.get_input_embeddings().weight.shape[-1]} "
        f"visual_prompt_length={args.visual_prompt_length} "
        f"visual_anchor_layer={args.visual_anchor_layer} trainable={trainable} "
        f"epochs={args.epochs} text_lr={args.learning_rate} "
        f"visual_lr={args.visual_learning_rate} "
        f"seed={args.seed} data_seed={args.data_seed}"
    )

    callback = EpochPromptCheckpointCallback(processor, args.output_dir)
    trainer = DualRatePromptTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
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
        data_collator=(
            PromptTuningCollator(processor)
            if dataset_name == "pathvqa"
            else DynamicPromptCollator(processor, dataset_name)
        ),
        processing_class=processor,
        callbacks=[callback],
        text_prompt_lr=args.learning_rate,
        visual_prompt_lr=args.visual_learning_rate,
    )
    result = trainer.train()
    final_dir = args.output_dir / "final"
    model.save_prompt(final_dir)
    processor.save_pretrained(final_dir)
    report = {
        "method": "static_prompt_tuning",
        "experiment": args.experiment_name,
        "dataset": display_name,
        "prompt_length": args.prompt_length,
        "visual_prompt_length": args.visual_prompt_length,
        "visual_anchor_layer": args.visual_anchor_layer,
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
    print(f"[{display_name.upper()}_PROMPT_TUNING_PASS] checkpoint={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
