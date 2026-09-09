#!/usr/bin/env python3
"""Train the CoCoOp-style conditional Prompt baseline on PathVQA."""

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
    TrainerCallback,
    TrainingArguments,
)

from pathvqa.data_pipeline import PathVQADataset
from pathvqa.train_prompt_tuning import PromptTuningCollator
from slake.cocoop_prompt_tuning import CoCoOpStylePromptTuningModel


class CoCoOpTrainer(Trainer):
    def __init__(
        self,
        *args,
        prompt_lr: float,
        meta_net_lr: float,
        **kwargs,
    ) -> None:
        self.prompt_lr = float(prompt_lr)
        self.meta_net_lr = float(meta_net_lr)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.model.trainable_parameter_groups()
        grouped = groups["soft_prompt"] + groups["meta_net"]
        active = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if {id(parameter) for parameter in grouped} != {
            id(parameter) for parameter in active
        }:
            raise RuntimeError(
                "CoCoOp-style optimizer groups do not match trainable parameters"
            )
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": groups["soft_prompt"],
                    "lr": self.prompt_lr,
                    "weight_decay": 0.0,
                    "group_name": "soft_prompt",
                },
                {
                    "params": groups["meta_net"],
                    "lr": self.meta_net_lr,
                    "weight_decay": 0.0,
                    "group_name": "meta_net",
                },
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        print(
            "[COCOOP_STYLE_OPTIMIZER] "
            f"prompt_lr={self.prompt_lr} meta_net_lr={self.meta_net_lr} "
            "weight_decay=0 scheduler=linear warmup_ratio=0.03"
        )
        return self.optimizer


class CoCoOpCallback(TrainerCallback):
    def __init__(self, processor, output_dir: Path) -> None:
        self.processor = processor
        self.output_dir = output_dir
        self.completed_epochs = set()
        self.diagnostics_path = output_dir / "cocoop_diagnostics.jsonl"

    @staticmethod
    def _gradient_norm(parameters) -> float:
        squares = [
            parameter.grad.detach().float().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        return float(torch.stack(squares).sum().sqrt()) if squares else 0.0

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or int(state.global_step) % 20:
            return control
        model = kwargs["model"]
        groups = model.trainable_parameter_groups()
        row = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "soft_prompt_grad_norm": self._gradient_norm(groups["soft_prompt"]),
            "meta_net_grad_norm": self._gradient_norm(groups["meta_net"]),
            "soft_prompt_norm": float(model.soft_prompt.detach().float().norm()),
        }
        row.update(
            {
                key: float(value.detach().float())
                for key, value in model.debug_context.items()
            }
        )
        static_norm = row.get("cocoop_static_prompt_norm_mean", 0.0)
        bias_norm = row.get("cocoop_prompt_bias_norm_mean", 0.0)
        row["cocoop_bias_to_static_ratio"] = (
            bias_norm / static_norm if static_norm > 0.0 else 0.0
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[COCOOP_STYLE_DIAGNOSTICS] " + json.dumps(row, ensure_ascii=False))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        epoch = max(1, int(round(float(state.epoch or 0.0))))
        if epoch in self.completed_epochs:
            return control
        self.completed_epochs.add(epoch)
        checkpoint = self.output_dir / "checkpoints" / f"epoch_{epoch}"
        kwargs["model"].save_cocoop(checkpoint)
        self.processor.save_pretrained(checkpoint)
        print(
            f"[PATHVQA_COCOOP_STYLE_EPOCH_CHECKPOINT] epoch={epoch} "
            f"global_step={int(state.global_step)} saved={checkpoint}"
        )
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoCoOp-style conditional Prompt baseline on PathVQA"
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-tmp/model")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/pathVQA"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--prompt-length", type=int, default=20)
    parser.add_argument("--bottleneck-dim", type=int, default=160)
    parser.add_argument("--prompt-learning-rate", type=float, default=0.3)
    parser.add_argument("--meta-net-learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--expected-trainable-parameters", type=int, default=873120)
    args = parser.parse_args()
    if args.prompt_length < 1 or args.bottleneck_dim < 1 or args.epochs < 1:
        parser.error("Prompt length, bottleneck dimension, and epochs must be positive")
    if args.prompt_learning_rate <= 0.0 or args.meta_net_learning_rate <= 0.0:
        parser.error("Learning rates must be positive")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("Batch size and gradient accumulation must be positive")
    if args.dataloader_workers < 0:
        parser.error("Dataloader workers must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    model = CoCoOpStylePromptTuningModel(
        base_model,
        tokenizer=processor.tokenizer,
        prompt_length=args.prompt_length,
        bottleneck_dim=args.bottleneck_dim,
        init_seed=args.seed,
    )
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
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    hidden_size = int(model.soft_prompt.shape[-1])
    expected_formula = (
        args.prompt_length * hidden_size
        + hidden_size * args.bottleneck_dim
        + args.bottleneck_dim
        + args.bottleneck_dim * hidden_size
        + hidden_size
    )
    if trainable != expected_formula or trainable != args.expected_trainable_parameters:
        raise RuntimeError(
            "CoCoOp-style trainable parameter audit failed: "
            f"formula={expected_formula} expected={args.expected_trainable_parameters} "
            f"actual={trainable}"
        )
    print(
        "[PATHVQA_COCOOP_STYLE_CONFIG] "
        f"experiment={args.experiment_name} prompt_length={args.prompt_length} "
        f"hidden_size={hidden_size} bottleneck={args.bottleneck_dim} "
        f"trainable={trainable} prompt_lr={args.prompt_learning_rate} "
        f"meta_net_lr={args.meta_net_learning_rate} seed={args.seed} "
        f"data_seed={args.data_seed} epochs={args.epochs} "
        "question_access=false visual_source=post_merger_llm_token_mean "
        "prompt_placement=before_full_chat"
    )

    callback = CoCoOpCallback(processor, args.output_dir)
    trainer = CoCoOpTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.prompt_learning_rate,
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
        prompt_lr=args.prompt_learning_rate,
        meta_net_lr=args.meta_net_learning_rate,
    )
    result = trainer.train()
    final_dir = args.output_dir / "final"
    model.save_cocoop(final_dir)
    processor.save_pretrained(final_dir)
    report = {
        "method": "cocoop_style_conditional_prompt_tuning",
        "approximation": "generative_mllm_unified_protocol",
        "experiment": args.experiment_name,
        "dataset": "PathVQA",
        "prompt_length": args.prompt_length,
        "bottleneck_dim": args.bottleneck_dim,
        "trainable_parameters": trainable,
        "prompt_learning_rate": args.prompt_learning_rate,
        "meta_net_learning_rate": args.meta_net_learning_rate,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "question_access": False,
        "visual_source": "post_merger_llm_visual_token_mean",
        "prompt_placement": "before_full_chat",
        "train_metrics": result.metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[PATHVQA_COCOOP_STYLE_PASS] checkpoint={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
