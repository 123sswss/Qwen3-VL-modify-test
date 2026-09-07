#!/usr/bin/env python3
"""Train the GRASP reproduction under the unified VQA protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments

from pathvqa.train_dynamic_prompt import (
    DynamicPromptCollator,
    _build_train_dataset,
    _dataset_display_name,
    _normalize_dataset_name,
)
from slake.grasp_prompt_tuning import GRASPPromptTuningModel


class GRASPTrainer(Trainer):
    def __init__(self, *args, grasp_lr: float, **kwargs):
        self.grasp_lr = float(grasp_lr)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.model.trainable_parameter_groups()
        parameters = groups["prompt_prototypes"] + groups["projections"]
        active = [p for p in self.model.parameters() if p.requires_grad]
        if {id(p) for p in parameters} != {id(p) for p in active}:
            raise RuntimeError("GRASP optimizer grouping does not match trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters, lr=self.grasp_lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
        )
        print(
            "[GRASP_OPTIMIZER] "
            f"lr={self.grasp_lr} weight_decay=0.01 tensors={len(parameters)} "
            "scheduler=linear warmup_ratio=0.1"
        )
        return self.optimizer


class GRASPCallback(TrainerCallback):
    def __init__(self, processor, output_dir: Path, dataset_name: str):
        self.processor = processor
        self.output_dir = output_dir
        self.prefix = _dataset_display_name(dataset_name).upper()
        self.completed_epochs = set()
        self.diagnostics_path = output_dir / "grasp_diagnostics.jsonl"

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or int(state.global_step) % 20:
            return control
        model = kwargs["model"]
        row = {"step": int(state.global_step), "epoch": float(state.epoch or 0)}
        for name, parameters in model.trainable_parameter_groups().items():
            gradients = [
                p.grad.detach().float().square().sum()
                for p in parameters
                if p.grad is not None
            ]
            row[f"{name}_grad_norm"] = (
                float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0
            )
        row["prompt_prototypes_norm"] = float(model.prompt_prototypes.detach().float().norm())
        for key, value in model.debug_context.items():
            row[key] = float(value.detach().float())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[GRASP_DIAGNOSTICS] " + json.dumps(row, ensure_ascii=False))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        epoch = max(1, int(round(float(state.epoch or 0))))
        if epoch in self.completed_epochs:
            return control
        self.completed_epochs.add(epoch)
        checkpoint = self.output_dir / "checkpoints" / f"epoch_{epoch}"
        kwargs["model"].save_grasp(checkpoint)
        self.processor.save_pretrained(checkpoint)
        print(
            f"[{self.prefix}_GRASP_EPOCH_CHECKPOINT] epoch={epoch} "
            f"global_step={int(state.global_step)} saved={checkpoint}"
        )
        return control


def parse_args(dataset_name: str) -> argparse.Namespace:
    default_root = {
        "pathvqa": Path("/root/autodl-tmp/dataset/pathVQA"),
        "slake": Path("/root/autodl-tmp/dataset/slake"),
        "electrical": Path("/root/autodl-tmp/dataset"),
    }[dataset_name]
    parser = argparse.ArgumentParser(description=f"GRASP reproduction on {dataset_name}")
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--bottleneck-dim", type=int, default=512)
    parser.add_argument("--prompt-init-std", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--expected-trainable-parameters", type=int)
    args = parser.parse_args()
    if min(
        args.blocks, args.bottleneck_dim, args.prompt_init_std, args.learning_rate,
        args.epochs, args.batch_size, args.gradient_accumulation
    ) <= 0:
        parser.error("GRASP dimensions, rates, and training settings must be positive")
    if int(args.blocks**0.5) ** 2 != args.blocks:
        parser.error("--blocks must be a perfect square")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    return args


def main(dataset_name: str = "pathvqa") -> int:
    dataset_name = _normalize_dataset_name(dataset_name)
    args = parse_args(dataset_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_model.config.use_cache = False
    model = GRASPPromptTuningModel(
        base_model,
        tokenizer=processor.tokenizer,
        block_count=args.blocks,
        bottleneck_dim=args.bottleneck_dim,
        prompt_init_std=args.prompt_init_std,
        init_seed=args.seed,
    )
    dataset = _build_train_dataset(dataset_name, args, processor)
    groups = model.trainable_parameter_groups()
    counts = {name: sum(p.numel() for p in parameters) for name, parameters in groups.items()}
    trainable = sum(counts.values())
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != actual:
        raise RuntimeError(f"GRASP parameter audit failed: grouped={trainable} actual={actual}")
    if args.expected_trainable_parameters is not None and trainable != args.expected_trainable_parameters:
        raise RuntimeError(
            f"GRASP expected parameter count {args.expected_trainable_parameters}, got {trainable}"
        )
    display_name = _dataset_display_name(dataset_name)
    print(
        f"[{display_name.upper()}_GRASP_CONFIG] experiment={args.experiment_name} "
        f"blocks={args.blocks} bottleneck={args.bottleneck_dim} alpha=1.5 "
        f"question=frozen_llm_last_hidden_mean visual=post_merger_grid "
        f"prompt_tokens=1 parameters={counts} total={trainable} "
        f"lr={args.learning_rate} weight_decay=0.01 warmup=0.1 "
        f"epochs={args.epochs} seed={args.seed} data_seed={args.data_seed}"
    )
    trainer = GRASPTrainer(
        model=model,
        grasp_lr=args.learning_rate,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
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
        data_collator=DynamicPromptCollator(processor, dataset_name=dataset_name),
        processing_class=processor,
        callbacks=[GRASPCallback(processor, args.output_dir, dataset_name)],
    )
    result = trainer.train()
    final_dir = args.output_dir / "final"
    model.save_grasp(final_dir)
    processor.save_pretrained(final_dir)
    report = {
        "method": "grasp_reimplementation",
        "source": "arXiv:2601.17089v1",
        "experiment": args.experiment_name,
        "dataset": display_name,
        "blocks": args.blocks,
        "bottleneck_dim": args.bottleneck_dim,
        "entmax_alpha": 1.5,
        "question_encoder": "frozen_llm_last_hidden_mean",
        "visual_source": "post_merger_grid",
        "trainable_parameters": counts,
        "total_trainable_parameters": trainable,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "epochs": args.epochs,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "train_metrics": result.metrics,
    }
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[{display_name.upper()}_GRASP_PASS] checkpoint={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
