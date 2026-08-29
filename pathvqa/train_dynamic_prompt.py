#!/usr/bin/env python3
"""Train the single-stage multimodal Dynamic Prompt method on PathVQA."""

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
from slake.dynamic_prompt_tuning import DynamicPromptTuningModel

MODEL_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "labels",
    "mmrl_gating_mask",
)


class DynamicPromptCollator:
    def __init__(self, processor: Any) -> None:
        self.base = PathVQADataCollator(processor)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base(features)
        return {
            key: batch[key]
            for key in MODEL_BATCH_KEYS
            if key in batch and batch[key] is not None
        }


class DynamicPromptTrainer(Trainer):
    def __init__(
        self,
        *args,
        prompt_lr: float,
        dynamic_lr: float,
        sparse_visual_lr: float,
        **kwargs,
    ):
        self.prompt_lr = float(prompt_lr)
        self.dynamic_lr = float(dynamic_lr)
        self.sparse_visual_lr = float(sparse_visual_lr)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.model.trainable_parameter_groups()
        prompt_parameters = groups["soft_prompt"]
        dynamic_parameters = groups["dynamic_prompt"]
        sparse_visual_parameters = groups.get("sparse_visual", [])
        active = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        grouped = prompt_parameters + dynamic_parameters + sparse_visual_parameters
        if {id(parameter) for parameter in active} != {
            id(parameter) for parameter in grouped
        }:
            raise RuntimeError(
                "Dynamic Prompt trainable parameters must belong to exactly one optimizer group"
            )
        optimizer_groups = [
            {
                "params": dynamic_parameters,
                "lr": self.dynamic_lr,
                "weight_decay": 0.0,
                "group_name": "dynamic_prompt",
            },
        ]
        if prompt_parameters:
            optimizer_groups.insert(
                0,
                {
                    "params": prompt_parameters,
                    "lr": self.prompt_lr,
                    "weight_decay": 0.0,
                    "group_name": "soft_prompt",
                },
            )
        if sparse_visual_parameters:
            optimizer_groups.append(
                {
                    "params": sparse_visual_parameters,
                    "lr": self.sparse_visual_lr,
                    "weight_decay": 0.0,
                    "group_name": "sparse_visual",
                }
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        print(
            "[DYNAMIC_PROMPT_OPTIMIZER] "
            f"soft_prompt_lr={self.prompt_lr} tensors={len(prompt_parameters)} "
            f"dynamic_prompt_lr={self.dynamic_lr} tensors={len(dynamic_parameters)} "
            f"sparse_visual_lr={self.sparse_visual_lr} "
            f"tensors={len(sparse_visual_parameters)} "
            "scheduler=linear warmup_ratio=0.03"
        )
        return self.optimizer


class DynamicPromptCallback(TrainerCallback):
    def __init__(
        self,
        processor: Any,
        output_dir: Path,
        logging_steps: int = 20,
    ) -> None:
        self.processor = processor
        self.output_dir = output_dir
        self.logging_steps = int(logging_steps)
        self.completed_epochs = set()
        self.diagnostics_path = output_dir / "dynamic_prompt_diagnostics.jsonl"

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        step = int(state.global_step)
        if step % self.logging_steps != 0:
            return control
        model = kwargs["model"]
        prompt_grad = (
            model.soft_prompt.grad if model.soft_prompt is not None else None
        )
        dynamic_grads = [
            parameter.grad.detach().float().pow(2).sum()
            for parameter in model.dynamic_prompt.parameters()
            if parameter.grad is not None
        ]
        dynamic_grad_norm = (
            torch.stack(dynamic_grads).sum().sqrt()
            if dynamic_grads
            else torch.tensor(0.0)
        )
        sparse_visual = getattr(model, "sparse_visual", None)
        sparse_grads = (
            [
                parameter.grad.detach().float().pow(2).sum()
                for parameter in sparse_visual.parameters()
                if parameter.grad is not None
            ]
            if sparse_visual is not None
            else []
        )
        sparse_grad_norm = (
            torch.stack(sparse_grads).sum().sqrt()
            if sparse_grads
            else torch.tensor(0.0)
        )
        row = {
            "step": step,
            "epoch": float(state.epoch or 0.0),
            "soft_prompt_norm": (
                float(model.soft_prompt.detach().float().norm().item())
                if model.soft_prompt is not None
                else 0.0
            ),
            "soft_prompt_grad_norm": (
                float(prompt_grad.detach().float().norm().item())
                if prompt_grad is not None
                else 0.0
            ),
            "dynamic_prompt_grad_norm": float(dynamic_grad_norm.item()),
            "sparse_visual_grad_norm": float(sparse_grad_norm.item()),
        }
        if (
            sparse_visual is not None
            and sparse_visual.map_shared_s_to_text
        ):
            shared_rep_grad = sparse_visual.shared_rep.grad
            row["shared_s_parameter_grad_norm"] = (
                float(shared_rep_grad.detach().float().norm().item())
                if shared_rep_grad is not None
                else 0.0
            )
            row["shared_s_text_prompt_grad_norm"] = float(
                sparse_visual.shared_s_text_prompt_grad_norm().item()
            )
        shared_attention = getattr(model, "shared_s_prompt_attention", None)
        shared_attention_grads = (
            [
                parameter.grad.detach().float().pow(2).sum()
                for parameter in shared_attention.parameters()
                if parameter.grad is not None
            ]
            if shared_attention is not None
            else []
        )
        row["shared_s_prompt_attention_grad_norm"] = (
            float(torch.stack(shared_attention_grads).sum().sqrt().item())
            if shared_attention_grads
            else 0.0
        )
        for key, value in model.debug_context.items():
            row[key] = float(value.detach().float().item())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[DYNAMIC_PROMPT_DIAGNOSTICS] " + json.dumps(row, ensure_ascii=False))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        epoch_id = max(1, int(round(float(state.epoch or 0.0))))
        if epoch_id in self.completed_epochs:
            return control
        self.completed_epochs.add(epoch_id)
        checkpoint_dir = self.output_dir / "checkpoints" / f"epoch_{epoch_id}"
        kwargs["model"].save_dynamic_prompt(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
        print(
            f"[PATHVQA_DYNAMIC_PROMPT_EPOCH_CHECKPOINT] epoch={epoch_id} "
            f"global_step={int(state.global_step)} saved={checkpoint_dir}"
        )
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal Dynamic Prompt Tuning for Qwen3-VL on PathVQA"
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
    parser.add_argument(
        "--experiment-name",
        default="pathvqa_dynamic_prompt_mean_ca256_len20",
    )
    parser.add_argument("--prompt-length", type=int, default=20)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--prompt-lr", type=float, default=0.3)
    parser.add_argument("--dynamic-lr", type=float, default=3e-4)
    parser.add_argument("--sparse-visual", action="store_true")
    parser.add_argument(
        "--sparse-visual-anchor-layers",
        type=int,
        nargs="+",
        default=(5, 11, 17),
    )
    parser.add_argument("--sparse-visual-rep-tokens", type=int, default=8)
    parser.add_argument("--sparse-visual-attention-dim", type=int, default=128)
    parser.add_argument("--sparse-visual-heads", type=int, default=4)
    parser.add_argument("--sparse-visual-lr", type=float, default=3e-5)
    parser.add_argument(
        "--shared-s-text-mode",
        choices=("none", "separate_residual", "direct_prompt"),
        default="none",
    )
    parser.add_argument("--shared-s-attention-dim", type=int, default=128)
    parser.add_argument("--shared-s-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--expected-trainable-parameters", type=int)
    args = parser.parse_args()
    if (
        min(
            args.prompt_length,
            args.attention_dim,
            args.attention_heads,
            args.epochs,
            args.batch_size,
            args.gradient_accumulation,
            args.sparse_visual_rep_tokens,
            args.sparse_visual_attention_dim,
            args.sparse_visual_heads,
            args.shared_s_attention_dim,
            args.shared_s_heads,
        )
        < 1
    ):
        parser.error("Prompt dimensions, epochs, and batch settings must be positive")
    if args.attention_dim % args.attention_heads != 0:
        parser.error("--attention-dim must be divisible by --attention-heads")
    if args.sparse_visual_attention_dim % args.sparse_visual_heads != 0:
        parser.error(
            "--sparse-visual-attention-dim must be divisible by --sparse-visual-heads"
        )
    if args.shared_s_attention_dim % args.shared_s_heads != 0:
        parser.error("--shared-s-attention-dim must be divisible by --shared-s-heads")
    if min(args.prompt_lr, args.dynamic_lr, args.sparse_visual_lr) <= 0.0:
        parser.error("Learning rates must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    if args.shared_s_text_mode != "none" and not args.sparse_visual:
        parser.error("Shared-S text modes require --sparse-visual")
    return args


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    model = DynamicPromptTuningModel(
        base_model,
        tokenizer=processor.tokenizer,
        prompt_length=args.prompt_length,
        init_seed=args.seed,
        attention_dim=args.attention_dim,
        num_heads=args.attention_heads,
        sparse_visual_anchor_layers=(
            tuple(args.sparse_visual_anchor_layers) if args.sparse_visual else None
        ),
        sparse_visual_rep_tokens=args.sparse_visual_rep_tokens,
        sparse_visual_attention_dim=args.sparse_visual_attention_dim,
        sparse_visual_heads=args.sparse_visual_heads,
        shared_s_text_mode=args.shared_s_text_mode,
        shared_s_attention_dim=args.shared_s_attention_dim,
        shared_s_heads=args.shared_s_heads,
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
    groups = model.trainable_parameter_groups()
    counts = {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in groups.items()
    }
    trainable = sum(counts.values())
    actual_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != actual_trainable:
        raise RuntimeError(
            f"Trainable parameter grouping failed: grouped={trainable} actual={actual_trainable}"
        )
    if (
        args.expected_trainable_parameters is not None
        and trainable != args.expected_trainable_parameters
    ):
        raise RuntimeError(
            "Trainable parameter audit failed: "
            f"expected={args.expected_trainable_parameters} actual={trainable}"
        )
    print(
        "[PATHVQA_DYNAMIC_PROMPT_CONFIG] "
        f"requested_prompt_length={args.prompt_length} "
        f"effective_prompt_length={model.prompt_length} "
        f"hidden_size={model.dynamic_prompt.hidden_size} "
        f"attention_dim={args.attention_dim} heads={args.attention_heads} "
        f"parameters={counts} total={trainable} "
        f"epochs={args.epochs} prompt_lr={args.prompt_lr} "
        f"dynamic_lr={args.dynamic_lr} seed={args.seed} data_seed={args.data_seed} "
        f"sparse_visual={args.sparse_visual} "
        f"sparse_anchors={list(args.sparse_visual_anchor_layers) if args.sparse_visual else []} "
        f"sparse_rep_tokens={args.sparse_visual_rep_tokens if args.sparse_visual else 0} "
        f"sparse_attention_dim={args.sparse_visual_attention_dim if args.sparse_visual else 0} "
        f"sparse_heads={args.sparse_visual_heads if args.sparse_visual else 0} "
        f"sparse_injection={'single_pass_insert_strip' if args.sparse_visual else 'none'} "
        f"shared_s_text_mode={args.shared_s_text_mode} "
        f"shared_s_text_merger={'main_visual_merger' if args.shared_s_text_mode != 'none' else 'none'} "
        f"shared_s_attention={args.shared_s_attention_dim}x{args.shared_s_heads} "
        f"train_soft_prompt={model.soft_prompt is not None} "
        "pretrained_prompt_checkpoint=False stage1=False"
    )

    callback = DynamicPromptCallback(processor, args.output_dir)
    trainer = DynamicPromptTrainer(
        model=model,
        prompt_lr=args.prompt_lr,
        dynamic_lr=args.dynamic_lr,
        sparse_visual_lr=args.sparse_visual_lr,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.prompt_lr,
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
        data_collator=DynamicPromptCollator(processor),
        processing_class=processor,
        callbacks=[callback],
    )
    result = trainer.train()
    final_dir = args.output_dir / "final"
    model.save_dynamic_prompt(final_dir)
    processor.save_pretrained(final_dir)
    report = {
        "method": "dynamic_multimodal_prompt_tuning",
        "experiment": args.experiment_name,
        "dataset": "PathVQA",
        "requested_prompt_length": args.prompt_length,
        "effective_prompt_length": model.prompt_length,
        "attention_dim": args.attention_dim,
        "attention_heads": args.attention_heads,
        "trainable_parameters": counts,
        "total_trainable_parameters": trainable,
        "prompt_learning_rate": args.prompt_lr,
        "dynamic_learning_rate": args.dynamic_lr,
        "sparse_visual_learning_rate": args.sparse_visual_lr,
        "shared_s_text_mode": args.shared_s_text_mode,
        "shared_s_text_merger": (
            "main_visual_merger" if args.shared_s_text_mode != "none" else None
        ),
        "shared_s_attention_dim": args.shared_s_attention_dim,
        "shared_s_attention_heads": args.shared_s_heads,
        "train_soft_prompt": model.soft_prompt is not None,
        "sparse_visual": (
            {
                "anchor_layers": list(args.sparse_visual_anchor_layers),
                "rep_token_count": args.sparse_visual_rep_tokens,
                "attention_dim": args.sparse_visual_attention_dim,
                "attention_heads": args.sparse_visual_heads,
                "injection_mode": "single_pass_insert_strip",
                "shared_s_text_merger": (
                    "main_visual_merger"
                    if args.shared_s_text_mode != "none"
                    else None
                ),
            }
            if args.sparse_visual
            else None
        ),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "train_metrics": result.metrics,
    }
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[PATHVQA_DYNAMIC_PROMPT_PASS] checkpoint={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
