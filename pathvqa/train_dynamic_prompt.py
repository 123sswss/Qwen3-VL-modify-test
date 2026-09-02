#!/usr/bin/env python3
"""Train the single-stage multimodal Dynamic Prompt method on VQA datasets."""

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
from slake.data_pipeline import SLAKEDataCollator, SLAKEDataset
from slake.dynamic_prompt_tuning import DynamicPromptTuningModel

MODEL_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "labels",
    "mmrl_gating_mask",
)
SUPPORTED_DATASETS = ("pathvqa", "slake")


def _normalize_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported Dynamic Prompt dataset={dataset_name!r}; "
            f"choices={SUPPORTED_DATASETS}"
        )
    return normalized


def _dataset_display_name(dataset_name: str) -> str:
    return "PathVQA" if dataset_name == "pathvqa" else "SLAKE"


class DynamicPromptCollator:
    def __init__(self, processor: Any, dataset_name: str = "pathvqa") -> None:
        dataset_name = _normalize_dataset_name(dataset_name)
        collator_class = (
            PathVQADataCollator if dataset_name == "pathvqa" else SLAKEDataCollator
        )
        self.base = collator_class(processor)

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
        workspace_lr: float,
        **kwargs,
    ):
        self.prompt_lr = float(prompt_lr)
        self.dynamic_lr = float(dynamic_lr)
        self.sparse_visual_lr = float(sparse_visual_lr)
        self.workspace_lr = float(workspace_lr)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.model.trainable_parameter_groups()
        prompt_parameters = groups["soft_prompt"]
        dynamic_parameters = groups["dynamic_prompt"]
        sparse_visual_parameters = groups.get("sparse_visual", [])
        workspace_parameters = groups.get("shared_workspace", [])
        active = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        grouped = (
            prompt_parameters
            + dynamic_parameters
            + sparse_visual_parameters
            + workspace_parameters
        )
        active_ids = [id(parameter) for parameter in active]
        grouped_ids = [id(parameter) for parameter in grouped]
        if (
            set(active_ids) != set(grouped_ids)
            or len(grouped_ids) != len(set(grouped_ids))
        ):
            raise RuntimeError(
                "Dynamic Prompt trainable parameters must belong to exactly one optimizer group"
            )
        optimizer_groups = []
        if dynamic_parameters:
            optimizer_groups.append(
                {
                    "params": dynamic_parameters,
                    "lr": self.dynamic_lr,
                    "weight_decay": 0.0,
                    "group_name": "dynamic_prompt",
                }
            )
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
        if workspace_parameters:
            optimizer_groups.append(
                {
                    "params": workspace_parameters,
                    "lr": self.workspace_lr,
                    "weight_decay": 0.0,
                    "group_name": "shared_workspace",
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
            f"workspace_lr={self.workspace_lr} "
            f"tensors={len(workspace_parameters)} "
            "scheduler=linear warmup_ratio=0.03"
        )
        return self.optimizer


class DynamicPromptCallback(TrainerCallback):
    def __init__(
        self,
        processor: Any,
        output_dir: Path,
        dataset_name: str = "pathvqa",
        logging_steps: int = 20,
    ) -> None:
        self.processor = processor
        self.output_dir = output_dir
        self.log_prefix = _dataset_display_name(
            _normalize_dataset_name(dataset_name)
        ).upper()
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
        parameter_groups = model.trainable_parameter_groups()
        prompt_grad = (
            model.soft_prompt.grad if model.soft_prompt is not None else None
        )
        workspace_text_anchor = getattr(model, "workspace_text_anchor", None)
        workspace_text_anchor_grad = (
            workspace_text_anchor.grad
            if workspace_text_anchor is not None
            else None
        )
        dynamic_attention = getattr(model, "dynamic_prompt", None)
        dynamic_grads = (
            [
                parameter.grad.detach().float().pow(2).sum()
                for parameter in dynamic_attention.parameters()
                if parameter.grad is not None
            ]
            if dynamic_attention is not None
            else []
        )
        dynamic_grad_norm = (
            torch.stack(dynamic_grads).sum().sqrt()
            if dynamic_grads
            else torch.tensor(0.0)
        )
        sparse_visual = getattr(model, "sparse_visual", None)
        sparse_grads = [
            parameter.grad.detach().float().pow(2).sum()
            for parameter in parameter_groups.get("sparse_visual", [])
            if parameter.grad is not None
        ]
        workspace_grads = [
            parameter.grad.detach().float().pow(2).sum()
            for parameter in parameter_groups.get("shared_workspace", [])
            if parameter.grad is not None
        ]
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
            "workspace_text_anchor_norm": (
                float(workspace_text_anchor.detach().float().norm().item())
                if workspace_text_anchor is not None
                else 0.0
            ),
            "workspace_text_anchor_grad_norm": (
                float(workspace_text_anchor_grad.detach().float().norm().item())
                if workspace_text_anchor_grad is not None
                else 0.0
            ),
            "dynamic_prompt_grad_norm": float(dynamic_grad_norm.item()),
            "sparse_visual_grad_norm": float(sparse_grad_norm.item()),
            "shared_workspace_grad_norm": (
                float(torch.stack(workspace_grads).sum().sqrt().item())
                if workspace_grads
                else 0.0
            ),
        }
        if (
            sparse_visual is not None
            and getattr(sparse_visual, "map_shared_s_to_text", False)
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
        if (
            sparse_visual is not None
            and getattr(sparse_visual, "text_anchor_tokens", 0)
        ):
            adapter_grads = [
                parameter.grad.detach().float().pow(2).sum()
                for name, parameter in sparse_visual.named_parameters()
                if name.startswith("text_anchor_") and parameter.grad is not None
            ]
            row["shared_s_text_owner_grad_norm"] = row["soft_prompt_grad_norm"]
            row["shared_s_visual_adapter_grad_norm"] = (
                float(torch.stack(adapter_grads).sum().sqrt().item())
                if adapter_grads
                else 0.0
            )
            row["shared_s_visual_to_s_gradient_blocked"] = 1.0
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
            f"[{self.log_prefix}_DYNAMIC_PROMPT_EPOCH_CHECKPOINT] epoch={epoch_id} "
            f"global_step={int(state.global_step)} saved={checkpoint_dir}"
        )
        return control


def parse_args(dataset_name: str = "pathvqa") -> argparse.Namespace:
    dataset_name = _normalize_dataset_name(dataset_name)
    display_name = _dataset_display_name(dataset_name)
    default_data_root = (
        Path("/root/autodl-tmp/dataset/pathVQA")
        if dataset_name == "pathvqa"
        else Path("/root/autodl-tmp/dataset/slake")
    )
    parser = argparse.ArgumentParser(
        description=f"Multimodal Dynamic Prompt Tuning for Qwen3-VL on {display_name}"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/model"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-name",
        default=f"{dataset_name}_dynamic_prompt_mean_ca256_len20",
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
        choices=(
            "none",
            "separate_residual",
            "direct_prompt",
            "text_owned_visual_readonly",
        ),
        default="none",
    )
    parser.add_argument("--shared-s-attention-dim", type=int, default=128)
    parser.add_argument("--shared-s-heads", type=int, default=4)
    parser.add_argument("--shared-s-visual-bottleneck-dim", type=int, default=128)
    parser.add_argument("--shared-workspace", action="store_true")
    parser.add_argument("--directional-concat-workspace", action="store_true")
    parser.add_argument(
        "--directional-visual-dynamic-write",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--directional-static-visual-write",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--directional-query-source",
        choices=("question_attention_pooling", "learned_static"),
        default="question_attention_pooling",
    )
    parser.add_argument("--workspace-tokens", type=int, default=32)
    parser.add_argument("--workspace-dim", type=int, default=1024)
    parser.add_argument("--workspace-heads", type=int, default=16)
    parser.add_argument("--workspace-ffn-dim", type=int, default=4096)
    parser.add_argument("--workspace-text-attention-dim", type=int, default=1024)
    parser.add_argument("--workspace-text-heads", type=int, default=16)
    parser.add_argument("--workspace-visual-attention-dim", type=int, default=1024)
    parser.add_argument("--workspace-visual-heads", type=int, default=16)
    parser.add_argument("--workspace-lr", type=float, default=1e-4)
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
            args.shared_s_visual_bottleneck_dim,
            args.workspace_tokens,
            args.workspace_dim,
            args.workspace_heads,
            args.workspace_ffn_dim,
            args.workspace_text_attention_dim,
            args.workspace_text_heads,
            args.workspace_visual_attention_dim,
            args.workspace_visual_heads,
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
    if args.workspace_dim % args.workspace_heads != 0:
        parser.error("--workspace-dim must be divisible by --workspace-heads")
    if args.workspace_text_attention_dim % args.workspace_text_heads != 0:
        parser.error(
            "--workspace-text-attention-dim must be divisible by its heads"
        )
    if args.workspace_visual_attention_dim % args.workspace_visual_heads != 0:
        parser.error(
            "--workspace-visual-attention-dim must be divisible by its heads"
        )
    if min(
        args.prompt_lr,
        args.dynamic_lr,
        args.sparse_visual_lr,
        args.workspace_lr,
    ) <= 0.0:
        parser.error("Learning rates must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    if args.shared_s_text_mode != "none" and not args.sparse_visual:
        parser.error("Shared-S text modes require --sparse-visual")
    if args.shared_workspace and not args.sparse_visual:
        parser.error("--shared-workspace requires --sparse-visual")
    if args.shared_workspace and args.shared_s_text_mode != "none":
        parser.error("--shared-workspace requires independent private P and visual S")
    if args.directional_concat_workspace and not args.sparse_visual:
        parser.error("--directional-concat-workspace requires --sparse-visual")
    if args.directional_concat_workspace and args.shared_workspace:
        parser.error(
            "--directional-concat-workspace and --shared-workspace are exclusive"
        )
    if args.directional_concat_workspace and args.shared_s_text_mode != "none":
        parser.error(
            "--directional-concat-workspace does not use Shared-S text modes"
        )
    if args.directional_concat_workspace and len(args.sparse_visual_anchor_layers) != 1:
        parser.error(
            "--directional-concat-workspace requires exactly one visual anchor"
        )
    if not args.directional_concat_workspace and not args.directional_visual_dynamic_write:
        parser.error(
            "--no-directional-visual-dynamic-write requires "
            "--directional-concat-workspace"
        )
    if not args.directional_concat_workspace and not args.directional_static_visual_write:
        parser.error(
            "--no-directional-static-visual-write requires "
            "--directional-concat-workspace"
        )
    if args.directional_visual_dynamic_write and not args.directional_static_visual_write:
        parser.error(
            "--directional-visual-dynamic-write requires "
            "--directional-static-visual-write"
        )
    if (
        not args.directional_concat_workspace
        and args.directional_query_source != "question_attention_pooling"
    ):
        parser.error(
            "--directional-query-source applies only to "
            "--directional-concat-workspace"
        )
    return args


def _build_train_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    processor: Any,
):
    if dataset_name == "pathvqa":
        return PathVQADataset(
            processor=processor,
            data_root=args.data_root,
            split="train",
            cache_dir=args.cache_dir,
            ce_enabled=True,
            seed=args.data_seed,
            deterministic_sampling=True,
            max_length=args.max_length,
        )
    return SLAKEDataset(
        processor=processor,
        questions_path=str(args.data_root / "train.json"),
        image_root=str(args.data_root / "imgs"),
        splits=("train",),
        ce_enabled=True,
        seed=args.data_seed,
        deterministic_sampling=True,
        max_length=args.max_length,
    )


def main(dataset_name: str = "pathvqa") -> int:
    dataset_name = _normalize_dataset_name(dataset_name)
    display_name = _dataset_display_name(dataset_name)
    log_prefix = display_name.upper()
    args = parse_args(dataset_name)
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
        shared_s_visual_bottleneck_dim=args.shared_s_visual_bottleneck_dim,
        shared_workspace=args.shared_workspace,
        workspace_tokens=args.workspace_tokens,
        workspace_dim=args.workspace_dim,
        workspace_heads=args.workspace_heads,
        workspace_ffn_dim=args.workspace_ffn_dim,
        workspace_text_attention_dim=args.workspace_text_attention_dim,
        workspace_text_heads=args.workspace_text_heads,
        workspace_visual_attention_dim=args.workspace_visual_attention_dim,
        workspace_visual_heads=args.workspace_visual_heads,
        directional_concat_workspace=args.directional_concat_workspace,
        directional_visual_dynamic_write=args.directional_visual_dynamic_write,
        directional_static_visual_write=args.directional_static_visual_write,
        directional_query_source=args.directional_query_source,
    )
    dataset = _build_train_dataset(dataset_name, args, processor)
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
        f"[{log_prefix}_DYNAMIC_PROMPT_CONFIG] "
        f"requested_prompt_length={args.prompt_length} "
        f"effective_prompt_length={model.prompt_length} "
        f"hidden_size={model.hidden_size} "
        f"attention_dim={args.attention_dim if model.dynamic_prompt is not None else 0} "
        f"heads={args.attention_heads if model.dynamic_prompt is not None else 0} "
        f"parameters={counts} total={trainable} "
        f"epochs={args.epochs} prompt_lr={args.prompt_lr} "
        f"dynamic_lr={args.dynamic_lr} seed={args.seed} data_seed={args.data_seed} "
        f"sparse_visual={args.sparse_visual} "
        f"sparse_anchors={list(args.sparse_visual_anchor_layers) if args.sparse_visual else []} "
        f"sparse_rep_tokens={args.sparse_visual_rep_tokens if args.sparse_visual and (not args.directional_concat_workspace or args.directional_static_visual_write) else 0} "
        f"sparse_attention_dim={args.sparse_visual_attention_dim if args.sparse_visual else 0} "
        f"sparse_heads={args.sparse_visual_heads if args.sparse_visual else 0} "
        f"sparse_injection={('directional_concat_single_pass_insert_strip' if args.directional_static_visual_write else 'directional_read_only_single_pass_hook') if args.directional_concat_workspace else ('single_pass_insert_strip' if args.sparse_visual else 'none')} "
        f"shared_s_text_mode={args.shared_s_text_mode} "
        f"shared_s_text_merger={'main_visual_merger' if args.shared_s_text_mode in ('separate_residual', 'direct_prompt') else 'none'} "
        f"shared_s_attention={args.shared_s_attention_dim}x{args.shared_s_heads} "
        f"shared_s_visual_bottleneck={args.shared_s_visual_bottleneck_dim} "
        f"shared_s_gradient_policy={'text_write_visual_readonly' if args.shared_s_text_mode == 'text_owned_visual_readonly' else 'joint_or_disabled'} "
        f"shared_workspace={args.shared_workspace} "
        f"directional_concat_workspace={args.directional_concat_workspace} "
        f"directional_visual_dynamic_write={args.directional_visual_dynamic_write if args.directional_concat_workspace else False} "
        f"directional_static_visual_write={args.directional_static_visual_write if args.directional_concat_workspace else False} "
        f"directional_query_source={args.directional_query_source if args.directional_concat_workspace else 'none'} "
        f"workspace_tokens={args.workspace_tokens if args.shared_workspace else 0} "
        f"directional_workspace_tokens={args.workspace_tokens if args.directional_concat_workspace else 0} "
        f"workspace_dim={args.workspace_dim if (args.shared_workspace or args.directional_concat_workspace) else 0} "
        f"workspace_blocks={len(args.sparse_visual_anchor_layers) if args.shared_workspace else 0} "
        f"workspace_heads={args.workspace_heads if (args.shared_workspace or args.directional_concat_workspace) else 0} "
        f"workspace_ffn_dim={args.workspace_ffn_dim if args.shared_workspace else 0} "
        f"workspace_text_ca={args.workspace_text_attention_dim if args.shared_workspace else 0}x{args.workspace_text_heads if args.shared_workspace else 0} "
        f"workspace_visual_ca={args.workspace_visual_attention_dim if args.shared_workspace else 0}x{args.workspace_visual_heads if args.shared_workspace else 0} "
        f"workspace_lr={args.workspace_lr} "
        f"train_soft_prompt={model.soft_prompt is not None} "
        f"text_workspace_anchor_tokens={model.workspace_prompt_length} "
        "pretrained_prompt_checkpoint=False stage1=False"
    )

    callback = DynamicPromptCallback(
        processor,
        args.output_dir,
        dataset_name=dataset_name,
    )
    trainer = DynamicPromptTrainer(
        model=model,
        prompt_lr=args.prompt_lr,
        dynamic_lr=args.dynamic_lr,
        sparse_visual_lr=args.sparse_visual_lr,
        workspace_lr=args.workspace_lr,
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
        data_collator=DynamicPromptCollator(processor, dataset_name=dataset_name),
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
        "dataset": display_name,
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
            "main_visual_merger"
            if args.shared_s_text_mode in ("separate_residual", "direct_prompt")
            else None
        ),
        "shared_s_attention_dim": args.shared_s_attention_dim,
        "shared_s_attention_heads": args.shared_s_heads,
        "shared_s_visual_bottleneck_dim": args.shared_s_visual_bottleneck_dim,
        "shared_s_gradient_policy": (
            "text_write_visual_readonly"
            if args.shared_s_text_mode == "text_owned_visual_readonly"
            else None
        ),
        "shared_workspace": (
            {
                "tokens": args.workspace_tokens,
                "dim": args.workspace_dim,
                "heads": args.workspace_heads,
                "ffn_dim": args.workspace_ffn_dim,
                "text_attention_dim": args.workspace_text_attention_dim,
                "text_attention_heads": args.workspace_text_heads,
                "visual_attention_dim": args.workspace_visual_attention_dim,
                "visual_attention_heads": args.workspace_visual_heads,
                "learning_rate": args.workspace_lr,
                "memory": "full_visual_plus_full_question_tokens",
                "residual_heads_zero_initialized": True,
            }
            if args.shared_workspace
            else None
        ),
        "directional_concat_workspace": (
            {
                "tokens": args.workspace_tokens,
                "dim": args.workspace_dim,
                "heads": args.workspace_heads,
                "anchor_layer": int(args.sparse_visual_anchor_layers[0]),
                "private_visual_prompt_tokens": (
                    args.sparse_visual_rep_tokens
                    if args.directional_static_visual_write
                    else 0
                ),
                "private_text_prompt_tokens": args.prompt_length,
                "text_workspace_anchor_tokens": args.workspace_tokens,
                "query": args.directional_query_source,
                "memory": "full_visual_tokens",
                "fusion": "text_query_visual_kv_cross_attention",
                "output_interface": (
                    "anchored_token_concat"
                    if args.directional_static_visual_write
                    else "text_prompt_only_with_visual_read_hook"
                ),
                "visual_dynamic_write": args.directional_visual_dynamic_write,
                "static_visual_write": args.directional_static_visual_write,
                "visual_output_interface": (
                    (
                        "dynamic_anchor_token_concat"
                        if args.directional_visual_dynamic_write
                        else "static_anchor_token_concat"
                    )
                    if args.directional_static_visual_write
                    else "read_only_layer17_hook"
                ),
                "text_output_interface": "dynamic_anchor_token_concat",
                "text_dynamic_projection_zero_initialized": True,
                "visual_dynamic_projection_zero_initialized": (
                    args.directional_visual_dynamic_write
                ),
                "zero_init_dynamic_projections": True,
                "learning_rate": args.workspace_lr,
            }
            if args.directional_concat_workspace
            else None
        ),
        "train_soft_prompt": model.soft_prompt is not None,
        "sparse_visual": (
            {
                "anchor_layers": list(args.sparse_visual_anchor_layers),
                "rep_token_count": (
                    args.sparse_visual_rep_tokens
                    if (
                        not args.directional_concat_workspace
                        or args.directional_static_visual_write
                    )
                    else 0
                ),
                "attention_dim": (
                    0
                    if args.directional_concat_workspace
                    else args.sparse_visual_attention_dim
                ),
                "attention_heads": (
                    0
                    if args.directional_concat_workspace
                    else args.sparse_visual_heads
                ),
                "injection_mode": (
                    (
                        "directional_concat_single_pass_insert_strip"
                        if args.directional_static_visual_write
                        else "directional_read_only_single_pass_hook"
                    )
                    if args.directional_concat_workspace
                    else "single_pass_insert_strip"
                ),
                "directional_concat_workspace": args.directional_concat_workspace,
                "shared_s_text_merger": (
                    "main_visual_merger"
                    if args.shared_s_text_mode
                    in ("separate_residual", "direct_prompt")
                    else None
                ),
                "text_owned_s_visual_readonly": (
                    args.shared_s_text_mode == "text_owned_visual_readonly"
                ),
                "text_anchor_bottleneck_dim": args.shared_s_visual_bottleneck_dim,
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
    print(f"[{log_prefix}_DYNAMIC_PROMPT_PASS] checkpoint={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
