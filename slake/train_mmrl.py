#!/usr/bin/env python3
"""Thin SLAKE entry point for the existing two-stage MMRL trainer."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Dict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset
from transformers import Qwen3VLForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
for import_root in (REPO_ROOT, TRAIN_DIR):
    import_text = str(import_root)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from live_epoch_eval import _LiveModelInterface
from data_pipeline import FourViewMMRLDataset
from mmrl_checkpoint import save_mmrl_checkpoint
from slake.data_pipeline import (
    SLAKEDataCollator,
    SLAKEDataset,
    SLAKEStage1Dataset,
)
from train_stages import build_model_and_processor, run_stage


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/dataset/SLAKE")
DEFAULT_MODEL_PATH = Path("/root/autodl-tmp/model")
TRAIN_MANIFEST_CANDIDATES = (
    "slake_train.json",
    "train.json",
    "train_questions.json",
)
DEFAULT_GENERAL_JSON = (
    Path("/root/autodl-tmp/dataset/llava_instruct_150k.json"),
    Path("/root/autodl-tmp/dataset/gen_test.json"),
    Path("/root/autodl-tmp/dataset/conversation_58k.json"),
)
DEFAULT_GENERAL_IMAGE_ROOTS = (
    Path("/root/autodl-tmp/dataset/gen/train2017"),
    Path("/root/autodl-tmp/dataset/gen/val2017"),
)


QUERY_ARCHITECTURES = (
    "layer_mlp_dynamic_query_static_kv",
    "layer_mlp_post_cross",
    "shared_direct_post_cross",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[SLAKE_REPRODUCIBILITY] seed={seed}")


def resolve_train_manifest(
    data_root: Path,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SLAKE train manifest not found: {path}")
        return path

    matches = [
        (data_root / candidate).resolve()
        for candidate in TRAIN_MANIFEST_CANDIDATES
        if (data_root / candidate).is_file()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "No SLAKE train manifest found. Pass --questions or create one of "
            f"{TRAIN_MANIFEST_CANDIDATES} under {data_root}."
        )
    raise RuntimeError(
        "Multiple SLAKE train manifests were found; pass --questions explicitly: "
        f"{matches}"
    )


def build_experiment_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "rp_space_length": args.rp_space_length,
        "memory_query_count": args.memory_query_count,
        "memory_attention_dim": args.memory_attention_dim,
        "projector_hidden_dim": args.projector_hidden_dim,
        "cross_attention_heads": args.cross_attention_heads,
        "query_architecture": args.query_architecture,
        "same_init_layer_projectors": args.same_init_layer_projectors,
        "mmrl_relation_loss_weight": args.relation_weight,
        "mmrl_cross_relation_loss_weight": args.cross_relation_weight,
        "mmrl_relation_max_tokens": args.relation_max_tokens,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "enable_deepstack_mmrl_residual": (
            args.enable_deepstack_mmrl_residual
        ),
        "deepstack_mmrl_residual_scale": (
            1.0 if args.enable_deepstack_mmrl_residual else 0.0
        ),
        "ablate_visual_gate": False,
        "ablate_direct_learnable_rep": False,
        "stage3_learning_rate": args.mmrl_lr,
        "stage3_mmrl_learning_rate": args.mmrl_lr,
        "stage3_schedule_epochs": args.stage3_epochs,
        "stage3_warmup_ratio": args.warmup_ratio,
        "stage3_lr_scheduler_type": args.scheduler,
        "stage3_epoch_lr_decay": (
            args.stage3_epoch_lr_decay if args.stage3_epochs > 1 else None
        ),
        "stage3_temperature_schedule_epochs": (
            1 if args.stage3_epochs > 1 else None
        ),
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_assistant_turn_policy": "all",
        "stage3_data_seed_offset": 0,
    }


def build_train_config(
    args: argparse.Namespace,
    experiment_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment_name": args.experiment_name,
        "experiment_cfg": experiment_cfg,
        "seed": args.seed,
        "data_sampling_seed": args.data_seed,
        "data_order_seed": args.data_seed,
        "deterministic_sampling": True,
        "eval_each_epoch": False,
        "save_each_epoch": args.stage3_epochs > 1,
        "checkpoint_base_model_path": str(args.model_path),
        "mmrl_relation_loss_weight": args.relation_weight,
        "mmrl_cross_relation_loss_weight": args.cross_relation_weight,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "dataloader_num_workers": args.dataloader_workers,
        "dataloader_pin_memory": True,
        "stage3_per_device_train_batch_size": args.stage3_batch_size,
        "stage3_gradient_accumulation_steps": args.stage3_gradient_accumulation,
        "stage3_dataloader_num_workers": args.stage3_dataloader_workers,
        "stage3_dataloader_pin_memory": False,
        "console_log_every": 1 if args.smoke_test else 25,
        "metric_smooth_window": 10 if args.smoke_test else 30,
        "metric_ema_alpha": 0.12,
        "metric_scatter_stride": 1 if args.smoke_test else 5,
        "save_debug_figure": False,
        "diag_every_steps": 2 if args.smoke_test else 250,
        "grad_spike_ema_alpha": 0.10,
        "grad_spike_ratio_threshold": 3.0,
        "grad_spike_abs_threshold": 0.5,
        "grad_spike_cooldown_steps": 20,
        "mmrl_diagnostics_keep_keys": [
            "ce_loss",
            "delta_to_org_ratio",
            "mmrl_delta_to_org_ratio",
            "mmrl_relation_loss_scaled",
            "mmrl_cross_relation_loss",
            "mmrl_cross_relation_loss_scaled",
            "dynamic_rep_base_norm_mean",
            "dynamic_rep_cross_delta_norm_mean",
            "dynamic_rep_cross_delta_ratio",
            "static_slot_memory_norm_mean",
            "text_dynamic_attention_entropy_norm",
            "text_dynamic_output_specificity_ratio",
            "text_dynamic_residual_gate_abs_mean",
            "visual_dynamic_attention_entropy_norm",
            "visual_dynamic_output_specificity_ratio",
            "visual_dynamic_residual_gate_abs_mean",
            "dynamic_query_residual_scale",
            "dynamic_query_residual_scale_grad_abs",
            "visual_memory_norm_mean",
            "text_memory_norm_mean",
            "visual_memory_pooling_grad_norm_mean",
            "text_memory_pooling_grad_norm_mean",
            "dynamic_query_projection_grad_norm_mean",
            "cross_attention_grad_norm_mean",
            "cross_attention_output_weight_norm",
            "deepstack_mmrl_residual_scale",
            "deepstack_delta_norm_mean",
            "deepstack_delta_to_org_ratio",
            "deepstack_residual_layers",
            "temperature",
        ],
        "learning_rate": {
            1: args.stage1_lr,
            3: args.mmrl_lr,
        },
        "epochs": {
            1: args.stage1_epochs,
            3: args.stage3_epochs,
        },
        "schedule_epochs": {
            3: args.stage3_epochs,
        },
        "stage3_warmup_ratio": args.warmup_ratio,
        "stage3_lr_scheduler_type": args.scheduler,
        "initial_temp": 1.0,
        "final_temp": 0.775,
    }


def build_dataset(
    args: argparse.Namespace,
    processor: Any,
    questions_path: Path,
    image_root: Path,
) -> tuple[SLAKEDataset, SLAKEDataCollator]:
    languages = None if args.language == "all" else (args.language,)
    base_types = tuple(args.base_type) or None
    sample_limit = 200 if args.smoke_test else args.sample_limit
    common = {
        "processor": processor,
        "questions_path": str(questions_path),
        "image_root": str(image_root),
        "total_limit": sample_limit,
        "languages": languages,
        "base_types": base_types,
        "splits": (args.split,),
        "seed": args.data_seed,
        "deterministic_sampling": True,
        "max_length": args.max_length,
    }
    dataset = SLAKEDataset(
        **common,
        ce_enabled=True,
    )
    if args.smoke_test and len(dataset) != 200:
        raise RuntimeError(
            "The 200-sample smoke test requires at least 200 matching records; "
            f"got {len(dataset)}"
        )
    return dataset, SLAKEDataCollator(processor)


def build_stage1_dataset(
    args: argparse.Namespace,
    processor: Any,
    slake_dataset: SLAKEDataset,
) -> tuple[ConcatDataset, SLAKEDataCollator, Dict[str, int]]:
    slake_positive = SLAKEStage1Dataset(slake_dataset)
    general_json = args.stage1_general_json or list(DEFAULT_GENERAL_JSON)
    general_image_roots = (
        args.stage1_general_image_root or list(DEFAULT_GENERAL_IMAGE_ROOTS)
    )
    general_negative = FourViewMMRLDataset(
        processor=processor,
        expert_json=[],
        expert_img_dir=[],
        general_json=[str(path) for path in general_json],
        general_img_dir=[str(path) for path in general_image_roots],
        total_limit=len(slake_dataset),
        enable_views=("general-mm", "general-text"),
        mode="slake_stage1",
        ce_enabled=False,
        seed=args.data_seed,
        deterministic_sampling=True,
        classifier_prompt_only=True,
    )
    if len(general_negative) < len(slake_dataset):
        raise RuntimeError(
            "Stage 1 general negatives are insufficient for a balanced gate warmup: "
            f"slake_raw={len(slake_dataset)} general_views={len(general_negative)}"
        )
    dataset = ConcatDataset((slake_positive, general_negative))
    counts = {
        "slake_raw": len(slake_dataset),
        "slake_positive_views": len(slake_positive),
        "general_negative_views": len(general_negative),
        "combined_views": len(dataset),
    }
    print(
        "[SLAKE_STAGE1_DATA] "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )
    return dataset, SLAKEDataCollator(processor), counts


def audit_template_and_collator(
    dataset: SLAKEDataset,
    collator: SLAKEDataCollator,
    processor: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    audit_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    features = []
    samples = []
    for index in audit_indices:
        feature = dataset[index]
        target_mask = feature["labels"] != -100
        pooling_mask = feature["mmrl_gating_mask"].bool()
        if not bool(target_mask.any()):
            raise RuntimeError(f"SLAKE sample {index} has no supervised answer tokens")
        if bool((target_mask & pooling_mask).any()):
            raise RuntimeError(
                f"SLAKE sample {index} leaks answer tokens into MMRL pooling"
            )

        target_text = processor.tokenizer.decode(
            feature["labels"][target_mask].tolist(),
            skip_special_tokens=True,
        ).strip()
        expected_answer = str(dataset.data[index]["answer"]).strip()
        if not target_text.startswith(expected_answer):
            raise RuntimeError(
                "SLAKE assistant boundary audit failed: "
                f"index={index} expected={expected_answer!r} decoded={target_text!r}"
            )
        samples.append(
            {
                "index": index,
                "question_id": dataset.data[index].get("question_id"),
                "answer": expected_answer,
                "decoded_target": target_text,
                "active_tokens": int(feature["attention_mask"].sum().item()),
                "supervised_tokens": int(target_mask.sum().item()),
            }
        )
        features.append(feature)

    batch = collator(features[:2])
    sequence_keys = (
        "input_ids",
        "attention_mask",
        "labels",
        "mmrl_gating_mask",
    )
    sequence_shape = tuple(batch["input_ids"].shape)
    for key in sequence_keys:
        if tuple(batch[key].shape) != sequence_shape:
            raise RuntimeError(
                f"SLAKE collator shape mismatch for {key}: "
                f"expected={sequence_shape} actual={tuple(batch[key].shape)}"
            )
    if sequence_shape[0] != 2 or sequence_shape[1] > dataset.max_length:
        raise RuntimeError(
            "SLAKE dynamic padding produced an invalid batch shape: "
            f"shape={sequence_shape} max_length={dataset.max_length}"
        )

    report = {
        "dataset_size": len(dataset),
        "audited_samples": samples,
        "batch_shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if torch.is_tensor(value)
        },
        "images_per_sample": list(batch["images_per_sample"]),
    }
    print(
        "[SLAKE_TEMPLATE_AUDIT_PASS] "
        f"dataset={len(dataset)} indices={audit_indices} "
        f"input_shape={tuple(batch['input_ids'].shape)} "
        f"pixel_shape={tuple(batch['pixel_values'].shape)}"
    )
    return report, batch


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def audit_model_forward(
    model: Any,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    device_batch = move_batch_to_device(batch, device)
    text_pooling_mask = model.model.build_mmrl_text_pooling_mask(
        input_ids=device_batch["input_ids"],
        attention_mask=device_batch["attention_mask"],
        mmrl_gating_mask=device_batch["mmrl_gating_mask"],
        audit_context="slake_smoke",
    )
    for token_name, token_id in model.model.mmrl_visual_template_token_ids.items():
        if bool((text_pooling_mask & device_batch["input_ids"].eq(token_id)).any()):
            raise RuntimeError(
                f"SLAKE text pooling still includes visual token {token_name}"
            )

    model.current_stage_id = 3
    model.current_stage_step = 0
    model.current_stage_progress = 0.0
    model.ce_loss_weight = 1.0
    model.alpha_loss_weight = 0.0
    model.enable_alpha_guide_loss = False
    was_training = model.training
    model.eval()
    model.model.MMRL.last_rep_shape = None
    model.model.MMRL.last_memory_shape = None
    with torch.no_grad():
        outputs = model(**device_batch)
    if outputs.loss is None or not bool(torch.isfinite(outputs.loss)):
        raise RuntimeError(f"SLAKE preflight forward produced invalid loss: {outputs.loss}")
    gate = model.model.visual.G_list
    if not torch.is_tensor(gate) or gate.numel() == 0:
        raise RuntimeError("SLAKE forward did not produce a visual gate value")
    if not bool(torch.isfinite(gate).all()):
        raise RuntimeError("SLAKE forward produced a non-finite visual gate value")
    mmrl = model.model.MMRL
    forced_dynamic_audit = False
    if mmrl.last_rep_shape is None or mmrl.last_memory_shape is None:
        forced_dynamic_audit = True
        original_ablation = model.model.visual.ablate_visual_gate
        model.model.visual.ablate_visual_gate = True
        try:
            with torch.no_grad():
                model(**device_batch)
        finally:
            model.model.visual.ablate_visual_gate = original_ablation
    rep_shape = mmrl.last_rep_shape
    memory_shape = mmrl.last_memory_shape
    expected_images = int(device_batch["image_grid_thw"].shape[0])
    expected_rep_shape = (
        expected_images,
        len(model.model.visual.insert_layers),
        mmrl.rp_space_length,
        mmrl.vision_token_dim,
    )
    expected_memory_tokens = (
        mmrl.rp_space_length
        if mmrl.use_dynamic_query_static_kv
        else 2 * mmrl.memory_query_count
    )
    expected_memory_shape = (
        expected_images,
        expected_memory_tokens,
        mmrl.vision_token_dim,
    )
    if rep_shape != expected_rep_shape:
        raise RuntimeError(
            "SLAKE dynamic Rep shape audit failed: "
            f"expected={expected_rep_shape} actual={rep_shape}"
        )
    if memory_shape != expected_memory_shape:
        raise RuntimeError(
            "SLAKE dynamic memory shape audit failed: "
            f"expected={expected_memory_shape} actual={memory_shape}"
        )
    cross_output_weight_norm = float(
        mmrl.cross_attention.output_projection.weight.detach().float().norm().item()
    )
    cross_output_bias_norm = float(
        mmrl.cross_attention.output_projection.bias.detach().float().norm().item()
    )
    if cross_output_weight_norm != 0.0 or cross_output_bias_norm != 0.0:
        raise RuntimeError(
            "Cross-Attention output projection changed before Stage 3: "
            f"weight_norm={cross_output_weight_norm} "
            f"bias_norm={cross_output_bias_norm}"
        )
    if was_training:
        model.train()

    report = {
        "loss": float(outputs.loss.detach().float().item()),
        "text_pooling_tokens": int(text_pooling_mask.sum().item()),
        "batch_size": int(device_batch["input_ids"].shape[0]),
        "gate_values": gate.detach().float().reshape(-1).cpu().tolist(),
        "dynamic_rep_shape": list(rep_shape),
        "dynamic_memory_shape": list(memory_shape),
        "cross_output_weight_norm": cross_output_weight_norm,
        "cross_output_bias_norm": cross_output_bias_norm,
        "forced_dynamic_audit": forced_dynamic_audit,
    }
    print(
        "[SLAKE_FORWARD_AUDIT_PASS] "
        f"loss={report['loss']:.6f} "
        f"text_pooling_tokens={report['text_pooling_tokens']} "
        f"dynamic_rep_shape={rep_shape} memory_shape={memory_shape} "
        f"forced_gate={forced_dynamic_audit}"
    )
    return report


def run_generation_checks(
    model: Any,
    processor: Any,
    dataset: SLAKEDataset,
    count: int,
) -> list[Dict[str, Any]]:
    if count <= 0:
        return []

    original_forward = model.forward
    was_training = model.training
    outer_temperature = getattr(model, "temperature_override", None)
    inner_temperature = getattr(model.model, "temperature_override", None)
    model.forward = MethodType(Qwen3VLForConditionalGeneration.forward, model)
    model.temperature_override = None
    model.model.temperature_override = None
    model.eval()
    interface = _LiveModelInterface(model, processor)
    rows = []
    try:
        for index in range(min(count, len(dataset))):
            sample = dataset.data[index]
            with Image.open(sample["image_path"]) as source_image:
                image = source_image.convert("RGB")
            output = interface.infer(
                image,
                sample["question"],
                max_new_tokens=16,
                temperature=0.0,
            )
            rows.append(
                {
                    "index": index,
                    "question_id": sample.get("question_id"),
                    "question": sample["question"],
                    "expected_answer": sample["answer"],
                    "generated": output,
                    "gate": interface.last_gate_value,
                    "timing": interface.last_generation_timing,
                }
            )
            print(
                "[SLAKE_GENERATION_CHECK] "
                f"index={index} G={interface.last_gate_value} "
                f"expected={sample['answer']!r} generated={output!r}"
            )
    finally:
        model.forward = original_forward
        model.temperature_override = outer_temperature
        model.model.temperature_override = inner_temperature
        if was_training:
            model.train()
    return rows


def snapshot_stage1_pooling(model: Any) -> Dict[str, Dict[str, torch.Tensor]]:
    visual = model.model.visual
    return {
        name: {
            key: value.detach().float().cpu().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in (
            ("hidden_state_pooling", visual.hidden_state_pooling),
            ("embedding_pooling", visual.embedding_pooling),
        )
    }


def audit_stage1_pooling_update(
    model: Any,
    before: Dict[str, Dict[str, torch.Tensor]],
    require_update: bool,
) -> Dict[str, float]:
    visual = model.model.visual
    deltas = {}
    for name, module in (
        ("hidden_state_pooling", visual.hidden_state_pooling),
        ("embedding_pooling", visual.embedding_pooling),
    ):
        after = {
            key: value.detach().float().cpu()
            for key, value in module.state_dict().items()
        }
        squared_delta = sum(
            (after[key] - before[name][key]).pow(2).sum()
            for key in after
        )
        deltas[name] = float(squared_delta.sqrt().item())
        if require_update and deltas[name] == 0.0:
            raise RuntimeError(f"Stage 1 did not update {name}")
    print(
        "[SLAKE_STAGE1_POOLING_UPDATE] "
        + " ".join(f"{name}_delta={value:.8f}" for name, value in deltas.items())
    )
    return deltas


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the existing MMRL architecture on an official SLAKE train manifest."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("MMRL_MODEL_PATH", DEFAULT_MODEL_PATH)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./slake_outputs/mmrl_train"),
    )
    parser.add_argument("--experiment-name", default="slake_mmrl_old_effective_anchor")
    parser.add_argument("--language", choices=("en", "zh", "all"), default="en")
    parser.add_argument("--base-type", action="append", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage3-epochs", type=int, default=1)
    parser.add_argument("--stage3-epoch-lr-decay", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--dataloader-workers", type=int, default=8)
    parser.add_argument("--stage3-batch-size", type=int, default=2)
    parser.add_argument("--stage3-gradient-accumulation", type=int, default=16)
    parser.add_argument("--stage3-dataloader-workers", type=int, default=4)
    parser.add_argument("--rp-space-length", type=int, default=40)
    parser.add_argument("--memory-query-count", type=int, default=128)
    parser.add_argument("--memory-attention-dim", type=int, default=128)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--cross-attention-heads", type=int, default=8)
    parser.add_argument(
        "--query-architecture",
        choices=QUERY_ARCHITECTURES,
        default="layer_mlp_post_cross",
    )
    parser.add_argument(
        "--same-init-layer-projectors",
        action="store_true",
        help="Initialize all independent layer MLPs from the same post-init weights.",
    )
    parser.add_argument(
        "--enable-deepstack-mmrl-residual",
        action="store_true",
        help="Route the complete gated MMRL state into the overlapping DeepStack layer.",
    )
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--mmrl-lr", type=float, default=6e-5)
    parser.add_argument("--relation-weight", type=float, default=0.050)
    parser.add_argument("--cross-relation-weight", type=float, default=0.0)
    parser.add_argument("--relation-max-tokens", type=int, default=64)
    parser.add_argument(
        "--scheduler",
        choices=("constant_with_warmup", "cosine"),
        default="constant_with_warmup",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument(
        "--stage1-general-json",
        type=Path,
        action="append",
        default=[],
        help="Repeat to override the default general-domain Stage 1 manifests.",
    )
    parser.add_argument(
        "--stage1-general-image-root",
        type=Path,
        action="append",
        default=[],
        help="Repeat to override the default general-domain Stage 1 image roots.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use exactly 200 samples and run template, forward, train, and generation checks.",
    )
    parser.add_argument("--generation-checks", type=int, default=2)
    parser.add_argument("--no-save-final", action="store_true")
    args = parser.parse_args()

    if args.sample_limit is not None and args.sample_limit < 1:
        parser.error("--sample-limit must be positive")
    if args.max_length < 32:
        parser.error("--max-length must be at least 32")
    if args.stage1_epochs < 1 or args.stage3_epochs < 1:
        parser.error("stage epochs must be positive")
    if not 0.0 < args.stage3_epoch_lr_decay <= 1.0:
        parser.error("--stage3-epoch-lr-decay must be in (0, 1]")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if (
        args.stage3_batch_size < 1
        or args.stage3_gradient_accumulation < 1
        or args.stage3_dataloader_workers < 0
    ):
        parser.error("Stage 3 batch/accumulation must be positive and workers non-negative")
    if args.rp_space_length < 1:
        parser.error("--rp-space-length must be positive")
    if args.memory_query_count < 1:
        parser.error("--memory-query-count must be positive")
    if args.memory_attention_dim < 1:
        parser.error("--memory-attention-dim must be positive")
    if args.projector_hidden_dim < 1:
        parser.error("--projector-hidden-dim must be positive")
    if args.cross_attention_heads < 1:
        parser.error("--cross-attention-heads must be positive")
    if args.relation_weight < 0.0:
        parser.error("--relation-weight must be non-negative")
    if args.cross_relation_weight < 0.0:
        parser.error("--cross-relation-weight must be non-negative")
    if (
        args.same_init_layer_projectors
        and args.query_architecture == "shared_direct_post_cross"
    ):
        parser.error(
            "--same-init-layer-projectors requires a layer MLP architecture"
        )
    if args.relation_max_tokens < 2:
        parser.error("--relation-max-tokens must be at least 2")
    if args.generation_checks < 0:
        parser.error("--generation-checks must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    questions_path = resolve_train_manifest(data_root, args.questions)
    image_root = (
        args.image_root.expanduser().resolve()
        if args.image_root is not None
        else (data_root / "imgs").resolve()
    )
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    experiment_cfg = build_experiment_config(args)
    train_cfg = build_train_config(args, experiment_cfg)
    print(
        "[SLAKE_TRAIN_CONFIG] "
        f"questions={questions_path} images={image_root} "
        f"limit={200 if args.smoke_test else args.sample_limit} "
        f"stages=(1,3) "
        f"rp_space_length={args.rp_space_length} "
        f"memory_query_count={args.memory_query_count} "
        f"memory_attention_dim={args.memory_attention_dim} "
        f"projector_hidden_dim={args.projector_hidden_dim} "
        f"cross_attention_heads={args.cross_attention_heads} "
        f"query_architecture={args.query_architecture} "
        f"same_init_layer_projectors={args.same_init_layer_projectors} "
        "deepstack_mmrl_residual="
        f"{args.enable_deepstack_mmrl_residual}:scale"
        f"{1.0 if args.enable_deepstack_mmrl_residual else 0.0} "
        f"relation={args.relation_weight} "
        f"cross_relation={args.cross_relation_weight} "
        f"stage1_batch={args.batch_size}x{args.gradient_accumulation} "
        f"stage3_batch={args.stage3_batch_size}x{args.stage3_gradient_accumulation} "
        f"seed={args.seed} data_seed={args.data_seed}"
    )

    model, processor = build_model_and_processor(
        str(model_path),
        experiment_cfg=experiment_cfg,
    )
    dataset, collator = build_dataset(
        args,
        processor,
        questions_path,
        image_root,
    )
    stage1_dataset, stage1_collator, stage1_counts = build_stage1_dataset(
        args,
        processor,
        dataset,
    )
    template_report, audit_batch = audit_template_and_collator(
        dataset,
        collator,
        processor,
    )

    data_cfg = {
        "stage1_dataset": stage1_dataset,
        "stage1_data_collator": stage1_collator,
        "stage3_dataset": dataset,
        "data_collator": collator,
    }
    pooling_before = snapshot_stage1_pooling(model)
    print("\n========== Running SLAKE Stage 1 ==========")
    run_stage(
        stage_id=1,
        model=model,
        processor=processor,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        output_dir=str(output_dir),
    )
    stage1_pooling_update = audit_stage1_pooling_update(
        model,
        pooling_before,
        require_update=True,
    )
    forward_report = audit_model_forward(model, audit_batch)

    print("\n========== Running SLAKE Stage 3 ==========")
    run_stage(
        stage_id=3,
        model=model,
        processor=processor,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        output_dir=str(output_dir),
    )

    generation_rows = run_generation_checks(
        model,
        processor,
        dataset,
        count=args.generation_checks,
    )
    checkpoint_manifest = None
    if not args.no_save_final:
        final_dir = output_dir / "final"
        checkpoint_manifest = save_mmrl_checkpoint(
            model,
            processor,
            final_dir,
            model_path,
            metadata={
                "experiment": args.experiment_name,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "dataset": "SLAKE",
                "language": args.language,
                "query_architecture": args.query_architecture,
                "same_init_layer_projectors": (
                    args.same_init_layer_projectors
                ),
                "enable_deepstack_mmrl_residual": (
                    args.enable_deepstack_mmrl_residual
                ),
                "relation_weight": args.relation_weight,
                "cross_relation_weight": args.cross_relation_weight,
                "relation_max_tokens": args.relation_max_tokens,
            },
        )
        print(f"[SLAKE_CHECKPOINT] saved={final_dir}")

    report = {
        "status": "pass",
        "smoke_test": args.smoke_test,
        "questions": questions_path,
        "image_root": image_root,
        "model_path": model_path,
        "output_dir": output_dir,
        "stages": [1, 3],
        "force_gate_open": False,
        "stage1_data": stage1_counts,
        "stage1_pooling_update": stage1_pooling_update,
        "experiment": experiment_cfg,
        "template_and_collator": template_report,
        "forward": forward_report,
        "generation": generation_rows,
        "checkpoint": checkpoint_manifest,
    }
    report_path = output_dir / "slake_train_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, ensure_ascii=False, indent=2)
    print(f"[SLAKE_TRAIN_PASS] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
