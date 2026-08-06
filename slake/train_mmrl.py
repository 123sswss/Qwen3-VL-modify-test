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
        "visual_residual_adapter_count": 4,
        "visual_adapter_reduction_factor": args.adapter_reduction_factor,
        "random_init_adapter_output_count": 0,
        "zero_init_adapter_router_output": False,
        "adapter_usage_balance_loss_weight": args.usage_weight,
        "adapter_sample_entropy_loss_weight": args.entropy_weight,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": args.effective_delta_weight,
        "adapter_diversity_loss_weight": 0.0,
        "adapter_sample_entropy_target": args.entropy_target,
        "adapter_common_mode_target": 0.94,
        "adapter_effective_delta_target_low": args.effective_delta_target_low,
        "adapter_effective_delta_target_high": args.effective_delta_target_high,
        "mmrl_relation_loss_weight": args.relation_weight,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "enable_adapter_router_identity_residual": args.enable_adapter_router_identity_residual,
        "direct_mmrl_output": False,
        "raw_visual_adapter": False,
        "enable_early_mmrl_guard": False,
        "enable_deepstack_mmrl_residual": False,
        "deepstack_mmrl_residual_scale": 0.0,
        "ablate_visual_gate": False,
        "ablate_direct_learnable_rep": False,
        "decouple_stage_pooling": args.decouple_stage_pooling,
        "stage3_learning_rate": args.mmrl_lr,
        "stage3_pooling_learning_rate": args.pooling_lr,
        "stage3_mmrl_learning_rate": args.mmrl_lr,
        "stage3_router_learning_rate": args.router_lr,
        "stage3_adapter_learning_rates": list(args.adapter_lrs),
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
        "visual_residual_adapter_count": 4,
        "adapter_usage_balance_loss_weight": args.usage_weight,
        "adapter_sample_entropy_loss_weight": args.entropy_weight,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": args.effective_delta_weight,
        "adapter_diversity_loss_weight": 0.0,
        "mmrl_relation_loss_weight": args.relation_weight,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "dataloader_num_workers": args.dataloader_workers,
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
            "adapter_effective_delta_ratio_mean",
            "final_to_gated_ratio",
            "adapter_route_entropy_norm",
            "adapter_usage_0",
            "adapter_usage_1",
            "adapter_usage_2",
            "adapter_usage_3",
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
    with torch.no_grad():
        outputs = model(**device_batch)
    if outputs.loss is None or not bool(torch.isfinite(outputs.loss)):
        raise RuntimeError(f"SLAKE preflight forward produced invalid loss: {outputs.loss}")
    gate = model.model.visual.G_list
    route_probs = model.model.visual.route_probs
    if not torch.is_tensor(gate) or gate.numel() == 0:
        raise RuntimeError("SLAKE forward did not produce a visual gate value")
    if not bool(torch.isfinite(gate).all()):
        raise RuntimeError("SLAKE forward produced a non-finite visual gate value")
    if not torch.is_tensor(route_probs) or route_probs.shape[-1] != 4:
        raise RuntimeError(
            "SLAKE Stage 3 must preserve the four-expert adapter router"
        )
    if was_training:
        model.train()

    report = {
        "loss": float(outputs.loss.detach().float().item()),
        "text_pooling_tokens": int(text_pooling_mask.sum().item()),
        "batch_size": int(device_batch["input_ids"].shape[0]),
        "gate_values": gate.detach().float().reshape(-1).cpu().tolist(),
        "route_probs": route_probs.detach().float().cpu().tolist(),
    }
    print(
        "[SLAKE_FORWARD_AUDIT_PASS] "
        f"loss={report['loss']:.6f} "
        f"text_pooling_tokens={report['text_pooling_tokens']}"
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--rp-space-length", type=int, default=40)
    parser.add_argument("--adapter-reduction-factor", type=int, default=4)

    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--pooling-lr", type=float, default=6e-5)
    parser.add_argument("--mmrl-lr", type=float, default=6e-5)
    parser.add_argument("--router-lr", type=float, default=8e-5)
    parser.add_argument(
        "--adapter-lrs",
        type=float,
        nargs=4,
        default=(4e-5, 6e-5, 8e-5, 1e-4),
        metavar=("A0", "A1", "A2", "A3"),
    )
    parser.add_argument("--usage-weight", type=float, default=0.0026)
    parser.add_argument("--entropy-weight", type=float, default=0.020)
    parser.add_argument("--entropy-target", type=float, default=0.72)
    parser.add_argument("--effective-delta-weight", type=float, default=0.0003)
    parser.add_argument("--effective-delta-target-low", type=float, default=0.52)
    parser.add_argument("--effective-delta-target-high", type=float, default=0.98)
    parser.add_argument(
        "--enable-adapter-router-identity-residual",
        action="store_true",
        help="Add the raw Rep-branch delta to the routed adapter correction.",
    )
    parser.add_argument("--relation-weight", type=float, default=0.010)
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
        "--decouple-stage-pooling",
        action="store_true",
        help="Use a frozen Stage 1 gate pooler and an independent Stage 3 router copy.",
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
    if args.rp_space_length < 1:
        parser.error("--rp-space-length must be positive")
    if args.adapter_reduction_factor < 1:
        parser.error("--adapter-reduction-factor must be positive")
    if args.generation_checks < 0:
        parser.error("--generation-checks must be non-negative")
    if args.effective_delta_target_low < 0:
        parser.error("--effective-delta-target-low must be non-negative")
    if args.effective_delta_target_high < args.effective_delta_target_low:
        parser.error(
            "--effective-delta-target-high must be greater than or equal to "
            "--effective-delta-target-low"
        )
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
        f"stages=(1,3) decouple_stage_pooling={args.decouple_stage_pooling} "
        f"rp_space_length={args.rp_space_length} "
        f"adapter_reduction_factor={args.adapter_reduction_factor} "
        f"identity_residual={args.enable_adapter_router_identity_residual} "
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
    if model.model.visual.decouple_stage_pooling:
        model.model.visual.reset_router_pooling_from_gate()
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
