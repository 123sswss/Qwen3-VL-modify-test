#!/usr/bin/env python3
"""PathVQA entry point for the existing two-stage MMRL trainer."""

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
from torch.utils.data import ConcatDataset
from transformers import Qwen3VLForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
for import_root in (REPO_ROOT, TRAIN_DIR):
    import_text = str(import_root)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from data_pipeline import FourViewMMRLDataset
from live_epoch_eval import _LiveModelInterface
from mmrl_checkpoint import load_mmrl_delta, load_mmrl_manifest, save_mmrl_checkpoint
from pathvqa.data_pipeline import (
    PathVQADataCollator,
    PathVQADataset,
    PathVQAStage1Dataset,
)
from slake.train_mmrl import (
    DEFAULT_GENERAL_IMAGE_ROOTS,
    DEFAULT_GENERAL_JSON,
    audit_model_forward,
    audit_stage1_pooling_update,
    build_experiment_config,
    build_train_config,
    json_safe,
    snapshot_stage1_pooling,
)
from train_stages import build_model_and_processor, run_stage


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/dataset/pathVQA")
DEFAULT_MODEL_PATH = Path("/root/autodl-tmp/model")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[PATHVQA_REPRODUCIBILITY] seed={seed}")


def build_dataset(
    args: argparse.Namespace,
    processor: Any,
) -> tuple[PathVQADataset, PathVQADataCollator]:
    dataset = PathVQADataset(
        processor=processor,
        data_root=args.data_root,
        split=args.split,
        cache_dir=args.cache_dir,
        total_limit=args.sample_limit,
        ce_enabled=True,
        seed=args.data_seed,
        deterministic_sampling=True,
        max_length=args.max_length,
    )
    return dataset, PathVQADataCollator(processor)


def build_stage1_dataset(
    args: argparse.Namespace,
    processor: Any,
    pathvqa_dataset: PathVQADataset,
) -> tuple[ConcatDataset, PathVQADataCollator, Dict[str, int]]:
    pathvqa_positive = PathVQAStage1Dataset(pathvqa_dataset)
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
        total_limit=len(pathvqa_dataset),
        enable_views=("general-mm", "general-text"),
        mode="pathvqa_stage1",
        ce_enabled=False,
        seed=args.data_seed,
        deterministic_sampling=True,
        classifier_prompt_only=True,
    )
    if len(general_negative) < len(pathvqa_dataset):
        raise RuntimeError(
            "Stage 1 general negatives are insufficient for a balanced gate warmup: "
            f"pathvqa_raw={len(pathvqa_dataset)} "
            f"general_views={len(general_negative)}"
        )
    dataset = ConcatDataset((pathvqa_positive, general_negative))
    counts = {
        "pathvqa_raw": len(pathvqa_dataset),
        "pathvqa_positive_views": len(pathvqa_positive),
        "general_negative_views": len(general_negative),
        "combined_views": len(dataset),
    }
    print(
        "[PATHVQA_STAGE1_DATA] "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )
    return dataset, PathVQADataCollator(processor), counts


def audit_template_and_collator(
    dataset: PathVQADataset,
    collator: PathVQADataCollator,
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
            raise RuntimeError(f"PathVQA sample {index} has no supervised answer tokens")
        if bool((target_mask & pooling_mask).any()):
            raise RuntimeError(
                f"PathVQA sample {index} leaks answer tokens into MMRL pooling"
            )
        target_text = processor.tokenizer.decode(
            feature["labels"][target_mask].tolist(),
            skip_special_tokens=True,
        ).strip()
        expected_answer = str(dataset.data[index]["answer"]).strip()
        if not target_text.startswith(expected_answer):
            raise RuntimeError(
                "PathVQA assistant boundary audit failed: "
                f"index={index} expected={expected_answer!r} decoded={target_text!r}"
            )
        samples.append(
            {
                "index": index,
                "question_id": dataset.data[index]["question_id"],
                "answer": expected_answer,
                "decoded_target": target_text,
                "active_tokens": int(feature["attention_mask"].sum().item()),
                "supervised_tokens": int(target_mask.sum().item()),
            }
        )
        features.append(feature)

    batch = collator(features[:2])
    sequence_shape = tuple(batch["input_ids"].shape)
    for key in ("input_ids", "attention_mask", "labels", "mmrl_gating_mask"):
        if tuple(batch[key].shape) != sequence_shape:
            raise RuntimeError(
                f"PathVQA collator shape mismatch for {key}: "
                f"expected={sequence_shape} actual={tuple(batch[key].shape)}"
            )
    if sequence_shape[0] != 2 or sequence_shape[1] > dataset.max_length:
        raise RuntimeError(
            "PathVQA dynamic padding produced an invalid batch shape: "
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
        "[PATHVQA_TEMPLATE_AUDIT_PASS] "
        f"dataset={len(dataset)} indices={audit_indices} "
        f"input_shape={tuple(batch['input_ids'].shape)} "
        f"pixel_shape={tuple(batch['pixel_values'].shape)}"
    )
    return report, batch


def run_generation_checks(
    model: Any,
    processor: Any,
    dataset: PathVQADataset,
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
            image = dataset.load_image(sample)
            try:
                output = interface.infer(
                    image,
                    sample["question"],
                    max_new_tokens=16,
                    temperature=0.0,
                )
            finally:
                image.close()
            rows.append(
                {
                    "index": index,
                    "question_id": sample["question_id"],
                    "question": sample["question"],
                    "expected_answer": sample["answer"],
                    "generated": output,
                    "gate": interface.last_gate_value,
                    "timing": interface.last_generation_timing,
                }
            )
            print(
                "[PATHVQA_GENERATION_CHECK] "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the existing MMRL architecture on PathVQA Parquet shards."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("MMRL_MODEL_PATH", DEFAULT_MODEL_PATH)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./pathvqa_outputs/mmrl_train"),
    )
    parser.add_argument(
        "--experiment-name",
        default="pathvqa_mmrl_layer_mlp_same_init_relation0050",
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage3-epochs", type=int, default=3)
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
        choices=("layer_mlp_post_cross", "shared_direct_post_cross"),
        default="layer_mlp_post_cross",
    )
    parser.add_argument("--expected-mmrl-parameters", type=int)
    parser.add_argument(
        "--same-init-layer-projectors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--disable-dynamic-cross-attention", action="store_true")
    parser.add_argument(
        "--memory-pooling-mode",
        choices=("multi_query", "mean", "text_guided"),
        default="multi_query",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("cross_attention", "concat_mlp"),
        default="cross_attention",
    )
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--mmrl-lr", type=float, default=6e-5)
    parser.add_argument("--soft-prompt-length", type=int, default=0)
    parser.add_argument("--soft-prompt-init-seed", type=int, default=44)
    parser.add_argument("--prompt-lr", type=float, default=0.3)
    parser.add_argument("--prompt-warmup-ratio", type=float, default=0.03)
    parser.add_argument("--expected-total-trainable-parameters", type=int)
    parser.add_argument("--relation-weight", type=float, default=0.050)
    parser.add_argument("--relation-max-tokens", type=int, default=64)
    parser.add_argument(
        "--scheduler",
        choices=("constant_with_warmup", "cosine"),
        default="constant_with_warmup",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--stage1-general-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--stage1-general-image-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--generation-checks", type=int, default=2)
    parser.add_argument("--stage1-checkpoint-in", type=Path)
    parser.add_argument("--stage1-checkpoint-out", type=Path)
    parser.add_argument("--no-save-final", action="store_true")
    args = parser.parse_args()
    args.smoke_test = False

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
    if min(
        args.rp_space_length,
        args.memory_query_count,
        args.memory_attention_dim,
        args.projector_hidden_dim,
        args.cross_attention_heads,
    ) < 1:
        parser.error("MMRL dimensions and attention heads must be positive")
    if args.fusion_mode == "concat_mlp" and args.memory_pooling_mode != "mean":
        parser.error("--fusion-mode concat_mlp requires --memory-pooling-mode mean")
    if args.relation_weight < 0.0:
        parser.error("--relation-weight must be non-negative")
    if args.relation_max_tokens < 2:
        parser.error("--relation-max-tokens must be at least 2")
    if args.generation_checks < 0:
        parser.error("--generation-checks must be non-negative")
    if args.expected_mmrl_parameters is not None and args.expected_mmrl_parameters < 1:
        parser.error("--expected-mmrl-parameters must be positive")
    if args.soft_prompt_length < 0:
        parser.error("--soft-prompt-length must be non-negative")
    if args.prompt_lr <= 0.0:
        parser.error("--prompt-lr must be positive")
    if not 0.0 <= args.prompt_warmup_ratio < 1.0:
        parser.error("--prompt-warmup-ratio must be in [0, 1)")
    return args


def validate_stage1_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_mmrl_manifest(args.stage1_checkpoint_in)
    metadata = dict(manifest.get("metadata", {}))
    expected = {
        "stage": 1,
        "dataset": "PathVQA",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "query_architecture": args.query_architecture,
        "memory_pooling_mode": args.memory_pooling_mode,
        "fusion_mode": args.fusion_mode,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Shared Stage1 checkpoint metadata mismatch: {mismatches}"
        )
    return manifest


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else (args.data_root / ".hf_cache").resolve()
    )
    args.model_path = args.model_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.stage1_checkpoint_in is not None:
        args.stage1_checkpoint_in = args.stage1_checkpoint_in.expanduser().resolve()
    if args.stage1_checkpoint_out is not None:
        args.stage1_checkpoint_out = args.stage1_checkpoint_out.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    experiment_cfg = build_experiment_config(args)
    train_cfg = build_train_config(args, experiment_cfg)
    print(
        "[PATHVQA_TRAIN_CONFIG] "
        f"data={args.data_root} cache={args.cache_dir} split={args.split} "
        f"limit={args.sample_limit} stages=(1,3) "
        f"rp_space_length={args.rp_space_length} "
        f"memory_query_count={args.memory_query_count} "
        f"memory_attention_dim={args.memory_attention_dim} "
        f"projector_hidden_dim={args.projector_hidden_dim} "
        f"cross_attention_heads={args.cross_attention_heads} "
        f"query_architecture={args.query_architecture} "
        f"same_init_layer_projectors={args.same_init_layer_projectors} "
        f"dynamic_cross_attention={not args.disable_dynamic_cross_attention} "
        f"memory_pooling_mode={args.memory_pooling_mode} "
        f"fusion_mode={args.fusion_mode} "
        "train_gate_mode=open_full_ce relation_mode=uniform "
        f"relation={args.relation_weight} "
        f"stage1_batch={args.batch_size}x{args.gradient_accumulation} "
        f"stage3_batch={args.stage3_batch_size}x{args.stage3_gradient_accumulation} "
        f"seed={args.seed} data_seed={args.data_seed} "
        f"stage1_checkpoint_in={args.stage1_checkpoint_in} "
        f"stage1_checkpoint_out={args.stage1_checkpoint_out}"
    )

    model, processor = build_model_and_processor(
        str(args.model_path),
        experiment_cfg=experiment_cfg,
    )
    dataset, collator = build_dataset(args, processor)
    stage1_dataset = None
    stage1_collator = None
    stage1_counts: Dict[str, int] = {}
    if args.stage1_checkpoint_in is None:
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

    stage1_checkpoint_manifest = None
    if args.stage1_checkpoint_in is None:
        pooling_before = snapshot_stage1_pooling(model)
        print("\n========== Running PathVQA Stage 1 ==========")
        run_stage(
            stage_id=1,
            model=model,
            processor=processor,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            output_dir=str(args.output_dir),
        )
        stage1_pooling_update = audit_stage1_pooling_update(
            model,
            pooling_before,
            require_update=True,
            dataset_label="PATHVQA",
        )
        if args.stage1_checkpoint_out is not None:
            stage1_checkpoint_manifest = save_mmrl_checkpoint(
                model,
                processor,
                args.stage1_checkpoint_out,
                args.model_path,
                metadata={
                    "experiment": args.experiment_name,
                    "stage": 1,
                    "dataset": "PathVQA",
                    "seed": args.seed,
                    "data_seed": args.data_seed,
                    "query_architecture": args.query_architecture,
                    "memory_pooling_mode": args.memory_pooling_mode,
                    "fusion_mode": args.fusion_mode,
                    "stage1_data": stage1_counts,
                },
            )
            print(
                "[PATHVQA_STAGE1_CHECKPOINT] "
                f"saved={args.stage1_checkpoint_out}"
            )
    else:
        stage1_checkpoint_manifest = validate_stage1_checkpoint(args)
        loaded_manifest = load_mmrl_delta(
            model,
            args.stage1_checkpoint_in,
            allow_missing_prefixes=("soft_prompt",),
        )
        if loaded_manifest.get("weights_sha256") != stage1_checkpoint_manifest.get(
            "weights_sha256"
        ):
            raise RuntimeError("Stage1 checkpoint changed between validation and load")
        stage1_counts = dict(
            stage1_checkpoint_manifest.get("metadata", {}).get("stage1_data", {})
        )
        stage1_pooling_update = None
        print(
            "[PATHVQA_STAGE1_REUSED] "
            f"checkpoint={args.stage1_checkpoint_in} "
            f"sha256={stage1_checkpoint_manifest.get('weights_sha256')}"
        )
    forward_report = audit_model_forward(
        model,
        audit_batch,
        dataset_label="PATHVQA",
    )

    print("\n========== Running PathVQA Stage 3 ==========")
    run_stage(
        stage_id=3,
        model=model,
        processor=processor,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        output_dir=str(args.output_dir),
    )
    generation_rows = run_generation_checks(
        model,
        processor,
        dataset,
        count=args.generation_checks,
    )

    checkpoint_manifest = None
    if not args.no_save_final:
        final_dir = args.output_dir / "final"
        checkpoint_manifest = save_mmrl_checkpoint(
            model,
            processor,
            final_dir,
            args.model_path,
            metadata={
                "experiment": args.experiment_name,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "dataset": "PathVQA",
                "split": args.split,
                "query_architecture": args.query_architecture,
                "same_init_layer_projectors": args.same_init_layer_projectors,
                "use_dynamic_cross_attention": (
                    not args.disable_dynamic_cross_attention
                ),
                "memory_pooling_mode": args.memory_pooling_mode,
                "fusion_mode": args.fusion_mode,
                "train_gate_mode": "open_full_ce",
                "relation_weight": args.relation_weight,
                "relation_max_tokens": args.relation_max_tokens,
            },
        )
        print(f"[PATHVQA_CHECKPOINT] saved={final_dir}")

    report = {
        "status": "pass",
        "data_root": args.data_root,
        "cache_dir": args.cache_dir,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "stages": [3] if args.stage1_checkpoint_in is not None else [1, 3],
        "stage1_data": stage1_counts,
        "stage1_pooling_update": stage1_pooling_update,
        "stage1_checkpoint": stage1_checkpoint_manifest,
        "stage1_checkpoint_source": args.stage1_checkpoint_in,
        "experiment": experiment_cfg,
        "template_and_collator": template_report,
        "forward": forward_report,
        "generation": generation_rows,
        "checkpoint": checkpoint_manifest,
    }
    report_path = args.output_dir / "pathvqa_train_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, ensure_ascii=False, indent=2)
    print(f"[PATHVQA_TRAIN_PASS] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
