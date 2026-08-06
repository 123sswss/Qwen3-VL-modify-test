#!/usr/bin/env python3
"""Train the three fixed visual PEFT baselines used by the SLAKE comparison."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slake.data_pipeline import SLAKEDataCollator, SLAKEDataset


try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


MODEL_PATH = Path("/root/autodl-tmp/model")
SLAKE_ROOT = Path("/root/autodl-tmp/dataset/slake")
TRAIN_QUESTIONS = SLAKE_ROOT / "train.json"
IMAGE_ROOT = SLAKE_ROOT / "imgs"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "visual_peft_comparison"

SEED = 44
DATA_SEED = 42
MAX_LENGTH = 2048
VISION_LAYER_COUNT = 24
LANGUAGE_LAYER_COUNT = 36
LAST_LAYER_COUNT = 8
LORA_DROPOUT = 0.05
DEFAULT_EPOCHS = 3
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "lora_visual_all_attention_r32": {
        "method": "lora",
        "target_scope": "visual",
        "last_n_layers": None,
        "rank": 32,
        "use_dora": False,
    },
    "lora_visual_last8_attention_r32": {
        "method": "lora",
        "target_scope": "visual",
        "last_n_layers": LAST_LAYER_COUNT,
        "rank": 32,
        "use_dora": False,
    },
    "dora_visual_all_attention_r32": {
        "method": "dora",
        "target_scope": "visual",
        "last_n_layers": None,
        "rank": 32,
        "use_dora": True,
    },
    "lora_visual_all_attention_r64": {
        "method": "lora",
        "target_scope": "visual",
        "last_n_layers": None,
        "rank": 64,
        "use_dora": False,
    },
    "lora_full_model_attention_r16": {
        "method": "lora",
        "target_scope": "full_model",
        "last_n_layers": None,
        "rank": 16,
        "use_dora": False,
    },
}

MODEL_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "labels",
)


class VisualPEFTCollator:
    """Remove MMRL-only metadata from the shared SLAKE batch."""

    def __init__(self, processor: Any) -> None:
        self.base_collator = SLAKEDataCollator(processor)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base_collator(features)
        return {
            key: batch[key]
            for key in MODEL_BATCH_KEYS
            if key in batch and batch[key] is not None
        }


class EpochAdapterCheckpointCallback(TrainerCallback):
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
        kwargs["model"].save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
        print(
            f"[SLAKE_PEFT_EPOCH_CHECKPOINT] epoch={epoch_id} "
            f"global_step={int(state.global_step)} saved={checkpoint_dir}"
        )
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one fixed SLAKE visual LoRA/DoRA baseline."
    )
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[SLAKE_PEFT_REPRODUCIBILITY] seed={seed} data_seed={DATA_SEED}")


def load_model_and_processor() -> tuple[nn.Module, Any]:
    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
    )
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    loaders = []
    if Qwen3VLForConditionalGeneration is not None:
        loaders.append(Qwen3VLForConditionalGeneration)
    if AutoModelForImageTextToText is not None:
        loaders.append(AutoModelForImageTextToText)
    loaders.append(AutoModelForCausalLM)

    last_error = None
    for loader in loaders:
        try:
            model = loader.from_pretrained(str(MODEL_PATH), **model_kwargs)
            print(f"[SLAKE_PEFT_MODEL] loader={loader.__name__}")
            return model, processor
        except Exception as exc:
            last_error = exc
            print(f"[SLAKE_PEFT_MODEL] loader={loader.__name__} failed: {exc}")
    raise RuntimeError(f"Failed to load base model from {MODEL_PATH}") from last_error


def extract_vision_layer_index(module_name: str) -> int | None:
    name = module_name.lower()
    if not any(
        keyword in name
        for keyword in ("visual", "vision", "vision_tower", "vision_model")
    ):
        return None
    patterns = (
        r"(?:visual|vision|vision_tower|vision_model).*?"
        r"\.(?:blocks|layers)\.(\d+)(?:\.|$)",
        r"(?:visual|vision|vision_tower|vision_model).*?"
        r"\.encoder\.layers\.(\d+)(?:\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def find_visual_attention_targets(
    model: nn.Module,
    last_n_layers: int | None,
) -> tuple[List[str], List[int]]:
    indexed_linears = [
        (name, module, extract_vision_layer_index(name))
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    layer_indexes = sorted(
        {
            layer_index
            for _, _, layer_index in indexed_linears
            if layer_index is not None
        }
    )
    expected_layers = list(range(VISION_LAYER_COUNT))
    if layer_indexes != expected_layers:
        raise RuntimeError(
            "Unexpected Qwen3-VL vision layer layout: "
            f"expected={expected_layers} actual={layer_indexes}"
        )

    selected_layers = (
        layer_indexes
        if last_n_layers is None
        else layer_indexes[-int(last_n_layers) :]
    )
    selected_layer_set = set(selected_layers)
    targets = []
    target_layer_counts: Counter[int] = Counter()
    for name, _, layer_index in indexed_linears:
        lowered = name.lower()
        if layer_index not in selected_layer_set:
            continue
        if not (
            ".attn." in lowered
            or ".attention." in lowered
            or "self_attn" in lowered
        ):
            continue
        if not lowered.endswith(
            ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj")
        ):
            continue
        targets.append(name)
        target_layer_counts[layer_index] += 1

    expected_target_count = len(selected_layers) * 2
    if len(targets) != expected_target_count:
        raise RuntimeError(
            "Visual attention target audit failed: "
            f"selected_layers={selected_layers} expected_targets={expected_target_count} "
            f"actual_targets={len(targets)} per_layer={dict(target_layer_counts)}"
        )
    if any(target_layer_counts[index] != 2 for index in selected_layers):
        raise RuntimeError(
            "Each visual layer must expose exactly qkv/proj attention targets; "
            f"per_layer={dict(target_layer_counts)}"
        )

    print(
        "[SLAKE_PEFT_TARGET_AUDIT] "
        f"selected_layers_0based={selected_layers} target_count={len(targets)}"
    )
    for name in targets:
        print(f"  - {name}")
    return targets, selected_layers


def extract_language_layer_index(module_name: str) -> int | None:
    match = re.search(
        r"(?:^|\.)language_model\.layers\.(\d+)(?:\.|$)",
        module_name.lower(),
    )
    return int(match.group(1)) if match else None


def find_language_attention_targets(
    model: nn.Module,
) -> tuple[List[str], List[int]]:
    targets = []
    target_layer_counts: Counter[int] = Counter()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_index = extract_language_layer_index(name)
        if layer_index is None:
            continue
        lowered = name.lower()
        if "self_attn" not in lowered:
            continue
        if not lowered.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            continue
        targets.append(name)
        target_layer_counts[layer_index] += 1

    layer_indexes = sorted(target_layer_counts)
    expected_layers = list(range(LANGUAGE_LAYER_COUNT))
    if layer_indexes != expected_layers:
        raise RuntimeError(
            "Unexpected Qwen3-VL language layer layout: "
            f"expected={expected_layers} actual={layer_indexes}"
        )
    expected_target_count = LANGUAGE_LAYER_COUNT * 4
    if len(targets) != expected_target_count:
        raise RuntimeError(
            "Language attention target audit failed: "
            f"expected_targets={expected_target_count} actual_targets={len(targets)} "
            f"per_layer={dict(target_layer_counts)}"
        )
    if any(target_layer_counts[index] != 4 for index in expected_layers):
        raise RuntimeError(
            "Each language layer must expose exactly q/k/v/o attention targets; "
            f"per_layer={dict(target_layer_counts)}"
        )

    print(
        "[SLAKE_PEFT_LANGUAGE_TARGET_AUDIT] "
        f"selected_layers_0based={layer_indexes} target_count={len(targets)}"
    )
    for name in targets:
        print(f"  - {name}")
    return targets, layer_indexes


def build_dataset(processor: Any) -> SLAKEDataset:
    dataset = SLAKEDataset(
        processor=processor,
        questions_path=str(TRAIN_QUESTIONS),
        image_root=str(IMAGE_ROOT),
        total_limit=None,
        languages=None,
        base_types=None,
        splits=("train",),
        ce_enabled=True,
        seed=DATA_SEED,
        deterministic_sampling=True,
        max_length=MAX_LENGTH,
    )
    print(
        "[SLAKE_PEFT_DATA] "
        f"questions={TRAIN_QUESTIONS} images={IMAGE_ROOT} "
        f"language=all samples={len(dataset)} max_length={MAX_LENGTH}"
    )
    return dataset


def count_parameters(model: nn.Module) -> Dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def main() -> int:
    args = parse_args()
    experiment = EXPERIMENTS[args.experiment]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else OUTPUT_ROOT / args.experiment
    )
    trainer_dir = output_dir / "trainer"
    final_dir = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    for required_path in (MODEL_PATH, TRAIN_QUESTIONS, IMAGE_ROOT):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path does not exist: {required_path}")

    seed_everything(SEED)
    model, processor = load_model_and_processor()
    visual_targets, selected_vision_layers = find_visual_attention_targets(
        model,
        experiment["last_n_layers"],
    )
    language_targets: List[str] = []
    selected_language_layers: List[int] = []
    if experiment["target_scope"] == "full_model":
        language_targets, selected_language_layers = (
            find_language_attention_targets(model)
        )
    target_modules = visual_targets + language_targets
    if len(target_modules) != len(set(target_modules)):
        raise RuntimeError("Duplicate PEFT target modules detected")
    expected_total = (
        192 if experiment["target_scope"] == "full_model" else len(visual_targets)
    )
    if len(target_modules) != expected_total:
        raise RuntimeError(
            "Combined PEFT target audit failed: "
            f"scope={experiment['target_scope']} expected={expected_total} "
            f"actual={len(target_modules)}"
        )
    print(
        "[SLAKE_PEFT_COMBINED_TARGET_AUDIT] "
        f"scope={experiment['target_scope']} visual={len(visual_targets)} "
        f"language={len(language_targets)} total={len(target_modules)}"
    )
    dataset = build_dataset(processor)

    rank = int(experiment["rank"])
    lora_alpha = rank * 2
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.config.use_cache = False
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        use_dora=bool(experiment["use_dora"]),
    )
    model = get_peft_model(model, peft_config)

    parameter_counts = count_parameters(model)
    print(
        "[SLAKE_PEFT_CONFIG] "
        f"experiment={args.experiment} method={experiment['method']} "
        f"target_scope={experiment['target_scope']} "
        f"epochs={args.epochs} micro_batch={MICRO_BATCH_SIZE} "
        f"gradient_accumulation={GRADIENT_ACCUMULATION_STEPS} "
        "gradient_checkpointing=False "
        f"rank={rank} alpha={lora_alpha} dropout={LORA_DROPOUT} "
        f"use_dora={experiment['use_dora']} trainable={parameter_counts['trainable']} "
        f"total={parameter_counts['total']} "
        f"ratio={parameter_counts['trainable_ratio']:.6%}"
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
        data_seed=DATA_SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VisualPEFTCollator(processor),
        processing_class=processor,
        callbacks=(
            [EpochAdapterCheckpointCallback(processor, output_dir)]
            if args.epochs > 1
            else None
        ),
    )
    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    report = {
        "status": "pass",
        "experiment": args.experiment,
        "method": experiment["method"],
        "target_scope": experiment["target_scope"],
        "use_dora": experiment["use_dora"],
        "rank": rank,
        "alpha": lora_alpha,
        "dropout": LORA_DROPOUT,
        "epochs": args.epochs,
        "per_device_train_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "gradient_checkpointing": False,
        "selected_vision_layers_0based": selected_vision_layers,
        "selected_language_layers_0based": selected_language_layers,
        "target_modules": target_modules,
        "parameter_counts": parameter_counts,
        "dataset_samples": len(dataset),
        "train_metrics": train_result.metrics,
        "model_path": str(MODEL_PATH),
        "questions": str(TRAIN_QUESTIONS),
        "image_root": str(IMAGE_ROOT),
        "final_checkpoint": str(final_dir),
    }
    report_path = output_dir / "train_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[SLAKE_PEFT_TRAIN_PASS] checkpoint={final_dir} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
