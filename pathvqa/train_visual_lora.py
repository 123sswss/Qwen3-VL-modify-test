#!/usr/bin/env python3
"""Train audited PathVQA Attention LoRA baselines."""

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

from pathvqa.data_pipeline import PathVQADataCollator, PathVQADataset


try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


VISION_LAYER_COUNT = 24
LANGUAGE_LAYER_COUNT = 36
DEFAULT_RANK = 128
LORA_DROPOUT = 0.05
LAST8_EXPECTED_TRAINABLE_PARAMETERS = 6_291_456
MODEL_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "labels",
)


class VisualLoRACollator:
    def __init__(self, processor: Any) -> None:
        self.base_collator = PathVQADataCollator(processor)

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
            f"[PATHVQA_LORA_EPOCH_CHECKPOINT] epoch={epoch_id} "
            f"global_step={int(state.global_step)} saved={checkpoint_dir}"
        )
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PathVQA LoRA on audited visual or full-model Attention."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/pathVQA"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("MMRL_MODEL_PATH", "/root/autodl-tmp/model")),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-name",
        default="pathvqa_lora_visual_all_attention_r128",
    )
    parser.add_argument(
        "--target-scope",
        choices=("visual", "full_model"),
        default="visual",
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--last-n-vision-layers", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--expected-trainable-parameters", type=int)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    if args.max_length < 32:
        parser.error("--max-length must be at least 32")
    if args.rank < 1:
        parser.error("--rank must be positive")
    if not 1 <= args.last_n_vision_layers <= VISION_LAYER_COUNT:
        parser.error(
            f"--last-n-vision-layers must be in [1, {VISION_LAYER_COUNT}]"
        )
    if (
        args.target_scope == "full_model"
        and args.last_n_vision_layers != VISION_LAYER_COUNT
    ):
        parser.error("Full-model Attention LoRA must include all visual layers")
    if (
        args.expected_trainable_parameters is not None
        and args.expected_trainable_parameters < 1
    ):
        parser.error("--expected-trainable-parameters must be positive")
    return args


def seed_everything(seed: int, data_seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[PATHVQA_LORA_REPRODUCIBILITY] seed={seed} data_seed={data_seed}")


def load_model_and_processor(model_path: Path) -> tuple[nn.Module, Any]:
    processor = AutoProcessor.from_pretrained(
        str(model_path),
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
            model = loader.from_pretrained(str(model_path), **model_kwargs)
            print(f"[PATHVQA_LORA_MODEL] loader={loader.__name__}")
            return model, processor
        except Exception as exc:
            last_error = exc
            print(f"[PATHVQA_LORA_MODEL] loader={loader.__name__} failed: {exc}")
    raise RuntimeError(f"Failed to load base model from {model_path}") from last_error


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
    last_n_layers: int = VISION_LAYER_COUNT,
) -> tuple[List[str], List[int]]:
    if not 1 <= last_n_layers <= VISION_LAYER_COUNT:
        raise ValueError(
            f"last_n_layers must be in [1, {VISION_LAYER_COUNT}], got {last_n_layers}"
        )
    indexed_linears = [
        (name, extract_vision_layer_index(name))
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    layer_indexes = sorted(
        {
            layer_index
            for _, layer_index in indexed_linears
            if layer_index is not None
        }
    )
    expected_layers = list(range(VISION_LAYER_COUNT))
    if layer_indexes != expected_layers:
        raise RuntimeError(
            "Unexpected Qwen3-VL vision layer layout: "
            f"expected={expected_layers} actual={layer_indexes}"
        )

    selected_layers = expected_layers[-last_n_layers:]
    targets = []
    per_layer: Counter[int] = Counter()
    for name, layer_index in indexed_linears:
        lowered = name.lower()
        if layer_index is None or layer_index not in selected_layers:
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
        per_layer[layer_index] += 1

    expected_target_count = last_n_layers * 2
    if len(targets) != expected_target_count:
        raise RuntimeError(
            "Visual attention target audit failed: "
            f"expected_targets={expected_target_count} actual={len(targets)} "
            f"per_layer={dict(per_layer)}"
        )
    if any(per_layer[index] != 2 for index in selected_layers):
        raise RuntimeError(
            "Each visual layer must expose exactly qkv/proj attention targets; "
            f"per_layer={dict(per_layer)}"
        )
    print(
        "[PATHVQA_LORA_TARGET_AUDIT] "
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
    per_layer: Counter[int] = Counter()
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
        per_layer[layer_index] += 1

    selected_layers = sorted(per_layer)
    expected_layers = list(range(LANGUAGE_LAYER_COUNT))
    if selected_layers != expected_layers:
        raise RuntimeError(
            "Unexpected Qwen3-VL language layer layout: "
            f"expected={expected_layers} actual={selected_layers}"
        )
    expected_target_count = LANGUAGE_LAYER_COUNT * 4
    if len(targets) != expected_target_count:
        raise RuntimeError(
            "Language attention target audit failed: "
            f"expected_targets={expected_target_count} actual={len(targets)} "
            f"per_layer={dict(per_layer)}"
        )
    if any(per_layer[index] != 4 for index in expected_layers):
        raise RuntimeError(
            "Each language layer must expose exactly q/k/v/o Attention targets; "
            f"per_layer={dict(per_layer)}"
        )
    print(
        "[PATHVQA_LORA_LANGUAGE_TARGET_AUDIT] "
        f"selected_layers_0based={selected_layers} target_count={len(targets)}"
    )
    for name in targets:
        print(f"  - {name}")
    return targets, selected_layers


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


def expected_lora_parameter_count(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
) -> int:
    modules = dict(model.named_modules())
    total = 0
    for name in target_modules:
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise RuntimeError(f"LoRA target is not a Linear module: {name}")
        total += rank * (module.in_features + module.out_features)
    return total


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else (data_root / ".hf_cache").resolve()
    )
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    trainer_dir = output_dir / "trainer"
    final_dir = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Base model path not found: {model_path}")

    seed_everything(args.seed, args.data_seed)
    model, processor = load_model_and_processor(model_path)
    visual_targets, selected_vision_layers = find_visual_attention_targets(
        model,
        last_n_layers=args.last_n_vision_layers,
    )
    language_targets: List[str] = []
    selected_language_layers: List[int] = []
    if args.target_scope == "full_model":
        language_targets, selected_language_layers = (
            find_language_attention_targets(model)
        )
    target_modules = visual_targets + language_targets
    if len(target_modules) != len(set(target_modules)):
        raise RuntimeError("Duplicate LoRA target modules detected")
    expected_target_count = len(selected_vision_layers) * 2
    if args.target_scope == "full_model":
        expected_target_count += LANGUAGE_LAYER_COUNT * 4
    if len(target_modules) != expected_target_count:
        raise RuntimeError(
            "Combined LoRA target audit failed: "
            f"scope={args.target_scope} expected={expected_target_count} "
            f"actual={len(target_modules)}"
        )
    print(
        "[PATHVQA_LORA_COMBINED_TARGET_AUDIT] "
        f"scope={args.target_scope} visual={len(visual_targets)} "
        f"language={len(language_targets)} total={len(target_modules)}"
    )
    predicted_trainable = expected_lora_parameter_count(
        model,
        target_modules,
        args.rank,
    )
    print(
        "[PATHVQA_LORA_PREDICTED_PARAMETER_AUDIT] "
        f"rank={args.rank} predicted_trainable={predicted_trainable}"
    )

    dataset = PathVQADataset(
        processor=processor,
        data_root=data_root,
        split="train",
        cache_dir=cache_dir,
        total_limit=None,
        ce_enabled=True,
        seed=args.data_seed,
        deterministic_sampling=True,
        max_length=args.max_length,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.config.use_cache = False
    lora_alpha = args.rank * 2
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        use_dora=False,
    )
    model = get_peft_model(model, peft_config)
    parameter_counts = count_parameters(model)
    if int(parameter_counts["trainable"]) != predicted_trainable:
        raise RuntimeError(
            "LoRA predicted/actual parameter mismatch: "
            f"predicted={predicted_trainable} "
            f"actual={int(parameter_counts['trainable'])}"
        )
    target_scope = (
        "full_model_attention"
        if args.target_scope == "full_model"
        else (
            "visual_all_attention"
            if args.last_n_vision_layers == VISION_LAYER_COUNT
            else f"visual_last{args.last_n_vision_layers}_attention"
        )
    )
    if (
        args.target_scope == "visual"
        and args.rank == DEFAULT_RANK
        and args.last_n_vision_layers == 8
    ):
        actual_trainable = int(parameter_counts["trainable"])
        if actual_trainable != LAST8_EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError(
                "Last8 visual LoRA parameter audit failed: "
                f"expected={LAST8_EXPECTED_TRAINABLE_PARAMETERS} "
                f"actual={actual_trainable}"
            )
        print(
            "[PATHVQA_LORA_EXPECTED_PARAMETER_AUDIT] "
            f"expected={LAST8_EXPECTED_TRAINABLE_PARAMETERS} "
            f"actual={actual_trainable} pass=True"
        )
    if args.expected_trainable_parameters is not None:
        actual_trainable = int(parameter_counts["trainable"])
        if actual_trainable != args.expected_trainable_parameters:
            raise RuntimeError(
                "LoRA parameter audit failed: "
                f"expected={args.expected_trainable_parameters} "
                f"actual={actual_trainable}"
            )
        print(
            "[PATHVQA_LORA_EXPECTED_PARAMETER_AUDIT] "
            f"expected={args.expected_trainable_parameters} "
            f"actual={actual_trainable} pass=True"
        )
    print(
        "[PATHVQA_LORA_CONFIG] "
        f"method=lora target_scope={target_scope} "
        f"epochs={args.epochs} micro_batch={args.batch_size} "
        f"gradient_accumulation={args.gradient_accumulation} "
        f"rank={args.rank} alpha={lora_alpha} dropout={LORA_DROPOUT} "
        f"learning_rate={args.learning_rate} "
        f"trainable={parameter_counts['trainable']} total={parameter_counts['total']} "
        f"ratio={parameter_counts['trainable_ratio']:.6%}"
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=args.epochs,
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
        fp16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.data_seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VisualLoRACollator(processor),
        processing_class=processor,
        callbacks=[EpochAdapterCheckpointCallback(processor, output_dir)],
    )
    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    report = {
        "status": "pass",
        "experiment": args.experiment_name,
        "method": "lora",
        "target_scope": target_scope,
        "rank": args.rank,
        "alpha": lora_alpha,
        "dropout": LORA_DROPOUT,
        "epochs": args.epochs,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "selected_vision_layers_0based": selected_vision_layers,
        "selected_language_layers_0based": selected_language_layers,
        "target_modules": target_modules,
        "parameter_counts": parameter_counts,
        "predicted_trainable_parameters": predicted_trainable,
        "dataset_samples": len(dataset),
        "train_metrics": train_result.metrics,
        "model_path": str(model_path),
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "final_checkpoint": str(final_dir),
    }
    report_path = output_dir / "train_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[PATHVQA_LORA_TRAIN_PASS] checkpoint={final_dir} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
