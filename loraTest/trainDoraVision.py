#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click DoRA training for Qwen3-VL visual encoder comparison.

Run from this folder:
    python trainDoraVision.py
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
from data_pipeline import FourViewMMRLDataset, MMRLDataCollator

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


CFG = {
    "model_path": "/root/autodl-tmp/model",
    "work_dir": "./runs/dora_vision_attn",
    "seed": 42,
    "max_length": 1024,
    "data": {
        "expert_json": [
            "/root/autodl-tmp/dataset/1json.json",
            "/root/autodl-tmp/dataset/2conv_c.json",
            "/root/autodl-tmp/dataset/1conv_c.json",
            "/root/autodl-tmp/dataset/4conv_c.json",
            "/root/autodl-tmp/dataset/14json.json",
            "/root/autodl-tmp/dataset/prof_test.json",
            "/root/autodl-tmp/dataset/test2_train.json",
            "/root/autodl-tmp/dataset/test7_train.json",
        ],
        "expert_img_dir": [
            "/root/autodl-tmp/dataset/1/train",
            "/root/autodl-tmp/dataset/2/train",
            "/root/autodl-tmp/dataset/4/train",
            "/root/autodl-tmp/dataset/14",
        ],
        "general_json": [
            "/root/autodl-tmp/dataset/llava_instruct_150k.json",
            "/root/autodl-tmp/dataset/gen_test.json",
            "/root/autodl-tmp/dataset/conversation_58k.json",
        ],
        "general_img_dir": [
            "/root/autodl-tmp/dataset/gen/train2017",
            "/root/autodl-tmp/dataset/gen/val2017",
        ],
        "total_limit": 20000,
        "enable_views": ("expert-mm",),
        "mode": "peft_ce",
        "ce_enabled": True,
        "deterministic_sampling": False,
    },
    "train": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "logging_steps": 20,
        "save_steps": 500,
        "save_total_limit": 2,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "report_to": "none",
    },
    "lora": {
        "ranks": [8, 16],
        "alpha_multiplier": 2,
        "dropout": 0.05,
        "bias": "none",
    },
}


def build_fair_dataset(processor: Any, data_cfg: Dict[str, Any]) -> FourViewMMRLDataset:
    return FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=data_cfg["total_limit"],
        enable_views=tuple(data_cfg["enable_views"]),
        mode=data_cfg["mode"],
        ce_enabled=data_cfg["ce_enabled"],
        seed=CFG["seed"],
        deterministic_sampling=data_cfg.get("deterministic_sampling", False),
    )


def build_fair_collator(processor: Any) -> MMRLDataCollator:
    return MMRLDataCollator(processor)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_processor(model_path: str) -> Any:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if CFG["train"]["bf16"] else "auto",
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
            model = loader.from_pretrained(model_path, **model_kwargs)
            return model, processor
        except Exception as exc:
            last_error = exc
            print(f"[model] loader {loader.__name__} failed: {exc}")
    raise RuntimeError(f"Failed to load model from {model_path}") from last_error


def find_vision_attention_linear_targets(model: nn.Module) -> List[str]:
    visual_keywords = ("visual", "vision", "vision_tower", "vision_model")
    blocked_keywords = ("lm_head", "language", "text", "embed_tokens")
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj")
    targets: List[str] = []
    for name, module in model.named_modules():
        lname = name.lower()
        if not isinstance(module, nn.Linear):
            continue
        if any(keyword in lname for keyword in blocked_keywords):
            continue
        if not any(keyword in lname for keyword in visual_keywords):
            continue
        if ".attn." not in lname and ".attention." not in lname and "self_attn" not in lname:
            continue
        if not lname.endswith(target_suffixes):
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError("No vision attention Linear modules found for DoRA. Please inspect Qwen3-VL module names.")
    print(f"[dora-vision] vision attention Linear target modules: {len(targets)}")
    for name in targets[:20]:
        print(f"  - {name}")
    if len(targets) > 20:
        print(f"  ... and {len(targets) - 20} more")
    return targets


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def print_trainable_banner(method: str, counts: Dict[str, int]) -> None:
    bang = "!" * 76
    ratio = counts["trainable"] / counts["total"] if counts["total"] else 0.0
    print(f"\n{bang}")
    print(f"!!! {method} TRAINABLE PARAMS: {counts['trainable']:,}")
    print(f"!!! {method} TOTAL PARAMS:     {counts['total']:,}")
    print(f"!!! {method} TRAINABLE RATIO:  {ratio:.6%}")
    print(f"{bang}\n")


def train_one_rank(rank: int, dataset: Dataset, target_modules: List[str]) -> None:
    model, processor = load_model_and_processor(CFG["model_path"])
    if CFG["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * CFG["lora"]["alpha_multiplier"],
        lora_dropout=CFG["lora"]["dropout"],
        bias=CFG["lora"]["bias"],
        target_modules=target_modules,
        use_dora=True,
    )
    model = get_peft_model(model, lora_cfg)
    counts = count_parameters(model)
    print_trainable_banner(f"DoRA rank {rank}", counts)
    model.print_trainable_parameters()

    output_dir = Path(CFG["work_dir"]) / f"rank{rank}"
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=CFG["train"]["num_train_epochs"],
        per_device_train_batch_size=CFG["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CFG["train"]["gradient_accumulation_steps"],
        learning_rate=CFG["train"]["learning_rate"],
        weight_decay=CFG["train"]["weight_decay"],
        warmup_ratio=CFG["train"]["warmup_ratio"],
        logging_steps=CFG["train"]["logging_steps"],
        save_steps=CFG["train"]["save_steps"],
        save_total_limit=CFG["train"]["save_total_limit"],
        bf16=CFG["train"]["bf16"],
        fp16=CFG["train"]["fp16"],
        gradient_checkpointing=CFG["train"]["gradient_checkpointing"],
        dataloader_num_workers=CFG["train"]["dataloader_num_workers"],
        remove_unused_columns=CFG["train"]["remove_unused_columns"],
        report_to=CFG["train"]["report_to"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=build_fair_collator(processor),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    print(f"[done] rank {rank} DoRA saved to {output_dir / 'final'}")


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    set_seed(CFG["seed"])
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)

    probe_model, processor = load_model_and_processor(CFG["model_path"])
    target_modules = find_vision_attention_linear_targets(probe_model)
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = build_fair_dataset(processor=processor, data_cfg=CFG["data"])
    for rank in CFG["lora"]["ranks"]:
        train_one_rank(rank, dataset, target_modules)
    print("All DoRA vision-attention rank experiments finished.")


if __name__ == "__main__":
    main()
