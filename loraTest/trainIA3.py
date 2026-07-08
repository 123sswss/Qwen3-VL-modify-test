#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click IA3 training for Qwen3-VL visual encoder comparison.

Run from this folder:
    python trainIA3.py
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import IA3Config, TaskType, get_peft_model

from trainLora import (
    build_fair_collator,
    build_fair_dataset,
    count_parameters,
    load_model_and_processor,
    print_trainable_banner,
    set_seed,
)


CFG = {
    "model_path": "/root/autodl-tmp/model",
    "work_dir": "./runs/ia3",
    "seed": 42,
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
}


def find_ia3_kv_ffn_targets(model: nn.Module) -> Tuple[List[str], List[str]]:
    blocked_keywords = ("lm_head", "embed_tokens")
    attention_suffixes = ("k_proj", "v_proj")
    feedforward_suffixes = ("down_proj", "fc2")
    target_modules: List[str] = []
    feedforward_modules: List[str] = []

    for name, module in model.named_modules():
        lname = name.lower()
        if not isinstance(module, nn.Linear):
            continue
        if any(keyword in lname for keyword in blocked_keywords):
            continue
        is_attention_target = lname.endswith(attention_suffixes)
        is_feedforward_target = lname.endswith(feedforward_suffixes)
        if not (is_attention_target or is_feedforward_target):
            continue
        target_modules.append(name)
        if is_feedforward_target:
            feedforward_modules.append(name)

    if not target_modules:
        raise RuntimeError("No k/v or FFN Linear modules found for IA3. Please inspect Qwen3-VL module names.")
    if not feedforward_modules:
        feedforward_modules = list(target_modules)

    print(f"[ia3] k/v + FFN target modules: {len(target_modules)}")
    print(f"[ia3] FFN feedforward modules: {len(feedforward_modules)}")
    for name in target_modules[:20]:
        print(f"  - {name}")
    if len(target_modules) > 20:
        print(f"  ... and {len(target_modules) - 20} more")
    return target_modules, feedforward_modules


def train_ia3(dataset: Dataset, target_modules: List[str], feedforward_modules: List[str]) -> None:
    model, processor = load_model_and_processor(CFG["model_path"])
    if CFG["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    ia3_cfg = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        feedforward_modules=feedforward_modules,
    )
    model = get_peft_model(model, ia3_cfg)
    counts = count_parameters(model)
    print_trainable_banner("IA3", counts)
    model.print_trainable_parameters()

    output_dir = Path(CFG["work_dir"])
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
    print(f"[done] IA3 saved to {output_dir / 'final'}")


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    set_seed(CFG["seed"])
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)

    probe_model, processor = load_model_and_processor(CFG["model_path"])
    target_modules, feedforward_modules = find_ia3_kv_ffn_targets(probe_model)
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = build_fair_dataset(processor=processor, data_cfg=CFG["data"])
    train_ia3(dataset, target_modules, feedforward_modules)
    print("IA3 experiment finished.")


if __name__ == "__main__":
    main()
