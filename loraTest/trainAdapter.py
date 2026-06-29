#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click bottleneck Adapter training for Qwen3-VL visual encoder comparison.

Run from this folder:
    python trainAdapter.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from trainLora import (
    QwenVLConversationDataset,
    QwenVLDataCollator,
    count_parameters,
    load_model_and_processor,
    print_trainable_banner,
    set_seed,
)


CFG = {
    "model_path": "/root/autodl-tmp/model",
    "work_dir": "./runs/adapter",
    "seed": 42,
    "data": {
        "json_files": [
            "/root/autodl-tmp/dataset/1json.json",
            "/root/autodl-tmp/dataset/2conv_c.json",
            "/root/autodl-tmp/dataset/1conv_c.json",
            "/root/autodl-tmp/dataset/4conv_c.json",
            "/root/autodl-tmp/dataset/14json.json",
            "/root/autodl-tmp/dataset/prof_test.json",
            "/root/autodl-tmp/dataset/test2_train.json",
            "/root/autodl-tmp/dataset/test7_train.json",
            "/root/autodl-tmp/dataset/llava_instruct_150k.json",
            "/root/autodl-tmp/dataset/gen_test.json",
            "/root/autodl-tmp/dataset/conversation_58k.json",
        ],
        "image_dirs": [
            "/root/autodl-tmp/dataset/1/train",
            "/root/autodl-tmp/dataset/2/train",
            "/root/autodl-tmp/dataset/4/train",
            "/root/autodl-tmp/dataset/14",
            "/root/autodl-tmp/dataset/gen/train2017",
            "/root/autodl-tmp/dataset/gen/val2017",
        ],
        "sample_limit": 20000,
        "shuffle": True,
    },
    "train": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "logging_steps": 20,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "report_to": "none",
    },
    "adapter": {
        "bottleneck_dim": 64,
        "dropout": 0.05,
        "target_limit": None,
    },
}


class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.act(self.down(x))))


class LinearWithResidualAdapter(nn.Module):
    def __init__(self, base: nn.Linear, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        dtype = base.weight.dtype
        device = base.weight.device
        self.adapter = ResidualBottleneckAdapter(base.out_features, bottleneck_dim, dropout, dtype=dtype, device=device)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        y = self.base(*args, **kwargs)
        return y + self.adapter(y)


def get_parent_module(model: nn.Module, module_name: str) -> Any:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def find_visual_adapter_targets(model: nn.Module) -> List[str]:
    visual_keywords = ("visual", "vision", "vision_tower", "vision_model")
    blocked_keywords = ("lm_head", "language", "text", "embed_tokens")
    targets: List[str] = []
    for name, module in model.named_modules():
        lname = name.lower()
        if not isinstance(module, nn.Linear):
            continue
        if not any(keyword in lname for keyword in visual_keywords):
            continue
        if any(keyword in lname for keyword in blocked_keywords):
            continue
        targets.append(name)
    limit = CFG["adapter"]["target_limit"]
    if limit is not None:
        targets = targets[: int(limit)]
    if not targets:
        raise RuntimeError("No visual Linear modules found for Adapter. Please inspect Qwen3-VL module names.")
    print(f"[adapter] visual Linear target modules: {len(targets)}")
    for name in targets[:20]:
        print(f"  - {name}")
    if len(targets) > 20:
        print(f"  ... and {len(targets) - 20} more")
    return targets


def inject_adapters(model: nn.Module, target_modules: List[str]) -> None:
    for name in target_modules:
        parent, child_name = get_parent_module(model, name)
        base = parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Adapter target is not nn.Linear: {name} -> {type(base)}")
        wrapped = LinearWithResidualAdapter(
            base,
            bottleneck_dim=CFG["adapter"]["bottleneck_dim"],
            dropout=CFG["adapter"]["dropout"],
        )
        if child_name.isdigit():
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)


def save_adapter_only(model: nn.Module, processor: Any, output_dir: Path, target_modules: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".adapter." in k}
    torch.save(adapter_state, output_dir / "adapter_model.bin")
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_modules": target_modules,
                "bottleneck_dim": CFG["adapter"]["bottleneck_dim"],
                "dropout": CFG["adapter"]["dropout"],
                "base_model_path": CFG["model_path"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    processor.save_pretrained(str(output_dir))


def train_adapter(dataset: Dataset, target_modules: List[str]) -> None:
    model, processor = load_model_and_processor(CFG["model_path"])
    if CFG["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    inject_adapters(model, target_modules)
    for name, param in model.named_parameters():
        param.requires_grad = ".adapter." in name

    counts = count_parameters(model)
    print_trainable_banner("Adapter", counts)

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
        save_strategy="no",
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
        data_collator=QwenVLDataCollator(),
        processing_class=processor,
    )
    trainer.train()
    save_adapter_only(model, processor, output_dir / "final", target_modules)
    print(f"[done] Adapter saved to {output_dir / 'final'}")


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    set_seed(CFG["seed"])
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)

    probe_model, processor = load_model_and_processor(CFG["model_path"])
    target_modules = find_visual_adapter_targets(probe_model)
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = QwenVLConversationDataset(processor=processor, cfg=CFG["data"])
    train_adapter(dataset, target_modules)
    print("Adapter experiment finished.")


if __name__ == "__main__":
    main()
