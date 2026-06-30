#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click LoRA training for Qwen3-VL visual encoder comparison.

Run from this folder:
    python trainLora.py
"""

import json
import os
import random
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
    "work_dir": "./runs/lora",
    "seed": 42,
    "max_length": 1024,
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
        "ranks": [8, 16, 32],
        "alpha_multiplier": 2,
        "dropout": 0.05,
        "bias": "none",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_files(paths: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[data] skip missing json: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[data] skip non-list json: {path}")
            continue
        for item in data:
            if isinstance(item, dict) and item.get("conversations"):
                copied = dict(item)
                copied["__source_json_path"] = path
                items.append(copied)
    return items


def build_image_index(image_dirs: List[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            print(f"[data] skip missing image dir: {image_dir}")
            continue
        for root, _, files in os.walk(image_dir):
            for filename in files:
                index.setdefault(filename, os.path.join(root, filename))
    return index


class QwenVLConversationDataset(Dataset):
    def __init__(self, processor: Any, cfg: Dict[str, Any]) -> None:
        self.processor = processor
        self.max_length = CFG["max_length"]
        self.image_index = build_image_index(cfg["image_dirs"])
        raw_items = load_json_files(cfg["json_files"])
        self.items = [x for x in raw_items if self._resolve_image(x) is not None]
        if cfg["shuffle"]:
            random.shuffle(self.items)
        limit = cfg["sample_limit"]
        if limit is not None:
            self.items = self.items[:limit]
        print(f"[data] usable multimodal samples: {len(self.items)}")
        if not self.items:
            raise RuntimeError("No usable multimodal training samples found.")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_image(self, item: Dict[str, Any]) -> Optional[str]:
        image_name = item.get("image")
        if not image_name:
            return None
        if os.path.isabs(image_name) and os.path.exists(image_name):
            return image_name
        return self.image_index.get(os.path.basename(image_name))

    def _to_qwen_messages(self, conversations: List[Dict[str, str]], image: Image.Image) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        image_used = False
        for turn in conversations:
            role = "user" if turn.get("from") == "human" else "assistant"
            text = str(turn.get("value", "")).strip()
            if role == "user" and not image_used:
                text = text.replace("<image>", "").strip()
                content = [{"type": "image", "image": image}, {"type": "text", "text": text}]
                image_used = True
            else:
                content = [{"type": "text", "text": text.replace("<image>", "").strip()}]
            messages.append({"role": role, "content": content})
        return messages

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        image_path = self._resolve_image(item)
        if image_path is None:
            raise RuntimeError(f"Image disappeared for sample index {idx}")
        image = Image.open(image_path).convert("RGB")
        messages = self._to_qwen_messages(item["conversations"], image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            images=image,
            text=text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1 else v for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_positions = (inputs["input_ids"] == im_start_id).nonzero(as_tuple=True)[0]
        if len(assistant_positions) >= 2:
            labels[: assistant_positions[1].item() + 2] = -100
        inputs["labels"] = labels
        return inputs


class QwenVLDataCollator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, Any] = {}
        keys = set().union(*(feature.keys() for feature in features))
        for key in keys:
            values = [feature.get(key) for feature in features]
            if any(value is None for value in values):
                continue
            if key == "pixel_values":
                batch[key] = torch.cat(values, dim=0)
            elif key == "image_grid_thw":
                batch[key] = torch.stack(values, dim=0) if values[0].dim() == 1 else torch.cat(values, dim=0)
            elif torch.is_tensor(values[0]):
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = values
        return batch


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


def find_full_linear_targets(model: nn.Module) -> List[str]:
    blocked_keywords = ("lm_head", "embed_tokens")
    targets: List[str] = []
    for name, module in model.named_modules():
        lname = name.lower()
        if not isinstance(module, nn.Linear):
            continue
        if any(keyword in lname for keyword in blocked_keywords):
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError("No full-model Linear modules found for LoRA. Please inspect Qwen3-VL module names.")
    print(f"[lora] full-model Linear target modules: {len(targets)}")
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
    )
    model = get_peft_model(model, lora_cfg)
    counts = count_parameters(model)
    print_trainable_banner(f"LoRA rank {rank}", counts)
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
        data_collator=QwenVLDataCollator(),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    print(f"[done] rank {rank} LoRA saved to {output_dir / 'final'}")


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    set_seed(CFG["seed"])
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)

    probe_model, processor = load_model_and_processor(CFG["model_path"])
    target_modules = find_full_linear_targets(probe_model)
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = QwenVLConversationDataset(processor=processor, cfg=CFG["data"])
    for rank in CFG["lora"]["ranks"]:
        train_one_rank(rank, dataset, target_modules)
    print("All LoRA rank experiments finished.")


if __name__ == "__main__":
    main()
