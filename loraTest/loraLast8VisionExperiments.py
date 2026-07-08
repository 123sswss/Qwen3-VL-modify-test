#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank-32 LoRA comparison focused on the last 8 vision-encoder layers.

Experiments:
  1. Full Linear LoRA on the last 8 vision encoder layers.
  2. Attention-only LoRA on the last 8 vision encoder layers.

Run from this folder:
    python loraLast8VisionExperiments.py
"""

import importlib.util
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

from trainLora import (
    build_fair_collator,
    build_fair_dataset,
    count_parameters,
    load_model_and_processor,
    print_trainable_banner,
    set_seed,
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


CFG = {
    "base_model_path": "/root/autodl-tmp/model",
    "work_dir": "./runs/lora_vision_last8",
    "seed": 42,
    "rank": 32,
    "last_n_layers": 8,
    "lora": {
        "alpha_multiplier": 2,
        "dropout": 0.05,
        "bias": "none",
    },
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
    "eval": {
        "enabled": True,
        "test_script_path": "../test/test.py",
        "json_paths": [
            "/root/autodl-tmp/dataset/test2_val.json",
            "/root/autodl-tmp/dataset/seen_simple/llava_test.json",
        ],
        "image_dirs": [
            "/root/autodl-tmp/dataset/2/train",
            "/root/autodl-tmp/dataset/seen_simple/image",
        ],
        "max_new_tokens": 256,
        "temperature": 0.2,
        "do_sample": False,
    },
    "experiments": [
        {
            "name": "last8_full_linear_rank32",
            "target_mode": "last8_full_linear",
            "description": "Full Linear LoRA on last 8 vision encoder layers",
        },
        {
            "name": "last8_attention_rank32",
            "target_mode": "last8_attention",
            "description": "Attention-only LoRA on last 8 vision encoder layers",
        },
    ],
}


def extract_vision_layer_index(module_name: str) -> Optional[int]:
    lname = module_name.lower()
    if not any(keyword in lname for keyword in ("visual", "vision", "vision_tower", "vision_model")):
        return None
    patterns = (
        r"(?:visual|vision|vision_tower|vision_model).*?\.(?:blocks|layers)\.(\d+)(?:\.|$)",
        r"(?:visual|vision|vision_tower|vision_model).*?\.encoder\.layers\.(\d+)(?:\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, lname)
        if match:
            return int(match.group(1))
    return None


def find_last8_vision_targets(model: nn.Module, target_mode: str) -> List[str]:
    blocked_keywords = ("lm_head", "language", "text", "embed_tokens")
    attention_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj")
    layer_indexes = sorted({
        idx
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        for idx in [extract_vision_layer_index(name)]
        if idx is not None
    })
    if not layer_indexes:
        raise RuntimeError("No indexed vision encoder Linear modules found. Inspect Qwen3-VL visual module names.")

    selected_layers = set(layer_indexes[-int(CFG["last_n_layers"]):])
    targets: List[str] = []
    for name, module in model.named_modules():
        lname = name.lower()
        if not isinstance(module, nn.Linear):
            continue
        if any(keyword in lname for keyword in blocked_keywords):
            continue
        layer_idx = extract_vision_layer_index(name)
        if layer_idx not in selected_layers:
            continue
        if target_mode == "last8_attention":
            is_attention_module = ".attn." in lname or ".attention." in lname or "self_attn" in lname
            if not is_attention_module or not lname.endswith(attention_suffixes):
                continue
        targets.append(name)

    if not targets:
        raise RuntimeError(f"No LoRA target modules found for target_mode={target_mode!r}.")

    print(f"[last8] detected vision layer indexes: {layer_indexes}")
    print(f"[last8] selected last {CFG['last_n_layers']} layers: {sorted(selected_layers)}")
    print(f"[last8] target_mode={target_mode} target_modules={len(targets)}")
    for name in targets[:40]:
        print(f"  - {name}")
    if len(targets) > 40:
        print(f"  ... and {len(targets) - 40} more")
    return targets


def train_experiment(exp_cfg: Dict[str, Any], dataset: Dataset, target_modules: List[str]) -> Path:
    rank = int(CFG["rank"])
    model, processor = load_model_and_processor(CFG["base_model_path"])
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
    print_trainable_banner(f"LoRA {exp_cfg['name']}", counts)
    model.print_trainable_parameters()

    output_dir = Path(CFG["work_dir"]) / exp_cfg["name"]
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
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[done] {exp_cfg['name']} saved to {final_dir}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_dir


def load_test_module(test_script_path: str) -> Any:
    script_path = Path(test_script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"test.py not found: {script_path}")
    spec = importlib.util.spec_from_file_location("qwen_lora_last8_eval_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import test module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_model(model_path: str) -> Any:
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
            return loader.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            last_error = exc
            print(f"[model] loader {loader.__name__} failed: {exc}")
    raise RuntimeError(f"Failed to load base model from {model_path}") from last_error


def move_inputs_to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            moved[key] = value.to(device=device)
    return moved


class LoraModelInterface:
    def __init__(self, adapter_path: Path, base_model_path: str) -> None:
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = load_base_model(base_model_path)
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[lora-last8] loaded adapter: {adapter_path}")

    def infer(self, image: Image.Image, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = move_inputs_to_device(inputs, self.device)
        generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": bool(CFG["eval"]["do_sample"])}
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def evaluate_experiment(exp_cfg: Dict[str, Any], adapter_dir: Path, test_module: Any) -> Dict[str, Any]:
    print(f"\n========== Evaluating {exp_cfg['name']} ==========")
    model = LoraModelInterface(adapter_dir, CFG["base_model_path"])
    summary = test_module.run_evaluation(
        CFG["eval"]["json_paths"],
        model,
        CFG["eval"]["image_dirs"],
        max_new_tokens=CFG["eval"]["max_new_tokens"],
        temperature=CFG["eval"]["temperature"],
    )
    print(f"[summary] {exp_cfg['name']} score={summary.get('score')} evaluated={summary.get('evaluated')}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    set_seed(CFG["seed"])
    Path(CFG["work_dir"]).mkdir(parents=True, exist_ok=True)

    probe_model, processor = load_model_and_processor(CFG["base_model_path"])
    targets_by_mode = {
        exp["target_mode"]: find_last8_vision_targets(probe_model, exp["target_mode"])
        for exp in CFG["experiments"]
    }
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = build_fair_dataset(processor=processor, data_cfg=CFG["data"])
    test_module = load_test_module(CFG["eval"]["test_script_path"]) if CFG["eval"]["enabled"] else None

    summaries = {}
    for exp in CFG["experiments"]:
        final_dir = train_experiment(exp, dataset, targets_by_mode[exp["target_mode"]])
        if test_module is not None:
            summaries[exp["name"]] = evaluate_experiment(exp, final_dir, test_module)

    if summaries:
        print("\n========== Last-8 vision LoRA experiment summaries ==========")
        for name, summary in summaries.items():
            print(f"{name}: score={summary.get('score')} evaluated={summary.get('evaluated')}")


if __name__ == "__main__":
    main()
