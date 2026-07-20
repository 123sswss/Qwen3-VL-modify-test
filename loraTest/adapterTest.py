#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained Qwen3-VL visual-encoder bottleneck Adapter with ../test/test.py.

Run from this folder:
    python adapterTest.py
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor

from generation_timing import generate_with_timing

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
    "adapter_path": "./runs/adapter/final",
    "test_script_path": "../test/test.py",
    "json_paths": [
        "/root/autodl-tmp/dataset/test2_val.json",
        "/root/autodl-tmp/dataset/seen_simple/llava_test.json",
    ],
    "image_dirs": [
        "/root/autodl-tmp/dataset/2/train",
        "/root/autodl-tmp/dataset/seen_simple/image",
    ],
    "generation": {
        "max_new_tokens": 256,
        "temperature": 0.2,
        "do_sample": False,
    },
    "model": {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    },
}


class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
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


def inject_adapters(model: nn.Module, target_modules: List[str], bottleneck_dim: int, dropout: float) -> None:
    for name in target_modules:
        parent, child_name = get_parent_module(model, name)
        base = parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Adapter target is not nn.Linear: {name} -> {type(base)}")
        wrapped = LinearWithResidualAdapter(base, bottleneck_dim=bottleneck_dim, dropout=dropout)
        if child_name.isdigit():
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)


def load_test_module(test_script_path: str) -> Any:
    script_path = Path(test_script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"test.py not found: {script_path}")
    spec = importlib.util.spec_from_file_location("qwen_adapter_eval_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import test module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_model(model_path: str) -> Any:
    model_kwargs = dict(CFG["model"])
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
            moved[key] = value.to(device=device, dtype=CFG["model"]["torch_dtype"])
        else:
            moved[key] = value.to(device=device)
    return moved


class AdapterModelInterface:
    def __init__(self, adapter_path: str, base_model_path: str) -> None:
        adapter_dir = Path(adapter_path).resolve()
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        config_path = adapter_dir / "adapter_config.json"
        state_path = adapter_dir / "adapter_model.bin"
        if not config_path.exists() or not state_path.exists():
            raise FileNotFoundError(f"Adapter config or state missing in {adapter_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = load_base_model(base_model_path)
        inject_adapters(
            self.model,
            adapter_cfg["target_modules"],
            bottleneck_dim=int(adapter_cfg["bottleneck_dim"]),
            dropout=float(adapter_cfg["dropout"]),
        )
        adapter_state = torch.load(state_path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(adapter_state, strict=False)
        if unexpected:
            print(f"[adapter] unexpected state keys: {unexpected[:10]}")
        print(f"[adapter] missing non-adapter keys ignored: {len(missing)}")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[adapter] loaded adapter: {adapter_dir}")

    def infer(self, image: Image.Image, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = move_inputs_to_device(inputs, self.device)
        generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": bool(CFG["generation"]["do_sample"])}
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids, self.last_generation_timing = generate_with_timing(
                self.model, inputs, generate_kwargs
            )
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    test_module = load_test_module(CFG["test_script_path"])
    model = AdapterModelInterface(CFG["adapter_path"], CFG["base_model_path"])
    summary = test_module.run_evaluation(
        CFG["json_paths"],
        model,
        CFG["image_dirs"],
        max_new_tokens=CFG["generation"]["max_new_tokens"],
        temperature=CFG["generation"]["temperature"],
    )
    print(f"[summary] adapter score={summary.get('score')} evaluated={summary.get('evaluated')}")


if __name__ == "__main__":
    main()
