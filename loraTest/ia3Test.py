#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained Qwen3-VL visual-encoder IA3 adapter with ../test/test.py.

Run from this folder:
    python ia3Test.py
"""

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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
    "adapter_path": "./runs/ia3/final",
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


def load_test_module(test_script_path: str) -> Any:
    script_path = Path(test_script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"test.py not found: {script_path}")
    spec = importlib.util.spec_from_file_location("qwen_ia3_eval_test", script_path)
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


class IA3ModelInterface:
    def __init__(self, adapter_path: str, base_model_path: str) -> None:
        adapter_dir = Path(adapter_path).resolve()
        if not adapter_dir.exists():
            raise FileNotFoundError(f"IA3 adapter not found: {adapter_dir}")
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = load_base_model(base_model_path)
        self.model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[ia3] loaded adapter: {adapter_dir}")

    def infer(self, image: Image.Image, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = move_inputs_to_device(inputs, self.device)
        generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": bool(CFG["generation"]["do_sample"])}
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    test_module = load_test_module(CFG["test_script_path"])
    model = IA3ModelInterface(CFG["adapter_path"], CFG["base_model_path"])
    summary = test_module.run_evaluation(
        CFG["json_paths"],
        model,
        CFG["image_dirs"],
        max_new_tokens=CFG["generation"]["max_new_tokens"],
        temperature=CFG["generation"]["temperature"],
    )
    print(f"[summary] ia3 score={summary.get('score')} evaluated={summary.get('evaluated')}")


if __name__ == "__main__":
    main()
