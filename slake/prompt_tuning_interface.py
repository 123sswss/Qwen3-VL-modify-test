"""Inference interface for static Prompt Tuning checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from slake.prompt_tuning import PROMPT_CONFIG_NAME, StaticPromptTuningModel

try:
    from generation_timing import generate_with_timing
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loraTest"))
    from generation_timing import generate_with_timing


def _move_inputs(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            moved[key] = value.to(device=device)
    return moved


class PromptTuningModelInterface:
    def __init__(self, checkpoint_path: str, base_model_path: str) -> None:
        checkpoint = Path(checkpoint_path).resolve()
        with (checkpoint / PROMPT_CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.processor = AutoProcessor.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = StaticPromptTuningModel(
            base_model,
            prompt_length=int(config["prompt_length"]),
        )
        self.model.load_prompt(checkpoint)
        self.model.eval()
        self.device = next(base_model.parameters()).device
        self.last_generation_timing = None
        print(
            "[prompt-tuning] "
            f"loaded={checkpoint} prompt_length={self.model.prompt_length}"
        )

    def infer(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = _move_inputs(dict(inputs), self.device)
        original_length = int(inputs["input_ids"].shape[-1])
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": True,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids, self.last_generation_timing = generate_with_timing(
                self.model, inputs, generate_kwargs
            )
        input_length = original_length + self.model.prompt_length
        generated = output_ids[:, input_length:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

