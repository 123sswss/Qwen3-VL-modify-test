"""PathVQA inference interface for the independent V0 checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from slake.visual_selection_offset import CONFIG_NAME, VisualSelectionOffsetModel, locate_question_mask

try:
    from generation_timing import generate_with_timing
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loraTest"))
    from generation_timing import generate_with_timing


class VisualSelectionOffsetInterface:
    requires_raw_question = True

    def __init__(self, checkpoint_path: str, base_model_path: str) -> None:
        checkpoint = Path(checkpoint_path)
        with (checkpoint / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        self.model = VisualSelectionOffsetModel(base, init_seed=int(config["init_seed"]))
        self.model.load_v0(checkpoint)
        self.model.eval()
        self.device = next(base.parameters()).device
        self.last_generation_timing = None
        print(f"[V0_INTERFACE] checkpoint={checkpoint} parameters={self.model._audit_parameters()}")

    def infer(
        self, image: Image.Image, prompt: str,
        max_new_tokens: int = 32, temperature: float = 0.0,
        *, question: str,
    ) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = dict(self.processor(images=image, text=formatted, return_tensors="pt"))
        raw_mask = locate_question_mask(
            inputs["input_ids"][0], question, self.processor.tokenizer,
            prompt_text=prompt,
        )
        inputs["question_mask"] = raw_mask.unsqueeze(0)
        original_length = inputs["input_ids"].shape[1]
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(
                    device=self.device,
                    dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
                )
            else:
                moved[key] = value
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.inference_mode():
            output, self.last_generation_timing = generate_with_timing(
                self.model, moved, kwargs,
            )
        self.last_generation_timing["generated_token_count"] -= 20
        generated = output[:, original_length + 20:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
