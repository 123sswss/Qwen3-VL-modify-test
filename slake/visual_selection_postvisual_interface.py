"""PathVQA inference for V3, retaining V1's original chat/question protocol."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from processingWithMMRL import Qwen3ProcessorWithV3
from slake.visual_selection_postvisual import CONFIG_NAME, VisualSelectionPostvisualModel
from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface
from slake.visual_selection_prefix_interface import generate_with_timing


class VisualSelectionPostvisualInterface(VisualSelectionPrefixInterface):
    def __init__(self, checkpoint_path: str, base_model_path: str) -> None:
        checkpoint = Path(checkpoint_path)
        with (checkpoint / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        native = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.processor = Qwen3ProcessorWithV3(
            image_processor=native.image_processor, tokenizer=native.tokenizer,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        self.model = VisualSelectionPostvisualModel(base, init_seed=int(config["init_seed"]))
        self.model.load_v3(checkpoint)
        self.model.eval()
        self.device = next(base.parameters()).device
        self.last_generation_timing = None
        print(f"[V3_INTERFACE] checkpoint={checkpoint} parameters={self.model._audit_parameters()}")

    def infer(self, image, prompt: str, max_new_tokens: int = 32,
              temperature: float = 0.0, *, question: str) -> str:
        inputs = self.prepare_inputs(image, prompt, question=question)
        prompt_length = inputs["input_ids"].shape[1]
        moved = {
            key: value.to(device=self.device,
                          dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
            if torch.is_tensor(value) else value for key, value in inputs.items()
        }
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "use_cache": True}
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.inference_mode():
            output, self.last_generation_timing = generate_with_timing(self.model, moved, kwargs)
        # The processor already reserved P20, so no second +20 here.
        generated = output[:, prompt_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
