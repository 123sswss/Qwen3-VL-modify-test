"""Inference interface for GRASP reproduction checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from slake.grasp_prompt_tuning import GRASP_CONFIG_NAME, GRASPPromptTuningModel

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


class GRASPModelInterface:
    def __init__(self, checkpoint_path: str, base_model_path: str) -> None:
        checkpoint = Path(checkpoint_path).resolve()
        with (checkpoint / GRASP_CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("question_source") != "raw_question_only" or config.get(
            "prompt_placement"
        ) != "before_visual_segment":
            raise ValueError(
                "This checkpoint uses the superseded GRASP approximation; "
                "retrain with raw-question encoding and paper-ordered Prompt placement"
            )
        self.processor = AutoProcessor.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = GRASPPromptTuningModel(
            base_model,
            tokenizer=self.processor.tokenizer,
            block_count=int(config["block_count"]),
            bottleneck_dim=int(config["bottleneck_dim"]),
            prompt_init_std=float(config["prompt_init_std"]),
            prompt_init_mode=str(config.get("prompt_init_mode", "gaussian")),
            position_encoding_mode=str(
                config.get("position_encoding", "fixed_2d_sincos")
            ),
            init_seed=int(config["init_seed"]),
        )
        self.model.load_grasp(checkpoint)
        self.model.eval()
        self.device = next(base_model.parameters()).device
        self.last_generation_timing = None
        print(
            "[grasp] "
            f"loaded={checkpoint} blocks={self.model.block_count} "
            f"bottleneck={self.model.bottleneck_dim} alpha=1.5 "
            f"prompt_init={self.model.prompt_init_mode} "
            f"position_encoding={self.model.position_encoding_mode} "
            "question=raw_question_only_frozen_llm_last_hidden_mean "
            "visual=post_merger_grid prompt_placement=before_visual_segment"
        )

    def reset_inference_state(self) -> None:
        self.last_generation_timing = None

    def infer(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _move_inputs(
            dict(self.processor(images=image, text=text, return_tensors="pt")),
            self.device,
        )
        question_inputs = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs["grasp_question_input_ids"] = question_inputs["input_ids"].to(
            self.device
        )
        inputs["grasp_question_attention_mask"] = question_inputs[
            "attention_mask"
        ].to(self.device)
        original_length = int(inputs["input_ids"].shape[-1])
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids, self.last_generation_timing = generate_with_timing(
                self.model, inputs, generate_kwargs
            )
        generated = output_ids[:, original_length + self.model.prompt_length :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


__all__ = ["GRASPModelInterface"]
