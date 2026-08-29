"""Inference interface for multimodal Dynamic Prompt checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from slake.dynamic_prompt_tuning import (
    DYNAMIC_PROMPT_CONFIG_NAME,
    DynamicPromptTuningModel,
)

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


class DynamicPromptTuningModelInterface:
    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: str,
        intervention: str = "normal",
        memory_lag: int = 32,
    ) -> None:
        checkpoint = Path(checkpoint_path).resolve()
        with (checkpoint / DYNAMIC_PROMPT_CONFIG_NAME).open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        if config.get("shared_s_memory", False):
            raise RuntimeError(
                "Legacy post-anchor Rep feedback checkpoints are intentionally "
                "unsupported by the corrected raw Shared-S implementation"
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
        sparse_visual = config.get("sparse_visual")
        shared_s_prompt_attention = config.get("shared_s_prompt_attention")
        shared_workspace = config.get("shared_workspace")
        self.model = DynamicPromptTuningModel(
            base_model,
            tokenizer=self.processor.tokenizer,
            prompt_length=int(
                config.get("requested_prompt_length", config["prompt_length"])
            ),
            init_seed=int(config.get("init_seed", 44)),
            attention_dim=int(config["attention_dim"]),
            num_heads=int(config["num_heads"]),
            sparse_visual_anchor_layers=(
                tuple(int(index) for index in sparse_visual["anchor_layers"])
                if sparse_visual is not None
                else None
            ),
            sparse_visual_rep_tokens=(
                int(sparse_visual["rep_token_count"])
                if sparse_visual is not None
                else 8
            ),
            sparse_visual_attention_dim=(
                int(sparse_visual["attention_dim"])
                if sparse_visual is not None
                else 128
            ),
            sparse_visual_heads=(
                int(sparse_visual["num_heads"]) if sparse_visual is not None else 4
            ),
            shared_s_text_mode=str(config.get("shared_s_text_mode", "none")),
            shared_s_attention_dim=(
                int(shared_s_prompt_attention["attention_dim"])
                if shared_s_prompt_attention is not None
                else 128
            ),
            shared_s_heads=(
                int(shared_s_prompt_attention["num_heads"])
                if shared_s_prompt_attention is not None
                else 4
            ),
            shared_s_visual_bottleneck_dim=(
                int(sparse_visual.get("text_anchor_bottleneck_dim", 128))
                if sparse_visual is not None
                else 128
            ),
            shared_workspace=shared_workspace is not None,
            workspace_tokens=(
                int(shared_workspace["tokens"])
                if shared_workspace is not None
                else 32
            ),
            workspace_dim=(
                int(shared_workspace["dim"])
                if shared_workspace is not None
                else 1024
            ),
            workspace_heads=(
                int(shared_workspace["heads"])
                if shared_workspace is not None
                else 16
            ),
            workspace_ffn_dim=(
                int(shared_workspace["ffn_dim"])
                if shared_workspace is not None
                else 4096
            ),
            workspace_text_attention_dim=(
                int(shared_workspace["text_attention_dim"])
                if shared_workspace is not None
                else 1024
            ),
            workspace_text_heads=(
                int(shared_workspace["text_attention_heads"])
                if shared_workspace is not None
                else 16
            ),
            workspace_visual_attention_dim=(
                int(shared_workspace["visual_attention_dim"])
                if shared_workspace is not None
                else 1024
            ),
            workspace_visual_heads=(
                int(shared_workspace["visual_attention_heads"])
                if shared_workspace is not None
                else 16
            ),
        )
        self.model.load_dynamic_prompt(checkpoint)
        self.model.eval()
        self.model.configure_inference_intervention(
            intervention,
            memory_lag=memory_lag,
        )
        self.device = next(base_model.parameters()).device
        self.last_generation_timing = None
        print(
            "[dynamic-prompt] "
            f"loaded={checkpoint} prompt_length={self.model.prompt_length} "
            f"attention_dim={self.model.dynamic_prompt.attention_dim} "
            f"heads={self.model.dynamic_prompt.num_heads} "
            f"sparse_visual={sparse_visual is not None} "
            f"shared_s_text_mode={self.model.shared_s_text_mode} "
            f"shared_workspace={self.model.shared_workspace_enabled} "
            f"train_soft_prompt={self.model.soft_prompt is not None} "
            f"intervention={intervention} memory_lag={memory_lag}"
        )

    def reset_inference_state(self) -> None:
        self.model.reset_inference_intervention_state()

    def inference_intervention_summary(self) -> Dict[str, Any]:
        return self.model.inference_intervention_summary()

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
        return self.processor.batch_decode(generated, skip_special_tokens=True)[
            0
        ].strip()
