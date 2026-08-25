"""Static soft-prompt support for frozen Qwen3-VL models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


PROMPT_CONFIG_NAME = "prompt_config.json"
PROMPT_WEIGHTS_NAME = "soft_prompt.pt"


class StaticPromptTuningModel(nn.Module):
    """Prepend trainable embeddings while leaving every base parameter frozen."""

    def __init__(
        self,
        base_model: nn.Module,
        prompt_length: int = 20,
        init_seed: int = 44,
    ) -> None:
        super().__init__()
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        self.base_model = base_model
        self.prompt_length = int(prompt_length)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        embeddings = self.base_model.get_input_embeddings().weight.detach()
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        sampled_rows = torch.randint(
            embeddings.shape[0],
            (self.prompt_length,),
            generator=generator,
        )
        initial_prompt = embeddings[sampled_rows.to(embeddings.device)].clone()
        # Keep the tiny trainable state in fp32 even when the frozen backbone is bf16.
        self.soft_prompt = nn.Parameter(initial_prompt.float())

        self.config = self.base_model.config
        self.generation_config = getattr(self.base_model, "generation_config", None)

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def _expand_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        labels = batch.pop("labels", None)
        batch_size = input_ids.shape[0]
        device = input_ids.device

        token_embeddings = self.get_input_embeddings()(input_ids)
        prompt = self.soft_prompt.to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat((prompt, token_embeddings), dim=1)

        # Dummy ids keep Qwen3-VL's image-placeholder mask aligned with inputs_embeds.
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        prompt_ids = torch.full(
            (batch_size, self.prompt_length),
            pad_id,
            dtype=input_ids.dtype,
            device=device,
        )
        prompt_mask = torch.ones(
            (batch_size, self.prompt_length),
            dtype=attention_mask.dtype,
            device=device,
        )
        expanded = {
            **batch,
            "input_ids": torch.cat((prompt_ids, input_ids), dim=1),
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.cat((prompt_mask, attention_mask), dim=1),
        }
        if labels is not None:
            ignored = torch.full(
                (batch_size, self.prompt_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            expanded["labels"] = torch.cat((ignored, labels), dim=1)
        return expanded

    def forward(self, **kwargs: Any) -> Any:
        return self.base_model(**self._expand_inputs(dict(kwargs)))

    def generate(self, **kwargs: Any) -> Any:
        return self.base_model.generate(**self._expand_inputs(dict(kwargs)))

    def save_prompt(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "static_prompt_tuning",
            "prompt_length": self.prompt_length,
            "hidden_size": int(self.soft_prompt.shape[-1]),
        }
        with (output_path / PROMPT_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        torch.save(
            {"soft_prompt": self.soft_prompt.detach().cpu()},
            output_path / PROMPT_WEIGHTS_NAME,
        )

    def load_prompt(self, checkpoint_dir: str | Path) -> None:
        checkpoint_path = Path(checkpoint_dir)
        with (checkpoint_path / PROMPT_CONFIG_NAME).open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        if int(config["prompt_length"]) != self.prompt_length:
            raise ValueError("Prompt length does not match checkpoint")
        state = torch.load(
            checkpoint_path / PROMPT_WEIGHTS_NAME,
            map_location="cpu",
            weights_only=True,
        )
        prompt = state["soft_prompt"]
        if tuple(prompt.shape) != tuple(self.soft_prompt.shape):
            raise ValueError(
                f"Soft prompt shape mismatch: {tuple(prompt.shape)} vs "
                f"{tuple(self.soft_prompt.shape)}"
            )
        self.soft_prompt.data.copy_(prompt.to(self.soft_prompt.device))
