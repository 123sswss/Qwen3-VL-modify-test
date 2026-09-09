"""CoCoOp-style image-conditioned soft prompts for frozen Qwen3-VL."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
from torch import nn

from slake.prompt_tuning import StaticPromptTuningModel


COCOOP_CONFIG_NAME = "cocoop_prompt_config.json"
COCOOP_WEIGHTS_NAME = "cocoop_prompt.pt"


class CoCoOpStylePromptTuningModel(StaticPromptTuningModel):
    """Condition every static context token on the current frozen image."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        prompt_length: int = 20,
        bottleneck_dim: int | None = None,
        init_seed: int = 44,
    ) -> None:
        super().__init__(base_model, prompt_length=prompt_length, init_seed=init_seed)
        hidden_size = int(self.soft_prompt.shape[-1])
        self.bottleneck_dim = int(bottleneck_dim or max(1, hidden_size // 16))
        if self.bottleneck_dim < 1:
            raise ValueError("CoCoOp bottleneck_dim must be positive")
        device = self.soft_prompt.device
        self.meta_net = nn.Sequential(
            nn.Linear(hidden_size, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, hidden_size),
        ).to(device=device)
        self.visual_token_ids = tuple(
            int(tokenizer.convert_tokens_to_ids(token))
            for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
        )
        if any(token_id < 0 for token_id in self.visual_token_ids):
            raise RuntimeError("CoCoOp-style Prompt could not resolve visual token ids")
        self.init_seed = int(init_seed)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._forward_audited = False

    def trainable_parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        return {
            "soft_prompt": [self.soft_prompt],
            "meta_net": list(self.meta_net.parameters()),
        }

    @staticmethod
    def _masked_visual_mean(
        embeddings: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        counts = visual_mask.sum(dim=1)
        if bool((counts == 0).any()):
            indices = (counts == 0).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"CoCoOp-style Prompt requires an image for every sample: {indices}"
            )
        weights = visual_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        return (embeddings * weights).sum(dim=1) / counts.to(
            dtype=embeddings.dtype
        ).unsqueeze(-1)

    @contextmanager
    def _inject_conditioned_prompt(
        self,
        expanded_ids: torch.Tensor,
    ) -> Iterator[None]:
        language_model = getattr(
            getattr(self.base_model, "model", None), "language_model", None
        )
        if language_model is None:
            raise RuntimeError(
                "CoCoOp-style Prompt requires base_model.model.language_model"
            )

        def condition_prompt(_module: nn.Module, args: Any, kwargs: Dict[str, Any]):
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None or inputs_embeds.ndim != 3:
                return args, kwargs
            if inputs_embeds.shape[1] != expanded_ids.shape[1]:
                # Generation decode steps reuse the conditioned Prompt in KV cache.
                return args, kwargs
            visual_mask = kwargs.get("visual_pos_masks")
            if visual_mask is None:
                visual_mask = expanded_ids.eq(self.visual_token_ids[0])
            visual_mask = visual_mask.to(
                device=inputs_embeds.device,
                dtype=torch.bool,
            )
            image_feature = self._masked_visual_mean(inputs_embeds, visual_mask)
            meta_device = next(self.meta_net.parameters()).device
            prompt_bias = self.meta_net(
                image_feature.float().to(device=meta_device)
            ).to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            conditioned = inputs_embeds.clone()
            conditioned[:, : self.prompt_length] = (
                conditioned[:, : self.prompt_length]
                + prompt_bias.unsqueeze(1)
            )
            kwargs["inputs_embeds"] = conditioned
            static_prompt = inputs_embeds[:, : self.prompt_length]
            self.debug_context = {
                "cocoop_visual_tokens_mean": visual_mask.sum(dim=1)
                .float()
                .mean()
                .detach(),
                "cocoop_image_feature_norm_mean": image_feature.float()
                .norm(dim=-1)
                .mean()
                .detach(),
                "cocoop_prompt_bias_norm_mean": prompt_bias.float()
                .norm(dim=-1)
                .mean()
                .detach(),
                "cocoop_static_prompt_norm_mean": static_prompt.float()
                .norm(dim=-1)
                .mean()
                .detach(),
            }
            if not self._forward_audited:
                print(
                    "[COCOOP_STYLE_FORWARD_AUDIT] "
                    f"prompt_tokens={self.prompt_length} "
                    f"visual_tokens={visual_mask.sum(dim=1).tolist()} "
                    f"meta_net={inputs_embeds.shape[-1]}x{self.bottleneck_dim}x"
                    f"{inputs_embeds.shape[-1]} question_access=false "
                    "visual_source=post_merger_llm_tokens "
                    "conditioning=shared_bias_per_prompt_token pass=True"
                )
                self._forward_audited = True
            return args, kwargs

        with self._inject_prompt_embeddings():
            handle = language_model.register_forward_pre_hook(
                condition_prompt,
                with_kwargs=True,
            )
            try:
                yield
            finally:
                handle.remove()

    def forward(self, **kwargs: Any) -> Any:
        expanded = self._expand_inputs(dict(kwargs))
        with self._inject_conditioned_prompt(expanded["input_ids"]):
            return self.base_model(**expanded)

    def generate(self, **kwargs: Any) -> Any:
        expanded = self._expand_inputs(dict(kwargs))
        with self._inject_conditioned_prompt(expanded["input_ids"]):
            return self.base_model.generate(**expanded)

    def save_cocoop(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "cocoop_style_conditional_prompt_tuning",
            "source": "CoCoOp_CVPR_2022",
            "approximation": "generative_mllm_unified_protocol",
            "prompt_length": self.prompt_length,
            "hidden_size": int(self.soft_prompt.shape[-1]),
            "bottleneck_dim": self.bottleneck_dim,
            "visual_source": "post_merger_llm_visual_token_mean",
            "question_access": False,
            "conditioning": "shared_image_bias_added_to_each_prompt_token",
            "prompt_placement": "before_full_chat",
            "init_seed": self.init_seed,
        }
        with (output / COCOOP_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        torch.save(
            {
                "soft_prompt": self.soft_prompt.detach().cpu(),
                "meta_net": self.meta_net.state_dict(),
            },
            output / COCOOP_WEIGHTS_NAME,
        )

    def load_cocoop(self, checkpoint_dir: str | Path) -> None:
        checkpoint = Path(checkpoint_dir)
        state = torch.load(
            checkpoint / COCOOP_WEIGHTS_NAME,
            map_location="cpu",
            weights_only=True,
        )
        prompt = state["soft_prompt"]
        if tuple(prompt.shape) != tuple(self.soft_prompt.shape):
            raise ValueError("CoCoOp-style soft Prompt shape mismatch")
        self.soft_prompt.data.copy_(prompt.to(self.soft_prompt.device))
        self.meta_net.load_state_dict(state["meta_net"], strict=True)
        self._forward_audited = True
