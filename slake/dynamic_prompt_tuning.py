"""Sample-conditioned soft prompts for frozen Qwen3-VL models."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import torch
from torch import nn


DYNAMIC_PROMPT_CONFIG_NAME = "dynamic_prompt_config.json"
DYNAMIC_PROMPT_WEIGHTS_NAME = "dynamic_prompt.pt"


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        indices = (counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"Dynamic Prompt {name} memory is empty for samples={indices}")
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / counts.to(values.dtype).unsqueeze(-1)


class DynamicPromptCrossAttention(nn.Module):
    """Read one visual and one textual memory into trainable prompt slots."""

    def __init__(
        self,
        hidden_size: int,
        attention_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or attention_dim < 1 or num_heads < 1:
            raise ValueError("Dynamic Prompt dimensions must be positive")
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")

        self.hidden_size = int(hidden_size)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.query_projection = nn.Linear(
            self.hidden_size, self.attention_dim, bias=False
        )
        self.key_projection = nn.Linear(
            self.hidden_size, self.attention_dim, bias=False
        )
        self.value_projection = nn.Linear(
            self.hidden_size, self.attention_dim, bias=False
        )
        self.output_projection = nn.Linear(
            self.attention_dim, self.hidden_size, bias=True
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._forward_audited = False

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        prompt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if prompt.ndim != 3 or prompt.shape[-1] != self.hidden_size:
            raise ValueError(
                f"prompt must have [B, P, {self.hidden_size}], got {tuple(prompt.shape)}"
            )
        if memory.ndim != 3 or tuple(memory.shape[1:]) != (2, self.hidden_size):
            raise ValueError(
                f"memory must have [B, 2, {self.hidden_size}], got {tuple(memory.shape)}"
            )

        parameter_dtype = self.query_projection.weight.dtype
        prompt_f = prompt.to(dtype=parameter_dtype)
        memory_f = memory.to(dtype=parameter_dtype)
        query = self._split_heads(
            self.query_projection(self.query_norm(prompt_f))
        )
        key = self._split_heads(
            self.key_projection(self.memory_norm(memory_f))
        )
        value = self._split_heads(
            self.value_projection(self.memory_norm(memory_f))
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(
            prompt.shape[0], prompt.shape[1], self.attention_dim
        )
        delta = self.output_projection(context).to(dtype=prompt.dtype)

        with torch.no_grad():
            attention_f = attention.detach().float().clamp_min(1e-8)
            entropy = -(attention_f * attention_f.log()).sum(dim=-1) / math.log(2.0)
            prompt_norm = prompt.detach().float().norm(dim=-1).mean()
            delta_norm = delta.detach().float().norm(dim=-1).mean()
            self.debug_context = {
                "dynamic_prompt_attention_entropy_norm": entropy.mean(),
                "dynamic_prompt_visual_attention_mean": attention_f[..., 0].mean(),
                "dynamic_prompt_delta_norm_mean": delta_norm,
                "dynamic_prompt_base_norm_mean": prompt_norm,
                "dynamic_prompt_delta_to_base_ratio": delta_norm
                / prompt_norm.clamp_min(1e-8),
            }
            if not self._forward_audited:
                max_abs = float(delta.detach().float().abs().max().item())
                if max_abs != 0.0:
                    raise RuntimeError(
                        "Dynamic Prompt output projection must start at exact zero; "
                        f"max_abs_delta={max_abs}"
                    )
                print(
                    "[DYNAMIC_PROMPT_ZERO_INIT_AUDIT] "
                    f"prompt_shape={tuple(prompt.shape)} memory_shape={tuple(memory.shape)} "
                    f"attention_dim={self.attention_dim} heads={self.num_heads} "
                    "max_abs_delta=0.0 pass=True"
                )
                self._forward_audited = True
        return delta


class DynamicPromptTuningModel(nn.Module):
    """Jointly train a static prompt and its multimodal residual generator."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        prompt_length: int = 20,
        init_seed: int = 44,
        attention_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        self.base_model = base_model
        self.prompt_length = int(prompt_length)
        self.init_seed = int(init_seed)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        embeddings = self.base_model.get_input_embeddings().weight.detach()
        generator = torch.Generator(device="cpu").manual_seed(self.init_seed)
        sampled_rows = torch.randint(
            embeddings.shape[0],
            (self.prompt_length,),
            generator=generator,
            device="cpu",
        )
        initial_prompt = embeddings[sampled_rows.to(embeddings.device)].clone()
        self.soft_prompt = nn.Parameter(initial_prompt.float())
        self.dynamic_prompt = DynamicPromptCrossAttention(
            hidden_size=int(embeddings.shape[-1]),
            attention_dim=attention_dim,
            num_heads=num_heads,
        ).to(device=embeddings.device)

        self.visual_template_token_ids = self._resolve_visual_token_ids(tokenizer)
        self.config = self.base_model.config
        self.generation_config = getattr(self.base_model, "generation_config", None)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._memory_audited = False

    @staticmethod
    def _resolve_visual_token_ids(tokenizer: Any) -> Sequence[int]:
        token_ids = []
        for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0:
                raise RuntimeError(f"Failed to resolve Dynamic Prompt token {token!r}")
            token_ids.append(int(token_id))
        return tuple(token_ids)

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def trainable_parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        return {
            "soft_prompt": [self.soft_prompt],
            "dynamic_prompt": list(self.dynamic_prompt.parameters()),
        }

    def _expand_inputs(
        self,
        batch: Dict[str, Any],
    ) -> tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        batch = dict(batch)
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        labels = batch.get("labels")
        context_mask = batch.pop("mmrl_gating_mask", None)
        if context_mask is None:
            context_mask = (
                (labels == -100) & attention_mask.bool()
                if labels is not None
                else attention_mask.bool()
            )
        elif context_mask.shape != input_ids.shape:
            raise ValueError("mmrl_gating_mask must match input_ids")
        if labels is not None and bool((context_mask.bool() & (labels != -100)).any()):
            raise RuntimeError(
                "Dynamic Prompt text memory must not include supervised answer tokens"
            )

        batch_size = input_ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        prompt_ids = torch.full(
            (batch_size, self.prompt_length),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        prompt_attention = torch.ones(
            (batch_size, self.prompt_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        prompt_context = torch.zeros(
            (batch_size, self.prompt_length),
            dtype=torch.bool,
            device=context_mask.device,
        )
        expanded_ids = torch.cat((prompt_ids, input_ids), dim=1)
        expanded_attention = torch.cat((prompt_attention, attention_mask), dim=1)
        expanded_context = torch.cat((prompt_context, context_mask.bool()), dim=1)
        expanded = {
            **batch,
            "input_ids": expanded_ids,
            "attention_mask": expanded_attention,
        }
        if labels is not None:
            ignored = torch.full(
                (batch_size, self.prompt_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            expanded["labels"] = torch.cat((ignored, labels), dim=1)
        return expanded, expanded_ids, expanded_context

    @contextmanager
    def _inject_prompt_embeddings(self) -> Iterator[None]:
        embeddings = self.get_input_embeddings()

        def replace_prompt(_module: nn.Module, _inputs: Any, output: torch.Tensor):
            if output.ndim != 3 or output.shape[1] < self.prompt_length:
                return output
            prompt = self.soft_prompt.to(
                device=output.device,
                dtype=output.dtype,
            ).unsqueeze(0).expand(output.shape[0], -1, -1)
            return torch.cat((prompt, output[:, self.prompt_length:]), dim=1)

        handle = embeddings.register_forward_hook(replace_prompt)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def _inject_dynamic_prompt(
        self,
        expanded_ids: torch.Tensor,
        expanded_context: torch.Tensor,
    ) -> Iterator[None]:
        model_core = getattr(self.base_model, "model", None)
        language_model = getattr(model_core, "language_model", None)
        if language_model is None:
            raise RuntimeError(
                "Dynamic Prompt requires base_model.model.language_model"
            )

        def replace_language_inputs(_module: nn.Module, args: Any, kwargs: Dict[str, Any]):
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None or inputs_embeds.ndim != 3:
                return args, kwargs
            if inputs_embeds.shape[1] != expanded_context.shape[1]:
                # Decode steps reuse the dynamic prompt already stored in KV cache.
                return args, kwargs

            visual_mask = kwargs.get("visual_pos_masks")
            if visual_mask is None:
                visual_mask = expanded_ids.eq(self.visual_template_token_ids[0])
            visual_mask = visual_mask.to(
                device=inputs_embeds.device,
                dtype=torch.bool,
            )
            text_mask = expanded_context.to(
                device=inputs_embeds.device,
                dtype=torch.bool,
            ) & ~visual_mask
            ids = expanded_ids.to(inputs_embeds.device)
            for token_id in self.visual_template_token_ids:
                text_mask = text_mask & ids.ne(token_id)
            text_mask[:, : self.prompt_length] = False

            visual_memory = _masked_mean(inputs_embeds, visual_mask, "visual")
            text_memory = _masked_mean(inputs_embeds, text_mask, "text")
            memory = torch.stack((visual_memory, text_memory), dim=1)
            prompt = inputs_embeds[:, : self.prompt_length]
            delta = self.dynamic_prompt(prompt, memory)
            dynamic = prompt + delta
            kwargs["inputs_embeds"] = torch.cat(
                (dynamic, inputs_embeds[:, self.prompt_length :]), dim=1
            )
            self.debug_context = {
                **self.dynamic_prompt.debug_context,
                "dynamic_prompt_visual_tokens_mean": visual_mask.sum(dim=1).float().mean(),
                "dynamic_prompt_text_tokens_mean": text_mask.sum(dim=1).float().mean(),
            }
            if not self._memory_audited:
                print(
                    "[DYNAMIC_PROMPT_MEMORY_AUDIT] "
                    f"visual_tokens={visual_mask.sum(dim=1).tolist()} "
                    f"text_tokens={text_mask.sum(dim=1).tolist()} "
                    "answer_overlap=0 pass=True"
                )
                self._memory_audited = True
            return args, kwargs

        handle = language_model.register_forward_pre_hook(
            replace_language_inputs,
            with_kwargs=True,
        )
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def _injection_context(
        self,
        expanded_ids: torch.Tensor,
        expanded_context: torch.Tensor,
    ) -> Iterator[None]:
        with self._inject_prompt_embeddings():
            with self._inject_dynamic_prompt(expanded_ids, expanded_context):
                yield

    def forward(self, **kwargs: Any) -> Any:
        expanded, expanded_ids, expanded_context = self._expand_inputs(kwargs)
        with self._injection_context(expanded_ids, expanded_context):
            return self.base_model(**expanded)

    def generate(self, **kwargs: Any) -> Any:
        expanded, expanded_ids, expanded_context = self._expand_inputs(kwargs)
        with self._injection_context(expanded_ids, expanded_context):
            return self.base_model.generate(**expanded)

    def save_dynamic_prompt(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "dynamic_multimodal_prompt_tuning",
            "prompt_length": self.prompt_length,
            "hidden_size": int(self.soft_prompt.shape[-1]),
            "attention_dim": self.dynamic_prompt.attention_dim,
            "num_heads": self.dynamic_prompt.num_heads,
            "init_seed": self.init_seed,
            "memory": "mean_visual_plus_mean_question",
        }
        with (output_path / DYNAMIC_PROMPT_CONFIG_NAME).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        torch.save(
            {
                "soft_prompt": self.soft_prompt.detach().cpu(),
                "dynamic_prompt": {
                    key: value.detach().cpu()
                    for key, value in self.dynamic_prompt.state_dict().items()
                },
            },
            output_path / DYNAMIC_PROMPT_WEIGHTS_NAME,
        )

    def load_dynamic_prompt(self, checkpoint_dir: str | Path) -> None:
        checkpoint_path = Path(checkpoint_dir)
        state = torch.load(
            checkpoint_path / DYNAMIC_PROMPT_WEIGHTS_NAME,
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
        self.dynamic_prompt.load_state_dict(state["dynamic_prompt"], strict=True)
        # A restored checkpoint is no longer expected to have a zero residual.
        self.dynamic_prompt._forward_audited = True
        print(
            "[DYNAMIC_PROMPT_CHECKPOINT_AUDIT] "
            f"loaded={checkpoint_path} zero_init_check=skipped_for_trained_state"
        )
