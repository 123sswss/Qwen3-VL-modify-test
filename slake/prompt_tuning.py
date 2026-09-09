"""Static soft-prompt support for frozen Qwen3-VL models."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import torch
from torch import nn

from slake.sparse_visual_mmrl import SparseVisualInjectionBlock


PROMPT_CONFIG_NAME = "prompt_config.json"
PROMPT_WEIGHTS_NAME = "soft_prompt.pt"


class StaticVisualPrompt(nn.Module):
    """Insert one shared trainable visual prefix into selected frozen blocks."""

    def __init__(
        self,
        visual_dim: int,
        prompt_length: int,
        anchor_layers: Sequence[int],
        init_seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        anchors = tuple(int(index) for index in anchor_layers)
        if visual_dim < 1 or prompt_length < 1 or not anchors:
            raise ValueError("Static Visual Prompt dimensions and anchors must be positive")
        if len(set(anchors)) != len(anchors) or min(anchors) < 0:
            raise ValueError("Static Visual Prompt anchors must be unique and non-negative")
        self.visual_dim = int(visual_dim)
        self.prompt_length = int(prompt_length)
        self.anchor_layers = anchors
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        initial = torch.randn(
            self.prompt_length,
            self.visual_dim,
            generator=generator,
            dtype=torch.float32,
        ) * 0.02
        self.visual_prompt = nn.Parameter(initial.to(device=device))
        self.active = False
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._installed = False
        self._forward_audited = False

    def install(self, visual_model: nn.Module) -> None:
        if self._installed:
            raise RuntimeError("Static Visual Prompt is already installed")
        blocks = getattr(visual_model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Static Visual Prompt requires visual_model.blocks")
        invalid = [index for index in self.anchor_layers if index >= len(blocks)]
        if invalid:
            raise ValueError(
                f"Static Visual Prompt anchors exceed depth: {invalid} >= {len(blocks)}"
            )
        for anchor in self.anchor_layers:
            blocks[anchor] = SparseVisualInjectionBlock(blocks[anchor], self, anchor)
        self._installed = True
        print(
            "[STATIC_VISUAL_PROMPT_LAYER_AUDIT] "
            f"vision_depth={len(blocks)} anchors_0based={list(self.anchor_layers)} "
            f"anchors_natural={[index + 1 for index in self.anchor_layers]} "
            f"tokens={self.prompt_length} dim={self.visual_dim} "
            "block_execution=single_pass insert_strip=True"
        )

    @contextmanager
    def activate(self) -> Iterator[None]:
        if self.active:
            raise RuntimeError("Static Visual Prompt context is already active")
        self.active = True
        self.debug_context = {}
        try:
            yield
        finally:
            self.active = False

    @staticmethod
    def _argument(
        args: Sequence[Any], kwargs: Dict[str, Any], position: int, name: str
    ) -> Any:
        return kwargs[name] if name in kwargs else (args[position] if len(args) > position else None)

    @staticmethod
    def _prefix_tensor_by_segment(
        tensor: torch.Tensor,
        lengths: Sequence[int],
        prefix_length: int,
        fill_value: float,
    ) -> torch.Tensor:
        prefix = torch.full(
            (prefix_length, *tensor.shape[1:]),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat(
            [
                torch.cat((prefix, segment), dim=0)
                for segment in torch.split(tensor, list(lengths), dim=0)
            ],
            dim=0,
        )

    def inject(
        self,
        layer_index: int,
        block: nn.Module,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self._argument(args, kwargs, 0, "hidden_states")
        cu_seqlens = self._argument(args, kwargs, 1, "cu_seqlens")
        if hidden_states is None or cu_seqlens is None:
            raise RuntimeError("Static Visual Prompt requires hidden_states/cu_seqlens")
        lengths_tensor = cu_seqlens[1:] - cu_seqlens[:-1]
        lengths = [int(value) for value in lengths_tensor.detach().cpu().tolist()]
        if not lengths or min(lengths) < 1 or sum(lengths) != int(hidden_states.shape[0]):
            raise RuntimeError("Static Visual Prompt received an invalid visual layout")
        prompt = self.visual_prompt.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        expanded_hidden = torch.cat(
            [torch.cat((prompt, segment), dim=0) for segment in torch.split(hidden_states, lengths)],
            dim=0,
        )
        offsets = torch.arange(
            cu_seqlens.numel(), device=cu_seqlens.device, dtype=cu_seqlens.dtype
        ) * self.prompt_length
        expanded_args = list(args)
        expanded_kwargs = dict(kwargs)
        if expanded_args:
            expanded_args[0] = expanded_hidden
        else:
            expanded_kwargs["hidden_states"] = expanded_hidden
        expanded_cu = cu_seqlens + offsets
        if "cu_seqlens" in expanded_kwargs:
            expanded_kwargs["cu_seqlens"] = expanded_cu
        elif len(expanded_args) > 1:
            expanded_args[1] = expanded_cu
        else:
            expanded_kwargs["cu_seqlens"] = expanded_cu

        position_embeddings = self._argument(args, kwargs, 3, "position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            expanded_position = (
                self._prefix_tensor_by_segment(cos, lengths, self.prompt_length, 1.0),
                self._prefix_tensor_by_segment(sin, lengths, self.prompt_length, 0.0),
            )
            if "position_embeddings" in expanded_kwargs:
                expanded_kwargs["position_embeddings"] = expanded_position
            elif len(expanded_args) > 3:
                expanded_args[3] = expanded_position
            else:
                expanded_kwargs["position_embeddings"] = expanded_position

        rotary_pos_emb = self._argument(args, kwargs, 2, "rotary_pos_emb")
        if rotary_pos_emb is not None:
            expanded_rotary = self._prefix_tensor_by_segment(
                rotary_pos_emb, lengths, self.prompt_length, 0.0
            )
            if "rotary_pos_emb" in expanded_kwargs:
                expanded_kwargs["rotary_pos_emb"] = expanded_rotary
            elif len(expanded_args) > 2:
                expanded_args[2] = expanded_rotary
            else:
                expanded_kwargs["rotary_pos_emb"] = expanded_rotary

        output = block(*expanded_args, **expanded_kwargs)
        if not torch.is_tensor(output):
            raise TypeError("Static Visual Prompt expects the visual block to return a tensor")
        segments = torch.split(
            output, [length + self.prompt_length for length in lengths], dim=0
        )
        stripped = torch.cat(
            [segment[self.prompt_length :] for segment in segments], dim=0
        )
        with torch.no_grad():
            input_norm = hidden_states.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
            prompt_norm = self.visual_prompt.detach().float().norm(dim=-1).mean()
            output_norm = stripped.detach().float().norm(dim=-1).mean()
            self.debug_context = {
                "static_visual_prompt_norm_mean": prompt_norm,
                "static_visual_prompt_to_input_ratio": prompt_norm / input_norm,
                "static_visual_output_to_input_ratio": output_norm / input_norm,
                "static_visual_anchor_layer": output_norm.new_tensor(float(layer_index)),
            }
            if not self._forward_audited:
                print(
                    "[STATIC_VISUAL_PROMPT_FORWARD_AUDIT] "
                    f"layer={layer_index} units={len(lengths)} tokens={self.prompt_length} "
                    "insert_before_block=True strip_after_block=True pass=True"
                )
                self._forward_audited = True
        return stripped


class StaticPromptTuningModel(nn.Module):
    """Prepend trainable embeddings while leaving every base parameter frozen."""

    def __init__(
        self,
        base_model: nn.Module,
        prompt_length: int = 20,
        init_seed: int = 44,
        visual_prompt_length: int = 0,
        visual_anchor_layers: Sequence[int] = (17,),
    ) -> None:
        super().__init__()
        if prompt_length < 0 or visual_prompt_length < 0:
            raise ValueError("Prompt lengths must be non-negative")
        if prompt_length == 0 and visual_prompt_length == 0:
            raise ValueError("At least one text or visual Prompt is required")
        self.base_model = base_model
        self.prompt_length = int(prompt_length)
        self.init_seed = int(init_seed)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        embeddings = self.base_model.get_input_embeddings().weight.detach()
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        if self.prompt_length:
            sampled_rows = torch.randint(
                embeddings.shape[0],
                (self.prompt_length,),
                generator=generator,
            )
            initial_prompt = embeddings[sampled_rows.to(embeddings.device)].clone()
            # Keep the tiny trainable state in fp32 even when the frozen backbone is bf16.
            self.soft_prompt = nn.Parameter(initial_prompt.float())
        else:
            self.register_parameter("soft_prompt", None)

        self.static_visual_prompt = None
        if visual_prompt_length:
            visual_model = getattr(getattr(self.base_model, "model", None), "visual", None)
            visual_config = getattr(visual_model, "config", None)
            visual_dim = getattr(visual_config, "hidden_size", None)
            if visual_model is None or visual_dim is None:
                raise RuntimeError(
                    "Static Visual Prompt requires base_model.model.visual.config.hidden_size"
                )
            self.static_visual_prompt = StaticVisualPrompt(
                visual_dim=int(visual_dim),
                prompt_length=int(visual_prompt_length),
                anchor_layers=visual_anchor_layers,
                init_seed=int(init_seed),
                device=next(visual_model.parameters()).device,
            )
            self.static_visual_prompt.install(visual_model)

        self.config = self.base_model.config
        self.generation_config = getattr(self.base_model, "generation_config", None)

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def _expand_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.prompt_length == 0:
            return batch
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        labels = batch.pop("labels", None)
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # The base model still builds multimodal embeddings from input_ids and pixels.
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

    @contextmanager
    def _inject_prompt_embeddings(self) -> Iterator[None]:
        if self.soft_prompt is None:
            yield
            return
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

    def forward(self, **kwargs: Any) -> Any:
        expanded = self._expand_inputs(dict(kwargs))
        with self._inject_prompt_embeddings():
            with self._inject_visual_prompt():
                return self.base_model(**expanded)

    def generate(self, **kwargs: Any) -> Any:
        expanded = self._expand_inputs(dict(kwargs))
        with self._inject_prompt_embeddings():
            with self._inject_visual_prompt():
                return self.base_model.generate(**expanded)

    @contextmanager
    def _inject_visual_prompt(self) -> Iterator[None]:
        if self.static_visual_prompt is None:
            yield
            return
        with self.static_visual_prompt.activate():
            yield

    def trainable_parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        return {
            "text_prompt": [self.soft_prompt] if self.soft_prompt is not None else [],
            "visual_prompt": (
                [self.static_visual_prompt.visual_prompt]
                if self.static_visual_prompt is not None
                else []
            ),
        }

    def save_prompt(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": (
                "dual_static_prompt_tuning"
                if self.soft_prompt is not None and self.static_visual_prompt is not None
                else (
                    "static_visual_prompt_tuning"
                    if self.static_visual_prompt is not None
                    else "static_prompt_tuning"
                )
            ),
            "prompt_length": self.prompt_length,
            "hidden_size": int(self.get_input_embeddings().weight.shape[-1]),
            "init_seed": self.init_seed,
            "static_visual_prompt": (
                {
                    "prompt_length": self.static_visual_prompt.prompt_length,
                    "visual_dim": self.static_visual_prompt.visual_dim,
                    "anchor_layers": list(self.static_visual_prompt.anchor_layers),
                    "injection_mode": "single_pass_insert_strip",
                }
                if self.static_visual_prompt is not None
                else None
            ),
        }
        with (output_path / PROMPT_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        torch.save(
            {
                "soft_prompt": (
                    self.soft_prompt.detach().cpu() if self.soft_prompt is not None else None
                ),
                "static_visual_prompt": (
                    self.static_visual_prompt.visual_prompt.detach().cpu()
                    if self.static_visual_prompt is not None
                    else None
                ),
            },
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
        if (prompt is None) != (self.soft_prompt is None):
            raise ValueError("Text Prompt presence does not match checkpoint")
        if self.soft_prompt is not None:
            if tuple(prompt.shape) != tuple(self.soft_prompt.shape):
                raise ValueError(
                    f"Soft prompt shape mismatch: {tuple(prompt.shape)} vs "
                    f"{tuple(self.soft_prompt.shape)}"
                )
            self.soft_prompt.data.copy_(prompt.to(self.soft_prompt.device))
        visual_prompt = state.get("static_visual_prompt")
        expected_visual = (
            self.static_visual_prompt.visual_prompt
            if self.static_visual_prompt is not None
            else None
        )
        if (visual_prompt is None) != (expected_visual is None):
            raise ValueError("Visual Prompt presence does not match checkpoint")
        if expected_visual is not None:
            if tuple(visual_prompt.shape) != tuple(expected_visual.shape):
                raise ValueError("Visual Prompt shape does not match checkpoint")
            expected_visual.data.copy_(visual_prompt.to(expected_visual.device))
