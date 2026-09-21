"""Final QDPT Dense-D768 Sandwich model for Qwen3-VL."""

from __future__ import annotations

import json
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from config import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    EXPECTED_TRAINABLE_PARAMETERS,
    PRIVATE_TEXT_TOKENS,
    PRIVATE_VISUAL_TOKENS,
    TEXT_ANCHOR_TOKENS,
    TEXT_DIM,
    VISUAL_ANCHOR_LAYER,
    VISUAL_ANCHOR_TOKENS,
    VISUAL_DIM,
    WORKSPACE_DIM,
    WORKSPACE_HEADS,
)


class WorkspaceTextProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(WORKSPACE_DIM)
        self.input_projection = nn.Linear(WORKSPACE_DIM, WORKSPACE_DIM)
        self.output_projection = nn.Linear(WORKSPACE_DIM, TEXT_DIM)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, workspace: torch.Tensor, anchor: torch.Tensor
    ) -> torch.Tensor:
        parameter = self.input_projection.weight
        hidden = torch.nn.functional.gelu(
            self.input_projection(
                self.norm(workspace.to(parameter.device, parameter.dtype))
            ),
            approximate="tanh",
        )
        delta = self.output_projection(hidden).to(workspace.dtype)
        return anchor.to(delta.device, delta.dtype).unsqueeze(0) + delta


class VisualInjectionBlock(nn.Module):
    """Insert visual Prompt tokens while executing the wrapped block once."""

    def __init__(self, block: nn.Module, adapter: "QDPTVisual") -> None:
        super().__init__()
        self.block = block
        object.__setattr__(self, "_adapter_ref", weakref.ref(adapter))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter = self._adapter_ref()
        if adapter is None or not adapter.active:
            return self.block(*args, **kwargs)
        return adapter.inject(self.block, args, kwargs)


class QDPTVisual(nn.Module):
    """Build question-guided Z10 from Layer17 visual K/V."""

    def __init__(self, visual_model: nn.Module) -> None:
        super().__init__()
        # Keep the initialization order used by the final experiments: A_v then S.
        self.workspace_visual_anchor = nn.Parameter(
            torch.empty(VISUAL_ANCHOR_TOKENS, VISUAL_DIM)
        )
        nn.init.normal_(self.workspace_visual_anchor, std=0.02)
        self.private_visual_prompt = nn.Parameter(
            torch.empty(PRIVATE_VISUAL_TOKENS, VISUAL_DIM)
        )
        nn.init.normal_(self.private_visual_prompt, std=0.02)

        self.workspace_text_query_norm = nn.LayerNorm(TEXT_DIM)
        self.workspace_text_value_projection = nn.Linear(
            TEXT_DIM, WORKSPACE_DIM, bias=False
        )
        self.workspace_text_score_projection = nn.Linear(
            TEXT_DIM, TEXT_ANCHOR_TOKENS, bias=False
        )
        self.workspace_query_norm = nn.LayerNorm(WORKSPACE_DIM)
        self.workspace_visual_memory_norm = nn.LayerNorm(VISUAL_DIM)
        self.workspace_visual_memory_projection = nn.Linear(
            VISUAL_DIM, WORKSPACE_DIM, bias=False
        )
        self.workspace_cross_attention = nn.MultiheadAttention(
            WORKSPACE_DIM, WORKSPACE_HEADS, batch_first=True
        )

        self.active = False
        self.text_tokens: torch.Tensor | None = None
        self.text_token_mask: torch.Tensor | None = None
        self.images_per_sample: torch.Tensor | None = None
        self.image_lengths: torch.Tensor | None = None
        self.image_to_sample: torch.Tensor | None = None
        self.workspace_by_image: torch.Tensor | None = None

        blocks = visual_model.blocks
        blocks[VISUAL_ANCHOR_LAYER] = VisualInjectionBlock(
            blocks[VISUAL_ANCHOR_LAYER], self
        )

    def private_parameters(self) -> list[nn.Parameter]:
        return [self.private_visual_prompt]

    def qdpt_parameters(self) -> list[nn.Parameter]:
        private_id = id(self.private_visual_prompt)
        return [parameter for parameter in self.parameters() if id(parameter) != private_id]

    @contextmanager
    def activate(
        self,
        text_tokens: torch.Tensor,
        text_token_mask: torch.Tensor,
        images_per_sample: torch.Tensor,
    ) -> Iterator[None]:
        lengths = text_token_mask.bool().sum(dim=1)
        max_length = int(lengths.max().item())
        compact = text_tokens.new_zeros(
            text_tokens.shape[0], max_length, text_tokens.shape[-1]
        )
        compact_mask = torch.zeros(
            text_tokens.shape[0],
            max_length,
            dtype=torch.bool,
            device=text_tokens.device,
        )
        for sample_index in range(text_tokens.shape[0]):
            selected = text_tokens[sample_index][text_token_mask[sample_index].bool()]
            compact[sample_index, : selected.shape[0]] = selected
            compact_mask[sample_index, : selected.shape[0]] = True

        self.active = True
        self.text_tokens = compact
        self.text_token_mask = compact_mask
        self.images_per_sample = images_per_sample.to(dtype=torch.long)
        self.image_lengths = None
        self.image_to_sample = None
        self.workspace_by_image = None
        try:
            yield
        finally:
            self.active = False
            self.text_tokens = None
            self.text_token_mask = None
            self.images_per_sample = None
            self.image_lengths = None
            self.image_to_sample = None
            self.workspace_by_image = None

    def prepare_visual(self, grid_thw: torch.Tensor) -> None:
        if not self.active:
            return
        self.image_lengths = grid_thw.prod(dim=-1).to(dtype=torch.long)
        sample_ids = torch.arange(
            self.images_per_sample.numel(), device=self.images_per_sample.device
        )
        self.image_to_sample = torch.repeat_interleave(
            sample_ids, self.images_per_sample
        )

    @staticmethod
    def _argument(
        args: Sequence[Any], kwargs: dict[str, Any], position: int, name: str
    ) -> Any:
        if name in kwargs:
            return kwargs[name]
        return args[position] if len(args) > position else None

    def _segment_layout(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> tuple[list[int], torch.Tensor]:
        lengths = [
            int(value)
            for value in (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
        ]
        starts = cu_seqlens[:-1].to(self.image_lengths.device, torch.long)
        image_ends = self.image_lengths.cumsum(dim=0)
        image_ids = torch.bucketize(starts, image_ends, right=True)
        return lengths, image_ids

    def _build_workspace(self, hidden_states: torch.Tensor) -> torch.Tensor:
        image_segments = torch.split(
            hidden_states, self.image_lengths.detach().cpu().tolist(), dim=0
        )
        visual_width = max(segment.shape[0] for segment in image_segments)
        dtype = self.workspace_cross_attention.in_proj_weight.dtype
        visual_tokens = hidden_states.new_zeros(
            len(image_segments), visual_width, VISUAL_DIM, dtype=dtype
        )
        visual_mask = torch.zeros(
            len(image_segments),
            visual_width,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        for image_index, segment in enumerate(image_segments):
            visual_tokens[image_index, : segment.shape[0]] = segment.to(dtype=dtype)
            visual_mask[image_index, : segment.shape[0]] = True

        image_to_sample = self.image_to_sample.to(hidden_states.device)
        text_tokens = self.text_tokens.to(hidden_states.device).index_select(
            0, image_to_sample
        ).to(dtype=dtype)
        text_mask = self.text_token_mask.to(hidden_states.device).index_select(
            0, image_to_sample
        )

        normalized_text = self.workspace_text_query_norm(text_tokens)
        text_values = self.workspace_text_value_projection(normalized_text)
        pooling_scores = self.workspace_text_score_projection(
            normalized_text
        ).transpose(1, 2)
        pooling_scores = pooling_scores.masked_fill(
            ~text_mask[:, None, :], torch.finfo(pooling_scores.dtype).min
        )
        pooling = torch.softmax(pooling_scores.float(), dim=-1).to(dtype=dtype)
        queries = torch.matmul(pooling, text_values)

        visual_memory = self.workspace_visual_memory_projection(
            self.workspace_visual_memory_norm(visual_tokens)
        )
        cross, _ = self.workspace_cross_attention(
            self.workspace_query_norm(queries),
            visual_memory,
            visual_memory,
            key_padding_mask=~visual_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.workspace_by_image = queries + cross
        return self.workspace_by_image

    def text_workspace(self) -> torch.Tensor:
        workspace = self.workspace_by_image
        sample_ids = self.image_to_sample.to(workspace.device)
        batch_size = int(self.images_per_sample.numel())
        combined = workspace.new_zeros(
            batch_size, TEXT_ANCHOR_TOKENS, WORKSPACE_DIM
        )
        combined = combined.index_add(0, sample_ids, workspace)
        counts = self.images_per_sample.to(
            workspace.device, workspace.dtype
        ).view(batch_size, 1, 1)
        return combined / counts.clamp_min(1.0)

    @staticmethod
    def _prefix_tensor_by_segment(
        tensor: torch.Tensor,
        lengths: list[int],
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
                for segment in torch.split(tensor, lengths, dim=0)
            ],
            dim=0,
        )

    def inject(
        self,
        block: nn.Module,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self._argument(args, kwargs, 0, "hidden_states")
        cu_seqlens = self._argument(args, kwargs, 1, "cu_seqlens")
        lengths, image_ids = self._segment_layout(hidden_states, cu_seqlens)
        self._build_workspace(hidden_states)

        private_visual = self.private_visual_prompt.to(
            hidden_states.device, hidden_states.dtype
        ).unsqueeze(0).expand(len(lengths), -1, -1)
        visual_anchor = self.workspace_visual_anchor.to(
            hidden_states.device, hidden_states.dtype
        ).unsqueeze(0).expand(len(lengths), -1, -1)
        prompts = torch.cat((private_visual, visual_anchor), dim=1)
        prompt_length = PRIVATE_VISUAL_TOKENS + VISUAL_ANCHOR_TOKENS
        hidden_segments = torch.split(hidden_states, lengths, dim=0)
        expanded_hidden = torch.cat(
            [
                torch.cat((prompt, segment), dim=0)
                for prompt, segment in zip(prompts, hidden_segments)
            ],
            dim=0,
        )
        offsets = torch.arange(
            cu_seqlens.numel(),
            device=cu_seqlens.device,
            dtype=cu_seqlens.dtype,
        ) * prompt_length
        expanded_cu = cu_seqlens + offsets

        expanded_args = list(args)
        expanded_kwargs = dict(kwargs)
        if expanded_args:
            expanded_args[0] = expanded_hidden
        else:
            expanded_kwargs["hidden_states"] = expanded_hidden
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
                self._prefix_tensor_by_segment(cos, lengths, prompt_length, 1.0),
                self._prefix_tensor_by_segment(sin, lengths, prompt_length, 0.0),
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
                rotary_pos_emb, lengths, prompt_length, 0.0
            )
            if "rotary_pos_emb" in expanded_kwargs:
                expanded_kwargs["rotary_pos_emb"] = expanded_rotary
            elif len(expanded_args) > 2:
                expanded_args[2] = expanded_rotary
            else:
                expanded_kwargs["rotary_pos_emb"] = expanded_rotary

        adapted_with_prompt = block(*expanded_args, **expanded_kwargs)
        adapted_segments = torch.split(
            adapted_with_prompt,
            [length + prompt_length for length in lengths],
            dim=0,
        )
        return torch.cat(
            [segment[prompt_length:] for segment in adapted_segments], dim=0
        )


class QDPTModel(nn.Module):
    def __init__(self, base_model: nn.Module, tokenizer: Any, seed: int = 44) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        embeddings = self.base_model.get_input_embeddings().weight.detach()
        if int(embeddings.shape[-1]) != TEXT_DIM:
            raise ValueError(f"Expected text width {TEXT_DIM}, got {embeddings.shape[-1]}")
        visual_model = self.base_model.model.visual
        if int(visual_model.config.hidden_size) != VISUAL_DIM:
            raise ValueError(
                f"Expected visual width {VISUAL_DIM}, got {visual_model.config.hidden_size}"
            )

        # The visual branch is constructed first in the experiment code.
        self.sparse_visual = QDPTVisual(visual_model).to(
            next(visual_model.parameters()).device
        )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        sampled_rows = torch.randint(
            embeddings.shape[0],
            (PRIVATE_TEXT_TOKENS + TEXT_ANCHOR_TOKENS,),
            generator=generator,
            device="cpu",
        )
        initial_prompt = embeddings[sampled_rows.to(embeddings.device)].clone()
        self.soft_prompt = nn.Parameter(initial_prompt[:PRIVATE_TEXT_TOKENS].float())
        self.workspace_text_anchor = nn.Parameter(
            initial_prompt[PRIVATE_TEXT_TOKENS:].float()
        )
        self.workspace_text_projection = WorkspaceTextProjection().to(
            embeddings.device
        )
        self.init_seed = int(seed)

        self.visual_token_ids = tuple(
            int(tokenizer.convert_tokens_to_ids(token))
            for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
        )
        self.prompt_length = PRIVATE_TEXT_TOKENS + TEXT_ANCHOR_TOKENS
        self.config = self.base_model.config
        self.generation_config = self.base_model.generation_config

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "language_prompt": [self.soft_prompt, self.workspace_text_anchor],
            "visual_prompt": self.sparse_visual.private_parameters(),
            "qdpt": self.sparse_visual.qdpt_parameters()
            + list(self.workspace_text_projection.parameters()),
        }

    def _expand_inputs(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        batch = dict(batch)
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        labels = batch.get("labels")
        context_mask = batch.pop("mmrl_gating_mask", attention_mask.bool())
        batch_size = input_ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        prompt_ids = torch.full(
            (batch_size, self.prompt_length),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        prompt_attention = torch.ones_like(prompt_ids, dtype=attention_mask.dtype)
        prompt_context = torch.zeros_like(prompt_ids, dtype=torch.bool)

        expanded_ids = []
        expanded_attention = []
        expanded_context = []
        expanded_labels = []
        for batch_index in range(batch_size):
            start = int(
                input_ids[batch_index]
                .eq(self.visual_token_ids[1])
                .nonzero(as_tuple=True)[0]
                .item()
            )
            end = int(
                input_ids[batch_index]
                .eq(self.visual_token_ids[2])
                .nonzero(as_tuple=True)[0]
                .item()
            )

            def insert(row, static_slots, dynamic_slots):
                return torch.cat(
                    (
                        row[:start],
                        static_slots,
                        row[start : end + 1],
                        dynamic_slots,
                        row[end + 1 :],
                    )
                )

            expanded_ids.append(
                insert(
                    input_ids[batch_index],
                    prompt_ids[batch_index, :PRIVATE_TEXT_TOKENS],
                    prompt_ids[batch_index, PRIVATE_TEXT_TOKENS:],
                )
            )
            expanded_attention.append(
                insert(
                    attention_mask[batch_index],
                    prompt_attention[batch_index, :PRIVATE_TEXT_TOKENS],
                    prompt_attention[batch_index, PRIVATE_TEXT_TOKENS:],
                )
            )
            expanded_context.append(
                insert(
                    context_mask[batch_index].bool(),
                    prompt_context[batch_index, :PRIVATE_TEXT_TOKENS],
                    prompt_context[batch_index, PRIVATE_TEXT_TOKENS:],
                )
            )
            if labels is not None:
                ignored = torch.full(
                    (self.prompt_length,), -100, dtype=labels.dtype, device=labels.device
                )
                expanded_labels.append(
                    insert(
                        labels[batch_index],
                        ignored[:PRIVATE_TEXT_TOKENS],
                        ignored[PRIVATE_TEXT_TOKENS:],
                    )
                )

        expanded_ids_tensor = torch.stack(expanded_ids)
        expanded_context_tensor = torch.stack(expanded_context)
        expanded = {
            **batch,
            "input_ids": expanded_ids_tensor,
            "attention_mask": torch.stack(expanded_attention),
        }
        if labels is not None:
            expanded["labels"] = torch.stack(expanded_labels)
        return expanded, expanded_ids_tensor, expanded_context_tensor

    def _prompt_masks(
        self, expanded_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        static_mask = torch.zeros_like(expanded_ids, dtype=torch.bool)
        dynamic_mask = torch.zeros_like(expanded_ids, dtype=torch.bool)
        for batch_index in range(expanded_ids.shape[0]):
            start = int(
                expanded_ids[batch_index]
                .eq(self.visual_token_ids[1])
                .nonzero(as_tuple=True)[0]
                .item()
            )
            end = int(
                expanded_ids[batch_index]
                .eq(self.visual_token_ids[2])
                .nonzero(as_tuple=True)[0]
                .item()
            )
            static_mask[
                batch_index, start - PRIVATE_TEXT_TOKENS : start
            ] = True
            dynamic_mask[
                batch_index, end + 1 : end + 1 + TEXT_ANCHOR_TOKENS
            ] = True
        return static_mask, dynamic_mask

    @contextmanager
    def _inject_prompt_embeddings(
        self, expanded_ids: torch.Tensor
    ) -> Iterator[None]:
        embeddings = self.get_input_embeddings()

        def replace_prompt(_module, _inputs, output):
            if output.ndim != 3 or output.shape[1] < self.prompt_length:
                return output
            static_mask, dynamic_mask = self._prompt_masks(
                expanded_ids.to(output.device)
            )
            static_prompt = self.soft_prompt.to(output.device, output.dtype)
            text_anchor = self.workspace_text_anchor.to(output.device, output.dtype)
            replaced = output.clone()
            replaced[static_mask] = static_prompt.unsqueeze(0).expand(
                output.shape[0], -1, -1
            ).reshape(-1, TEXT_DIM)
            replaced[dynamic_mask] = text_anchor.unsqueeze(0).expand(
                output.shape[0], -1, -1
            ).reshape(-1, TEXT_DIM)
            return replaced

        handle = embeddings.register_forward_hook(replace_prompt)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def _inject_visual(
        self,
        expanded_ids: torch.Tensor,
        expanded_context: torch.Tensor,
    ) -> Iterator[None]:
        ids = expanded_ids.to(self.get_input_embeddings().weight.device)
        text_mask = expanded_context.to(ids.device, torch.bool)
        for token_id in self.visual_token_ids:
            text_mask = text_mask & ids.ne(token_id)
        text_tokens = self.get_input_embeddings()(ids)
        images_per_sample = ids.eq(self.visual_token_ids[2]).sum(dim=1)
        visual_model = self.base_model.model.visual

        def prepare_visual(_module, args, kwargs):
            grid_thw = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
            self.sparse_visual.prepare_visual(grid_thw)
            return args, kwargs

        with self.sparse_visual.activate(text_tokens, text_mask, images_per_sample):
            handle = visual_model.register_forward_pre_hook(
                prepare_visual, with_kwargs=True
            )
            try:
                yield
            finally:
                handle.remove()

    @contextmanager
    def _inject_dynamic_text(self, expanded_ids: torch.Tensor) -> Iterator[None]:
        language_model = self.base_model.model.language_model

        def replace_language_inputs(_module, args, kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None or inputs_embeds.shape[1] != expanded_ids.shape[1]:
                return args, kwargs
            _, dynamic_mask = self._prompt_masks(expanded_ids.to(inputs_embeds.device))
            dynamic_prompt = self.workspace_text_projection(
                self.sparse_visual.text_workspace(), self.workspace_text_anchor
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            replaced = inputs_embeds.clone()
            replaced[dynamic_mask] = dynamic_prompt.reshape(-1, TEXT_DIM)
            kwargs["inputs_embeds"] = replaced
            return args, kwargs

        handle = language_model.register_forward_pre_hook(
            replace_language_inputs, with_kwargs=True
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
        with self._inject_prompt_embeddings(expanded_ids):
            with self._inject_visual(expanded_ids, expanded_context):
                with self._inject_dynamic_text(expanded_ids):
                    yield

    def forward(self, **kwargs: Any) -> Any:
        expanded, expanded_ids, expanded_context = self._expand_inputs(kwargs)
        with self._injection_context(expanded_ids, expanded_context):
            return self.base_model(**expanded)

    def generate(self, **kwargs: Any) -> Any:
        expanded, expanded_ids, expanded_context = self._expand_inputs(kwargs)
        with self._injection_context(expanded_ids, expanded_context):
            return self.base_model.generate(**expanded)

    def save_adapter(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "directional_concat_workspace_prompt_tuning",
            "prompt_length": PRIVATE_TEXT_TOKENS + TEXT_ANCHOR_TOKENS,
            "requested_prompt_length": PRIVATE_TEXT_TOKENS,
            "hidden_size": TEXT_DIM,
            "attention_dim": 256,
            "num_heads": 8,
            "init_seed": self.init_seed,
            "shared_s_text_mode": "none",
            "shared_workspace": None,
            "directional_concat_workspace": {
                "tokens": TEXT_ANCHOR_TOKENS,
                "dim": WORKSPACE_DIM,
                "heads": WORKSPACE_HEADS,
                "anchor_layer": VISUAL_ANCHOR_LAYER,
                "anchor_layers": [VISUAL_ANCHOR_LAYER],
                "private_visual_prompt_tokens": PRIVATE_VISUAL_TOKENS,
                "static_visual_prompt_tokens": 0,
                "private_text_prompt_tokens": PRIVATE_TEXT_TOKENS,
                "text_workspace_anchor_tokens": TEXT_ANCHOR_TOKENS,
                "text_prompt_placement": "static_before_visual_dynamic_after_visual",
                "text_projection_hidden_dim": WORKSPACE_DIM,
                "query": "question_attention_pooling",
                "visual_conditioning": "cross_attention",
                "visual_dynamic_write": False,
                "static_visual_write": True,
                "unified_static_visual_prompt": False,
                "direct_visual_z_tokens": False,
            },
            "sparse_visual": {
                "anchor_layers": [VISUAL_ANCHOR_LAYER],
                "rep_token_count": PRIVATE_VISUAL_TOKENS,
                "attention_dim": 0,
                "num_heads": 0,
                "injection_mode": "directional_concat_single_pass_insert_strip",
            },
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        }
        with (output_dir / ADAPTER_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        torch.save(
            {
                "soft_prompt": self.soft_prompt.detach().cpu(),
                "workspace_text_anchor": self.workspace_text_anchor.detach().cpu(),
                "workspace_text_projection": {
                    key: value.detach().cpu()
                    for key, value in self.workspace_text_projection.state_dict().items()
                },
                "sparse_visual": {
                    key: value.detach().cpu()
                    for key, value in self.sparse_visual.state_dict().items()
                },
            },
            output_dir / ADAPTER_WEIGHTS_NAME,
        )

    def load_adapter(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        with (checkpoint_dir / ADAPTER_CONFIG_NAME).open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)
        if config["method"] != "directional_concat_workspace_prompt_tuning":
            raise ValueError("Checkpoint is not the final QDPT Dense-D768 Sandwich")
        state = torch.load(
            checkpoint_dir / ADAPTER_WEIGHTS_NAME,
            map_location="cpu",
            weights_only=True,
        )
        self.soft_prompt.data.copy_(state["soft_prompt"].to(self.soft_prompt.device))
        self.workspace_text_anchor.data.copy_(
            state["workspace_text_anchor"].to(self.workspace_text_anchor.device)
        )
        self.workspace_text_projection.load_state_dict(
            state["workspace_text_projection"], strict=True
        )
        self.sparse_visual.load_state_dict(state["sparse_visual"], strict=True)


def trainable_parameter_count(model: QDPTModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
