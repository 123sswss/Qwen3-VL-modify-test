"""Sparse local Rep-token injection for a frozen Qwen3-VL vision encoder."""

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SparseVisualRepAttention(nn.Module):
    """Generate visual-width Rep tokens from visual and textual mean memories."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        attention_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads

        self.query_norm = nn.LayerNorm(self.visual_dim)
        self.visual_memory_norm = nn.LayerNorm(self.visual_dim)
        self.text_memory_norm = nn.LayerNorm(self.text_dim)
        self.query_projection = nn.Linear(
            self.visual_dim, self.attention_dim, bias=False
        )
        self.visual_key_projection = nn.Linear(
            self.visual_dim, self.attention_dim, bias=False
        )
        self.visual_value_projection = nn.Linear(
            self.visual_dim, self.attention_dim, bias=False
        )
        self.text_key_projection = nn.Linear(
            self.text_dim, self.attention_dim, bias=False
        )
        self.text_value_projection = nn.Linear(
            self.text_dim, self.attention_dim, bias=False
        )
        self.output_projection = nn.Linear(
            self.attention_dim, self.visual_dim, bias=True
        )
        self.debug_context: Dict[str, torch.Tensor] = {}

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        visual_memory: torch.Tensor,
        text_memory: torch.Tensor,
    ) -> torch.Tensor:
        if queries.ndim != 3 or queries.shape[-1] != self.visual_dim:
            raise ValueError(
                f"queries must have [N, R, {self.visual_dim}], got "
                f"{tuple(queries.shape)}"
            )
        if visual_memory.shape != (queries.shape[0], self.visual_dim):
            raise ValueError("visual memory shape does not match Rep query units")
        if text_memory.shape != (queries.shape[0], self.text_dim):
            raise ValueError("text memory shape does not match Rep query units")

        dtype = self.query_projection.weight.dtype
        query_f = self.query_norm(queries.to(dtype=dtype))
        visual_f = self.visual_memory_norm(visual_memory.to(dtype=dtype))
        text_f = self.text_memory_norm(text_memory.to(dtype=dtype))

        query = self._split_heads(self.query_projection(query_f))
        visual_key = self.visual_key_projection(visual_f)
        text_key = self.text_key_projection(text_f)
        visual_value = self.visual_value_projection(visual_f)
        text_value = self.text_value_projection(text_f)
        key = self._split_heads(torch.stack((visual_key, text_key), dim=1))
        value = self._split_heads(torch.stack((visual_value, text_value), dim=1))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        context = torch.matmul(attention, value)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(queries.shape[0], queries.shape[1], self.attention_dim)
        )
        delta = self.output_projection(context)

        with torch.no_grad():
            attention_f = attention.detach().float().clamp_min(1e-8)
            entropy = -(attention_f * attention_f.log()).sum(dim=-1) / math.log(2.0)
            self.debug_context = {
                "attention_entropy_norm": entropy.mean(),
                "visual_attention_mean": attention_f[..., 0].mean(),
            }
        return queries + delta


class SparseVisualInjectionBlock(nn.Module):
    """Insert local Rep tokens while executing the wrapped block exactly once."""

    def __init__(
        self,
        block: nn.Module,
        adapter: "SparseVisualMMRL",
        layer_index: int,
    ) -> None:
        super().__init__()
        self.block = block
        self.layer_index = int(layer_index)
        object.__setattr__(self, "_adapter_ref", weakref.ref(adapter))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter = self._adapter_ref()
        if adapter is None or not adapter.active:
            return self.block(*args, **kwargs)
        return adapter.inject(
            self.layer_index,
            self.block,
            args,
            kwargs,
        )


class SparseVisualMMRL(nn.Module):
    """Inject independently generated Rep tokens at sparse visual anchors."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        anchor_layers: Sequence[int] = (5, 11, 17),
        rep_token_count: int = 8,
        attention_dim: int = 128,
        num_heads: int = 4,
        map_shared_s_to_text: bool = False,
        text_anchor_tokens: int | None = None,
        text_anchor_bottleneck_dim: int = 128,
    ) -> None:
        super().__init__()
        anchors = tuple(int(index) for index in anchor_layers)
        if not anchors or len(set(anchors)) != len(anchors):
            raise ValueError(f"anchor_layers must be non-empty and unique: {anchors}")
        if min(anchors) < 0:
            raise ValueError("anchor layer indexes must be non-negative")
        if min(visual_dim, text_dim, rep_token_count, attention_dim, num_heads) < 1:
            raise ValueError("Sparse Visual MMRL dimensions must be positive")
        if text_anchor_tokens is not None and text_anchor_tokens < 1:
            raise ValueError("text_anchor_tokens must be positive when enabled")
        if text_anchor_bottleneck_dim < 1:
            raise ValueError("text_anchor_bottleneck_dim must be positive")
        if map_shared_s_to_text and text_anchor_tokens is not None:
            raise ValueError(
                "Raw visual-S text mapping and text-owned S are mutually exclusive"
            )

        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.anchor_layers = anchors
        self.rep_token_count = int(rep_token_count)
        self.map_shared_s_to_text = bool(map_shared_s_to_text)
        self.text_anchor_tokens = int(text_anchor_tokens or 0)
        self.text_anchor_bottleneck_dim = int(text_anchor_bottleneck_dim)
        if self.text_anchor_tokens:
            self.register_parameter("shared_rep", None)
            self.text_anchor_token_mixer = nn.Parameter(
                torch.zeros(self.rep_token_count, self.text_anchor_tokens)
            )
            self.text_anchor_down_projection = nn.Linear(
                self.text_dim,
                self.text_anchor_bottleneck_dim,
                bias=False,
            )
            self.text_anchor_up_projection = nn.Linear(
                self.text_anchor_bottleneck_dim,
                self.visual_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(self.text_anchor_down_projection.weight)
            nn.init.normal_(self.text_anchor_up_projection.weight, std=0.002)
            with torch.no_grad():
                for slot in range(self.rep_token_count):
                    center = round(
                        (slot + 0.5)
                        * self.text_anchor_tokens
                        / self.rep_token_count
                        - 0.5
                    )
                    self.text_anchor_token_mixer[slot, center] = 4.0
        else:
            self.shared_rep = nn.Parameter(
                torch.empty(self.rep_token_count, self.visual_dim)
            )
            self.register_parameter("text_anchor_token_mixer", None)
            self.text_anchor_down_projection = None
            self.text_anchor_up_projection = None
        self.layer_embeddings = nn.Parameter(
            torch.empty(len(self.anchor_layers), 1, self.visual_dim)
        )
        if self.shared_rep is not None:
            nn.init.normal_(self.shared_rep, std=0.02)
        nn.init.normal_(self.layer_embeddings, std=0.02)
        self.rep_attention = SparseVisualRepAttention(
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
        )

        self.active = False
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._text_memory: torch.Tensor | None = None
        self._images_per_sample: torch.Tensor | None = None
        self._image_lengths: torch.Tensor | None = None
        self._image_to_sample: torch.Tensor | None = None
        self._read_only_shared_rep: torch.Tensor | None = None
        self._text_anchor_debug: Dict[str, torch.Tensor] = {}
        self._last_shared_s_text_prompt: torch.Tensor | None = None
        self._debug_rows: list[Dict[str, torch.Tensor]] = []
        self._installed = False
        self._forward_audited = False
        object.__setattr__(self, "_shared_memory_merger_ref", None)
        self._shared_memory_merge_unit = 0

    def install(self, visual_model: nn.Module) -> None:
        if self._installed:
            raise RuntimeError("Sparse Visual MMRL blocks are already installed")
        blocks = getattr(visual_model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Sparse Visual MMRL requires visual_model.blocks")
        depth = len(blocks)
        invalid = [index for index in self.anchor_layers if index >= depth]
        if invalid:
            raise ValueError(
                f"Sparse Visual anchors {invalid} exceed vision depth {depth}"
            )
        if self.map_shared_s_to_text:
            merger = getattr(visual_model, "merger", None)
            if merger is None:
                raise RuntimeError("Shared S-Memory requires visual_model.merger")
            merge_unit = int(getattr(visual_model, "spatial_merge_unit", 0))
            if merge_unit < 1 or self.rep_token_count % merge_unit != 0:
                raise ValueError(
                    "Shared S tokens must be divisible by the visual merge unit: "
                    f"rep_tokens={self.rep_token_count} merge_unit={merge_unit}"
                )
            object.__setattr__(
                self,
                "_shared_memory_merger_ref",
                weakref.ref(merger),
            )
            self._shared_memory_merge_unit = merge_unit
        for index in self.anchor_layers:
            blocks[index] = SparseVisualInjectionBlock(blocks[index], self, index)
        self._installed = True
        print(
            "[SPARSE_VISUAL_LAYER_AUDIT] "
            f"vision_depth={depth} anchors_0based={list(self.anchor_layers)} "
            f"anchors_natural={[index + 1 for index in self.anchor_layers]} "
            "local_insert_strip=True block_execution=single_pass "
            f"map_shared_s_to_text={self.map_shared_s_to_text} "
            f"text_owned_s_visual_readonly={bool(self.text_anchor_tokens)} "
            f"text_anchor_bottleneck={self.text_anchor_bottleneck_dim if self.text_anchor_tokens else 0}"
        )

    def adapt_read_only_text_anchor(self, text_anchor: torch.Tensor) -> torch.Tensor:
        """Map detached LLM-space S tokens into visual Rep-token bases."""
        if not self.text_anchor_tokens:
            raise RuntimeError("Text-owned Shared-S visual adapter is disabled")
        if tuple(text_anchor.shape) != (self.text_anchor_tokens, self.text_dim):
            raise ValueError(
                "Text-owned Shared-S must have "
                f"[{self.text_anchor_tokens}, {self.text_dim}], got "
                f"{tuple(text_anchor.shape)}"
            )
        if (
            self.text_anchor_token_mixer is None
            or self.text_anchor_down_projection is None
            or self.text_anchor_up_projection is None
        ):
            raise RuntimeError("Text-owned Shared-S visual adapter is incomplete")

        parameter = self.text_anchor_down_projection.weight
        normalized = F.layer_norm(
            text_anchor.detach().to(device=parameter.device, dtype=parameter.dtype),
            (self.text_dim,),
        )
        mixing = torch.softmax(self.text_anchor_token_mixer.float(), dim=-1).to(
            dtype=normalized.dtype
        )
        pooled = torch.matmul(mixing, normalized)
        bottleneck = F.gelu(
            self.text_anchor_down_projection(pooled),
            approximate="tanh",
        )
        visual_rep = self.text_anchor_up_projection(bottleneck)

        with torch.no_grad():
            probabilities = mixing.detach().float().clamp_min(1e-8)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            entropy = entropy / math.log(float(self.text_anchor_tokens))
            self._text_anchor_debug = {
                "shared_s_visual_adapter_token_entropy_norm": entropy.mean(),
                "shared_s_visual_adapter_rep_norm_mean": visual_rep.detach()
                .float()
                .norm(dim=-1)
                .mean(),
                "shared_s_visual_adapter_source_norm_mean": text_anchor.detach()
                .float()
                .norm(dim=-1)
                .mean(),
            }
        return visual_rep

    @contextmanager
    def activate(
        self,
        text_memory: torch.Tensor,
        images_per_sample: torch.Tensor,
        text_anchor: torch.Tensor | None = None,
    ) -> Iterator[None]:
        if self.active:
            raise RuntimeError("Sparse Visual MMRL context is already active")
        self.active = True
        self._text_memory = text_memory
        self._images_per_sample = images_per_sample.to(dtype=torch.long)
        self._image_lengths = None
        self._image_to_sample = None
        if self.text_anchor_tokens:
            if text_anchor is None:
                raise RuntimeError("Text-owned Shared-S requires a text anchor")
            self._read_only_shared_rep = self.adapt_read_only_text_anchor(text_anchor)
        elif text_anchor is not None:
            raise RuntimeError("A text anchor was provided while its adapter is disabled")
        else:
            self._read_only_shared_rep = None
        self._last_shared_s_text_prompt = None
        self._debug_rows = []
        self.debug_context = {}
        try:
            yield
        finally:
            self.active = False
            self._text_memory = None
            self._images_per_sample = None
            self._image_lengths = None
            self._image_to_sample = None
            self._read_only_shared_rep = None

    def prepare_visual(self, grid_thw: torch.Tensor) -> None:
        if not self.active:
            return
        if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError("Sparse Visual MMRL requires grid_thw=[images,3]")
        if self._images_per_sample is None:
            raise RuntimeError("Sparse Visual image counts are unavailable")
        if int(self._images_per_sample.sum().item()) != int(grid_thw.shape[0]):
            raise RuntimeError(
                "Sparse Visual image count mismatch: "
                f"template={int(self._images_per_sample.sum().item())} "
                f"grid={int(grid_thw.shape[0])}"
            )
        self._image_lengths = grid_thw.prod(dim=-1).to(dtype=torch.long)
        sample_ids = torch.arange(
            self._images_per_sample.numel(),
            device=self._images_per_sample.device,
        )
        self._image_to_sample = torch.repeat_interleave(
            sample_ids, self._images_per_sample
        )

    def _argument(
        self,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        position: int,
        name: str,
    ) -> Any:
        if name in kwargs:
            return kwargs[name]
        return args[position] if len(args) > position else None

    def _segment_memories(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        if self._text_memory is None or self._image_lengths is None:
            raise RuntimeError(
                "Sparse Visual context was not prepared before a block ran"
            )
        if self._image_to_sample is None:
            raise RuntimeError("Sparse Visual image-to-sample mapping is unavailable")
        lengths_tensor = cu_seqlens[1:] - cu_seqlens[:-1]
        if bool((lengths_tensor <= 0).any()):
            raise RuntimeError("Sparse Visual attention segments must be non-empty")
        lengths = [int(value) for value in lengths_tensor.detach().cpu().tolist()]
        if sum(lengths) != int(hidden_states.shape[0]):
            raise RuntimeError(
                "Sparse Visual segment lengths do not match hidden states: "
                f"segments={sum(lengths)} hidden={hidden_states.shape[0]}"
            )
        if int(self._image_lengths.sum().item()) != int(hidden_states.shape[0]):
            raise RuntimeError(
                "Sparse Visual image lengths do not match hidden states: "
                f"images={int(self._image_lengths.sum().item())} "
                f"hidden={hidden_states.shape[0]}"
            )

        starts = cu_seqlens[:-1].to(device=self._image_lengths.device, dtype=torch.long)
        ends = cu_seqlens[1:].to(device=self._image_lengths.device, dtype=torch.long)
        image_ends = self._image_lengths.cumsum(dim=0)
        image_ids = torch.bucketize(starts, image_ends, right=True)
        if bool((image_ids >= image_ends.numel()).any()):
            raise RuntimeError("Sparse Visual segment starts outside image boundaries")
        if bool((ends > image_ends.index_select(0, image_ids)).any()):
            raise RuntimeError(
                "Sparse Visual attention segment crosses an image boundary"
            )
        sample_ids = self._image_to_sample.to(image_ids.device).index_select(
            0, image_ids
        )
        image_memories = torch.stack(
            [
                image_states.mean(dim=0)
                for image_states in torch.split(
                    hidden_states,
                    self._image_lengths.detach().cpu().tolist(),
                    dim=0,
                )
            ],
            dim=0,
        )
        visual_memory = image_memories.index_select(
            0, image_ids.to(image_memories.device)
        )
        text_memory_all = self._text_memory.to(hidden_states.device)
        text_memory = text_memory_all.index_select(
            0, sample_ids.to(text_memory_all.device)
        )
        return visual_memory, text_memory, lengths, sample_ids

    @property
    def shared_s_text_prompt_length(self) -> int:
        if not self.map_shared_s_to_text or self._shared_memory_merge_unit < 1:
            raise RuntimeError("Raw Shared-S text mapping is disabled")
        return self.rep_token_count // self._shared_memory_merge_unit

    def shared_s_text_prompt(self) -> torch.Tensor:
        """Map the raw, unconditioned S bank into the native LLM input space."""
        merger_ref = self._shared_memory_merger_ref
        merger = merger_ref() if merger_ref is not None else None
        if not self.map_shared_s_to_text or merger is None:
            raise RuntimeError("Raw Shared-S text mapping is unavailable")
        merger_parameter = next(merger.parameters())
        if self.shared_rep is None:
            raise RuntimeError("Raw visual Shared-S is unavailable")
        raw_s = self.shared_rep.to(
            device=merger_parameter.device,
            dtype=merger_parameter.dtype,
        )
        mapped = merger(raw_s).view(self.shared_s_text_prompt_length, self.text_dim)
        if mapped.requires_grad:
            mapped.retain_grad()
        self._last_shared_s_text_prompt = mapped
        return mapped

    def shared_s_text_prompt_grad_norm(self) -> torch.Tensor:
        prompt = self._last_shared_s_text_prompt
        if prompt is None or prompt.grad is None:
            parameter = next(self.parameters())
            return parameter.new_tensor(0.0)
        return prompt.grad.detach().float().norm()

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

    def _expanded_block_inputs(
        self,
        reps: torch.Tensor,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        lengths: Sequence[int],
        cu_seqlens: torch.Tensor,
    ) -> tuple[list[Any], Dict[str, Any]]:
        hidden_states = self._argument(args, kwargs, 0, "hidden_states")
        hidden_segments = torch.split(hidden_states, list(lengths), dim=0)
        expanded_hidden = torch.cat(
            [
                torch.cat((rep, segment), dim=0)
                for rep, segment in zip(reps, hidden_segments)
            ],
            dim=0,
        )
        offsets = (
            torch.arange(
                cu_seqlens.numel(),
                device=cu_seqlens.device,
                dtype=cu_seqlens.dtype,
            )
            * self.rep_token_count
        )
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
                self._prefix_tensor_by_segment(cos, lengths, self.rep_token_count, 1.0),
                self._prefix_tensor_by_segment(sin, lengths, self.rep_token_count, 0.0),
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
                rotary_pos_emb, lengths, self.rep_token_count, 0.0
            )
            if "rotary_pos_emb" in expanded_kwargs:
                expanded_kwargs["rotary_pos_emb"] = expanded_rotary
            elif len(expanded_args) > 2:
                expanded_args[2] = expanded_rotary
            else:
                expanded_kwargs["rotary_pos_emb"] = expanded_rotary
        return expanded_args, expanded_kwargs

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
            raise RuntimeError(
                "Sparse Visual block did not receive hidden_states/cu_seqlens"
            )
        visual_memory, text_memory, lengths, _sample_ids = self._segment_memories(
            hidden_states, cu_seqlens
        )
        layer_position = self.anchor_layers.index(layer_index)
        shared_rep = (
            self._read_only_shared_rep
            if self.text_anchor_tokens
            else self.shared_rep
        )
        if shared_rep is None:
            raise RuntimeError("Sparse Visual Shared-S base is unavailable")
        queries = shared_rep.unsqueeze(0) + self.layer_embeddings[layer_position]
        queries = queries.expand(len(lengths), -1, -1)
        reps = self.rep_attention(queries, visual_memory, text_memory).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        expanded_args, expanded_kwargs = self._expanded_block_inputs(
            reps, args, kwargs, lengths, cu_seqlens
        )
        adapted_with_rep = block(*expanded_args, **expanded_kwargs)
        if not torch.is_tensor(adapted_with_rep):
            raise TypeError("Sparse Visual expects vision blocks to return a tensor")
        adapted_segments = torch.split(
            adapted_with_rep,
            [length + self.rep_token_count for length in lengths],
            dim=0,
        )
        adapted = torch.cat(
            [
                segment[self.rep_token_count :]
                for segment in adapted_segments
            ],
            dim=0,
        )

        with torch.no_grad():
            input_norm = hidden_states.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
            rep_norm = reps.detach().float().norm(dim=-1).mean()
            output_norm = adapted.detach().float().norm(dim=-1).mean()
            self._debug_rows.append(
                {
                    "layer": adapted.new_tensor(float(layer_index)),
                    "rep_to_input_norm_ratio": rep_norm / input_norm,
                    "output_to_input_norm_ratio": output_norm / input_norm,
                    **self.rep_attention.debug_context,
                }
            )
            if not self._forward_audited:
                print(
                    "[SPARSE_VISUAL_SINGLE_PASS_AUDIT] "
                    f"layer={layer_index} units={len(lengths)} reps={self.rep_token_count} "
                    "insert_before_block=True strip_after_block=True pass=True"
                )
                if layer_index == self.anchor_layers[-1]:
                    self._forward_audited = True
        self._refresh_debug_context()
        return adapted

    def _refresh_debug_context(self) -> None:
        if not self._debug_rows:
            self.debug_context = {}
            return
        debug: Dict[str, torch.Tensor] = {}
        for row in self._debug_rows:
            layer = int(row["layer"].item())
            debug[f"sparse_visual_layer{layer}_rep_to_input_norm_ratio"] = row[
                "rep_to_input_norm_ratio"
            ]
            debug[f"sparse_visual_layer{layer}_output_to_input_norm_ratio"] = row[
                "output_to_input_norm_ratio"
            ]
            debug[f"sparse_visual_layer{layer}_attention_entropy_norm"] = row[
                "attention_entropy_norm"
            ]
            debug[f"sparse_visual_layer{layer}_visual_attention_mean"] = row[
                "visual_attention_mean"
            ]
        debug["sparse_visual_rep_to_input_norm_ratio_mean"] = torch.stack(
            [row["rep_to_input_norm_ratio"] for row in self._debug_rows]
        ).mean()
        debug["sparse_visual_output_to_input_norm_ratio_mean"] = torch.stack(
            [row["output_to_input_norm_ratio"] for row in self._debug_rows]
        ).mean()
        debug.update(self._text_anchor_debug)
        if self._last_shared_s_text_prompt is not None:
            debug["shared_s_text_prompt_norm_mean"] = (
                self._last_shared_s_text_prompt.detach().float().norm(dim=-1).mean()
            )
            parameter = next(self.parameters())
            debug["shared_s_text_prompt_tokens"] = parameter.new_tensor(
                float(self.shared_s_text_prompt_length)
            )
        self.debug_context = debug
