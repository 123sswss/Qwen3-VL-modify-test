"""Sparse local Rep-token injection for a frozen Qwen3-VL vision encoder."""

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn


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
    """Reuse one frozen block for its base and local Rep-conditioned paths."""

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
        original = self.block(*args, **kwargs)
        adapter = self._adapter_ref()
        if adapter is None or not adapter.active:
            return original
        return adapter.inject(
            self.layer_index,
            self.block,
            original,
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
        relation_weight: float = 0.05,
        relation_max_tokens: int = 64,
    ) -> None:
        super().__init__()
        anchors = tuple(int(index) for index in anchor_layers)
        if not anchors or len(set(anchors)) != len(anchors):
            raise ValueError(f"anchor_layers must be non-empty and unique: {anchors}")
        if min(anchors) < 0:
            raise ValueError("anchor layer indexes must be non-negative")
        if min(visual_dim, text_dim, rep_token_count, attention_dim, num_heads) < 1:
            raise ValueError("Sparse Visual MMRL dimensions must be positive")
        if relation_weight < 0.0:
            raise ValueError("relation_weight must be non-negative")

        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.anchor_layers = anchors
        self.rep_token_count = int(rep_token_count)
        self.relation_weight = float(relation_weight)
        self.relation_max_tokens = int(relation_max_tokens)
        self.shared_rep = nn.Parameter(
            torch.empty(self.rep_token_count, self.visual_dim)
        )
        self.layer_embeddings = nn.Parameter(
            torch.empty(len(self.anchor_layers), 1, self.visual_dim)
        )
        self.residual_scales = nn.Parameter(torch.zeros(len(self.anchor_layers)))
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
        self._relation_losses: list[torch.Tensor] = []
        self._debug_rows: list[Dict[str, torch.Tensor]] = []
        self._installed = False
        self._forward_audited = False

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
        for index in self.anchor_layers:
            blocks[index] = SparseVisualInjectionBlock(blocks[index], self, index)
        self._installed = True
        print(
            "[SPARSE_VISUAL_LAYER_AUDIT] "
            f"vision_depth={depth} anchors_0based={list(self.anchor_layers)} "
            f"anchors_natural={[index + 1 for index in self.anchor_layers]} "
            "local_insert_strip=True shared_frozen_block=True"
        )

    @contextmanager
    def activate(
        self,
        text_memory: torch.Tensor,
        images_per_sample: torch.Tensor,
    ) -> Iterator[None]:
        if self.active:
            raise RuntimeError("Sparse Visual MMRL context is already active")
        self.active = True
        self._text_memory = text_memory
        self._images_per_sample = images_per_sample.to(dtype=torch.long)
        self._image_lengths = None
        self._image_to_sample = None
        self._relation_losses = []
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

    @property
    def relation_loss(self) -> torch.Tensor:
        if self._relation_losses:
            return torch.stack(self._relation_losses).mean()
        return self.shared_rep.new_tensor(0.0)

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
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
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
        return visual_memory, text_memory, lengths

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

    def _relation(
        self,
        adapted: torch.Tensor,
        original: torch.Tensor,
        lengths: Sequence[int],
    ) -> torch.Tensor:
        losses = []
        for adapted_segment, original_segment in zip(
            torch.split(adapted, list(lengths), dim=0),
            torch.split(original, list(lengths), dim=0),
        ):
            count = int(adapted_segment.shape[0])
            if count < 2:
                continue
            if count > self.relation_max_tokens:
                indices = (
                    torch.linspace(
                        0,
                        count - 1,
                        steps=self.relation_max_tokens,
                        device=adapted.device,
                    )
                    .round()
                    .long()
                )
                adapted_segment = adapted_segment.index_select(0, indices)
                original_segment = original_segment.index_select(0, indices)
            adapted_unit = F.normalize(adapted_segment.float(), dim=-1, eps=1e-6)
            original_unit = F.normalize(
                original_segment.detach().float(), dim=-1, eps=1e-6
            )
            losses.append(
                F.mse_loss(
                    adapted_unit @ adapted_unit.transpose(0, 1),
                    original_unit @ original_unit.transpose(0, 1),
                )
            )
        return torch.stack(losses).mean() if losses else adapted.new_tensor(0.0)

    def inject(
        self,
        layer_index: int,
        block: nn.Module,
        original: torch.Tensor,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self._argument(args, kwargs, 0, "hidden_states")
        cu_seqlens = self._argument(args, kwargs, 1, "cu_seqlens")
        if hidden_states is None or cu_seqlens is None:
            raise RuntimeError(
                "Sparse Visual block did not receive hidden_states/cu_seqlens"
            )
        if not torch.is_tensor(original):
            raise TypeError("Sparse Visual expects vision blocks to return a tensor")
        visual_memory, text_memory, lengths = self._segment_memories(
            hidden_states, cu_seqlens
        )
        layer_position = self.anchor_layers.index(layer_index)
        queries = self.shared_rep.unsqueeze(0) + self.layer_embeddings[layer_position]
        queries = queries.expand(len(lengths), -1, -1)
        reps = self.rep_attention(queries, visual_memory, text_memory).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        expanded_args, expanded_kwargs = self._expanded_block_inputs(
            reps, args, kwargs, lengths, cu_seqlens
        )
        adapted_with_rep = block(*expanded_args, **expanded_kwargs)
        adapted = torch.cat(
            [
                segment[self.rep_token_count :]
                for segment in torch.split(
                    adapted_with_rep,
                    [length + self.rep_token_count for length in lengths],
                    dim=0,
                )
            ],
            dim=0,
        )
        scale = torch.tanh(self.residual_scales[layer_position]).to(
            device=original.device, dtype=original.dtype
        )
        delta = adapted - original
        mixed = original + scale * delta
        relation = self._relation(mixed, original, lengths)
        self._relation_losses.append(relation)

        with torch.no_grad():
            original_norm = (
                original.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
            )
            applied_delta_norm = (mixed - original).detach().float().norm(dim=-1).mean()
            self._debug_rows.append(
                {
                    "layer": original.new_tensor(float(layer_index)),
                    "scale": scale.detach().float(),
                    "delta_ratio": applied_delta_norm / original_norm,
                    "relation": relation.detach().float(),
                    **self.rep_attention.debug_context,
                }
            )
            if not self._forward_audited:
                max_abs = float((mixed - original).detach().float().abs().max().item())
                if max_abs != 0.0:
                    raise RuntimeError(
                        "Sparse Visual residual scales must initialize to exact zero; "
                        f"max_abs_delta={max_abs}"
                    )
                print(
                    "[SPARSE_VISUAL_ZERO_INIT_AUDIT] "
                    f"layer={layer_index} units={len(lengths)} reps={self.rep_token_count} "
                    "max_abs_delta=0.0 pass=True"
                )
                if layer_index == self.anchor_layers[-1]:
                    self._forward_audited = True
        self._refresh_debug_context()
        return mixed

    def _refresh_debug_context(self) -> None:
        if not self._debug_rows:
            self.debug_context = {}
            return
        debug: Dict[str, torch.Tensor] = {}
        for row in self._debug_rows:
            layer = int(row["layer"].item())
            debug[f"sparse_visual_layer{layer}_scale"] = row["scale"]
            debug[f"sparse_visual_layer{layer}_delta_ratio"] = row["delta_ratio"]
            debug[f"sparse_visual_layer{layer}_relation_loss"] = row["relation"]
            debug[f"sparse_visual_layer{layer}_attention_entropy_norm"] = row[
                "attention_entropy_norm"
            ]
            debug[f"sparse_visual_layer{layer}_visual_attention_mean"] = row[
                "visual_attention_mean"
            ]
        debug["sparse_visual_relation_loss"] = self.relation_loss.detach().float()
        debug["sparse_visual_scale_abs_mean"] = torch.stack(
            [row["scale"].abs() for row in self._debug_rows]
        ).mean()
        debug["sparse_visual_delta_ratio_mean"] = torch.stack(
            [row["delta_ratio"] for row in self._debug_rows]
        ).mean()
        self.debug_context = debug
