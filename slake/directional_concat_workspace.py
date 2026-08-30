"""Directional text-to-vision workspace with token-concatenated prompts."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Sequence

import torch
from torch import nn

from slake.sparse_visual_mmrl import SparseVisualInjectionBlock


class ZeroInitWorkspaceProjection(nn.Module):
    """Map workspace slots to anchored prompt tokens with a zero initial delta."""

    def __init__(self, workspace_dim: int, output_dim: int) -> None:
        super().__init__()
        if workspace_dim < 1 or output_dim < 1:
            raise ValueError("Workspace projection dimensions must be positive")
        self.workspace_dim = int(workspace_dim)
        self.output_dim = int(output_dim)
        self.norm = nn.LayerNorm(self.workspace_dim)
        self.input_projection = nn.Linear(self.workspace_dim, self.workspace_dim)
        self.output_projection = nn.Linear(self.workspace_dim, self.output_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._forward_audited = False

    def forward(
        self,
        workspace: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        if workspace.ndim != 3 or workspace.shape[-1] != self.workspace_dim:
            raise ValueError("Workspace projection received invalid slots")
        if anchor.shape != (workspace.shape[1], self.output_dim):
            raise ValueError(
                "Workspace text anchor must match slot count and output width"
            )
        parameter = self.input_projection.weight
        dtype = parameter.dtype
        hidden = torch.nn.functional.gelu(
            self.input_projection(
                self.norm(
                    workspace.to(device=parameter.device, dtype=dtype)
                )
            ),
            approximate="tanh",
        )
        delta = self.output_projection(hidden).to(dtype=workspace.dtype)
        anchored = anchor.to(
            device=delta.device,
            dtype=delta.dtype,
        ).unsqueeze(0) + delta

        with torch.no_grad():
            anchor_norm = anchor.detach().float().norm(dim=-1).mean()
            delta_norm = delta.detach().float().norm(dim=-1).mean()
            self.debug_context = {
                "anchor_norm_mean": anchor_norm,
                "delta_norm_mean": delta_norm,
                "delta_to_anchor_ratio": delta_norm / anchor_norm.clamp_min(1e-8),
            }
            if not self._forward_audited:
                max_abs = float(delta.detach().float().abs().max().item())
                if max_abs != 0.0:
                    raise RuntimeError(
                        "Workspace text projection must start at exact zero; "
                        f"max_abs_delta={max_abs}"
                    )
                print(
                    "[DIRECTIONAL_WORKSPACE_TEXT_ZERO_INIT_AUDIT] "
                    f"workspace_shape={tuple(workspace.shape)} "
                    f"anchor_shape={tuple(anchor.shape)} max_abs_delta=0.0 pass=True"
                )
                self._forward_audited = True
        return anchored


class DirectionalConcatWorkspaceVisual(nn.Module):
    """Build Z from question-derived queries and inject anchored visual tokens."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        anchor_layer: int,
        private_prompt_tokens: int = 8,
        workspace_tokens: int = 10,
        workspace_dim: int = 1024,
        workspace_heads: int = 16,
    ) -> None:
        super().__init__()
        if min(
            visual_dim,
            text_dim,
            private_prompt_tokens,
            workspace_tokens,
            workspace_dim,
            workspace_heads,
        ) < 1:
            raise ValueError("Directional Workspace dimensions must be positive")
        if anchor_layer < 0:
            raise ValueError("Directional Workspace anchor must be non-negative")
        if workspace_dim != visual_dim:
            raise ValueError(
                "Directional Workspace currently keeps visual width unchanged"
            )
        if workspace_dim % workspace_heads != 0:
            raise ValueError("workspace_dim must be divisible by workspace_heads")

        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.anchor_layers = (int(anchor_layer),)
        self.private_prompt_tokens = int(private_prompt_tokens)
        self.workspace_tokens = int(workspace_tokens)
        self.workspace_dim = int(workspace_dim)
        self.workspace_heads = int(workspace_heads)
        self.visual_prompt_tokens = (
            self.private_prompt_tokens + self.workspace_tokens
        )

        self.private_visual_prompt = nn.Parameter(
            torch.empty(self.private_prompt_tokens, self.visual_dim)
        )
        self.workspace_visual_anchor = nn.Parameter(
            torch.empty(self.workspace_tokens, self.visual_dim)
        )
        nn.init.normal_(self.private_visual_prompt, std=0.02)
        nn.init.normal_(self.workspace_visual_anchor, std=0.02)

        self.workspace_text_query_norm = nn.LayerNorm(self.text_dim)
        self.workspace_text_value_projection = nn.Linear(
            self.text_dim, self.workspace_dim, bias=False
        )
        self.workspace_text_score_projection = nn.Linear(
            self.text_dim, self.workspace_tokens, bias=False
        )
        self.workspace_query_norm = nn.LayerNorm(self.workspace_dim)
        self.workspace_visual_memory_norm = nn.LayerNorm(self.visual_dim)
        self.workspace_cross_attention = nn.MultiheadAttention(
            self.workspace_dim,
            self.workspace_heads,
            batch_first=True,
        )
        self.workspace_visual_delta = ZeroInitWorkspaceProjection(
            self.workspace_dim,
            self.visual_dim,
        )

        self.active = False
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._text_tokens: torch.Tensor | None = None
        self._text_token_mask: torch.Tensor | None = None
        self._images_per_sample: torch.Tensor | None = None
        self._image_lengths: torch.Tensor | None = None
        self._image_to_sample: torch.Tensor | None = None
        self._workspace_by_image: torch.Tensor | None = None
        self._installed = False
        self._forward_audited = False

    def private_parameters(self) -> list[nn.Parameter]:
        return [self.private_visual_prompt]

    def workspace_parameters(self) -> list[nn.Parameter]:
        private_id = id(self.private_visual_prompt)
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) != private_id
        ]

    def install(self, visual_model: nn.Module) -> None:
        if self._installed:
            raise RuntimeError("Directional Workspace is already installed")
        blocks = getattr(visual_model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Directional Workspace requires visual_model.blocks")
        anchor = self.anchor_layers[0]
        if anchor >= len(blocks):
            raise ValueError(
                f"Directional Workspace anchor {anchor} exceeds depth {len(blocks)}"
            )
        blocks[anchor] = SparseVisualInjectionBlock(blocks[anchor], self, anchor)
        self._installed = True
        print(
            "[DIRECTIONAL_CONCAT_WORKSPACE_LAYER_AUDIT] "
            f"vision_depth={len(blocks)} anchor_0based={anchor} "
            f"anchor_natural={anchor + 1} private_visual_tokens="
            f"{self.private_prompt_tokens} workspace_tokens={self.workspace_tokens} "
            f"workspace_dim={self.workspace_dim} heads={self.workspace_heads} "
            "text_query_visual_kv=True token_concat=True block_execution=single_pass"
        )

    @contextmanager
    def activate(
        self,
        _text_memory: torch.Tensor,
        images_per_sample: torch.Tensor,
        text_anchor: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
    ) -> Iterator[None]:
        if self.active:
            raise RuntimeError("Directional Workspace context is already active")
        if text_anchor is not None:
            raise ValueError("Directional Workspace does not consume a text anchor")
        if text_tokens is None or text_token_mask is None:
            raise RuntimeError("Directional Workspace requires full question tokens")
        if text_tokens.ndim != 3 or text_token_mask.shape != text_tokens.shape[:2]:
            raise ValueError("Directional Workspace question tokens have invalid shapes")
        lengths = text_token_mask.to(dtype=torch.bool).sum(dim=1)
        if bool((lengths == 0).any()):
            raise RuntimeError("Directional Workspace question memory is empty")
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
        self._text_tokens = compact
        self._text_token_mask = compact_mask
        self._images_per_sample = images_per_sample.to(dtype=torch.long)
        self._image_lengths = None
        self._image_to_sample = None
        self._workspace_by_image = None
        self.debug_context = {}
        try:
            yield
        finally:
            self.active = False
            self._text_tokens = None
            self._text_token_mask = None
            self._images_per_sample = None
            self._image_lengths = None
            self._image_to_sample = None
            self._workspace_by_image = None

    def prepare_visual(self, grid_thw: torch.Tensor) -> None:
        if not self.active:
            return
        if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError("Directional Workspace requires grid_thw=[images,3]")
        if self._images_per_sample is None:
            raise RuntimeError("Directional Workspace image counts are unavailable")
        if int(self._images_per_sample.sum().item()) != int(grid_thw.shape[0]):
            raise RuntimeError("Directional Workspace image count mismatch")
        self._image_lengths = grid_thw.prod(dim=-1).to(dtype=torch.long)
        sample_ids = torch.arange(
            self._images_per_sample.numel(),
            device=self._images_per_sample.device,
        )
        self._image_to_sample = torch.repeat_interleave(
            sample_ids,
            self._images_per_sample,
        )

    @staticmethod
    def _argument(
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        position: int,
        name: str,
    ) -> Any:
        if name in kwargs:
            return kwargs[name]
        return args[position] if len(args) > position else None

    def _segment_layout(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        if self._image_lengths is None or self._image_to_sample is None:
            raise RuntimeError("Directional Workspace visual context was not prepared")
        lengths_tensor = cu_seqlens[1:] - cu_seqlens[:-1]
        if bool((lengths_tensor <= 0).any()):
            raise RuntimeError("Directional Workspace segments must be non-empty")
        lengths = [int(value) for value in lengths_tensor.detach().cpu().tolist()]
        if sum(lengths) != int(hidden_states.shape[0]):
            raise RuntimeError("Directional Workspace segments do not match hidden states")
        if int(self._image_lengths.sum().item()) != int(hidden_states.shape[0]):
            raise RuntimeError("Directional Workspace image lengths do not match hidden states")
        starts = cu_seqlens[:-1].to(
            device=self._image_lengths.device,
            dtype=torch.long,
        )
        ends = cu_seqlens[1:].to(
            device=self._image_lengths.device,
            dtype=torch.long,
        )
        image_ends = self._image_lengths.cumsum(dim=0)
        image_ids = torch.bucketize(starts, image_ends, right=True)
        if bool((image_ids >= image_ends.numel()).any()):
            raise RuntimeError("Directional Workspace segment starts outside images")
        if bool((ends > image_ends.index_select(0, image_ids)).any()):
            raise RuntimeError("Directional Workspace segment crosses image boundary")
        return lengths, image_ids

    def _build_workspace(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self._image_lengths is None
            or self._image_to_sample is None
            or self._text_tokens is None
            or self._text_token_mask is None
        ):
            raise RuntimeError("Directional Workspace context is incomplete")
        image_segments = torch.split(
            hidden_states,
            self._image_lengths.detach().cpu().tolist(),
            dim=0,
        )
        visual_width = max(segment.shape[0] for segment in image_segments)
        dtype = self.workspace_cross_attention.in_proj_weight.dtype
        visual_tokens = hidden_states.new_zeros(
            len(image_segments),
            visual_width,
            self.visual_dim,
            dtype=dtype,
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

        image_to_sample = self._image_to_sample.to(hidden_states.device)
        text_tokens = self._text_tokens.to(hidden_states.device).index_select(
            0, image_to_sample
        ).to(dtype=dtype)
        text_mask = self._text_token_mask.to(hidden_states.device).index_select(
            0, image_to_sample
        )
        normalized_text = self.workspace_text_query_norm(text_tokens)
        text_values = self.workspace_text_value_projection(normalized_text)
        pooling_scores = self.workspace_text_score_projection(
            normalized_text
        ).transpose(1, 2)
        pooling_scores = pooling_scores.masked_fill(
            ~text_mask[:, None, :],
            torch.finfo(pooling_scores.dtype).min,
        )
        pooling = torch.softmax(pooling_scores.float(), dim=-1).to(dtype=dtype)
        queries = torch.matmul(pooling, text_values)

        normalized_visual = self.workspace_visual_memory_norm(visual_tokens)
        cross, cross_weights = self.workspace_cross_attention(
            self.workspace_query_norm(queries),
            normalized_visual,
            normalized_visual,
            key_padding_mask=~visual_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        workspace = queries + cross
        self._workspace_by_image = workspace

        with torch.no_grad():
            pooling_f = pooling.detach().float().clamp_min(1e-8)
            text_counts = text_mask.sum(dim=-1).clamp_min(2).float()
            text_entropy = -(pooling_f * pooling_f.log()).sum(dim=-1)
            text_entropy = text_entropy / text_counts.log()[:, None]
            cross_f = cross_weights.detach().float().clamp_min(1e-8)
            cross_f = cross_f * visual_mask[:, None, None, :]
            visual_counts = visual_mask.sum(dim=-1).clamp_min(2).float()
            visual_entropy = -(cross_f * cross_f.log()).sum(dim=-1)
            visual_entropy = visual_entropy / visual_counts.log()[:, None, None]
            query_norm = queries.detach().float().norm(dim=-1).mean()
            cross_norm = cross.detach().float().norm(dim=-1).mean()
            normalized_slots = torch.nn.functional.normalize(
                workspace.detach().float(),
                dim=-1,
            )
            pairwise = torch.matmul(
                normalized_slots,
                normalized_slots.transpose(1, 2),
            )
            off_diagonal = ~torch.eye(
                self.workspace_tokens,
                dtype=torch.bool,
                device=pairwise.device,
            )
            self.debug_context = {
                "workspace_text_pooling_entropy_norm": text_entropy.mean(),
                "workspace_visual_attention_entropy_norm": visual_entropy.mean(),
                "workspace_cross_delta_norm_mean": cross_norm,
                "workspace_text_query_norm_mean": query_norm,
                "workspace_cross_delta_to_query_ratio": cross_norm
                / query_norm.clamp_min(1e-8),
                "workspace_norm_mean": workspace.detach()
                .float()
                .norm(dim=-1)
                .mean(),
                "workspace_slot_pairwise_cosine_mean": pairwise[
                    :, off_diagonal
                ].mean(),
                "workspace_visual_tokens_mean": visual_mask.sum(dim=-1)
                .float()
                .mean(),
                "workspace_text_tokens_mean": text_mask.sum(dim=-1)
                .float()
                .mean(),
            }
        return workspace

    def shared_workspace_text_memory(self) -> torch.Tensor:
        if self._workspace_by_image is None:
            raise RuntimeError("Directional Workspace is unavailable")
        if self._image_to_sample is None or self._images_per_sample is None:
            raise RuntimeError("Directional Workspace sample mapping is unavailable")
        workspace = self._workspace_by_image
        sample_ids = self._image_to_sample.to(workspace.device)
        batch_size = int(self._images_per_sample.numel())
        combined = workspace.new_zeros(
            batch_size,
            self.workspace_tokens,
            self.workspace_dim,
        )
        combined = combined.index_add(0, sample_ids, workspace)
        counts = self._images_per_sample.to(
            device=workspace.device,
            dtype=workspace.dtype,
        ).view(batch_size, 1, 1)
        return combined / counts.clamp_min(1.0)

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
        offsets = torch.arange(
            cu_seqlens.numel(),
            device=cu_seqlens.device,
            dtype=cu_seqlens.dtype,
        ) * self.visual_prompt_tokens
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

        position_embeddings = self._argument(
            args, kwargs, 3, "position_embeddings"
        )
        if position_embeddings is not None:
            cos, sin = position_embeddings
            expanded_position = (
                self._prefix_tensor_by_segment(
                    cos, lengths, self.visual_prompt_tokens, 1.0
                ),
                self._prefix_tensor_by_segment(
                    sin, lengths, self.visual_prompt_tokens, 0.0
                ),
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
                rotary_pos_emb,
                lengths,
                self.visual_prompt_tokens,
                0.0,
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
                "Directional Workspace block requires hidden_states/cu_seqlens"
            )
        lengths, image_ids = self._segment_layout(hidden_states, cu_seqlens)
        workspace_by_image = self._build_workspace(hidden_states)
        workspace = workspace_by_image.index_select(
            0, image_ids.to(workspace_by_image.device)
        )
        workspace_visual = self.workspace_visual_delta(
            workspace,
            self.workspace_visual_anchor,
        ).to(device=hidden_states.device, dtype=hidden_states.dtype)
        private_visual = self.private_visual_prompt.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).unsqueeze(0).expand(len(lengths), -1, -1)
        reps = torch.cat((private_visual, workspace_visual), dim=1)
        expanded_args, expanded_kwargs = self._expanded_block_inputs(
            reps,
            args,
            kwargs,
            lengths,
            cu_seqlens,
        )
        adapted_with_prompt = block(*expanded_args, **expanded_kwargs)
        if not torch.is_tensor(adapted_with_prompt):
            raise TypeError("Directional Workspace visual block must return a tensor")
        adapted_segments = torch.split(
            adapted_with_prompt,
            [length + self.visual_prompt_tokens for length in lengths],
            dim=0,
        )
        adapted = torch.cat(
            [segment[self.visual_prompt_tokens :] for segment in adapted_segments],
            dim=0,
        )

        with torch.no_grad():
            input_norm = hidden_states.detach().float().norm(dim=-1).mean()
            output_norm = adapted.detach().float().norm(dim=-1).mean()
            private_norm = self.private_visual_prompt.detach().float().norm(
                dim=-1
            ).mean()
            workspace_anchor_norm = self.workspace_visual_anchor.detach().float().norm(
                dim=-1
            ).mean()
            self.debug_context.update(
                {
                    **{
                        f"workspace_visual_{key}": value
                        for key, value in self.workspace_visual_delta.debug_context.items()
                    },
                    "workspace_private_visual_prompt_norm_mean": private_norm,
                    "workspace_visual_anchor_norm_mean": workspace_anchor_norm,
                    "workspace_visual_output_to_input_ratio": output_norm
                    / input_norm.clamp_min(1e-8),
                    "workspace_visual_prompt_to_input_ratio": reps.detach()
                    .float()
                    .norm(dim=-1)
                    .mean()
                    / input_norm.clamp_min(1e-8),
                }
            )
            if not self._forward_audited:
                print(
                    "[DIRECTIONAL_CONCAT_WORKSPACE_FORWARD_AUDIT] "
                    f"layer={layer_index} units={len(lengths)} "
                    f"private_tokens={self.private_prompt_tokens} "
                    f"workspace_tokens={self.workspace_tokens} "
                    "insert_before_block=True strip_after_block=True pass=True"
                )
                self._forward_audited = True
        return adapted

    def workspace_inference_intervention_summary(self) -> Dict[str, Any]:
        return {
            "mode": "directional_concat",
            "anchor_layers": list(self.anchor_layers),
        }
