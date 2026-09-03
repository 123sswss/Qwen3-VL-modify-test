"""Directional text-to-vision workspace with token-concatenated prompts."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Sequence

import torch
from torch import nn

from slake.sparse_visual_mmrl import SparseVisualInjectionBlock


DIRECTIONAL_VISUAL_MEMORY_INTERVENTIONS = (
    "normal",
    "previous-distinct-image",
)

DIRECTIONAL_QUESTION_QUERY_INTERVENTIONS = (
    "normal",
    "previous-distinct-question",
)

DIRECTIONAL_QUERY_SOURCES = (
    "question_attention_pooling",
    "learned_static",
)


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
        delta_scale: float = 1.0,
    ) -> torch.Tensor:
        if workspace.ndim != 3 or workspace.shape[-1] != self.workspace_dim:
            raise ValueError("Workspace projection received invalid slots")
        if anchor.shape != (workspace.shape[1], self.output_dim):
            raise ValueError(
                "Workspace text anchor must match slot count and output width"
            )
        scale = float(delta_scale)
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("Workspace delta_scale must be finite and in [0, 1]")
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
        scaled_delta = delta * scale
        anchored = anchor.to(
            device=delta.device,
            dtype=delta.dtype,
        ).unsqueeze(0) + scaled_delta

        with torch.no_grad():
            anchor_norm = anchor.detach().float().norm(dim=-1).mean()
            raw_delta_norm = delta.detach().float().norm(dim=-1).mean()
            delta_norm = scaled_delta.detach().float().norm(dim=-1).mean()
            self.debug_context = {
                "anchor_norm_mean": anchor_norm,
                "raw_delta_norm_mean": raw_delta_norm,
                "delta_norm_mean": delta_norm,
                "delta_to_anchor_ratio": delta_norm / anchor_norm.clamp_min(1e-8),
                "delta_scale": delta.new_tensor(scale),
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
        anchor_layer: int | None = None,
        private_prompt_tokens: int = 8,
        workspace_tokens: int = 10,
        workspace_dim: int = 1024,
        workspace_heads: int = 16,
        visual_dynamic_write: bool = True,
        static_visual_write: bool = True,
        unified_static_visual_prompt: bool = False,
        direct_visual_z_tokens: bool = False,
        query_source: str = "question_attention_pooling",
        anchor_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if min(
            visual_dim,
            text_dim,
            workspace_tokens,
            workspace_dim,
            workspace_heads,
        ) < 1:
            raise ValueError("Directional Workspace dimensions must be positive")
        if private_prompt_tokens < 0 or (
            static_visual_write
            and not direct_visual_z_tokens
            and private_prompt_tokens < 1
        ):
            raise ValueError(
                "private_prompt_tokens must be positive when static visual "
                "write is enabled, and non-negative otherwise"
            )
        if anchor_layers is None:
            if anchor_layer is None:
                raise ValueError("Directional Workspace requires visual anchors")
            anchors = (int(anchor_layer),)
        else:
            if anchor_layer is not None:
                raise ValueError("Specify anchor_layer or anchor_layers, not both")
            anchors = tuple(int(index) for index in anchor_layers)
        if not anchors or any(index < 0 for index in anchors):
            raise ValueError("Directional Workspace anchors must be non-negative")
        if len(set(anchors)) != len(anchors) or tuple(sorted(anchors)) != anchors:
            raise ValueError(
                "Directional Workspace anchors must be unique and ascending"
            )
        if workspace_dim % workspace_heads != 0:
            raise ValueError("workspace_dim must be divisible by workspace_heads")
        if query_source not in DIRECTIONAL_QUERY_SOURCES:
            raise ValueError(
                f"Unsupported Directional query_source={query_source!r}; "
                f"choices={DIRECTIONAL_QUERY_SOURCES}"
            )
        if visual_dynamic_write and not static_visual_write:
            raise ValueError(
                "Directional visual dynamic write requires static visual write"
            )
        if direct_visual_z_tokens and not static_visual_write:
            raise ValueError(
                "Direct visual Z tokens require static visual write"
            )
        if direct_visual_z_tokens and visual_dynamic_write:
            raise ValueError(
                "Direct visual Z tokens and anchored visual dynamic write "
                "are mutually exclusive"
            )
        if unified_static_visual_prompt and (
            not static_visual_write
            or visual_dynamic_write
            or direct_visual_z_tokens
        ):
            raise ValueError(
                "Unified static visual Prompt requires static-only visual write"
            )

        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.anchor_layers = anchors
        self.unified_static_visual_prompt = bool(unified_static_visual_prompt)
        self.private_prompt_tokens = (
            int(private_prompt_tokens)
            if static_visual_write
            and not direct_visual_z_tokens
            and not self.unified_static_visual_prompt
            else 0
        )
        self.static_visual_prompt_tokens = (
            int(private_prompt_tokens) if self.unified_static_visual_prompt else 0
        )
        self.workspace_tokens = int(workspace_tokens)
        self.workspace_dim = int(workspace_dim)
        self.workspace_heads = int(workspace_heads)
        self.visual_dynamic_write = bool(visual_dynamic_write)
        self.static_visual_write = bool(static_visual_write)
        self.direct_visual_z_tokens = bool(direct_visual_z_tokens)
        self.query_source = str(query_source)
        if not self.static_visual_write:
            self.visual_prompt_tokens = 0
        elif self.direct_visual_z_tokens:
            self.visual_prompt_tokens = 2 * self.workspace_tokens
        elif self.unified_static_visual_prompt:
            self.visual_prompt_tokens = self.static_visual_prompt_tokens
        else:
            self.visual_prompt_tokens = (
                self.private_prompt_tokens + self.workspace_tokens
            )

        if self.unified_static_visual_prompt:
            downstream_rng_state = torch.random.get_rng_state()
            self.static_visual_prompt = nn.Parameter(
                torch.empty(self.static_visual_prompt_tokens, self.visual_dim)
            )
            nn.init.normal_(self.static_visual_prompt, std=0.02)
            # Keep downstream Directional initialization identical to the
            # historical A10-then-S8 seed44 control despite changing V length.
            torch.random.set_rng_state(downstream_rng_state)
            nn.init.normal_(
                torch.empty(self.workspace_tokens, self.visual_dim),
                std=0.02,
            )
            nn.init.normal_(torch.empty(8, self.visual_dim), std=0.02)
            self.register_parameter("private_visual_prompt", None)
            self.register_parameter("workspace_visual_anchor", None)
        elif self.static_visual_write:
            self.register_parameter("static_visual_prompt", None)
            self.workspace_visual_anchor = nn.Parameter(
                torch.empty(self.workspace_tokens, self.visual_dim)
            )
            nn.init.normal_(self.workspace_visual_anchor, std=0.02)
            if self.direct_visual_z_tokens:
                self.register_parameter("private_visual_prompt", None)
            else:
                self.private_visual_prompt = nn.Parameter(
                    torch.empty(self.private_prompt_tokens, self.visual_dim)
                )
                nn.init.normal_(self.private_visual_prompt, std=0.02)
        else:
            self.register_parameter("static_visual_prompt", None)
            self.register_parameter("private_visual_prompt", None)
            self.register_parameter("workspace_visual_anchor", None)

        self.workspace_text_query_norm = nn.LayerNorm(self.text_dim)
        self.workspace_text_value_projection = nn.Linear(
            self.text_dim, self.workspace_dim, bias=False
        )
        if self.query_source == "question_attention_pooling":
            self.workspace_text_score_projection = nn.Linear(
                self.text_dim, self.workspace_tokens, bias=False
            )
            self.register_parameter("learned_static_query", None)
        else:
            self.workspace_text_score_projection = None
            self.learned_static_query = nn.Parameter(
                torch.empty(self.workspace_tokens, self.text_dim)
            )
            nn.init.normal_(self.learned_static_query, std=0.02)
        self.workspace_query_norm = nn.LayerNorm(self.workspace_dim)
        self.workspace_visual_memory_norm = nn.LayerNorm(self.visual_dim)
        self.workspace_visual_memory_projection = (
            nn.Identity()
            if self.workspace_dim == self.visual_dim
            else nn.Linear(self.visual_dim, self.workspace_dim, bias=False)
        )
        self.workspace_cross_attention = nn.MultiheadAttention(
            self.workspace_dim,
            self.workspace_heads,
            batch_first=True,
        )
        self.workspace_visual_delta = (
            ZeroInitWorkspaceProjection(
                self.workspace_dim,
                self.visual_dim,
            )
            if self.visual_dynamic_write and self.static_visual_write
            else None
        )
        self.direct_visual_z_projection = (
            nn.Sequential(
                nn.LayerNorm(self.workspace_dim),
                nn.Linear(self.workspace_dim, self.visual_dim),
            )
            if self.direct_visual_z_tokens
            else None
        )

        self.active = False
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._text_tokens: torch.Tensor | None = None
        self._text_token_mask: torch.Tensor | None = None
        self._images_per_sample: torch.Tensor | None = None
        self._image_lengths: torch.Tensor | None = None
        self._image_to_sample: torch.Tensor | None = None
        self._workspace_by_image: torch.Tensor | None = None
        self._layer_debug_context: Dict[str, torch.Tensor] = {}
        self._installed = False
        self._forward_audited = False
        self._visual_delta_scale = 1.0
        self._visual_memory_mode = "normal"
        self._question_query_mode = "normal"
        self._last_visual_memory: torch.Tensor | None = None
        self._previous_distinct_visual_memory: torch.Tensor | None = None
        self._visual_memory_images_seen = 0
        self._visual_memory_images_mismatched = 0
        self._visual_memory_images_natural = 0
        self._visual_memory_mismatch_rate = torch.tensor(0.0)
        self._visual_memory_source_cosine = torch.tensor(1.0)
        self._last_question_query: torch.Tensor | None = None
        self._previous_distinct_question_query: torch.Tensor | None = None
        self._question_query_units_seen = 0
        self._question_query_units_mismatched = 0
        self._question_query_units_natural = 0
        self._question_query_mismatch_rate = torch.tensor(0.0)
        self._question_query_source_cosine = torch.tensor(1.0)

    def private_parameters(self) -> list[nn.Parameter]:
        return (
            [self.private_visual_prompt]
            if self.private_visual_prompt is not None
            else []
        )

    def workspace_parameters(self) -> list[nn.Parameter]:
        private_id = (
            id(self.private_visual_prompt)
            if self.private_visual_prompt is not None
            else None
        )
        return [
            parameter
            for parameter in self.parameters()
            if private_id is None or id(parameter) != private_id
        ]

    def _conditioning_intervention_scope(self) -> str:
        visual_active = self._visual_memory_mode != "normal"
        question_active = self._question_query_mode != "normal"
        if visual_active and question_active:
            return "directional_ca_conditioning_only"
        if visual_active:
            return "directional_ca_visual_kv_only"
        if question_active:
            return "directional_ca_question_q_only"
        return "none"

    def configure_inference_intervention(
        self,
        visual_delta_scale: float = 1.0,
        visual_memory_mode: str = "normal",
        question_query_mode: str = "normal",
    ) -> None:
        scale = float(visual_delta_scale)
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError(
                "Directional visual delta scale must be finite and in [0, 1]"
            )
        if (
            not self.visual_dynamic_write
            and not self.direct_visual_z_tokens
            and scale != 1.0
        ):
            raise ValueError(
                "Directional visual delta scale requires visual_dynamic_write"
            )
        mode = str(visual_memory_mode)
        if mode not in DIRECTIONAL_VISUAL_MEMORY_INTERVENTIONS:
            raise ValueError(
                "Unsupported Directional visual memory intervention="
                f"{mode!r}; choices={DIRECTIONAL_VISUAL_MEMORY_INTERVENTIONS}"
            )
        query_mode = str(question_query_mode)
        if query_mode not in DIRECTIONAL_QUESTION_QUERY_INTERVENTIONS:
            raise ValueError(
                "Unsupported Directional question query intervention="
                f"{query_mode!r}; choices={DIRECTIONAL_QUESTION_QUERY_INTERVENTIONS}"
            )
        if self.training and (
            scale != 1.0 or mode != "normal" or query_mode != "normal"
        ):
            raise RuntimeError(
                "Directional conditioning interventions are inference-only"
            )
        if self.query_source == "learned_static" and query_mode != "normal":
            raise ValueError(
                "Question-query mismatch requires question_attention_pooling"
            )
        self._visual_delta_scale = scale
        self._visual_memory_mode = mode
        self._question_query_mode = query_mode
        self.reset_inference_intervention_state()
        intervention_scope = self._conditioning_intervention_scope()
        print(
            "[DIRECTIONAL_CONCAT_WORKSPACE_VISUAL_INTERVENTION] "
            f"visual_delta_scale={scale} visual_memory_mode={mode} "
            f"question_query_mode={query_mode} "
            f"visual_dynamic_write={self.visual_dynamic_write} "
            "original_image_input_unchanged=True "
            "original_question_input_unchanged=True "
            f"intervention_scope={intervention_scope} "
            "anchors_preserved=True "
            "token_count_preserved=True"
        )

    def reset_inference_intervention_state(self) -> None:
        self._last_visual_memory = None
        self._previous_distinct_visual_memory = None
        self._visual_memory_images_seen = 0
        self._visual_memory_images_mismatched = 0
        self._visual_memory_images_natural = 0
        self._visual_memory_mismatch_rate = torch.tensor(0.0)
        self._visual_memory_source_cosine = torch.tensor(1.0)
        self._last_question_query = None
        self._previous_distinct_question_query = None
        self._question_query_units_seen = 0
        self._question_query_units_mismatched = 0
        self._question_query_units_natural = 0
        self._question_query_mismatch_rate = torch.tensor(0.0)
        self._question_query_source_cosine = torch.tensor(1.0)

    @staticmethod
    def _same_visual_memory(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> bool:
        return left.shape == right.shape and bool(torch.equal(left, right))

    def _apply_visual_memory_intervention(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._visual_memory_mode == "normal":
            self._visual_memory_mismatch_rate = visual_tokens.new_tensor(0.0)
            self._visual_memory_source_cosine = visual_tokens.new_tensor(1.0)
            return visual_tokens, visual_mask
        if self.training:
            raise RuntimeError(
                "Directional visual memory intervention is inference-only"
            )

        selected_memories = []
        mismatch_flags = []
        source_cosines = []
        for image_index in range(visual_tokens.shape[0]):
            current = visual_tokens[image_index][visual_mask[image_index]].detach()
            if current.numel() == 0:
                raise RuntimeError("Directional visual memory cannot be empty")
            cached_current = current.clone()
            source = current
            mismatched = False
            if self._last_visual_memory is None:
                self._last_visual_memory = cached_current
            elif self._same_visual_memory(current, self._last_visual_memory):
                if self._previous_distinct_visual_memory is not None:
                    source = self._previous_distinct_visual_memory
                    mismatched = True
            else:
                self._previous_distinct_visual_memory = self._last_visual_memory
                self._last_visual_memory = cached_current
                source = self._previous_distinct_visual_memory
                mismatched = True

            selected_memories.append(source)
            mismatch_flags.append(mismatched)
            current_mean = current.float().mean(dim=0)
            source_mean = source.float().mean(dim=0)
            source_cosines.append(
                torch.nn.functional.cosine_similarity(
                    current_mean,
                    source_mean,
                    dim=0,
                )
            )

        width = max(memory.shape[0] for memory in selected_memories)
        selected_tokens = visual_tokens.new_zeros(
            len(selected_memories),
            width,
            self.visual_dim,
        )
        selected_mask = torch.zeros(
            len(selected_memories),
            width,
            dtype=torch.bool,
            device=visual_tokens.device,
        )
        for image_index, memory in enumerate(selected_memories):
            selected_tokens[image_index, : memory.shape[0]] = memory.to(
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
            selected_mask[image_index, : memory.shape[0]] = True

        mismatched_count = sum(mismatch_flags)
        image_count = len(mismatch_flags)
        self._visual_memory_images_seen += image_count
        self._visual_memory_images_mismatched += mismatched_count
        self._visual_memory_images_natural += image_count - mismatched_count
        self._visual_memory_mismatch_rate = visual_tokens.new_tensor(
            mismatched_count / image_count
        )
        self._visual_memory_source_cosine = torch.stack(source_cosines).mean().to(
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        return selected_tokens, selected_mask

    def _apply_question_query_intervention(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if self._question_query_mode == "normal":
            self._question_query_mismatch_rate = queries.new_tensor(0.0)
            self._question_query_source_cosine = queries.new_tensor(1.0)
            return queries
        if self.training:
            raise RuntimeError(
                "Directional question query intervention is inference-only"
            )

        selected_queries = []
        mismatch_flags = []
        source_cosines = []
        for unit_index in range(queries.shape[0]):
            current = queries[unit_index].detach()
            cached_current = current.clone()
            source = current
            mismatched = False
            if self._last_question_query is None:
                self._last_question_query = cached_current
            elif self._same_visual_memory(current, self._last_question_query):
                if self._previous_distinct_question_query is not None:
                    source = self._previous_distinct_question_query
                    mismatched = True
            else:
                self._previous_distinct_question_query = self._last_question_query
                self._last_question_query = cached_current
                source = self._previous_distinct_question_query
                mismatched = True

            selected_queries.append(source)
            mismatch_flags.append(mismatched)
            source_cosines.append(
                torch.nn.functional.cosine_similarity(
                    current.float().flatten(),
                    source.float().flatten(),
                    dim=0,
                )
            )

        mismatched_count = sum(mismatch_flags)
        unit_count = len(mismatch_flags)
        self._question_query_units_seen += unit_count
        self._question_query_units_mismatched += mismatched_count
        self._question_query_units_natural += unit_count - mismatched_count
        self._question_query_mismatch_rate = queries.new_tensor(
            mismatched_count / unit_count
        )
        self._question_query_source_cosine = torch.stack(source_cosines).mean().to(
            device=queries.device,
            dtype=queries.dtype,
        )
        return torch.stack(selected_queries).to(
            device=queries.device,
            dtype=queries.dtype,
        )

    def install(self, visual_model: nn.Module) -> None:
        if self._installed:
            raise RuntimeError("Directional Workspace is already installed")
        blocks = getattr(visual_model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Directional Workspace requires visual_model.blocks")
        invalid = [index for index in self.anchor_layers if index >= len(blocks)]
        if invalid:
            raise ValueError(
                "Directional Workspace anchors exceed visual depth: "
                f"anchors={invalid} depth={len(blocks)}"
            )
        for anchor in self.anchor_layers:
            blocks[anchor] = SparseVisualInjectionBlock(blocks[anchor], self, anchor)
        self._installed = True
        print(
            "[DIRECTIONAL_CONCAT_WORKSPACE_LAYER_AUDIT] "
            f"vision_depth={len(blocks)} anchors_0based={list(self.anchor_layers)} "
            f"anchors_natural={[index + 1 for index in self.anchor_layers]} "
            "parameter_sharing=all_directional_and_visual_prompt_parameters "
            f"private_visual_tokens="
            f"{self.private_prompt_tokens} workspace_tokens={self.workspace_tokens} "
            f"direct_visual_z_tokens_count="
            f"{self.workspace_tokens if self.direct_visual_z_tokens else 0} "
            f"workspace_dim={self.workspace_dim} heads={self.workspace_heads} "
            f"visual_dynamic_write={self.visual_dynamic_write} "
            f"direct_visual_z_tokens={self.direct_visual_z_tokens} "
            f"static_visual_write={self.static_visual_write} "
            f"query_source={self.query_source} text_query_visual_kv=True "
            f"token_concat={self.static_visual_write} block_execution=single_pass"
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
        self._layer_debug_context = {}
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
            self._layer_debug_context = {}

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
        if self.query_source == "question_attention_pooling":
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
            natural_queries = torch.matmul(pooling, text_values)
        else:
            static_query = self.learned_static_query.to(
                device=hidden_states.device,
                dtype=dtype,
            )
            natural_queries = self.workspace_text_value_projection(
                self.workspace_text_query_norm(static_query)
            ).unsqueeze(0).expand(len(image_segments), -1, -1)
            pooling = None
        queries = self._apply_question_query_intervention(natural_queries)

        attention_visual_tokens, attention_visual_mask = (
            self._apply_visual_memory_intervention(visual_tokens, visual_mask)
        )
        normalized_visual = self.workspace_visual_memory_projection(
            self.workspace_visual_memory_norm(attention_visual_tokens)
        )
        cross, cross_weights = self.workspace_cross_attention(
            self.workspace_query_norm(queries),
            normalized_visual,
            normalized_visual,
            key_padding_mask=~attention_visual_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        workspace = queries + cross
        self._workspace_by_image = workspace

        with torch.no_grad():
            if pooling is not None:
                pooling_f = pooling.detach().float()
                text_mask_f = text_mask[:, None, :]
                pooling_safe = torch.where(
                    text_mask_f,
                    pooling_f.clamp_min(1e-8),
                    torch.ones_like(pooling_f),
                )
                text_counts = text_mask.sum(dim=-1).clamp_min(2).float()
                text_entropy = -(
                    pooling_f * pooling_safe.log() * text_mask_f
                ).sum(dim=-1)
                text_entropy = text_entropy / text_counts.log()[:, None]
                text_tokens_used = text_mask.sum(dim=-1).float().mean()
            else:
                text_entropy = queries.new_zeros(queries.shape[:2], dtype=torch.float32)
                text_tokens_used = queries.new_tensor(0.0)
            cross_f = cross_weights.detach().float()
            visual_mask_f = attention_visual_mask[:, None, None, :]
            cross_safe = torch.where(
                visual_mask_f,
                cross_f.clamp_min(1e-8),
                torch.ones_like(cross_f),
            )
            visual_counts = attention_visual_mask.sum(dim=-1).clamp_min(2).float()
            visual_entropy = -(
                cross_f * cross_safe.log() * visual_mask_f
            ).sum(dim=-1)
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
                "workspace_question_query_mismatch_rate": (
                    self._question_query_mismatch_rate
                ),
                "workspace_question_query_source_cosine_mean": (
                    self._question_query_source_cosine
                ),
                "workspace_cross_delta_to_query_ratio": cross_norm
                / query_norm.clamp_min(1e-8),
                "workspace_norm_mean": workspace.detach()
                .float()
                .norm(dim=-1)
                .mean(),
                "workspace_slot_pairwise_cosine_mean": pairwise[
                    :, off_diagonal
                ].mean(),
                "workspace_visual_tokens_mean": attention_visual_mask.sum(dim=-1)
                .float()
                .mean(),
                "workspace_visual_projected_norm_mean": normalized_visual.detach()
                .float()
                .norm(dim=-1)
                .mean(),
                "workspace_visual_projection_enabled": workspace.new_tensor(
                    float(self.workspace_dim != self.visual_dim)
                ),
                "workspace_query_is_static": workspace.new_tensor(
                    float(self.query_source == "learned_static")
                ),
                "workspace_visual_memory_mismatch_rate": (
                    self._visual_memory_mismatch_rate
                ),
                "workspace_visual_memory_source_cosine_mean": (
                    self._visual_memory_source_cosine
                ),
                "workspace_text_tokens_mean": text_tokens_used,
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
        if not self.static_visual_write:
            adapted = block(*args, **kwargs)
            if not torch.is_tensor(adapted):
                raise TypeError("Directional Workspace visual block must return a tensor")
            with torch.no_grad():
                input_norm = hidden_states.detach().float().norm(dim=-1).mean()
                output_norm = adapted.detach().float().norm(dim=-1).mean()
                zero = input_norm.new_tensor(0.0)
                self.debug_context.update(
                    {
                        "workspace_visual_dynamic_write_enabled": zero,
                        "workspace_direct_visual_z_enabled": zero,
                        "workspace_static_visual_write_enabled": zero,
                        "workspace_private_visual_prompt_norm_mean": zero,
                        "workspace_visual_anchor_norm_mean": zero,
                        "workspace_visual_raw_delta_norm_mean": zero,
                        "workspace_visual_delta_norm_mean": zero,
                        "workspace_visual_delta_to_anchor_ratio": zero,
                        "workspace_visual_delta_scale": zero,
                        "workspace_visual_output_to_input_ratio": output_norm
                        / input_norm.clamp_min(1e-8),
                        "workspace_visual_prompt_to_input_ratio": zero,
                    }
                )
                self._capture_layer_debug(layer_index)
                if not self._forward_audited:
                    print(
                        "[DIRECTIONAL_CONCAT_WORKSPACE_FORWARD_AUDIT] "
                        f"layer={layer_index} units={len(lengths)} "
                        "private_tokens=0 workspace_tokens=0 "
                        "visual_dynamic_write=False static_visual_write=False "
                        "insert_before_block=False strip_after_block=False "
                        "read_visual_kv_before_block=True pass=True"
                    )
                    self._forward_audited = True
            return adapted
        workspace = workspace_by_image.index_select(
            0, image_ids.to(workspace_by_image.device)
        )
        if self.unified_static_visual_prompt:
            reps = self.static_visual_prompt.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).unsqueeze(0).expand(len(lengths), -1, -1)
            zero = workspace.detach().new_tensor(0.0)
            workspace_visual_debug = {
                "workspace_visual_raw_delta_norm_mean": zero,
                "workspace_visual_delta_norm_mean": zero,
                "workspace_visual_delta_to_anchor_ratio": zero,
                "workspace_visual_delta_scale": zero,
            }
        elif self.direct_visual_z_tokens:
            workspace_visual = self.workspace_visual_anchor.unsqueeze(0).expand(
                workspace.shape[0], -1, -1
            )
            raw_direct_visual_z = self.direct_visual_z_projection(workspace)
            direct_visual_z = raw_direct_visual_z * self._visual_delta_scale
            direct_visual_z = direct_visual_z.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            workspace_visual_debug = {
                "workspace_visual_raw_delta_norm_mean": raw_direct_visual_z.detach()
                .float()
                .norm(dim=-1)
                .mean(),
                "workspace_visual_delta_norm_mean": direct_visual_z.detach()
                .float()
                .norm(dim=-1)
                .mean(),
                "workspace_visual_delta_to_anchor_ratio": direct_visual_z.detach()
                .float()
                .norm(dim=-1)
                .mean()
                / self.workspace_visual_anchor.detach()
                .float()
                .norm(dim=-1)
                .mean()
                .clamp_min(1e-8),
                "workspace_visual_delta_scale": workspace.new_tensor(
                    self._visual_delta_scale
                ),
                "workspace_direct_visual_z_norm_mean": direct_visual_z.detach()
                .float()
                .norm(dim=-1)
                .mean(),
            }
        elif self.workspace_visual_delta is not None:
            workspace_visual = self.workspace_visual_delta(
                workspace,
                self.workspace_visual_anchor,
                delta_scale=self._visual_delta_scale,
            )
            workspace_visual_debug = {
                f"workspace_visual_{key}": value
                for key, value in self.workspace_visual_delta.debug_context.items()
            }
        else:
            workspace_visual = self.workspace_visual_anchor.unsqueeze(0).expand(
                workspace.shape[0], -1, -1
            )
            zero = workspace.detach().new_tensor(0.0)
            workspace_visual_debug = {
                "workspace_visual_raw_delta_norm_mean": zero,
                "workspace_visual_delta_norm_mean": zero,
                "workspace_visual_delta_to_anchor_ratio": zero,
                "workspace_visual_delta_scale": zero,
            }
        if not self.unified_static_visual_prompt:
            workspace_visual = workspace_visual.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if self.direct_visual_z_tokens:
                reps = torch.cat((workspace_visual, direct_visual_z), dim=1)
            else:
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
            private_norm = (
                hidden_states.new_tensor(0.0)
                if self.private_visual_prompt is None
                else self.private_visual_prompt.detach().float().norm(
                    dim=-1
                ).mean()
            )
            unified_static_norm = (
                hidden_states.new_tensor(0.0)
                if self.static_visual_prompt is None
                else self.static_visual_prompt.detach().float().norm(dim=-1).mean()
            )
            workspace_anchor_norm = (
                hidden_states.new_tensor(0.0)
                if self.workspace_visual_anchor is None
                else self.workspace_visual_anchor.detach().float().norm(dim=-1).mean()
            )
            self.debug_context.update(
                {
                    **workspace_visual_debug,
                    "workspace_visual_dynamic_write_enabled": hidden_states.new_tensor(
                        float(
                            self.visual_dynamic_write
                            or self.direct_visual_z_tokens
                        )
                    ),
                    "workspace_direct_visual_z_enabled": hidden_states.new_tensor(
                        float(self.direct_visual_z_tokens)
                    ),
                    "workspace_static_visual_write_enabled": hidden_states.new_tensor(
                        1.0
                    ),
                    "workspace_private_visual_prompt_norm_mean": private_norm,
                    "workspace_unified_static_visual_prompt_norm_mean": (
                        unified_static_norm
                    ),
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
            self._capture_layer_debug(layer_index)
            if not self._forward_audited:
                print(
                    "[DIRECTIONAL_CONCAT_WORKSPACE_FORWARD_AUDIT] "
                    f"layer={layer_index} units={len(lengths)} "
                    f"private_tokens={self.private_prompt_tokens} "
                    f"unified_static_tokens={self.static_visual_prompt_tokens} "
                    f"workspace_tokens={self.workspace_tokens} "
                    f"direct_z_tokens="
                    f"{self.workspace_tokens if self.direct_visual_z_tokens else 0} "
                    f"visual_dynamic_write={self.visual_dynamic_write} "
                    f"direct_visual_z_tokens={self.direct_visual_z_tokens} "
                    "static_visual_write=True insert_before_block=True "
                    "strip_after_block=True pass=True"
                )
                self._forward_audited = True
        return adapted

    def _capture_layer_debug(self, layer_index: int) -> None:
        current = dict(self.debug_context)
        self._layer_debug_context.update(
            {
                f"{key}_layer{layer_index}": value
                for key, value in current.items()
                if not key.rsplit("_layer", 1)[-1].isdigit()
            }
        )
        self.debug_context.update(self._layer_debug_context)

    def workspace_inference_intervention_summary(self) -> Dict[str, Any]:
        return {
            "mode": "directional_concat",
            "anchor_layers": list(self.anchor_layers),
            "static_visual_write": self.static_visual_write,
            "unified_static_visual_prompt": self.unified_static_visual_prompt,
            "unified_static_visual_rng_control": (
                self.unified_static_visual_prompt
            ),
            "direct_visual_z_tokens": self.direct_visual_z_tokens,
            "query_source": self.query_source,
            "visual_delta_scale": self._visual_delta_scale,
            "visual_dynamic_write": self.visual_dynamic_write,
            "visual_memory_mode": self._visual_memory_mode,
            "visual_memory_images_seen": self._visual_memory_images_seen,
            "visual_memory_images_mismatched": (
                self._visual_memory_images_mismatched
            ),
            "visual_memory_images_natural": self._visual_memory_images_natural,
            "question_query_mode": self._question_query_mode,
            "question_query_units_seen": self._question_query_units_seen,
            "question_query_units_mismatched": (
                self._question_query_units_mismatched
            ),
            "question_query_units_natural": self._question_query_units_natural,
            "original_image_input_unchanged": True,
            "original_question_input_unchanged": True,
            "visual_memory_intervention_scope": "directional_ca_visual_kv_only",
            "question_query_intervention_scope": "directional_ca_question_q_only",
            "intervention_scope": self._conditioning_intervention_scope(),
            "anchors_preserved": True,
            "token_count_preserved": True,
        }
