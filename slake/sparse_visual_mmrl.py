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


class LatentResidualAttention(nn.Module):
    """Read a latent workspace without replacing a branch's private base."""

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        attention_dim: int,
        num_heads: int,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if min(query_dim, memory_dim, attention_dim, num_heads) < 1:
            raise ValueError("Latent attention dimensions must be positive")
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.query_dim = int(query_dim)
        self.memory_dim = int(memory_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.memory_norm = nn.LayerNorm(self.memory_dim)
        self.query_projection = nn.Linear(
            self.query_dim, self.attention_dim, bias=False
        )
        self.key_projection = nn.Linear(
            self.memory_dim, self.attention_dim, bias=False
        )
        self.value_projection = nn.Linear(
            self.memory_dim, self.attention_dim, bias=False
        )
        self.output_projection = nn.Linear(
            self.attention_dim, self.query_dim, bias=True
        )
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._forward_audited = False

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if queries.ndim != 3 or queries.shape[-1] != self.query_dim:
            raise ValueError(
                f"queries must have [B,Q,{self.query_dim}], got {tuple(queries.shape)}"
            )
        if memory.ndim != 3 or memory.shape[-1] != self.memory_dim:
            raise ValueError(
                f"memory must have [B,M,{self.memory_dim}], got {tuple(memory.shape)}"
            )
        if queries.shape[0] != memory.shape[0]:
            raise ValueError("Latent attention batch dimensions must match")

        dtype = self.query_projection.weight.dtype
        query = self._split_heads(
            self.query_projection(self.query_norm(queries.to(dtype=dtype)))
        )
        memory_f = self.memory_norm(memory.to(dtype=dtype))
        key = self._split_heads(self.key_projection(memory_f))
        value = self._split_heads(self.value_projection(memory_f))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        context = torch.matmul(attention, value)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(queries.shape[0], queries.shape[1], self.attention_dim)
        )
        delta = self.output_projection(context).to(dtype=queries.dtype)

        with torch.no_grad():
            attention_f = attention.detach().float().clamp_min(1e-8)
            entropy = -(attention_f * attention_f.log()).sum(dim=-1)
            if memory.shape[1] > 1:
                entropy = entropy / math.log(float(memory.shape[1]))
            query_norm = queries.detach().float().norm(dim=-1).mean()
            delta_norm = delta.detach().float().norm(dim=-1).mean()
            self.debug_context = {
                "attention_entropy_norm": entropy.mean(),
                "delta_norm_mean": delta_norm,
                "base_norm_mean": query_norm,
                "delta_to_base_ratio": delta_norm / query_norm.clamp_min(1e-8),
            }
            if not self._forward_audited:
                max_abs = float(delta.detach().float().abs().max().item())
                if max_abs != 0.0:
                    raise RuntimeError(
                        "Latent residual output must start at exact zero; "
                        f"max_abs_delta={max_abs}"
                    )
                self._forward_audited = True
        return delta


class FullTokenWorkspaceBlock(nn.Module):
    """Update shared slots from full visual and question-token memories."""

    def __init__(
        self,
        workspace_dim: int,
        num_heads: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        if min(workspace_dim, num_heads, ffn_dim) < 1:
            raise ValueError("Workspace block dimensions must be positive")
        if workspace_dim % num_heads != 0:
            raise ValueError("workspace_dim must be divisible by num_heads")
        self.workspace_dim = int(workspace_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.query_norm = nn.LayerNorm(self.workspace_dim)
        self.memory_norm = nn.LayerNorm(self.workspace_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.workspace_dim,
            self.num_heads,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(self.workspace_dim)
        self.self_attention = nn.MultiheadAttention(
            self.workspace_dim,
            self.num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.workspace_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.workspace_dim, self.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, self.workspace_dim),
        )
        self.debug_context: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        workspace: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        visual_width: int,
    ) -> torch.Tensor:
        if workspace.ndim != 3 or workspace.shape[-1] != self.workspace_dim:
            raise ValueError("Workspace slots have an invalid shape")
        if memory.ndim != 3 or memory.shape[-1] != self.workspace_dim:
            raise ValueError("Workspace memory has an invalid shape")
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError("Workspace memory mask has an invalid shape")
        if visual_width < 1 or visual_width >= memory.shape[1]:
            raise ValueError("Workspace memory must contain visual and text tokens")

        query = self.query_norm(workspace)
        normalized_memory = self.memory_norm(memory)
        cross, weights = self.cross_attention(
            query,
            normalized_memory,
            normalized_memory,
            key_padding_mask=~memory_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        workspace = workspace + cross
        self_input = self.self_norm(workspace)
        self_delta, _ = self.self_attention(
            self_input,
            self_input,
            self_input,
            need_weights=False,
        )
        workspace = workspace + self_delta
        workspace = workspace + self.ffn(self.ffn_norm(workspace))

        with torch.no_grad():
            probabilities = weights.detach().float().clamp_min(1e-8)
            valid = memory_mask[:, None, None, :]
            probabilities = probabilities * valid
            valid_counts = memory_mask.sum(dim=-1).clamp_min(2).float()
            entropy = -(
                probabilities * probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1)
            entropy = entropy / valid_counts.log()[:, None, None]
            self.debug_context = {
                "attention_entropy_norm": entropy.mean(),
                "visual_attention_mean": probabilities[..., :visual_width]
                .sum(dim=-1)
                .mean(),
                "text_attention_mean": probabilities[..., visual_width:]
                .sum(dim=-1)
                .mean(),
                "workspace_norm_mean": workspace.detach()
                .float()
                .norm(dim=-1)
                .mean(),
            }
        return workspace


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
        shared_workspace: bool = False,
        workspace_tokens: int = 32,
        workspace_dim: int = 1024,
        workspace_heads: int = 16,
        workspace_ffn_dim: int = 4096,
        workspace_visual_attention_dim: int = 1024,
        workspace_visual_heads: int = 16,
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
        if min(
            workspace_tokens,
            workspace_dim,
            workspace_heads,
            workspace_ffn_dim,
            workspace_visual_attention_dim,
            workspace_visual_heads,
        ) < 1:
            raise ValueError("Shared workspace dimensions must be positive")
        if shared_workspace and workspace_dim != visual_dim:
            raise ValueError(
                "The first full-capacity workspace keeps visual width unchanged: "
                f"workspace_dim={workspace_dim} visual_dim={visual_dim}"
            )
        if workspace_dim % workspace_heads != 0:
            raise ValueError("workspace_dim must be divisible by workspace_heads")
        if workspace_visual_attention_dim % workspace_visual_heads != 0:
            raise ValueError(
                "workspace_visual_attention_dim must be divisible by heads"
            )
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
        self.shared_workspace_enabled = bool(shared_workspace)
        self.workspace_tokens = int(workspace_tokens)
        self.workspace_dim = int(workspace_dim)
        self.workspace_heads = int(workspace_heads)
        self.workspace_ffn_dim = int(workspace_ffn_dim)
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
        if self.shared_workspace_enabled:
            if self.text_anchor_tokens or self.map_shared_s_to_text:
                raise ValueError(
                    "The full workspace requires independent private P and visual S"
                )
            self.workspace_seed = nn.Parameter(
                torch.empty(self.workspace_tokens, self.workspace_dim)
            )
            self.workspace_layer_embeddings = nn.Parameter(
                torch.empty(len(self.anchor_layers), 1, self.workspace_dim)
            )
            self.workspace_text_norm = nn.LayerNorm(self.text_dim)
            self.workspace_text_projection = nn.Linear(
                self.text_dim, self.workspace_dim, bias=True
            )
            self.workspace_visual_type = nn.Parameter(
                torch.empty(1, 1, self.workspace_dim)
            )
            self.workspace_text_type = nn.Parameter(
                torch.empty(1, 1, self.workspace_dim)
            )
            self.workspace_blocks = nn.ModuleList(
                FullTokenWorkspaceBlock(
                    self.workspace_dim,
                    self.workspace_heads,
                    self.workspace_ffn_dim,
                )
                for _ in self.anchor_layers
            )
            self.workspace_visual_attentions = nn.ModuleList(
                LatentResidualAttention(
                    query_dim=self.visual_dim,
                    memory_dim=self.workspace_dim,
                    attention_dim=workspace_visual_attention_dim,
                    num_heads=workspace_visual_heads,
                    zero_init_output=True,
                )
                for _ in self.anchor_layers
            )
            nn.init.normal_(self.workspace_seed, std=0.02)
            nn.init.normal_(self.workspace_layer_embeddings, std=0.02)
            nn.init.normal_(self.workspace_visual_type, std=0.02)
            nn.init.normal_(self.workspace_text_type, std=0.02)
        else:
            self.register_parameter("workspace_seed", None)
            self.register_parameter("workspace_layer_embeddings", None)
            self.workspace_text_norm = None
            self.workspace_text_projection = None
            self.register_parameter("workspace_visual_type", None)
            self.register_parameter("workspace_text_type", None)
            self.workspace_blocks = nn.ModuleList()
            self.workspace_visual_attentions = nn.ModuleList()

        self.active = False
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._text_memory: torch.Tensor | None = None
        self._text_tokens: torch.Tensor | None = None
        self._text_token_mask: torch.Tensor | None = None
        self._images_per_sample: torch.Tensor | None = None
        self._image_lengths: torch.Tensor | None = None
        self._image_to_sample: torch.Tensor | None = None
        self._read_only_shared_rep: torch.Tensor | None = None
        self._workspace_by_image: torch.Tensor | None = None
        self._workspace_visual_write_disabled_layers: frozenset[int] = frozenset()
        self._workspace_update_bypassed_layers: frozenset[int] = frozenset()
        self._workspace_layer_inputs: Dict[int, torch.Tensor] = {}
        self._workspace_layer_states: Dict[int, torch.Tensor] = {}
        self._workspace_visual_deltas: Dict[int, torch.Tensor] = {}
        self._text_anchor_debug: Dict[str, torch.Tensor] = {}
        self._last_shared_s_text_prompt: torch.Tensor | None = None
        self._debug_rows: list[Dict[str, torch.Tensor]] = []
        self._installed = False
        self._forward_audited = False
        object.__setattr__(self, "_shared_memory_merger_ref", None)
        self._shared_memory_merge_unit = 0

    def configure_workspace_inference_intervention(
        self,
        visual_write_disabled_layers: Sequence[int] = (),
        update_bypassed_layers: Sequence[int] = (),
    ) -> None:
        if self.active:
            raise RuntimeError("Cannot reconfigure an active Workspace intervention")
        disabled = frozenset(int(layer) for layer in visual_write_disabled_layers)
        bypassed = frozenset(int(layer) for layer in update_bypassed_layers)
        requested = disabled | bypassed
        if requested and not self.shared_workspace_enabled:
            raise ValueError("Workspace interventions require shared_workspace")
        invalid = requested.difference(self.anchor_layers)
        if invalid:
            raise ValueError(
                "Workspace intervention layers must be configured anchors: "
                f"invalid={sorted(invalid)} anchors={list(self.anchor_layers)}"
            )
        self._workspace_visual_write_disabled_layers = disabled
        self._workspace_update_bypassed_layers = bypassed
        print(
            "[SHARED_WORKSPACE_INTERVENTION] "
            f"visual_write_disabled={sorted(disabled)} "
            f"update_bypassed={sorted(bypassed)}"
        )

    def workspace_inference_intervention_summary(self) -> Dict[str, Any]:
        return {
            "visual_write_disabled_layers": sorted(
                self._workspace_visual_write_disabled_layers
            ),
            "update_bypassed_layers": sorted(
                self._workspace_update_bypassed_layers
            ),
        }

    def shared_workspace_parameters(self) -> list[nn.Parameter]:
        if not self.shared_workspace_enabled:
            return []
        return [
            parameter
            for name, parameter in self.named_parameters()
            if name.startswith("workspace_")
        ]

    def private_parameters(self) -> list[nn.Parameter]:
        workspace_ids = {
            id(parameter) for parameter in self.shared_workspace_parameters()
        }
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in workspace_ids
        ]

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
            f"text_anchor_bottleneck={self.text_anchor_bottleneck_dim if self.text_anchor_tokens else 0} "
            f"shared_workspace={self.shared_workspace_enabled} "
            f"workspace_tokens={self.workspace_tokens if self.shared_workspace_enabled else 0} "
            f"workspace_dim={self.workspace_dim if self.shared_workspace_enabled else 0}"
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
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
    ) -> Iterator[None]:
        if self.active:
            raise RuntimeError("Sparse Visual MMRL context is already active")
        self.active = True
        self._text_memory = text_memory
        self._images_per_sample = images_per_sample.to(dtype=torch.long)
        if self.shared_workspace_enabled:
            if text_tokens is None or text_token_mask is None:
                raise RuntimeError("Shared workspace requires full question tokens")
            if text_tokens.ndim != 3 or text_token_mask.shape != text_tokens.shape[:2]:
                raise ValueError("Shared workspace question tokens have invalid shapes")
            lengths = text_token_mask.to(dtype=torch.bool).sum(dim=1)
            if bool((lengths == 0).any()):
                raise RuntimeError("Shared workspace question memory is empty")
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
            self._text_tokens = compact
            self._text_token_mask = compact_mask
        else:
            self._text_tokens = None
            self._text_token_mask = None
        self._image_lengths = None
        self._image_to_sample = None
        self._workspace_by_image = None
        self._workspace_layer_inputs = {}
        self._workspace_layer_states = {}
        self._workspace_visual_deltas = {}
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
            self._text_tokens = None
            self._text_token_mask = None
            self._images_per_sample = None
            self._image_lengths = None
            self._image_to_sample = None
            self._read_only_shared_rep = None
            self._workspace_by_image = None
            self._workspace_layer_inputs = {}
            self._workspace_layer_states = {}
            self._workspace_visual_deltas = {}

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor,
        torch.Tensor,
    ]:
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
        return visual_memory, text_memory, lengths, sample_ids, image_ids

    def _update_shared_workspace(
        self,
        layer_position: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.shared_workspace_enabled:
            raise RuntimeError("Shared workspace is disabled")
        if (
            self._image_lengths is None
            or self._image_to_sample is None
            or self._text_tokens is None
            or self._text_token_mask is None
            or self.workspace_seed is None
            or self.workspace_layer_embeddings is None
            or self.workspace_text_norm is None
            or self.workspace_text_projection is None
            or self.workspace_visual_type is None
            or self.workspace_text_type is None
        ):
            raise RuntimeError("Shared workspace context is incomplete")

        image_segments = torch.split(
            hidden_states,
            self._image_lengths.detach().cpu().tolist(),
            dim=0,
        )
        visual_width = max(segment.shape[0] for segment in image_segments)
        workspace_dtype = next(
            self.workspace_blocks[layer_position].parameters()
        ).dtype
        visual_tokens = torch.zeros(
            len(image_segments),
            visual_width,
            self.visual_dim,
            dtype=workspace_dtype,
            device=hidden_states.device,
        )
        visual_mask = torch.zeros(
            len(image_segments),
            visual_width,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        for image_index, segment in enumerate(image_segments):
            visual_tokens[image_index, : segment.shape[0]] = segment.to(
                dtype=workspace_dtype
            )
            visual_mask[image_index, : segment.shape[0]] = True

        image_to_sample = self._image_to_sample.to(hidden_states.device)
        text_tokens = self._text_tokens.to(hidden_states.device).index_select(
            0, image_to_sample
        )
        text_mask = self._text_token_mask.to(hidden_states.device).index_select(
            0, image_to_sample
        )
        projected_text = self.workspace_text_projection(
            self.workspace_text_norm(text_tokens.to(dtype=workspace_dtype))
        )
        visual_tokens = visual_tokens + self.workspace_visual_type.to(
            device=hidden_states.device,
            dtype=workspace_dtype,
        )
        projected_text = projected_text + self.workspace_text_type.to(
            device=hidden_states.device,
            dtype=workspace_dtype,
        )
        memory = torch.cat((visual_tokens, projected_text), dim=1)
        memory_mask = torch.cat((visual_mask, text_mask), dim=1)

        if self._workspace_by_image is None:
            workspace = self.workspace_seed.to(
                device=hidden_states.device,
                dtype=workspace_dtype,
            ).unsqueeze(0).expand(len(image_segments), -1, -1)
        else:
            workspace = self._workspace_by_image.to(
                device=hidden_states.device,
                dtype=workspace_dtype,
            )
        workspace = workspace + self.workspace_layer_embeddings[layer_position].to(
            device=hidden_states.device,
            dtype=workspace_dtype,
        )
        layer_index = self.anchor_layers[layer_position]
        self._workspace_layer_inputs[layer_index] = workspace.detach()
        if layer_index in self._workspace_update_bypassed_layers:
            if self.training:
                raise RuntimeError("Workspace update bypass is inference-only")
            zero = workspace.new_tensor(0.0)
            self.workspace_blocks[layer_position].debug_context = {
                "attention_entropy_norm": zero,
                "visual_attention_mean": zero,
                "text_attention_mean": zero,
                "workspace_norm_mean": workspace.detach().float().norm(dim=-1).mean(),
            }
        else:
            workspace = self.workspace_blocks[layer_position](
                workspace,
                memory,
                memory_mask,
                visual_width,
            )
        self._workspace_layer_states[layer_index] = workspace.detach()
        self._workspace_by_image = workspace
        return workspace

    def shared_workspace_text_memory(self) -> torch.Tensor:
        """Return final image workspaces averaged into one bank per sample."""
        if not self.shared_workspace_enabled or self._workspace_by_image is None:
            raise RuntimeError("Final shared workspace is unavailable")
        if self._image_to_sample is None or self._images_per_sample is None:
            raise RuntimeError("Shared workspace sample mapping is unavailable")
        workspace = self._workspace_by_image
        sample_ids = self._image_to_sample.to(workspace.device)
        batch_size = int(self._images_per_sample.numel())
        combined = workspace.new_zeros(
            batch_size, self.workspace_tokens, self.workspace_dim
        )
        combined = combined.index_add(0, sample_ids, workspace)
        counts = self._images_per_sample.to(
            device=workspace.device,
            dtype=workspace.dtype,
        ).view(batch_size, 1, 1)
        return combined / counts.clamp_min(1.0)

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
        (
            visual_memory,
            text_memory,
            lengths,
            _sample_ids,
            image_ids,
        ) = self._segment_memories(hidden_states, cu_seqlens)
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
        workspace_debug = {}
        if self.shared_workspace_enabled:
            workspace_by_image = self._update_shared_workspace(
                layer_position,
                hidden_states,
            )
            workspace = workspace_by_image.index_select(
                0, image_ids.to(workspace_by_image.device)
            )
            shared_delta = self.workspace_visual_attentions[layer_position](
                reps,
                workspace,
            ).to(device=hidden_states.device, dtype=hidden_states.dtype)
            self._workspace_visual_deltas[layer_index] = shared_delta.detach()
            visual_write_disabled = (
                layer_index in self._workspace_visual_write_disabled_layers
            )
            if visual_write_disabled:
                if self.training:
                    raise RuntimeError("Workspace visual-write disable is inference-only")
                shared_delta = torch.zeros_like(shared_delta)
            reps = reps + shared_delta
            block_debug = self.workspace_blocks[layer_position].debug_context
            attention_debug = self.workspace_visual_attentions[
                layer_position
            ].debug_context
            workspace_debug = {
                "workspace_attention_entropy_norm": block_debug[
                    "attention_entropy_norm"
                ],
                "workspace_visual_attention_mean": block_debug[
                    "visual_attention_mean"
                ],
                "workspace_text_attention_mean": block_debug[
                    "text_attention_mean"
                ],
                "workspace_norm_mean": block_debug["workspace_norm_mean"],
                "workspace_visual_residual_entropy_norm": attention_debug[
                    "attention_entropy_norm"
                ],
                "workspace_visual_delta_norm_mean": attention_debug[
                    "delta_norm_mean"
                ],
                "workspace_visual_delta_to_base_ratio": attention_debug[
                    "delta_to_base_ratio"
                ] if not visual_write_disabled else reps.new_tensor(0.0),
                "workspace_visual_write_disabled": reps.new_tensor(
                    float(visual_write_disabled)
                ),
                "workspace_update_bypassed": reps.new_tensor(
                    float(layer_index in self._workspace_update_bypassed_layers)
                ),
            }
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
                    **workspace_debug,
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
            if self.shared_workspace_enabled:
                debug[f"workspace_layer{layer}_attention_entropy_norm"] = row[
                    "workspace_attention_entropy_norm"
                ]
                debug[f"workspace_layer{layer}_visual_attention_mean"] = row[
                    "workspace_visual_attention_mean"
                ]
                debug[f"workspace_layer{layer}_text_attention_mean"] = row[
                    "workspace_text_attention_mean"
                ]
                debug[f"workspace_layer{layer}_norm_mean"] = row[
                    "workspace_norm_mean"
                ]
                debug[f"workspace_layer{layer}_visual_residual_entropy_norm"] = row[
                    "workspace_visual_residual_entropy_norm"
                ]
                debug[f"workspace_layer{layer}_visual_delta_norm_mean"] = row[
                    "workspace_visual_delta_norm_mean"
                ]
                debug[f"workspace_layer{layer}_visual_delta_to_base_ratio"] = row[
                    "workspace_visual_delta_to_base_ratio"
                ]
                debug[f"workspace_layer{layer}_visual_write_disabled"] = row[
                    "workspace_visual_write_disabled"
                ]
                debug[f"workspace_layer{layer}_update_bypassed"] = row[
                    "workspace_update_bypassed"
                ]
        debug["sparse_visual_rep_to_input_norm_ratio_mean"] = torch.stack(
            [row["rep_to_input_norm_ratio"] for row in self._debug_rows]
        ).mean()
        debug["sparse_visual_output_to_input_norm_ratio_mean"] = torch.stack(
            [row["output_to_input_norm_ratio"] for row in self._debug_rows]
        ).mean()
        if self.shared_workspace_enabled:
            debug["workspace_visual_delta_to_base_ratio_mean"] = torch.stack(
                [
                    row["workspace_visual_delta_to_base_ratio"]
                    for row in self._debug_rows
                ]
            ).mean()
            for previous_layer, current_layer in zip(
                self.anchor_layers,
                self.anchor_layers[1:],
            ):
                previous_state = self._workspace_layer_states.get(previous_layer)
                current_state = self._workspace_layer_states.get(current_layer)
                previous_input = self._workspace_layer_inputs.get(previous_layer)
                current_input = self._workspace_layer_inputs.get(current_layer)
                previous_visual_delta = self._workspace_visual_deltas.get(
                    previous_layer
                )
                current_visual_delta = self._workspace_visual_deltas.get(current_layer)
                prefix = f"workspace_transition_{previous_layer}_to_{current_layer}"
                if previous_state is not None and current_state is not None:
                    debug[f"{prefix}_state_cosine_mean"] = F.cosine_similarity(
                        F.layer_norm(
                            previous_state.float(),
                            (previous_state.shape[-1],),
                        ),
                        F.layer_norm(
                            current_state.float(),
                            (current_state.shape[-1],),
                        ),
                        dim=-1,
                    ).mean()
                if (
                    previous_state is not None
                    and current_state is not None
                    and previous_input is not None
                    and current_input is not None
                ):
                    previous_update = previous_state.float() - previous_input.float()
                    current_update = current_state.float() - current_input.float()
                    debug[f"{prefix}_update_cosine_mean"] = F.cosine_similarity(
                        previous_update,
                        current_update,
                        dim=-1,
                    ).mean()
                if (
                    previous_visual_delta is not None
                    and current_visual_delta is not None
                    and previous_visual_delta.shape == current_visual_delta.shape
                ):
                    debug[f"{prefix}_visual_delta_cosine_mean"] = (
                        F.cosine_similarity(
                            previous_visual_delta.float(),
                            current_visual_delta.float(),
                            dim=-1,
                        ).mean()
                    )
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
