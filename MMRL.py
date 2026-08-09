import math
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _grouped_token_geometry(
    states: torch.Tensor,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Summarize token diversity independently inside each sample and layer."""
    if states.ndim != 4:
        raise ValueError(
            f"grouped token geometry expects [B,L,R,D], got {tuple(states.shape)}"
        )
    states = states.detach().float()
    with torch.no_grad(), torch.autocast(
        device_type=states.device.type,
        enabled=False,
    ):
        token_count = states.shape[2]
        if token_count < 2:
            return {}

        norms = states.norm(dim=-1)
        unit = states / norms.unsqueeze(-1).clamp_min(1e-8)
        cosine = torch.matmul(unit, unit.transpose(-2, -1))
        triangle = torch.triu_indices(
            token_count,
            token_count,
            offset=1,
            device=states.device,
        )
        pairwise = cosine[..., triangle[0], triangle[1]]

        def _rank_statistics(token_states: torch.Tensor):
            gram = torch.matmul(token_states, token_states.transpose(-2, -1))
            singular_values = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
            singular_sum = singular_values.sum(dim=-1)
            distribution = singular_values / singular_sum.unsqueeze(-1).clamp_min(1e-8)
            has_signal = (singular_sum > 1e-8).to(distribution.dtype)
            effective_rank = torch.exp(-(
                distribution * distribution.clamp_min(1e-8).log()
            ).sum(dim=-1)) * has_signal
            descending = singular_values.flip(-1)
            top1_ratio = descending[..., 0] / singular_sum.clamp_min(1e-8)
            top5_ratio = descending[..., :min(5, token_count)].sum(
                dim=-1
            ) / singular_sum.clamp_min(1e-8)
            return effective_rank, top1_ratio, top5_ratio

        effective_rank, top1_ratio, top5_ratio = _rank_statistics(states)
        centered_states = states - states.mean(dim=-2, keepdim=True)
        centered_effective_rank, _, _ = _rank_statistics(centered_states)
        effective_rank_fraction = effective_rank / float(token_count)
        centered_effective_rank_fraction = centered_effective_rank / float(
            max(token_count - 1, 1)
        )
        common_mode_ratio = states.mean(dim=-2).norm(dim=-1) / norms.mean(
            dim=-1
        ).clamp_min(1e-8)

        return {
            f"{prefix}_token_pair_cos_mean": pairwise.mean(),
            f"{prefix}_token_pair_cos_max": pairwise.max(),
            f"{prefix}_token_effective_rank_mean": effective_rank.mean(),
            f"{prefix}_token_effective_rank_min": effective_rank.min(),
            f"{prefix}_token_effective_rank_fraction_mean": (
                effective_rank_fraction.mean()
            ),
            f"{prefix}_token_effective_rank_fraction_min": (
                effective_rank_fraction.min()
            ),
            f"{prefix}_token_centered_effective_rank_mean": (
                centered_effective_rank.mean()
            ),
            f"{prefix}_token_centered_effective_rank_min": (
                centered_effective_rank.min()
            ),
            f"{prefix}_token_centered_effective_rank_fraction_mean": (
                centered_effective_rank_fraction.mean()
            ),
            f"{prefix}_token_centered_effective_rank_fraction_min": (
                centered_effective_rank_fraction.min()
            ),
            f"{prefix}_token_singular_top1_ratio_mean": top1_ratio.mean(),
            f"{prefix}_token_singular_top5_ratio_mean": top5_ratio.mean(),
            f"{prefix}_token_common_mode_ratio_mean": common_mode_ratio.mean(),
            f"{prefix}_token_common_mode_ratio_max": common_mode_ratio.max(),
        }


class MultiQueryAttentionPooling(nn.Module):
    """Compress a masked sequence into a fixed number of learned memory tokens."""

    POOLING_MODES = {"independent", "competitive"}

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        query_count: int,
        attention_dim: int,
        pooling_mode: str = "independent",
    ):
        super().__init__()
        if query_count < 1 or attention_dim < 1:
            raise ValueError(
                "query_count and attention_dim must be positive, got "
                f"query_count={query_count} attention_dim={attention_dim}"
            )
        self.query_count = int(query_count)
        self.attention_dim = int(attention_dim)
        self.pooling_mode = str(pooling_mode).strip().lower()
        if self.pooling_mode not in self.POOLING_MODES:
            raise ValueError(
                f"Unsupported memory pooling mode={self.pooling_mode!r}; "
                f"choices={sorted(self.POOLING_MODES)}"
            )
        self.queries = nn.Parameter(
            torch.empty(self.query_count, self.attention_dim)
        )
        nn.init.normal_(self.queries, std=0.02)
        self.key_projection = nn.Linear(input_dim, self.attention_dim)
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.debug_context = {}

    @staticmethod
    def _sample_slots(states: torch.Tensor, limit: int = 16) -> torch.Tensor:
        slot_count = states.shape[-2]
        if slot_count <= limit:
            return states
        indices = torch.linspace(
            0,
            slot_count - 1,
            steps=limit,
            device=states.device,
        ).round().long()
        return states.index_select(-2, indices)

    @staticmethod
    def _specificity_ratio(
        states: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = states.detach().float()
        if valid_mask is None:
            centered = states - states.mean(dim=-2, keepdim=True)
            centered_rms = centered.square().mean(dim=(-2, -1)).sqrt()
            full_rms = states.square().mean(dim=(-2, -1)).sqrt()
            return (centered_rms / full_rms.clamp_min(1e-8)).mean()

        mask = valid_mask.to(device=states.device, dtype=states.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (states * mask.unsqueeze(-1)).sum(dim=-2) / count
        centered = (states - mean.unsqueeze(-2)) * mask.unsqueeze(-1)
        centered_energy = centered.square().sum(dim=(-2, -1))
        full_energy = (states * mask.unsqueeze(-1)).square().sum(dim=(-2, -1))
        return (centered_energy / full_energy.clamp_min(1e-8)).sqrt().mean()

    @staticmethod
    def slot_diversity_loss(
        pooled: torch.Tensor,
        cosine_max: float,
    ) -> torch.Tensor:
        slot_count = pooled.shape[1]
        if slot_count < 2:
            return pooled.sum() * 0.0
        normalized = F.normalize(pooled.float(), dim=-1, eps=1e-8)
        cosine = torch.matmul(normalized, normalized.transpose(-2, -1))
        off_diagonal = ~torch.eye(
            slot_count,
            device=pooled.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        return F.relu(cosine - float(cosine_max))[off_diagonal.expand_as(
            cosine
        )].mean()

    def _update_debug_context(
        self,
        input_states: torch.Tensor,
        valid_mask: torch.Tensor,
        weights: torch.Tensor,
        pooled: torch.Tensor,
    ) -> None:
        if not self.training:
            self.debug_context = {}
            return
        with torch.no_grad():
            sampled_queries = self._sample_slots(self.queries).float()
            sampled_weights = self._sample_slots(weights.detach()).float()
            sampled_pooled = self._sample_slots(pooled.detach()).float()
            valid_count = valid_mask.sum(dim=-1).clamp_min(1).float()
            entropy = -(
                sampled_weights
                * sampled_weights.clamp_min(1e-8).log()
            ).sum(dim=-1)
            entropy_norm = entropy / valid_count.log().clamp_min(1.0).unsqueeze(1)
            stats = {
                "source_specificity_ratio": self._specificity_ratio(
                    input_states,
                    valid_mask,
                ),
                "attention_entropy_norm": entropy_norm.mean(),
                "output_specificity_ratio": self._specificity_ratio(
                    sampled_pooled,
                ),
            }
            stats.update(_grouped_token_geometry(
                sampled_queries[None, None],
                "query",
            ))
            stats.update(_grouped_token_geometry(
                sampled_weights[:, None],
                "attention",
            ))
            stats.update(_grouped_token_geometry(
                sampled_pooled[:, None],
                "output",
            ))
            self.debug_context = stats

    def forward(
        self,
        input_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_states.ndim != 3:
            raise ValueError(
                "attention pooling expects input_states=[B,S,D], got "
                f"{tuple(input_states.shape)}"
            )
        if valid_mask.shape != input_states.shape[:2]:
            raise ValueError(
                "attention pooling mask must match [B,S], got "
                f"states={tuple(input_states.shape)} mask={tuple(valid_mask.shape)}"
            )

        valid_mask = valid_mask.to(device=input_states.device, dtype=torch.bool)
        valid_per_sample = valid_mask.sum(dim=1)
        if bool((valid_per_sample == 0).any()):
            empty = (valid_per_sample == 0).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                "attention pooling received samples without valid tokens: "
                f"indices={empty}"
            )

        keys = self.key_projection(input_states)
        values = self.value_projection(input_states)
        queries = self.queries.to(device=keys.device, dtype=keys.dtype)
        scores = torch.einsum("qa,bsa->bqs", queries, keys)
        scores = scores.float() / math.sqrt(self.attention_dim)
        if self.pooling_mode == "independent":
            scores = scores.masked_fill(
                ~valid_mask[:, None, :],
                float("-inf"),
            )
            weights = torch.softmax(scores, dim=-1)
        else:
            assignment = torch.softmax(scores, dim=1)
            assignment = assignment.masked_fill(
                ~valid_mask[:, None, :],
                0.0,
            )
            weights = assignment / assignment.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        weights = weights.to(dtype=values.dtype)
        pooled = torch.einsum("bqs,bsd->bqd", weights, values)
        pooled = self.output_norm(pooled)
        self._update_debug_context(
            input_states,
            valid_mask,
            weights,
            pooled,
        )
        return pooled


class ZeroInitProjection(nn.Module):
    """A trainable projection that remains zero through HF post_init()."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class ResidualCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "cross-attention hidden_dim must be divisible by num_heads, got "
                f"hidden_dim={hidden_dim} num_heads={num_heads}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_projection = ZeroInitProjection(self.hidden_dim)
        self.debug_context = {}

    def _split_heads(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = states.shape
        return states.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_split_index: int | None = None,
        query_group_shape: tuple[int, int] | None = None,
        memory_logit_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if queries.ndim != 3 or memory.ndim != 3:
            raise ValueError(
                "cross-attention expects [B,S,D] tensors, got "
                f"queries={tuple(queries.shape)} memory={tuple(memory.shape)}"
            )
        if queries.shape[0] != memory.shape[0]:
            raise ValueError(
                "cross-attention batch mismatch: "
                f"queries={queries.shape[0]} memory={memory.shape[0]}"
            )

        query_states = self._split_heads(
            self.query_projection(self.query_norm(queries))
        )
        normalized_memory = self.memory_norm(memory)
        key_states = self._split_heads(self.key_projection(normalized_memory))
        value_states = self._split_heads(self.value_projection(normalized_memory))
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        scores = scores.float() / math.sqrt(self.head_dim)
        if memory_logit_bias is not None:
            expected_bias_shape = memory.shape[:2]
            if tuple(memory_logit_bias.shape) != tuple(expected_bias_shape):
                raise ValueError(
                    "memory_logit_bias must match [B,M], got "
                    f"bias={tuple(memory_logit_bias.shape)} "
                    f"memory={tuple(memory.shape)}"
                )
            scores = scores + memory_logit_bias.to(
                device=scores.device,
                dtype=scores.dtype,
            )[:, None, None, :]
        weights = torch.softmax(scores, dim=-1).to(dtype=value_states.dtype)
        if self.training:
            with torch.no_grad():
                full_weights = weights.detach().float()
                query_stride = max(full_weights.shape[-2] // 40, 1)
                sampled = full_weights[..., ::query_stride, :]
                entropy = -(
                    sampled * sampled.clamp_min(1e-8).log()
                ).sum(dim=-1)
                entropy_norm = entropy / math.log(float(sampled.shape[-1]))
                stats = {
                    "cross_attention_entropy_norm": entropy_norm.mean(),
                    "cross_attention_peak_mean": sampled.max(dim=-1).values.mean(),
                }
                if memory_split_index is not None:
                    split = int(memory_split_index)
                    if not 0 < split < sampled.shape[-1]:
                        raise ValueError(
                            "memory_split_index must divide visual and text memory, got "
                            f"split={split} memory_tokens={sampled.shape[-1]}"
                        )
                    visual_mass = sampled[..., :split].sum(dim=-1)
                    text_mass = sampled[..., split:].sum(dim=-1)
                    stats["cross_attention_visual_mass_mean"] = visual_mass.mean()
                    stats["cross_attention_text_mass_mean"] = text_mass.mean()
                    if queries.shape[1] > 0:
                        full_visual_mass = full_weights[..., :split].sum(
                            dim=-1
                        )
                        stats["cross_attention_visual_mass_query_std"] = (
                            full_visual_mass.mean(dim=(0, 1)).std(unbiased=False)
                        )
                if query_group_shape is not None:
                    group_count, group_size = map(int, query_group_shape)
                    if group_count * group_size != queries.shape[1]:
                        raise ValueError(
                            "query_group_shape does not match Cross-Attention "
                            f"queries: groups={query_group_shape} "
                            f"query_count={queries.shape[1]}"
                        )
                    grouped_attention = full_weights.mean(dim=1).reshape(
                        queries.shape[0],
                        group_count,
                        group_size,
                        full_weights.shape[-1],
                    )
                    stats.update(_grouped_token_geometry(
                        grouped_attention,
                        "cross_attention_map",
                    ))
                self.debug_context = stats
        else:
            self.debug_context = {}
        context = torch.matmul(weights, value_states)
        context = context.transpose(1, 2).contiguous().view(
            queries.shape[0],
            queries.shape[1],
            self.hidden_dim,
        )
        delta = self.output_projection(context)
        return queries + delta, delta


class MMRL(nn.Module):
    QUERY_ARCHITECTURES = {
        "layer_mlp_post_cross",
        "layer_linear_post_cross",
        "lowdim_cross_layer_linear",
        "shared_direct_post_cross",
    }
    REP_UPDATE_MODES = {"replace", "persistent_delta"}
    INFERENCE_MEMORY_COLLAPSE_MODES = {"none", "visual", "text", "both"}

    def __init__(self, config):
        super().__init__()
        cfg = SimpleNamespace(**config.mmrl_config)
        self.query_architecture = str(cfg.MMRL_QUERY_ARCHITECTURE)
        self.rep_update_mode = str(cfg.MMRL_REP_UPDATE_MODE)
        if self.query_architecture not in self.QUERY_ARCHITECTURES:
            raise ValueError(
                "Unsupported MMRL_QUERY_ARCHITECTURE="
                f"{self.query_architecture!r}; choices={sorted(self.QUERY_ARCHITECTURES)}"
            )
        if self.rep_update_mode not in self.REP_UPDATE_MODES:
            raise ValueError(
                "Unsupported MMRL_REP_UPDATE_MODE="
                f"{self.rep_update_mode!r}; choices={sorted(self.REP_UPDATE_MODES)}"
            )
        if (
            self.rep_update_mode == "persistent_delta"
            and self.query_architecture != "layer_mlp_post_cross"
        ):
            raise ValueError(
                "persistent_delta currently requires layer_mlp_post_cross"
            )
        self.use_direct_learnable_rep = bool(
            getattr(cfg, "ABLATE_DIRECT_LEARNABLE_REP", False)
        )
        self.use_direct_shared_rep = (
            self.query_architecture == "shared_direct_post_cross"
        )
        if self.use_direct_learnable_rep and self.use_direct_shared_rep:
            raise ValueError(
                "shared_direct_post_cross and ABLATE_DIRECT_LEARNABLE_REP "
                "are mutually exclusive"
            )
        self.insert_layer_count = len(cfg.INSERT_LAYER)
        self.rp_space_length = int(cfg.RP_SPACE_LENGTH)
        self.rp_space_dim = int(cfg.RP_SPACE_DIM)
        self.vision_token_dim = int(cfg.vision_token_dim)
        self.text_token_dim = int(cfg.text_token_dim)
        self.memory_query_count = int(cfg.MMRL_MEMORY_QUERY_COUNT)
        self.memory_attention_dim = int(cfg.MMRL_MEMORY_ATTENTION_DIM)
        self.memory_pooling_mode = str(
            getattr(cfg, "MMRL_MEMORY_POOLING_MODE", "independent")
        ).strip().lower()
        self.memory_slot_diversity_weight = float(
            getattr(cfg, "MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT", 0.0)
        )
        self.memory_slot_cosine_max = float(
            getattr(cfg, "MMRL_MEMORY_SLOT_COSINE_MAX", 0.995)
        )
        if self.memory_pooling_mode not in MultiQueryAttentionPooling.POOLING_MODES:
            raise ValueError(
                "Unsupported MMRL_MEMORY_POOLING_MODE="
                f"{self.memory_pooling_mode!r}; choices="
                f"{sorted(MultiQueryAttentionPooling.POOLING_MODES)}"
            )
        if self.memory_slot_diversity_weight < 0.0:
            raise ValueError(
                "MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT must be non-negative, got "
                f"{self.memory_slot_diversity_weight}"
            )
        if not -1.0 <= self.memory_slot_cosine_max <= 1.0:
            raise ValueError(
                "MMRL_MEMORY_SLOT_COSINE_MAX must be in [-1, 1], got "
                f"{self.memory_slot_cosine_max}"
            )
        self.projector_hidden_dim = int(cfg.MMRL_PROJECTOR_HIDDEN_DIM)
        self.cross_attention_dim = (
            self.rp_space_dim
            if self.query_architecture == "lowdim_cross_layer_linear"
            else self.vision_token_dim
        )
        self.layer_lora_rank = int(getattr(cfg, "MMRL_LAYER_LORA_RANK", 0))
        if self.layer_lora_rank < 0:
            raise ValueError(
                f"MMRL_LAYER_LORA_RANK must be non-negative, got {self.layer_lora_rank}"
            )
        if self.layer_lora_rank > 0 and not self.use_direct_shared_rep:
            raise ValueError(
                "MMRL_LAYER_LORA_RANK requires "
                "MMRL_QUERY_ARCHITECTURE='shared_direct_post_cross'"
            )
        if (
            self.use_direct_learnable_rep
            and self.query_architecture == "lowdim_cross_layer_linear"
        ):
            raise ValueError(
                "ABLATE_DIRECT_LEARNABLE_REP is incompatible with "
                "lowdim_cross_layer_linear"
            )

        positive_dimensions = {
            "insert_layer_count": self.insert_layer_count,
            "rp_space_length": self.rp_space_length,
            "rp_space_dim": self.rp_space_dim,
            "vision_token_dim": self.vision_token_dim,
            "text_token_dim": self.text_token_dim,
            "memory_query_count": self.memory_query_count,
            "memory_attention_dim": self.memory_attention_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
            "cross_attention_dim": self.cross_attention_dim,
        }
        invalid_dimensions = {
            name: value
            for name, value in positive_dimensions.items()
            if value < 1
        }
        if invalid_dimensions:
            raise ValueError(
                f"MMRL dimensions must be positive: {invalid_dimensions}"
            )

        shared_rep_dim = (
            self.vision_token_dim
            if self.use_direct_shared_rep
            else self.rp_space_dim
        )
        self.shared_represent_space = nn.Parameter(
            torch.empty(self.rp_space_length, shared_rep_dim)
        )
        nn.init.normal_(self.shared_represent_space, std=0.02)

        if self.use_direct_learnable_rep:
            self.v_r_token_projector = nn.ModuleList()
            self.direct_v_tokens = nn.ParameterList([
                nn.Parameter(
                    torch.empty(self.rp_space_length, self.vision_token_dim)
                )
                for _ in range(self.insert_layer_count)
            ])
            for parameter in self.direct_v_tokens:
                nn.init.normal_(parameter, std=0.02)
        elif self.use_direct_shared_rep:
            self.v_r_token_projector = nn.ModuleList()
            self.direct_v_tokens = nn.ParameterList()
        elif self.query_architecture == "layer_mlp_post_cross":
            self.v_r_token_projector = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.rp_space_dim, self.projector_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.projector_hidden_dim, self.vision_token_dim),
                )
                for _ in range(self.insert_layer_count)
            ])
            self.direct_v_tokens = nn.ParameterList()
        else:
            self.v_r_token_projector = nn.ModuleList([
                nn.Linear(self.rp_space_dim, self.vision_token_dim)
                for _ in range(self.insert_layer_count)
            ])
            self.direct_v_tokens = nn.ParameterList()

        self.layer_embeddings = nn.Parameter(
            torch.empty(self.insert_layer_count, 1, self.cross_attention_dim)
        )
        nn.init.normal_(self.layer_embeddings, std=0.02)

        if self.layer_lora_rank > 0:
            self.layer_lora_A = nn.Parameter(torch.empty(
                self.insert_layer_count,
                self.vision_token_dim,
                self.layer_lora_rank,
            ))
            self.layer_lora_B = nn.Parameter(torch.zeros(
                self.insert_layer_count,
                self.layer_lora_rank,
                self.vision_token_dim,
            ))
            for layer_matrix in self.layer_lora_A:
                nn.init.kaiming_uniform_(layer_matrix, a=math.sqrt(5))
        else:
            self.register_parameter("layer_lora_A", None)
            self.register_parameter("layer_lora_B", None)

        self.visual_memory_pooling = MultiQueryAttentionPooling(
            input_dim=self.vision_token_dim,
            output_dim=self.cross_attention_dim,
            query_count=self.memory_query_count,
            attention_dim=self.memory_attention_dim,
            pooling_mode=self.memory_pooling_mode,
        )
        self.text_memory_pooling = MultiQueryAttentionPooling(
            input_dim=self.text_token_dim,
            output_dim=self.cross_attention_dim,
            query_count=self.memory_query_count,
            attention_dim=self.memory_attention_dim,
            pooling_mode=self.memory_pooling_mode,
        )
        self.cross_attention = ResidualCrossAttention(
            hidden_dim=self.cross_attention_dim,
            num_heads=int(cfg.MMRL_CROSS_ATTENTION_HEADS),
        )

        self.cached_base_queries = None
        self.debug_context = {}
        self.last_rep_shape = None
        self.last_memory_shape = None
        self._printed_shape_audit = False
        self.inference_memory_collapse_mode = "none"
        self._printed_memory_collapse_audit = False
        self.low_rank_debug_context = {}
        self.memory_slot_diversity_loss = torch.tensor(0.0)

    def set_inference_memory_collapse_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        if mode not in self.INFERENCE_MEMORY_COLLAPSE_MODES:
            raise ValueError(
                "Unsupported inference memory collapse mode "
                f"{mode!r}; choices={sorted(self.INFERENCE_MEMORY_COLLAPSE_MODES)}"
            )
        self.inference_memory_collapse_mode = mode
        self._printed_memory_collapse_audit = False
        print(f"[MMRL_INFERENCE_MEMORY_COLLAPSE] mode={mode}")

    def _apply_inference_memory_collapse(
        self,
        visual_memory: torch.Tensor,
        text_memory: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        mode = self.inference_memory_collapse_mode
        if mode == "none":
            return visual_memory, text_memory, None, None
        if self.training:
            raise RuntimeError(
                "Memory collapse is an inference-only ablation and cannot run "
                "while MMRL is in training mode"
            )

        original_visual_shape = tuple(visual_memory.shape)
        original_text_shape = tuple(text_memory.shape)
        visual_logit_bias = torch.zeros(
            visual_memory.shape[:2],
            device=visual_memory.device,
            dtype=torch.float32,
        )
        text_logit_bias = torch.zeros(
            text_memory.shape[:2],
            device=text_memory.device,
            dtype=torch.float32,
        )
        if mode in {"visual", "both"}:
            visual_count = visual_memory.shape[1]
            visual_memory = visual_memory.mean(dim=1, keepdim=True)
            visual_logit_bias = torch.full(
                visual_memory.shape[:2],
                math.log(float(visual_count)),
                device=visual_memory.device,
                dtype=torch.float32,
            )
        if mode in {"text", "both"}:
            text_count = text_memory.shape[1]
            text_memory = text_memory.mean(dim=1, keepdim=True)
            text_logit_bias = torch.full(
                text_memory.shape[:2],
                math.log(float(text_count)),
                device=text_memory.device,
                dtype=torch.float32,
            )
        if not self._printed_memory_collapse_audit:
            print("[MMRL_INFERENCE_MEMORY_COLLAPSE_AUDIT]")
            print(f"  mode={mode}")
            print(
                f"  visual_memory={original_visual_shape}"
                f"->{tuple(visual_memory.shape)}"
            )
            print(
                f"  text_memory={original_text_shape}"
                f"->{tuple(text_memory.shape)}"
            )
            print("  multiplicity_logit_correction=True")
            self._printed_memory_collapse_audit = True
        return (
            visual_memory,
            text_memory,
            visual_logit_bias,
            text_logit_bias,
        )

    @staticmethod
    def _layer_geometry_metrics(
        states: torch.Tensor,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        states = states.detach().float()
        with torch.no_grad(), torch.autocast(
            device_type=states.device.type,
            enabled=False,
        ):
            if states.ndim == 4:
                flat = states.permute(1, 0, 2, 3).reshape(states.shape[1], -1)
            elif states.ndim == 3:
                flat = states.reshape(states.shape[0], -1)
            else:
                raise ValueError(
                    f"layer geometry expects [L,R,D] or [B,L,R,D], got {tuple(states.shape)}"
                )

            norms = flat.norm(dim=-1)
            unit = flat / norms.unsqueeze(-1).clamp_min(1e-8)
            cosine = unit @ unit.transpose(0, 1)
            triangle = torch.triu_indices(
                cosine.shape[0],
                cosine.shape[1],
                offset=1,
                device=cosine.device,
            )
            pairwise = cosine[triangle[0], triangle[1]]

            gram = flat @ flat.transpose(0, 1)
            singular_values = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
            singular_sum = singular_values.sum()
            safe_sum = singular_sum.clamp_min(1e-8)
            distribution = singular_values / safe_sum
            has_signal = (singular_sum > 1e-8).to(distribution.dtype)
            effective_rank = torch.exp(-(
                distribution * distribution.clamp_min(1e-8).log()
            ).sum()) * has_signal
            descending = singular_values.flip(0)
            top1_ratio = descending[0] / safe_sum * has_signal
            top2_ratio = descending[:2].sum() / safe_sum * has_signal

            return {
                f"{prefix}_layer_cos_mean": pairwise.mean(),
                f"{prefix}_layer_cos_max": pairwise.max(),
                f"{prefix}_effective_rank": effective_rank,
                f"{prefix}_singular_top1_ratio": top1_ratio,
                f"{prefix}_singular_top2_ratio": top2_ratio,
            }

    def _compute_base_queries(self) -> torch.Tensor:
        if self.use_direct_learnable_rep:
            projected = torch.stack(list(self.direct_v_tokens), dim=0)
        elif self.use_direct_shared_rep:
            projected = self.shared_represent_space.unsqueeze(0).expand(
                self.insert_layer_count,
                -1,
                -1,
            )
            if self.layer_lora_rank > 0:
                low_rank_hidden = torch.einsum(
                    "td,ldr->ltr",
                    self.shared_represent_space,
                    self.layer_lora_A,
                )
                low_rank_delta = torch.einsum(
                    "ltr,lrd->ltd",
                    low_rank_hidden,
                    self.layer_lora_B,
                )
                projected = projected + low_rank_delta
                if self.training:
                    with torch.no_grad():
                        base_norm = self.shared_represent_space.detach().float().norm(
                            dim=-1
                        ).mean()
                        delta_norm = low_rank_delta.detach().float().norm(
                            dim=-1
                        ).mean()
                        self.low_rank_debug_context = {
                            "low_rank_delta_norm_mean": delta_norm,
                            "low_rank_delta_to_base_ratio": (
                                delta_norm / base_norm.clamp_min(1e-8)
                            ),
                            **self._layer_geometry_metrics(
                                low_rank_delta,
                                "low_rank_delta",
                            ),
                        }
                else:
                    self.low_rank_debug_context = {}
            else:
                self.low_rank_debug_context = {}
        elif self.query_architecture == "lowdim_cross_layer_linear":
            projected = self.shared_represent_space.unsqueeze(0).expand(
                self.insert_layer_count,
                -1,
                -1,
            )
            self.low_rank_debug_context = {}
        else:
            projected = torch.stack([
                projector(self.shared_represent_space)
                for projector in self.v_r_token_projector
            ], dim=0)
        return projected + self.layer_embeddings

    def _project_lowdim_layers(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 3 and states.shape[0] == self.insert_layer_count:
            return torch.stack([
                projector(states[layer_index])
                for layer_index, projector in enumerate(
                    self.v_r_token_projector
                )
            ], dim=0)
        if states.ndim == 4 and states.shape[1] == self.insert_layer_count:
            return torch.stack([
                projector(states[:, layer_index])
                for layer_index, projector in enumerate(
                    self.v_r_token_projector
                )
            ], dim=1)
        raise ValueError(
            "low-dimensional layer projection expects [L,R,D] or "
            "[B,L,R,D], got "
            f"{tuple(states.shape)}"
        )

    def _get_base_queries(self) -> torch.Tensor:
        if self.training:
            self.cached_base_queries = None
            return self._compute_base_queries()
        if self.cached_base_queries is None:
            with torch.no_grad():
                self.cached_base_queries = self._compute_base_queries().detach()
        return self.cached_base_queries

    def _pack_visual_states(
        self,
        visual_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if lengths.numel() == 0 or bool((lengths <= 0).any()):
            raise RuntimeError(
                f"invalid visual sequence lengths: {lengths.tolist()}"
            )
        sequences = torch.split(visual_states, lengths.cpu().tolist(), dim=0)
        padded = pad_sequence(sequences, batch_first=True)
        positions = torch.arange(
            padded.shape[1],
            device=visual_states.device,
        )
        valid_mask = positions.unsqueeze(0) < lengths.to(
            device=visual_states.device,
        ).unsqueeze(1)
        return padded, valid_mask

    def forward(
        self,
        visual_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        images_per_sample: Sequence[int],
    ) -> list[torch.Tensor]:
        visual_padded, visual_mask = self._pack_visual_states(
            visual_states,
            cu_seqlens,
        )
        visual_memory = self.visual_memory_pooling(
            visual_padded,
            visual_mask,
        )
        text_memory = self.text_memory_pooling(text_states, text_mask)
        should_measure_slot_diversity = (
            self.training
            and (
                self.memory_query_count <= 32
                or self.memory_slot_diversity_weight > 0.0
            )
        )
        if should_measure_slot_diversity:
            visual_slot_loss = MultiQueryAttentionPooling.slot_diversity_loss(
                visual_memory,
                self.memory_slot_cosine_max,
            )
            text_slot_loss = MultiQueryAttentionPooling.slot_diversity_loss(
                text_memory,
                self.memory_slot_cosine_max,
            )
            self.memory_slot_diversity_loss = 0.5 * (
                visual_slot_loss + text_slot_loss
            )
        else:
            self.memory_slot_diversity_loss = visual_memory.sum() * 0.0
        (
            visual_memory,
            text_memory,
            visual_logit_bias,
            text_logit_bias,
        ) = self._apply_inference_memory_collapse(visual_memory, text_memory)

        image_counts = torch.as_tensor(
            images_per_sample,
            device=text_states.device,
            dtype=torch.long,
        )
        image_count = visual_memory.shape[0]
        if image_counts.numel() != text_memory.shape[0]:
            raise RuntimeError(
                "images_per_sample must match the text batch, got "
                f"counts={image_counts.numel()} text_batch={text_memory.shape[0]}"
            )
        if bool((image_counts < 0).any()) or int(image_counts.sum().item()) != image_count:
            raise RuntimeError(
                "images_per_sample does not match packed visual sequences: "
                f"counts={image_counts.tolist()} images={image_count}"
            )
        expanded_text_memory = torch.repeat_interleave(
            text_memory,
            image_counts,
            dim=0,
        )
        memory = torch.cat([visual_memory, expanded_text_memory], dim=1)
        memory_logit_bias = None
        if visual_logit_bias is not None and text_logit_bias is not None:
            expanded_text_logit_bias = torch.repeat_interleave(
                text_logit_bias,
                image_counts,
                dim=0,
            )
            memory_logit_bias = torch.cat(
                [visual_logit_bias, expanded_text_logit_bias],
                dim=1,
            )

        base_queries = self._get_base_queries().to(
            device=visual_states.device,
            dtype=visual_states.dtype,
        )
        flat_queries = base_queries.reshape(
            self.insert_layer_count * self.rp_space_length,
            self.cross_attention_dim,
        ).unsqueeze(0).expand(image_count, -1, -1)
        dynamic_cross_queries, cross_delta = self.cross_attention(
            flat_queries,
            memory,
            memory_split_index=visual_memory.shape[1],
            query_group_shape=(
                self.insert_layer_count,
                self.rp_space_length,
            ),
            memory_logit_bias=memory_logit_bias,
        )
        dynamic_cross_queries = dynamic_cross_queries.reshape(
            image_count,
            self.insert_layer_count,
            self.rp_space_length,
            self.cross_attention_dim,
        )
        cross_delta = cross_delta.reshape(
            image_count,
            self.insert_layer_count,
            self.rp_space_length,
            self.cross_attention_dim,
        )
        expanded_base_queries = base_queries.unsqueeze(0).expand(
            image_count,
            -1,
            -1,
            -1,
        )
        if self.query_architecture == "lowdim_cross_layer_linear":
            projected_base_queries = self._project_lowdim_layers(
                base_queries
            ).unsqueeze(0).expand(image_count, -1, -1, -1)
            output_cross_delta = torch.stack([
                F.linear(
                    cross_delta[:, layer_index],
                    projector.weight,
                    bias=None,
                )
                for layer_index, projector in enumerate(
                    self.v_r_token_projector
                )
            ], dim=1)
            dynamic_queries = projected_base_queries + output_cross_delta
        else:
            dynamic_queries = dynamic_cross_queries
            projected_base_queries = expanded_base_queries
            output_cross_delta = cross_delta

        if self.rep_update_mode == "persistent_delta":
            insertion_queries = torch.cat(
                [dynamic_queries[:, :1], output_cross_delta[:, 1:]],
                dim=1,
            )
        else:
            insertion_queries = dynamic_queries

        self.last_memory_shape = tuple(memory.shape)
        self.last_rep_shape = tuple(insertion_queries.shape)
        if not self._printed_shape_audit:
            print("[MMRL_DYNAMIC_REP_AUDIT]")
            print(f"  query_architecture={self.query_architecture}")
            print(f"  rep_update_mode={self.rep_update_mode}")
            print(f"  cross_attention_dim={self.cross_attention_dim}")
            print(f"  visual_memory_shape={tuple(visual_memory.shape)}")
            print(f"  text_memory_shape={tuple(text_memory.shape)}")
            print(f"  combined_memory_shape={self.last_memory_shape}")
            print(f"  dynamic_rep_shape={self.last_rep_shape}")
            print(
                "  cross_output_weight_norm="
                f"{float(self.cross_attention.output_projection.weight.detach().float().norm().item())}"
            )
            print(
                "  cross_output_bias_norm="
                f"{float(self.cross_attention.output_projection.bias.detach().float().norm().item())}"
            )
            self._printed_shape_audit = True

        if self.training:
            with torch.no_grad():
                base_norm = projected_base_queries.detach().float().norm(
                    dim=-1
                ).mean()
                delta_norm = output_cross_delta.detach().float().norm(
                    dim=-1
                ).mean()
                self.debug_context = {
                    "dynamic_rep_base_norm_mean": base_norm,
                    "dynamic_rep_cross_delta_norm_mean": delta_norm,
                    "dynamic_rep_cross_delta_ratio": delta_norm
                    / base_norm.clamp_min(1e-8),
                    "visual_memory_norm_mean": visual_memory.detach().float().norm(
                        dim=-1
                    ).mean(),
                    "text_memory_norm_mean": text_memory.detach().float().norm(
                        dim=-1
                    ).mean(),
                    "memory_slot_diversity_loss": (
                        self.memory_slot_diversity_loss.detach()
                    ),
                    **{
                        f"visual_pooling_{key}": value
                        for key, value in self.visual_memory_pooling.debug_context.items()
                    },
                    **{
                        f"text_pooling_{key}": value
                        for key, value in self.text_memory_pooling.debug_context.items()
                    },
                    **self.low_rank_debug_context,
                    **self.cross_attention.debug_context,
                    **self._layer_geometry_metrics(
                        dynamic_queries,
                        "dynamic_rep",
                    ),
                    **self._layer_geometry_metrics(
                        output_cross_delta,
                        "cross_delta",
                    ),
                    **_grouped_token_geometry(
                        dynamic_queries,
                        "dynamic_rep",
                    ),
                    **_grouped_token_geometry(
                        output_cross_delta,
                        "cross_delta",
                    ),
                }
        else:
            self.debug_context = {}

        return [
            insertion_queries[:, layer_index]
            for layer_index in range(self.insert_layer_count)
        ]
