import math
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class MultiQueryAttentionPooling(nn.Module):
    """Compress a masked sequence into learned memory tokens."""

    def __init__(self, input_dim, output_dim, query_count, attention_dim):
        super().__init__()
        if query_count < 1 or attention_dim < 1:
            raise ValueError(
                "query_count and attention_dim must be positive, got "
                f"query_count={query_count} attention_dim={attention_dim}"
            )
        self.query_count = int(query_count)
        self.attention_dim = int(attention_dim)
        self.queries = nn.Parameter(torch.empty(self.query_count, self.attention_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.key_projection = nn.Linear(input_dim, self.attention_dim)
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, input_states, valid_mask):
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
        scores = scores.masked_fill(~valid_mask[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(dtype=values.dtype)
        pooled = torch.einsum("bqs,bsd->bqd", weights, values)
        return self.output_norm(pooled)


class MaskedMeanPooling(nn.Module):
    """Compress a masked sequence into one projected memory token."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, input_states, valid_mask):
        if input_states.ndim != 3:
            raise ValueError(
                "mean pooling expects input_states=[B,S,D], got "
                f"{tuple(input_states.shape)}"
            )
        if valid_mask.shape != input_states.shape[:2]:
            raise ValueError(
                "mean pooling mask must match [B,S], got "
                f"states={tuple(input_states.shape)} mask={tuple(valid_mask.shape)}"
            )
        valid_mask = valid_mask.to(device=input_states.device, dtype=torch.bool)
        valid_per_sample = valid_mask.sum(dim=1)
        if bool((valid_per_sample == 0).any()):
            empty = (valid_per_sample == 0).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                "mean pooling received samples without valid tokens: "
                f"indices={empty}"
            )

        mask = valid_mask.unsqueeze(-1).to(dtype=input_states.dtype)
        pooled = (input_states * mask).sum(dim=1)
        pooled = pooled / valid_per_sample.unsqueeze(-1).to(input_states.dtype)
        projected = self.value_projection(pooled).unsqueeze(1)
        return self.output_norm(projected)


class TextGuidedVisualPooling(nn.Module):
    """Fuse text with visual evidence while using roles only for routing."""

    def __init__(
        self,
        text_dim,
        visual_dim,
        output_dim,
        slot_count,
        attention_dim,
    ):
        super().__init__()
        if slot_count < 1 or attention_dim < 1:
            raise ValueError(
                "slot_count and attention_dim must be positive, got "
                f"slot_count={slot_count} attention_dim={attention_dim}"
            )
        self.slot_count = int(slot_count)
        self.attention_dim = int(attention_dim)
        self.text_input_norm = nn.LayerNorm(text_dim)
        self.text_query_projection = nn.Linear(text_dim, output_dim)
        self.role_queries = nn.Parameter(
            torch.empty(self.slot_count, output_dim)
        )
        nn.init.normal_(self.role_queries, std=0.02)
        self.role_query_norm = nn.LayerNorm(
            output_dim,
            elementwise_affine=False,
        )
        self.text_query_norm = nn.LayerNorm(output_dim)
        self.query_projection = nn.Linear(output_dim, self.attention_dim)
        self.visual_input_norm = nn.LayerNorm(visual_dim)
        self.key_projection = nn.Linear(visual_dim, self.attention_dim)
        self.value_projection = nn.Linear(visual_dim, self.attention_dim)
        self.context_output_projection = nn.Linear(
            self.attention_dim,
            output_dim,
        )
        self.context_fusion_norm = nn.LayerNorm(
            output_dim,
            elementwise_affine=False,
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.debug_context = {}

    def forward(
        self,
        visual_states,
        visual_mask,
        text_states,
        text_mask,
        images_per_sample,
    ):
        if visual_states.ndim != 3 or text_states.ndim != 3:
            raise ValueError(
                "text-guided pooling expects [B,S,D] states, got "
                f"visual={tuple(visual_states.shape)} "
                f"text={tuple(text_states.shape)}"
            )
        if visual_mask.shape != visual_states.shape[:2]:
            raise ValueError(
                "visual mask must match visual [B,S], got "
                f"states={tuple(visual_states.shape)} "
                f"mask={tuple(visual_mask.shape)}"
            )
        if text_mask.shape != text_states.shape[:2]:
            raise ValueError(
                "text mask must match text [B,S], got "
                f"states={tuple(text_states.shape)} mask={tuple(text_mask.shape)}"
            )

        visual_mask = visual_mask.to(
            device=visual_states.device,
            dtype=torch.bool,
        )
        text_mask = text_mask.to(device=text_states.device, dtype=torch.bool)
        visual_valid = visual_mask.sum(dim=1)
        text_valid = text_mask.sum(dim=1)
        if bool((visual_valid == 0).any()) or bool((text_valid == 0).any()):
            raise RuntimeError(
                "text-guided pooling received an empty sequence: "
                f"visual_valid={visual_valid.tolist()} "
                f"text_valid={text_valid.tolist()}"
            )

        image_counts = torch.as_tensor(
            images_per_sample,
            device=text_states.device,
            dtype=torch.long,
        )
        if image_counts.numel() != text_states.shape[0]:
            raise RuntimeError(
                "images_per_sample must match text batch in text-guided pooling, "
                f"counts={image_counts.numel()} text_batch={text_states.shape[0]}"
            )
        if bool((image_counts < 0).any()) or int(image_counts.sum().item()) != visual_states.shape[0]:
            raise RuntimeError(
                "images_per_sample must match visual batch in text-guided pooling, "
                f"counts={image_counts.tolist()} visual_batch={visual_states.shape[0]}"
            )

        text_weights = text_mask.unsqueeze(-1).to(dtype=text_states.dtype)
        text_summary = (text_states * text_weights).sum(dim=1)
        text_summary = text_summary / text_valid.unsqueeze(-1).to(text_states.dtype)
        text_query = self.text_query_projection(
            self.text_input_norm(text_summary)
        )
        text_query = self.text_query_norm(text_query)
        text_query = torch.repeat_interleave(text_query, image_counts, dim=0)

        role_queries = self.role_query_norm(self.role_queries).to(
            device=text_query.device,
            dtype=text_query.dtype,
        )
        routing_query = (
            text_query.unsqueeze(1) + role_queries.unsqueeze(0)
        ) / math.sqrt(2.0)

        normalized_visual = self.visual_input_norm(visual_states)
        queries = self.query_projection(routing_query)
        keys = self.key_projection(normalized_visual)
        values = self.value_projection(normalized_visual)
        scores = torch.einsum("bqa,bsa->bqs", queries, keys)
        scores = scores.float() / math.sqrt(self.attention_dim)
        scores = scores.masked_fill(~visual_mask[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(dtype=values.dtype)
        context = torch.einsum("bqs,bsa->bqa", weights, values)
        context_output = self.context_output_projection(context)
        normalized_context = self.context_fusion_norm(context_output)
        text_value = text_query.unsqueeze(1).expand(
            -1,
            self.slot_count,
            -1,
        )
        pooled = self.output_norm(
            (text_value + normalized_context) / math.sqrt(2.0)
        )

        if self.training:
            with torch.no_grad():
                weights_float = weights.detach().float()
                entropy = -(
                    weights_float
                    * weights_float.clamp_min(1e-8).log()
                ).sum(dim=-1)
                max_entropy = visual_valid.detach().float().log().unsqueeze(1)
                entropy_norm = torch.where(
                    max_entropy > 0,
                    entropy / max_entropy.clamp_min(1e-8),
                    torch.zeros_like(entropy),
                )
                pooled_float = pooled.detach().float()
                query_norm = routing_query.detach().float().norm(dim=-1).mean()
                context_norm = context_output.detach().float().norm(dim=-1).mean()
                fusion_text = text_value.detach().float()
                fusion_context = normalized_context.detach().float()
                pooled_norm = pooled_float.norm(dim=-1).mean().clamp_min(1e-8)
                centered = pooled_float - pooled_float.mean(dim=1, keepdim=True)
                slot_specificity = centered.norm(dim=-1).mean() / pooled_norm

                pooled_unit = F.normalize(pooled_float, dim=-1, eps=1e-8)
                pairwise = pooled_unit @ pooled_unit.transpose(1, 2)
                if self.slot_count > 1:
                    pair_indices = torch.triu_indices(
                        self.slot_count,
                        self.slot_count,
                        offset=1,
                        device=pairwise.device,
                    )
                    pairwise_cos = pairwise[
                        :, pair_indices[0], pair_indices[1]
                    ].mean()
                else:
                    pairwise_cos = pairwise.new_tensor(1.0)

                self.debug_context = {
                    "text_guided_visual_attention_entropy_norm": (
                        entropy_norm.mean()
                    ),
                    "text_guided_visual_attention_peak_mean": (
                        weights_float.max(dim=-1).values.mean()
                    ),
                    "text_guided_visual_slot_specificity_ratio": slot_specificity,
                    "text_guided_visual_slot_pairwise_cos_mean": pairwise_cos,
                    "text_guided_visual_query_norm_mean": query_norm,
                    "text_guided_visual_context_norm_mean": context_norm,
                    "text_guided_visual_context_to_query_ratio": (
                        context_norm / query_norm.clamp_min(1e-8)
                    ),
                    "text_guided_visual_fusion_text_norm_mean": (
                        fusion_text.norm(dim=-1).mean()
                    ),
                    "text_guided_visual_fusion_context_norm_mean": (
                        fusion_context.norm(dim=-1).mean()
                    ),
                    "text_guided_visual_fusion_cos_mean": (
                        F.cosine_similarity(
                            fusion_text,
                            fusion_context,
                            dim=-1,
                            eps=1e-8,
                        ).mean()
                    ),
                }
        else:
            self.debug_context = {}

        return pooled


class ZeroInitProjection(nn.Module):
    """A trainable projection that remains zero through HF post_init()."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)


class ResidualCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
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

    def _split_heads(self, states):
        batch_size, sequence_length, _ = states.shape
        return states.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, queries, memory):
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
        weights = torch.softmax(scores, dim=-1).to(dtype=value_states.dtype)
        context = torch.matmul(weights, value_states)
        context = context.transpose(1, 2).contiguous().view(
            queries.shape[0], queries.shape[1], self.hidden_dim
        )
        delta = self.output_projection(context)
        return queries + delta, delta


class ResidualConcatMLPFusion(nn.Module):
    """Fuse each prompt with mean-pooled visual and text tokens without attention."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(3 * self.hidden_dim)
        self.input_projection = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.output_projection = ZeroInitProjection(self.hidden_dim)

    def forward(self, queries, memory):
        if queries.ndim != 3 or memory.ndim != 3:
            raise ValueError(
                "concat-MLP fusion expects [B,S,D] tensors, got "
                f"queries={tuple(queries.shape)} memory={tuple(memory.shape)}"
            )
        if queries.shape[0] != memory.shape[0]:
            raise ValueError(
                "concat-MLP fusion batch mismatch: "
                f"queries={queries.shape[0]} memory={memory.shape[0]}"
            )
        if queries.shape[-1] != self.hidden_dim or memory.shape[-1] != self.hidden_dim:
            raise ValueError(
                "concat-MLP fusion hidden dimension mismatch: "
                f"expected={self.hidden_dim} queries={queries.shape[-1]} "
                f"memory={memory.shape[-1]}"
            )
        if memory.shape[1] != 2:
            raise ValueError(
                "concat-MLP fusion requires exactly two mean-pooled memory tokens "
                f"(visual, text), got {memory.shape[1]}"
            )

        query_count = queries.shape[1]
        visual = memory[:, 0:1].expand(-1, query_count, -1)
        text = memory[:, 1:2].expand(-1, query_count, -1)
        fused = torch.cat((queries, visual, text), dim=-1)
        hidden = F.gelu(self.input_projection(self.input_norm(fused)))
        delta = self.output_projection(hidden)
        return queries + delta, delta


class MMRL(nn.Module):
    QUERY_ARCHITECTURES = {
        "layer_mlp_post_cross",
        "shared_direct_post_cross",
    }

    def __init__(self, config):
        super().__init__()
        cfg = SimpleNamespace(**config.mmrl_config)
        self.query_architecture = str(
            getattr(cfg, "MMRL_QUERY_ARCHITECTURE", "layer_mlp_post_cross")
        )
        if self.query_architecture not in self.QUERY_ARCHITECTURES:
            raise ValueError(
                "Unsupported MMRL_QUERY_ARCHITECTURE="
                f"{self.query_architecture!r}; choices="
                f"{sorted(self.QUERY_ARCHITECTURES)}"
            )
        self.same_init_layer_projectors = bool(
            getattr(cfg, "MMRL_SAME_INIT_LAYER_PROJECTORS", True)
        )
        self.insert_layer_count = len(cfg.INSERT_LAYER)
        self.rp_space_length = int(cfg.RP_SPACE_LENGTH)
        self.rp_space_dim = int(cfg.RP_SPACE_DIM)
        self.vision_token_dim = int(cfg.vision_token_dim)
        self.text_token_dim = int(cfg.text_token_dim)
        self.cross_attention_dim = self.vision_token_dim
        self.memory_query_count = int(cfg.MMRL_MEMORY_QUERY_COUNT)
        self.memory_attention_dim = int(cfg.MMRL_MEMORY_ATTENTION_DIM)
        self.projector_hidden_dim = int(cfg.MMRL_PROJECTOR_HIDDEN_DIM)
        self.use_dynamic_cross_attention = bool(
            getattr(cfg, "MMRL_USE_DYNAMIC_CROSS_ATTENTION", True)
        )
        self.memory_pooling_mode = str(
            getattr(cfg, "MMRL_MEMORY_POOLING_MODE", "multi_query")
        )
        if self.memory_pooling_mode not in {
            "multi_query",
            "mean",
            "text_guided",
        }:
            raise ValueError(
                "MMRL_MEMORY_POOLING_MODE must be 'multi_query', 'mean', or "
                "'text_guided', got "
                f"{self.memory_pooling_mode!r}"
            )
        self.fusion_mode = str(
            getattr(cfg, "MMRL_FUSION_MODE", "cross_attention")
        )
        if self.fusion_mode not in {"cross_attention", "concat_mlp"}:
            raise ValueError(
                "MMRL_FUSION_MODE must be 'cross_attention' or 'concat_mlp', got "
                f"{self.fusion_mode!r}"
            )
        if self.fusion_mode == "concat_mlp" and self.memory_pooling_mode != "mean":
            raise ValueError(
                "concat_mlp fusion requires MMRL_MEMORY_POOLING_MODE='mean', got "
                f"{self.memory_pooling_mode!r}"
            )

        dimensions = {
            "insert_layer_count": self.insert_layer_count,
            "rp_space_length": self.rp_space_length,
            "rp_space_dim": self.rp_space_dim,
            "vision_token_dim": self.vision_token_dim,
            "text_token_dim": self.text_token_dim,
            "memory_query_count": self.memory_query_count,
            "memory_attention_dim": self.memory_attention_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
        }
        invalid = {name: value for name, value in dimensions.items() if value < 1}
        if invalid:
            raise ValueError(f"MMRL dimensions must be positive: {invalid}")

        shared_rep_dim = (
            self.vision_token_dim
            if self.query_architecture == "shared_direct_post_cross"
            else self.rp_space_dim
        )
        self.shared_represent_space = nn.Parameter(
            torch.empty(self.rp_space_length, shared_rep_dim)
        )
        nn.init.normal_(self.shared_represent_space, std=0.02)
        if self.query_architecture == "layer_mlp_post_cross":
            self.v_r_token_projector = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.rp_space_dim, self.projector_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.projector_hidden_dim, self.vision_token_dim),
                )
                for _ in range(self.insert_layer_count)
            ])
        else:
            self.v_r_token_projector = nn.ModuleList()

        self.layer_embeddings = nn.Parameter(
            torch.empty(self.insert_layer_count, 1, self.vision_token_dim)
        )
        nn.init.normal_(self.layer_embeddings, std=0.02)
        if self.memory_pooling_mode == "multi_query":
            self.visual_memory_pooling = MultiQueryAttentionPooling(
                self.vision_token_dim,
                self.vision_token_dim,
                self.memory_query_count,
                self.memory_attention_dim,
            )
            self.text_memory_pooling = MultiQueryAttentionPooling(
                self.text_token_dim,
                self.vision_token_dim,
                self.memory_query_count,
                self.memory_attention_dim,
            )
        elif self.memory_pooling_mode == "mean":
            self.visual_memory_pooling = MaskedMeanPooling(
                self.vision_token_dim,
                self.vision_token_dim,
            )
            self.text_memory_pooling = MaskedMeanPooling(
                self.text_token_dim,
                self.vision_token_dim,
            )
        else:
            self.visual_memory_pooling = TextGuidedVisualPooling(
                self.text_token_dim,
                self.vision_token_dim,
                self.vision_token_dim,
                self.memory_query_count,
                self.memory_attention_dim,
            )
            self.text_memory_pooling = None
        if self.fusion_mode == "cross_attention":
            self.cross_attention = ResidualCrossAttention(
                self.vision_token_dim,
                int(cfg.MMRL_CROSS_ATTENTION_HEADS),
            )
        else:
            self.cross_attention = ResidualConcatMLPFusion(
                self.vision_token_dim,
            )
        fusion_parameter_count = sum(
            parameter.numel() for parameter in self.cross_attention.parameters()
        )
        expected_fusion_parameters = (
            4 * self.vision_token_dim * self.vision_token_dim
            + 8 * self.vision_token_dim
        )
        if fusion_parameter_count != expected_fusion_parameters:
            raise RuntimeError(
                "fusion parameter budget mismatch: "
                f"mode={self.fusion_mode} expected={expected_fusion_parameters} "
                f"actual={fusion_parameter_count}"
            )
        print(
            "[MMRL_FUSION_PARAMETER_AUDIT] "
            f"mode={self.fusion_mode} parameters={fusion_parameter_count}"
        )

        self.cached_base_queries = None
        self.debug_context = {}
        self.last_rep_shape = None
        self.last_memory_shape = None
        self._printed_shape_audit = False

    def synchronize_layer_projector_initialization(self):
        if self.query_architecture == "shared_direct_post_cross":
            if len(self.v_r_token_projector) != 0:
                raise RuntimeError(
                    "shared-direct queries must not contain layer projectors"
                )
            print(
                "[MMRL_SAME_INIT_AUDIT] "
                "skipped=True reason=shared_direct_post_cross"
            )
            return
        if not self.same_init_layer_projectors:
            return
        if len(self.v_r_token_projector) != self.insert_layer_count:
            raise RuntimeError(
                "same initialization requires one independent projector per layer"
            )
        source = self.v_r_token_projector[0]
        with torch.no_grad():
            for projector in self.v_r_token_projector[1:]:
                projector.load_state_dict(source.state_dict(), strict=True)

        source_parameters = list(source.parameters())
        max_difference = 0.0
        shares_storage = False
        for projector in self.v_r_token_projector[1:]:
            for source_parameter, target_parameter in zip(
                source_parameters,
                projector.parameters(),
            ):
                max_difference = max(
                    max_difference,
                    float(
                        (source_parameter - target_parameter)
                        .detach()
                        .float()
                        .abs()
                        .max()
                        .item()
                    ),
                )
                shares_storage = (
                    shares_storage
                    or source_parameter.data_ptr() == target_parameter.data_ptr()
                )
        if max_difference != 0.0 or shares_storage:
            raise RuntimeError(
                "layer projector synchronization audit failed: "
                f"max_difference={max_difference} shares_storage={shares_storage}"
            )
        print(
            "[MMRL_SAME_INIT_AUDIT] "
            f"projectors={len(self.v_r_token_projector)} "
            "max_difference=0.0 shared_storage=False"
        )

    def _compute_base_queries(self):
        if self.query_architecture == "shared_direct_post_cross":
            projected = self.shared_represent_space.unsqueeze(0).expand(
                self.insert_layer_count,
                -1,
                -1,
            )
            return projected + self.layer_embeddings
        projected = torch.stack([
            projector(self.shared_represent_space)
            for projector in self.v_r_token_projector
        ], dim=0)
        return projected + self.layer_embeddings

    def _get_base_queries(self):
        if self.training:
            self.cached_base_queries = None
            return self._compute_base_queries()
        if self.cached_base_queries is None:
            with torch.no_grad():
                self.cached_base_queries = self._compute_base_queries().detach()
        return self.cached_base_queries

    def _pack_visual_states(self, visual_states, cu_seqlens):
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if lengths.numel() == 0 or bool((lengths <= 0).any()):
            raise RuntimeError(f"invalid visual sequence lengths: {lengths.tolist()}")
        sequences = torch.split(visual_states, lengths.cpu().tolist(), dim=0)
        padded = pad_sequence(sequences, batch_first=True)
        positions = torch.arange(padded.shape[1], device=visual_states.device)
        valid_mask = positions.unsqueeze(0) < lengths.to(
            device=visual_states.device
        ).unsqueeze(1)
        return padded, valid_mask

    def forward(
        self,
        visual_states,
        cu_seqlens,
        text_states,
        text_mask,
        images_per_sample: Sequence[int],
    ):
        image_counts = torch.as_tensor(
            images_per_sample, device=text_states.device, dtype=torch.long
        )
        image_count = int(cu_seqlens.numel() - 1)
        if image_counts.numel() != text_states.shape[0]:
            raise RuntimeError(
                "images_per_sample must match the text batch, got "
                f"counts={image_counts.numel()} text_batch={text_states.shape[0]}"
            )
        if bool((image_counts < 0).any()) or int(image_counts.sum().item()) != image_count:
            raise RuntimeError(
                "images_per_sample does not match packed visual sequences: "
                f"counts={image_counts.tolist()} images={image_count}"
            )
        base_queries = self._get_base_queries().to(
            device=visual_states.device, dtype=visual_states.dtype
        )
        flat_queries = base_queries.reshape(
            self.insert_layer_count * self.rp_space_length,
            self.vision_token_dim,
        ).unsqueeze(0).expand(image_count, -1, -1)
        visual_memory = None
        text_memory = None
        memory = None
        if self.use_dynamic_cross_attention:
            visual_padded, visual_mask = self._pack_visual_states(
                visual_states, cu_seqlens
            )
            if self.memory_pooling_mode == "text_guided":
                visual_memory = self.visual_memory_pooling(
                    visual_padded,
                    visual_mask,
                    text_states,
                    text_mask,
                    image_counts,
                )
                memory = visual_memory
            else:
                visual_memory = self.visual_memory_pooling(
                    visual_padded,
                    visual_mask,
                )
                text_memory = self.text_memory_pooling(text_states, text_mask)
                expanded_text_memory = torch.repeat_interleave(
                    text_memory,
                    image_counts,
                    dim=0,
                )
                memory = torch.cat(
                    [visual_memory, expanded_text_memory],
                    dim=1,
                )
            dynamic_queries, cross_delta = self.cross_attention(flat_queries, memory)
        else:
            dynamic_queries = flat_queries
            cross_delta = torch.zeros_like(flat_queries)
        dynamic_queries = dynamic_queries.reshape(
            image_count,
            self.insert_layer_count,
            self.rp_space_length,
            self.vision_token_dim,
        )

        self.last_memory_shape = tuple(memory.shape) if memory is not None else None
        self.last_rep_shape = tuple(dynamic_queries.shape)
        if not self._printed_shape_audit:
            print("[MMRL_DYNAMIC_REP_AUDIT]")
            print(f"  query_architecture={self.query_architecture}")
            print(f"  dynamic_cross_attention={self.use_dynamic_cross_attention}")
            print(f"  memory_pooling_mode={self.memory_pooling_mode}")
            print(f"  fusion_mode={self.fusion_mode}")
            print(
                "  visual_memory_shape="
                f"{None if visual_memory is None else tuple(visual_memory.shape)}"
            )
            print(
                "  text_memory_shape="
                f"{None if text_memory is None else tuple(text_memory.shape)}"
            )
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
                base_norm = flat_queries.detach().float().norm(dim=-1).mean()
                delta_norm = cross_delta.detach().float().norm(dim=-1).mean()
                self.debug_context = {
                    "dynamic_rep_base_norm_mean": base_norm,
                    "dynamic_rep_cross_delta_norm_mean": delta_norm,
                    "dynamic_rep_cross_delta_ratio": (
                        delta_norm / base_norm.clamp_min(1e-8)
                    ),
                }
                if visual_memory is not None:
                    self.debug_context["visual_memory_norm_mean"] = (
                        visual_memory.detach().float().norm(dim=-1).mean()
                    )
                if text_memory is not None:
                    self.debug_context["text_memory_norm_mean"] = (
                        text_memory.detach().float().norm(dim=-1).mean()
                    )
                pooling_debug = getattr(
                    self.visual_memory_pooling,
                    "debug_context",
                    {},
                )
                self.debug_context.update(pooling_debug)
        else:
            self.debug_context = {}

        return [
            dynamic_queries[:, layer_index]
            for layer_index in range(self.insert_layer_count)
        ]
