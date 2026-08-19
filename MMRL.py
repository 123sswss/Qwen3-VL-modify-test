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


class MMRL(nn.Module):
    QUERY_ARCHITECTURES = {
        "layer_mlp_post_cross",
        "shared_direct_post_cross",
    }

    def __init__(self, config):
        super().__init__()
        cfg = SimpleNamespace(**config.mmrl_config)
        default_architecture = (
            "shared_direct_post_cross"
            if bool(getattr(cfg, "DIRECT_SHARED_REP", False))
            else "layer_mlp_post_cross"
        )
        self.query_architecture = str(
            getattr(cfg, "MMRL_QUERY_ARCHITECTURE", default_architecture)
        )
        if self.query_architecture not in self.QUERY_ARCHITECTURES:
            raise ValueError(
                "Unsupported MMRL_QUERY_ARCHITECTURE="
                f"{self.query_architecture!r}; choices={sorted(self.QUERY_ARCHITECTURES)}"
            )
        self.use_direct_shared_rep = (
            self.query_architecture == "shared_direct_post_cross"
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
            if self.use_direct_shared_rep
            else self.rp_space_dim
        )
        self.shared_represent_space = nn.Parameter(
            torch.empty(self.rp_space_length, shared_rep_dim)
        )
        nn.init.normal_(self.shared_represent_space, std=0.02)
        if self.use_direct_shared_rep:
            self.v_r_token_projector = nn.ModuleList()
        else:
            self.v_r_token_projector = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.rp_space_dim, self.projector_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.projector_hidden_dim, self.vision_token_dim),
                )
                for _ in range(self.insert_layer_count)
            ])
        self.direct_v_tokens = nn.ParameterList()

        self.layer_embeddings = nn.Parameter(
            torch.empty(self.insert_layer_count, 1, self.vision_token_dim)
        )
        nn.init.normal_(self.layer_embeddings, std=0.02)
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
        self.cross_attention = ResidualCrossAttention(
            self.vision_token_dim,
            int(cfg.MMRL_CROSS_ATTENTION_HEADS),
        )

        self.cached_base_queries = None
        self.debug_context = {}
        self.last_rep_shape = None
        self.last_memory_shape = None
        self._printed_shape_audit = False

    def _compute_base_queries(self):
        if self.use_direct_shared_rep:
            projected = self.shared_represent_space.unsqueeze(0).expand(
                self.insert_layer_count, -1, -1
            )
        else:
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
        visual_padded, visual_mask = self._pack_visual_states(
            visual_states, cu_seqlens
        )
        visual_memory = self.visual_memory_pooling(visual_padded, visual_mask)
        text_memory = self.text_memory_pooling(text_states, text_mask)

        image_counts = torch.as_tensor(
            images_per_sample, device=text_states.device, dtype=torch.long
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
            text_memory, image_counts, dim=0
        )
        memory = torch.cat([visual_memory, expanded_text_memory], dim=1)

        base_queries = self._get_base_queries().to(
            device=visual_states.device, dtype=visual_states.dtype
        )
        flat_queries = base_queries.reshape(
            self.insert_layer_count * self.rp_space_length,
            self.vision_token_dim,
        ).unsqueeze(0).expand(image_count, -1, -1)
        dynamic_queries, cross_delta = self.cross_attention(flat_queries, memory)
        dynamic_queries = dynamic_queries.reshape(
            image_count,
            self.insert_layer_count,
            self.rp_space_length,
            self.vision_token_dim,
        )

        self.last_memory_shape = tuple(memory.shape)
        self.last_rep_shape = tuple(dynamic_queries.shape)
        if not self._printed_shape_audit:
            print("[MMRL_DYNAMIC_REP_AUDIT]")
            print(f"  query_architecture={self.query_architecture}")
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
                base_norm = flat_queries.detach().float().norm(dim=-1).mean()
                delta_norm = cross_delta.detach().float().norm(dim=-1).mean()
                self.debug_context = {
                    "dynamic_rep_base_norm_mean": base_norm,
                    "dynamic_rep_cross_delta_norm_mean": delta_norm,
                    "dynamic_rep_cross_delta_ratio": (
                        delta_norm / base_norm.clamp_min(1e-8)
                    ),
                    "visual_memory_norm_mean": (
                        visual_memory.detach().float().norm(dim=-1).mean()
                    ),
                    "text_memory_norm_mean": (
                        text_memory.detach().float().norm(dim=-1).mean()
                    ),
                }
        else:
            self.debug_context = {}

        return [
            dynamic_queries[:, layer_index]
            for layer_index in range(self.insert_layer_count)
        ]
