from multiprocessing import pool
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

import MMRLGating
import utils

from itertools import accumulate

class MMRLVitBlock(qwen3_vl.Qwen3VLVisionBlock):
    def __init__(self, config):
        super(MMRLVitBlock, self).__init__(config)
        self.INSERT_METHOD = config.mmrl_config["insert_method"]
    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                r_token: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[torch.Tensor] = None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                first_insert: Optional[bool] = False,
                **kwargs):
        assert r_token is not None
        num_r_tokens = r_token.shape[0]
        full_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
        if first_insert:
            hidden_states_split = torch.split(hidden_states, full_seq_lengths, dim=0)
            new_hidden_states_list = [torch.cat([r_token, s], dim=0) for s in hidden_states_split]
            hidden_states = torch.cat(new_hidden_states_list, dim=0)
            position_embeddings, cu_seqlens, rotary_pos_emb = _resize_cu_and_pos(
                r_token, cu_seqlens, position_embeddings, rotary_pos_emb
            )
        else:
            hidden_states_split = torch.split(hidden_states, full_seq_lengths, dim=0)
            new_hidden_states_list = []
            updated_seq = None
            for seq_hidden in hidden_states_split:
                if self.INSERT_METHOD == "replace":
                    updated_seq = torch.cat([r_token, seq_hidden[num_r_tokens:]], dim=0)
                elif self.INSERT_METHOD == "add":
                    prefix = seq_hidden[:num_r_tokens] + r_token
                    suffix = seq_hidden[num_r_tokens:]
                    updated_seq = torch.cat([prefix, suffix], dim=0)
                new_hidden_states_list.append(updated_seq)
            hidden_states = torch.cat(new_hidden_states_list, dim=0)
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states),
                                                  cu_seqlens=cu_seqlens,
                                                  rotary_pos_emb=rotary_pos_emb,
                                                  position_embeddings=position_embeddings,
                                                  **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def _resize_cu_and_pos(r_token, cu_seqlens, position_embeddings, rotary_pos_emb):
    num_r_tokens = r_token.shape[0]
    total_pic_num = cu_seqlens.shape[0] - 1
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
    original_cos, original_sin = position_embeddings
    dtype = original_cos.dtype
    device = original_cos.device
    pos_emb_dim = original_cos.shape[-1]
    rotary_dim = rotary_pos_emb.shape[-1]
    r_token_cos = torch.ones(num_r_tokens, pos_emb_dim, device=device, dtype=dtype)
    r_token_sin = torch.zeros(num_r_tokens, pos_emb_dim, device=device, dtype=dtype)
    r_token_rotary = torch.zeros(num_r_tokens, rotary_dim, device=device, dtype=rotary_pos_emb.dtype)
    cos_split = torch.split(original_cos, seq_lengths, dim=0)
    sin_split = torch.split(original_sin, seq_lengths, dim=0)
    rotary_split = torch.split(rotary_pos_emb, seq_lengths, dim=0)
    new_cos_list = []
    new_sin_list = []
    new_rotary_list = []
    for i in range(total_pic_num):
        current_img_cos = cos_split[i]  # [Seq_Len_i, Dim]
        current_img_sin = sin_split[i]  # [Seq_Len_i, Dim]
        current_img_rot = rotary_split[i]  # [Seq_Len_i, Dim]

        new_cos_list.append(torch.cat([r_token_cos, current_img_cos], dim=0))
        new_sin_list.append(torch.cat([r_token_sin, current_img_sin], dim=0))
        new_rotary_list.append(torch.cat([r_token_rotary, current_img_rot], dim=0))
    new_cos = torch.cat(new_cos_list, dim=0)
    new_sin = torch.cat(new_sin_list, dim=0)
    new_rotary = torch.cat(new_rotary_list, dim=0)
    updated_position_embeddings = (new_cos, new_sin)
    if total_pic_num > 0:
        offsets = torch.arange(total_pic_num + 1, device=cu_seqlens.device,
                               dtype=cu_seqlens.dtype) * num_r_tokens
        updated_cu_seqlens = cu_seqlens + offsets
    else:
        updated_cu_seqlens = cu_seqlens
    return updated_position_embeddings, updated_cu_seqlens, new_rotary

def _strip_r_token(hidden_states, original_lens, num_r_token):
    current_lens = [l + num_r_token for l in original_lens]
    hidden_states_split = torch.split(hidden_states, current_lens, dim=0)
    clean_list = []
    for seq in hidden_states_split:
        clean_list.append(seq[num_r_token:])
    return torch.cat(clean_list, dim=0)

class zeroInit(nn.Module):
    def __init__(self, dim):
        super(zeroInit, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 4),
                                 nn.ReLU(),
                                 nn.Linear(dim // 4, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class ResidualRouter(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim, adapter_count):
        super().__init__()
        self.adapter_count = int(max(adapter_count, 1))
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, self.adapter_count)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.relu = nn.ReLU()

    def compute_logits(
        self,
        pooled_vision_states: torch.Tensor,
        pooled_text_states: torch.Tensor,
        alpha_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pooled_vision_states is None or pooled_text_states is None:
            raise ValueError("pooled_vision_states and pooled_text_states must not be None")
        if pooled_vision_states.numel() == 0:
            return pooled_vision_states.new_zeros((0, self.adapter_count))

        if alpha_logits is None:
            alpha_prob = pooled_vision_states.new_zeros((pooled_vision_states.shape[0], 1))
        else:
            alpha_prob = torch.sigmoid(alpha_logits).to(dtype=pooled_vision_states.dtype)

        v_feat = self.relu(self.vision_proj(pooled_vision_states))
        t_feat = self.relu(self.text_proj(pooled_text_states))
        fused = torch.cat([v_feat, t_feat, alpha_prob], dim=-1)
        fused = self.relu(self.fusion(fused))
        return self.output_head(fused)

    def forward(
        self,
        pooled_vision_states: torch.Tensor,
        pooled_text_states: torch.Tensor,
        alpha_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.compute_logits(pooled_vision_states, pooled_text_states, alpha_logits)
        return torch.softmax(logits, dim=-1)

class VisionWithMMRL(qwen3_vl.Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.cfg = SimpleNamespace(**config.mmrl_config)
        self.blocks = nn.ModuleList([qwen3_vl.Qwen3VLVisionBlock(config)
                                     for _ in range(config.depth)])
        self.insert_layers = list(self.cfg.INSERT_LAYER)
        self._validate_layer_indexes("INSERT_LAYER", self.insert_layers, len(self.blocks))
        self.deepstack_visual_indexes = list(self.deepstack_visual_indexes)
        self._validate_layer_indexes("deepstack_visual_indexes", self.deepstack_visual_indexes, len(self.blocks))
        self.cfg.INSERT_LAYER = self.insert_layers
        self.blocks_with_rep = nn.ModuleList([MMRLVitBlock(config)
                                              for _ in self.insert_layers])
        self._printed_forward_layer_entry_audit = False
        self._printed_forward_layer_hit_audit = False
        self._print_layer_audit()
        self.hidden_state_pooling = utils.attention_pooling(self.cfg.vision_token_dim,
                                                            self.cfg.POOLING_DIM)
        self.embedding_pooling = utils.attention_pooling(self.cfg.text_token_dim,
                                                         self.cfg.POOLING_DIM)
        self.Task_classifier = MMRLGating.Task_classifier(self.cfg)
        self.visionGating = MMRLGating.HardConcreteGate(self.cfg.gating_temperature)
        # self.text_gating = MMRLGating.textGating(self.cfg,
        #                                          self.cfg.text_gating_epsilon,
        #                                          self.cfg.gating_temperature)
        self.visual_residual_adapter_count = int(max(
            getattr(self.cfg, "VISUAL_RESIDUAL_ADAPTER_COUNT", 4),
            1,
        ))
        self.adapter_router = ResidualRouter(
            self.cfg.vision_token_dim,
            self.cfg.text_token_dim,
            self.cfg.GATING_MID_DIM,
            self.visual_residual_adapter_count,
        )
        self.residual_adapters = nn.ModuleList([
            zeroInit(self.cfg.vision_token_dim)
            for _ in range(self.visual_residual_adapter_count)
        ])
        self.adapter_usage_balance_loss = torch.tensor(0.0)
        self.adapter_sample_entropy_loss = torch.tensor(0.0)
        self.adapter_common_mode_loss = torch.tensor(0.0)
        self.adapter_effective_delta_loss = torch.tensor(0.0)
        self.prototype_anchor_loss = torch.tensor(0.0)
        self.adapter_diversity_loss = torch.tensor(0.0)
        self.adapter_sample_entropy_target = float(getattr(self.cfg, "ADAPTER_SAMPLE_ENTROPY_TARGET", 0.40))
        self.adapter_common_mode_target = float(getattr(self.cfg, "ADAPTER_COMMON_MODE_TARGET", 0.85))
        self.adapter_effective_delta_target_low = float(getattr(self.cfg, "ADAPTER_EFFECTIVE_DELTA_TARGET_LOW", 0.78))
        self.adapter_effective_delta_target_high = float(getattr(self.cfg, "ADAPTER_EFFECTIVE_DELTA_TARGET_HIGH", 1.10))
        self.prototype_anchor_temperature = float(getattr(self.cfg, "PROTOTYPE_ANCHOR_TEMPERATURE", 0.20))
        self.prototype_anchor_momentum = float(getattr(self.cfg, "PROTOTYPE_ANCHOR_MOMENTUM", 0.95))
        self.prototype_anchor_min_confidence = float(getattr(self.cfg, "PROTOTYPE_ANCHOR_MIN_CONFIDENCE", 0.40))
        self.prototype_anchor_assignment_power = float(getattr(self.cfg, "PROTOTYPE_ANCHOR_ASSIGNMENT_POWER", 2.0))
        self.prototype_anchor_init_noise = float(getattr(self.cfg, "PROTOTYPE_ANCHOR_INIT_NOISE", 0.10))
        self.adapter_diversity_target_low = float(getattr(self.cfg, "ADAPTER_DIVERSITY_TARGET_LOW", 0.30))
        self.adapter_diversity_target_high = float(getattr(self.cfg, "ADAPTER_DIVERSITY_TARGET_HIGH", 0.58))
        self.adapter_diversity_upper_weight = float(getattr(self.cfg, "ADAPTER_DIVERSITY_UPPER_WEIGHT", 2.0))
        self.adapter_diversity_worst_pair_weight = float(
            getattr(self.cfg, "ADAPTER_DIVERSITY_WORST_PAIR_WEIGHT", 1.0)
        )
        self.enable_adapter_router_identity_residual = bool(
            getattr(self.cfg, "ENABLE_ADAPTER_ROUTER_IDENTITY_RESIDUAL", False)
        )
        self.direct_mmrl_output = bool(getattr(self.cfg, "DIRECT_MMRL_OUTPUT", False))
        self.enable_deepstack_mmrl_residual = bool(getattr(self.cfg, "ENABLE_DEEPSTACK_MMRL_RESIDUAL", False))
        self.deepstack_mmrl_residual_scale = float(getattr(self.cfg, "DEEPSTACK_MMRL_RESIDUAL_SCALE", 0.0))
        self.adapter_effective_delta_ratio_mean = torch.tensor(float("nan"))
        self.adapter_effective_delta_ratio_min = torch.tensor(float("nan"))
        self.adapter_effective_delta_ratio_max = torch.tensor(float("nan"))
        self.route_proto_kl = torch.tensor(float("nan"))
        self.route_proto_agreement = torch.tensor(float("nan"))
        self.prototype_usage_max = torch.tensor(float("nan"))
        self.prototype_usage_min = torch.tensor(float("nan"))
        self.adapter_pairwise_cos_mean = torch.tensor(float("nan"))
        self.adapter_pairwise_cos_max = torch.tensor(float("nan"))
        self.adapter_pairwise_cos_below_band = torch.tensor(float("nan"))
        self.adapter_pairwise_cos_above_band = torch.tensor(float("nan"))
        self.adapter_diversity_mean_component = torch.tensor(float("nan"))
        self.adapter_diversity_worst_component = torch.tensor(float("nan"))
        self.deepstack_delta_norm_mean = torch.tensor(float("nan"))
        self.deepstack_delta_to_org_ratio = torch.tensor(float("nan"))
        self.deepstack_residual_layers = torch.tensor(0.0)
        for idx in range(self.visual_residual_adapter_count):
            setattr(self, f"prototype_usage_{idx}", torch.tensor(float("nan")))
        self.register_buffer(
            "prototype_vectors",
            torch.zeros(self.visual_residual_adapter_count, self.cfg.vision_token_dim),
            persistent=True,
        )
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.visual_residual_adapter_count, dtype=torch.bool),
            persistent=True,
        )
        slot_offsets = torch.zeros(self.visual_residual_adapter_count, self.cfg.vision_token_dim)
        for idx in range(self.visual_residual_adapter_count):
            slot_offsets[idx, idx::self.visual_residual_adapter_count] = 1.0
            if idx % 2 == 1:
                slot_offsets[idx, (idx + 1)::self.visual_residual_adapter_count] = -1.0
        slot_offsets = F.normalize(slot_offsets, dim=-1)
        self.register_buffer(
            "prototype_slot_offsets",
            slot_offsets,
            persistent=False,
        )
        self.alpha_list = []
        self.G_list = []
        self.route_probs = None
        self._route_utility_ema = None
        self._route_utility_ema_initialized = False
        self._route_utility_step_metrics = {}
        self._route_utility_hook_warned = False
        self.k_results = None
        self.k_mask_results = None
        self.tax_loss = None
        self.capacity_prior_loss = None


        self.null_image_token = nn.Parameter(torch.zeros(1, self.cfg.vision_token_dim))
        nn.init.normal_(self.null_image_token, std=0.02)
        self.ablate_visual_gate = bool(getattr(self.cfg, "ABLATE_VISUAL_GATE", False))
        self.debug_context = {}

    @staticmethod
    def _validate_layer_indexes(name, indexes, depth):
        if not indexes:
            raise ValueError(f"{name} must not be empty")
        if len(set(indexes)) != len(indexes):
            raise ValueError(f"{name} contains duplicate indexes: {indexes}")
        for idx in indexes:
            if not isinstance(idx, int):
                raise TypeError(f"{name} must contain int indexes, got {idx!r}")
            if idx < 0 or idx >= depth:
                raise ValueError(
                    f"{name} index {idx} out of range for vision depth {depth}. "
                    "Layer indexes must be 0-based Python indexes."
                )

    def _print_layer_audit(self):
        overlap = sorted(set(self.deepstack_visual_indexes) & set(self.insert_layers))
        print("[MMRL_LAYER_INDEX_AUDIT]")
        print(f"  vision_depth={len(self.blocks)}")
        print(f"  insert_layers_0based={self.insert_layers}")
        print(f"  insert_layers_natural={[idx + 1 for idx in self.insert_layers]}")
        print(f"  blocks_with_rep_count={len(self.blocks_with_rep)}")
        print(f"  deepstack_visual_indexes_0based={self.deepstack_visual_indexes}")
        print(f"  deepstack_visual_indexes_natural={[idx + 1 for idx in self.deepstack_visual_indexes]}")
        print(f"  deepstack_mmrl_overlap_0based={overlap}")
        print(f"  deepstack_mmrl_overlap_natural={[idx + 1 for idx in overlap]}")

    def _pool_tokens_by_image_mean(self, token_states: torch.Tensor, cu_seqlens: torch.Tensor):
        if token_states is None or cu_seqlens is None or cu_seqlens.numel() <= 1 or token_states.numel() == 0:
            return None
        pooled = []
        starts = cu_seqlens[:-1].tolist()
        ends = cu_seqlens[1:].tolist()
        for s, e in zip(starts, ends):
            s, e = int(s), int(e)
            if e <= s:
                continue
            pooled.append(token_states[s:e].mean(dim=0))
        if not pooled:
            return None
        return torch.stack(pooled, dim=0)

    def _compute_residual_debug_metrics(
        self,
        final_delta: torch.Tensor,
        gated_delta: torch.Tensor,
        org_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        route_probs: Optional[torch.Tensor] = None,
    ):
        device = final_delta.device
        nan = torch.tensor(float("nan"), device=device)
        metrics = {
            "gated_delta_norm_mean": nan,
            "gated_delta_norm_std": nan,
            "gated_delta_norm_p95": nan,
            "gated_delta_norm_max": nan,
            "final_delta_norm_std": nan,
            "final_delta_norm_p95": nan,
            "final_delta_norm_max": nan,
            "final_to_gated_ratio": nan,
            "delta_transform_cos_mean": nan,
            "delta_transform_cos_std": nan,
            "delta_pool_norm_mean": nan,
            "delta_pool_norm_std": nan,
            "delta_pool_pairwise_cos_mean": nan,
            "delta_pool_pairwise_cos_std": nan,
            "delta_pool_common_mode_ratio": nan,
            "delta_pool_specificity_ratio": nan,
            "mmrl_delta_pool_common_mode_ratio": nan,
            "mmrl_delta_pool_specificity_ratio": nan,
            "delta_org_pool_cos_mean": nan,
            "delta_org_pool_cos_std": nan,
            "delta_token_norm_entropy": nan,
            "delta_token_norm_entropy_norm": nan,
            "delta_token_top10_mass": nan,
            "adapter_route_entropy": nan,
            "adapter_route_entropy_norm": nan,
            "adapter_route_confidence": nan,
            "adapter_usage_max": nan,
            "adapter_usage_min": nan,
        }

        with torch.no_grad():
            final_delta = final_delta.detach().float()
            gated_delta = gated_delta.detach().float()
            org_hidden_states = org_hidden_states.detach().float()

            if torch.is_tensor(route_probs) and route_probs.numel() > 0:
                route_probs = route_probs.detach().float()
                route_usage = route_probs.mean(dim=0)
                route_usage = route_usage / route_usage.sum().clamp_min(1e-8)
                route_entropy = -(route_usage * route_usage.clamp_min(1e-8).log()).sum()
                metrics["adapter_route_entropy"] = route_entropy.to(device)
                if route_usage.numel() > 1:
                    metrics["adapter_route_entropy_norm"] = (
                        route_entropy / torch.log(route_usage.new_tensor(float(route_usage.numel())))
                    ).to(device)
                else:
                    metrics["adapter_route_entropy_norm"] = route_usage.new_tensor(0.0).to(device)
                metrics["adapter_route_confidence"] = route_probs.max(dim=-1).values.mean().to(device)
                metrics["adapter_usage_max"] = route_usage.max().to(device)
                metrics["adapter_usage_min"] = route_usage.min().to(device)
                for idx, value in enumerate(route_usage.tolist()):
                    metrics[f"adapter_usage_{idx}"] = torch.tensor(float(value), device=device)

            final_token_norm = final_delta.norm(dim=-1)
            gated_token_norm = gated_delta.norm(dim=-1)

            metrics["gated_delta_norm_mean"] = gated_token_norm.mean()
            metrics["gated_delta_norm_std"] = gated_token_norm.std(unbiased=False)
            metrics["gated_delta_norm_p95"] = torch.quantile(gated_token_norm, 0.95) if gated_token_norm.numel() > 0 else nan
            metrics["gated_delta_norm_max"] = gated_token_norm.max() if gated_token_norm.numel() > 0 else nan
            metrics["final_delta_norm_std"] = final_token_norm.std(unbiased=False)
            metrics["final_delta_norm_p95"] = torch.quantile(final_token_norm, 0.95) if final_token_norm.numel() > 0 else nan
            metrics["final_delta_norm_max"] = final_token_norm.max() if final_token_norm.numel() > 0 else nan
            metrics["final_to_gated_ratio"] = final_token_norm.mean() / gated_token_norm.mean().clamp_min(1e-8)

            valid_transform = (final_token_norm > 1e-8) & (gated_token_norm > 1e-8)
            if valid_transform.any():
                transform_cos = F.cosine_similarity(final_delta[valid_transform], gated_delta[valid_transform], dim=-1)
                metrics["delta_transform_cos_mean"] = transform_cos.mean()
                metrics["delta_transform_cos_std"] = transform_cos.std(unbiased=False)

            pooled_delta = self._pool_tokens_by_image_mean(final_delta, cu_seqlens)
            pooled_mmrl_delta = self._pool_tokens_by_image_mean(gated_delta, cu_seqlens)
            pooled_org = self._pool_tokens_by_image_mean(org_hidden_states, cu_seqlens)
            if pooled_delta is not None:
                pooled_delta_norm = pooled_delta.norm(dim=-1)
                metrics["delta_pool_norm_mean"] = pooled_delta_norm.mean()
                metrics["delta_pool_norm_std"] = pooled_delta_norm.std(unbiased=False)
                pooled_delta_mean_norm = pooled_delta_norm.mean().clamp_min(1e-8)
                pooled_delta_common = pooled_delta.mean(dim=0)
                pooled_delta_specific = pooled_delta - pooled_delta_common
                metrics["delta_pool_common_mode_ratio"] = pooled_delta_common.norm() / pooled_delta_mean_norm
                metrics["delta_pool_specificity_ratio"] = (
                    pooled_delta_specific.norm(dim=-1).mean() / pooled_delta_mean_norm
                )

                if pooled_delta.shape[0] >= 2:
                    delta_unit = pooled_delta / pooled_delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    pairwise = delta_unit @ delta_unit.transpose(0, 1)
                    tri = torch.triu_indices(pairwise.shape[0], pairwise.shape[1], offset=1, device=pairwise.device)
                    pairwise_vals = pairwise[tri[0], tri[1]]
                    if pairwise_vals.numel() > 0:
                        metrics["delta_pool_pairwise_cos_mean"] = pairwise_vals.mean()
                        metrics["delta_pool_pairwise_cos_std"] = pairwise_vals.std(unbiased=False)

            if pooled_mmrl_delta is not None:
                pooled_mmrl_norm = pooled_mmrl_delta.norm(dim=-1)
                pooled_mmrl_mean_norm = pooled_mmrl_norm.mean().clamp_min(1e-8)
                pooled_mmrl_common = pooled_mmrl_delta.mean(dim=0)
                pooled_mmrl_specific = pooled_mmrl_delta - pooled_mmrl_common
                metrics["mmrl_delta_pool_common_mode_ratio"] = (
                    pooled_mmrl_common.norm() / pooled_mmrl_mean_norm
                )
                metrics["mmrl_delta_pool_specificity_ratio"] = (
                    pooled_mmrl_specific.norm(dim=-1).mean() / pooled_mmrl_mean_norm
                )

            if pooled_delta is not None and pooled_org is not None:
                valid_pool = (pooled_delta.norm(dim=-1) > 1e-8) & (pooled_org.norm(dim=-1) > 1e-8)
                if valid_pool.any():
                    pool_cos = F.cosine_similarity(pooled_delta[valid_pool], pooled_org[valid_pool], dim=-1)
                    metrics["delta_org_pool_cos_mean"] = pool_cos.mean()
                    metrics["delta_org_pool_cos_std"] = pool_cos.std(unbiased=False)

            token_mass = final_token_norm.clamp_min(0.0)
            total_mass = token_mass.sum()
            if total_mass > 1e-8:
                probs = token_mass / total_mass
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
                metrics["delta_token_norm_entropy"] = entropy
                if probs.numel() > 1:
                    metrics["delta_token_norm_entropy_norm"] = entropy / torch.log(
                        probs.new_tensor(float(probs.numel()))
                    )
                topk = min(10, probs.numel())
                metrics["delta_token_top10_mass"] = probs.topk(topk).values.sum()

        return metrics

    def _adapter_weight_norm(self, device):
        vals = []
        for adapter in self.residual_adapters:
            for p in adapter.parameters():
                vals.append(p.detach().float().norm())
        if not vals:
            return torch.tensor(float("nan"), device=device)
        return torch.stack(vals).norm().to(device)

    def _pool_adapter_outputs_by_image_mean(self, adapter_outputs: torch.Tensor, cu_seqlens: torch.Tensor):
        if adapter_outputs is None or cu_seqlens is None or cu_seqlens.numel() <= 1 or adapter_outputs.numel() == 0:
            return None
        pooled = []
        starts = cu_seqlens[:-1].tolist()
        ends = cu_seqlens[1:].tolist()
        for s, e in zip(starts, ends):
            s, e = int(s), int(e)
            if e <= s:
                continue
            pooled.append(adapter_outputs[s:e].mean(dim=0))
        if not pooled:
            return None
        return torch.stack(pooled, dim=0)

    def _prepare_route_utility_metrics(self, reference: torch.Tensor):
        device = reference.device
        if (
            self._route_utility_ema is None
            or self._route_utility_ema.shape[0] != self.visual_residual_adapter_count
            or self._route_utility_ema.device != device
        ):
            self._route_utility_ema = torch.full(
                (self.visual_residual_adapter_count,),
                float("nan"),
                device=device,
                dtype=torch.float32,
            )
            self._route_utility_ema_initialized = False

        nan = torch.tensor(float("nan"), device=device, dtype=torch.float32)
        self._route_utility_step_metrics = {
            "route_utility_agreement": nan.clone(),
            "route_utility_regret": nan.clone(),
            "utility_margin": nan.clone(),
            "utility_grad_norm": nan.clone(),
            "route_utility_valid_fraction": nan.clone(),
        }
        for idx in range(self.visual_residual_adapter_count):
            self._route_utility_step_metrics[f"adapter_utility_ema_{idx}"] = self._route_utility_ema[idx]

    def _register_route_utility_monitor(
        self,
        hidden_states: torch.Tensor,
        adapter_outputs: torch.Tensor,
        final_delta: torch.Tensor,
        delta: torch.Tensor,
        G_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ):
        self._prepare_route_utility_metrics(hidden_states)
        if (
            not self.training
            or not hidden_states.requires_grad
            or adapter_outputs is None
            or not torch.is_tensor(self.route_probs)
            or self.route_probs.numel() == 0
        ):
            return

        metric_slots = self._route_utility_step_metrics
        ema_state = self._route_utility_ema
        adapter_outputs_detached = adapter_outputs.detach()
        final_delta_detached = final_delta.detach()
        delta_detached = delta.detach()
        gate_detached = G_mask.detach()
        routes_detached = self.route_probs.detach()
        seqlens_detached = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().to(torch.long)
        include_identity = self.enable_adapter_router_identity_residual

        # This tensor is upstream of the merger/LLM but not the auxiliary losses,
        # so its hook observes task-CE credit without changing the backward pass.
        def _route_utility_hook(grad):
            try:
                with torch.no_grad():
                    grad_f = grad.detach().float()
                    final_f = final_delta_detached.float()
                    gate_f = gate_detached.float().view(-1)
                    routes_f = routes_detached.float()

                    image_count = min(int(seqlens_detached.numel()), int(routes_f.shape[0]))
                    token_count = min(
                        int(grad_f.shape[0]),
                        int(final_f.shape[0]),
                        int(adapter_outputs_detached.shape[0]),
                        int(gate_f.shape[0]),
                        int(seqlens_detached[:image_count].sum().item()) if image_count > 0 else 0,
                    )
                    if image_count <= 0 or token_count <= 0:
                        return grad

                    image_ids = torch.repeat_interleave(
                        torch.arange(image_count, device=grad_f.device),
                        seqlens_detached[:image_count].to(device=grad_f.device),
                    )[:token_count]
                    grad_f = grad_f[:token_count]
                    final_f = final_f[:token_count]
                    gate_f = gate_f[:token_count]

                    grad_sq = grad_f.square().sum(dim=-1)
                    final_sq = final_f.square().sum(dim=-1)
                    grad_sq_by_image = torch.zeros(image_count, device=grad_f.device).index_add_(
                        0, image_ids, grad_sq
                    )
                    final_sq_by_image = torch.zeros(image_count, device=grad_f.device).index_add_(
                        0, image_ids, final_sq
                    )
                    grad_norm = grad_sq_by_image.clamp_min(0.0).sqrt()
                    final_norm = final_sq_by_image.clamp_min(0.0).sqrt()
                    denominator = grad_norm * final_norm

                    mix_dot = (grad_f * final_f).sum(dim=-1)
                    identity_dot = None
                    if include_identity:
                        identity_dot = (
                            grad_f * delta_detached[:token_count].float()
                        ).sum(dim=-1) * gate_f

                    utility_numerator = torch.zeros(
                        image_count,
                        self.visual_residual_adapter_count,
                        device=grad_f.device,
                        dtype=torch.float32,
                    )
                    for idx in range(self.visual_residual_adapter_count):
                        expert_dot = (
                            grad_f * adapter_outputs_detached[:token_count, idx].float()
                        ).sum(dim=-1) * gate_f
                        if identity_dot is not None:
                            expert_dot = expert_dot + identity_dot
                        token_utility = -(expert_dot - mix_dot)
                        utility_numerator[:, idx].index_add_(0, image_ids, token_utility)

                    utility = utility_numerator / denominator.clamp_min(1e-12).unsqueeze(-1)
                    valid = (
                        (grad_norm > 1e-12)
                        & (final_norm > 1e-12)
                        & torch.isfinite(utility).all(dim=-1)
                    )
                    metric_slots["route_utility_valid_fraction"].copy_(valid.float().mean())
                    finite_grad_norm = grad_norm[torch.isfinite(grad_norm)]
                    if finite_grad_norm.numel() > 0:
                        metric_slots["utility_grad_norm"].copy_(finite_grad_norm.mean())
                    if not valid.any():
                        return grad

                    utility = utility[valid]
                    routes = routes_f[:image_count][valid]
                    routes = routes / routes.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    batch_utility = utility.mean(dim=0)
                    if not self._route_utility_ema_initialized:
                        ema_state.copy_(batch_utility)
                        self._route_utility_ema_initialized = True
                    else:
                        ema_state.mul_(0.90).add_(batch_utility, alpha=0.10)

                    utility_best, utility_winner = utility.max(dim=-1)
                    route_winner = routes.argmax(dim=-1)
                    routed_top_utility = utility.gather(1, route_winner.unsqueeze(-1)).squeeze(-1)
                    metric_slots["route_utility_agreement"].copy_(
                        (utility_winner == route_winner).float().mean()
                    )
                    metric_slots["route_utility_regret"].copy_(
                        (utility_best - routed_top_utility).clamp_min(0.0).mean()
                    )
                    if utility.shape[-1] > 1:
                        top_two = utility.topk(k=2, dim=-1).values
                        metric_slots["utility_margin"].copy_((top_two[:, 0] - top_two[:, 1]).mean())
            except Exception as exc:
                if not self._route_utility_hook_warned:
                    print(f"[ROUTE_UTILITY_MONITOR_WARN] disabled for failed batch: {exc}")
                    self._route_utility_hook_warned = True
            return grad

        hidden_states.register_hook(_route_utility_hook)

    @torch.no_grad()
    def _initialize_prototypes_from_batch(self, features: torch.Tensor):
        if features is None or features.numel() == 0:
            return
        features = F.normalize(features.detach().float(), dim=-1)
        n = features.shape[0]
        if n <= 0:
            return

        offsets = self.prototype_slot_offsets.to(device=features.device, dtype=features.dtype)
        init_noise = max(float(self.prototype_anchor_init_noise), 0.0)
        proto = []
        for idx in range(self.visual_residual_adapter_count):
            base = features[idx % n]
            proto.append(F.normalize(base + offsets[idx] * init_noise, dim=-1))
        proto = torch.stack(proto, dim=0)
        self.prototype_vectors.copy_(proto.to(device=self.prototype_vectors.device, dtype=self.prototype_vectors.dtype))
        self.prototype_initialized.fill_(True)

    @torch.no_grad()
    def _update_prototypes(self, pooled_vision_states: torch.Tensor, route_probs: torch.Tensor):
        if pooled_vision_states is None or route_probs is None:
            return
        if pooled_vision_states.numel() == 0 or route_probs.numel() == 0:
            return

        features = F.normalize(pooled_vision_states.detach().float(), dim=-1)
        routes = route_probs.detach().float()
        n = min(features.shape[0], routes.shape[0])
        if n <= 0:
            return
        features = features[:n]
        routes = routes[:n]

        if not bool(self.prototype_initialized.all().item()):
            self._initialize_prototypes_from_batch(features)

        confidence, winners = routes.max(dim=-1)
        active_sample = confidence >= float(self.prototype_anchor_min_confidence)
        if not active_sample.any():
            return

        assignment_power = max(float(self.prototype_anchor_assignment_power), 1.0)
        weights = routes.clamp_min(1e-8).pow(assignment_power)

        proto = self.prototype_vectors.to(device=features.device, dtype=features.dtype).clone()
        initialized = self.prototype_initialized.to(device=features.device)
        momentum = float(self.prototype_anchor_momentum)
        for idx in range(self.visual_residual_adapter_count):
            mask = active_sample & (winners == idx)
            if not mask.any():
                continue
            slot_weights = weights[mask, idx]
            batch_proto = (features[mask] * slot_weights.unsqueeze(-1)).sum(dim=0)
            batch_proto = batch_proto / slot_weights.sum().clamp_min(1e-8)
            batch_proto = F.normalize(batch_proto, dim=-1)
            proto[idx].mul_(momentum).add_(batch_proto, alpha=1.0 - momentum)
            proto[idx] = F.normalize(proto[idx], dim=-1)
            initialized[idx] = True
        self.prototype_vectors.copy_(proto.to(device=self.prototype_vectors.device, dtype=self.prototype_vectors.dtype))
        self.prototype_initialized.copy_(initialized.to(device=self.prototype_initialized.device))

    def _compute_prototype_anchor_loss(self, pooled_vision_states, route_probs, dtype, device):
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        nan = torch.tensor(float("nan"), device=device, dtype=dtype)
        self.route_proto_kl = nan
        self.route_proto_agreement = nan
        self.prototype_usage_max = nan
        self.prototype_usage_min = nan

        if pooled_vision_states is None or route_probs is None:
            return zero
        if pooled_vision_states.numel() == 0 or route_probs.numel() == 0:
            return zero
        initialized_before = bool(self.prototype_initialized.any().item())
        if not initialized_before:
            self._update_prototypes(pooled_vision_states, route_probs)
        if not bool(self.prototype_initialized.any().item()):
            return zero

        features = pooled_vision_states.detach().float()
        routes = route_probs.float()
        n = min(features.shape[0], routes.shape[0])
        if n <= 0:
            return zero
        features = F.normalize(features[:n], dim=-1)
        prototypes = self.prototype_vectors.detach().to(device=features.device, dtype=features.dtype)
        prototypes = F.normalize(prototypes, dim=-1)
        valid = self.prototype_initialized.to(device=features.device)
        logits = features @ prototypes.transpose(0, 1)
        logits = logits / max(float(self.prototype_anchor_temperature), 1e-4)
        if (~valid).any():
            logits[:, ~valid] = -1e4
        proto_probs = torch.softmax(logits, dim=-1)
        proto_probs = proto_probs.detach()
        routes = routes[:n].clamp_min(1e-8)
        loss = (proto_probs * (proto_probs.clamp_min(1e-8).log() - routes.log())).sum(dim=-1).mean()

        with torch.no_grad():
            usage = proto_probs.mean(dim=0)
            self.route_proto_kl = loss.detach().to(device=device, dtype=dtype)
            self.route_proto_agreement = (
                routes.detach().argmax(dim=-1) == proto_probs.argmax(dim=-1)
            ).float().mean().to(device=device, dtype=dtype)
            self.prototype_usage_max = usage.max().to(device=device, dtype=dtype)
            self.prototype_usage_min = usage.min().to(device=device, dtype=dtype)
            for idx, value in enumerate(usage.tolist()):
                setattr(self, f"prototype_usage_{idx}", torch.tensor(float(value), device=device, dtype=dtype))

        if self.training and initialized_before:
            self._update_prototypes(pooled_vision_states, route_probs)

        return loss.to(device=device, dtype=dtype)

    def _compute_adapter_diversity_loss(self, adapter_outputs, cu_seqlens, dtype, device):
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        nan = torch.tensor(float("nan"), device=device, dtype=dtype)
        self.adapter_pairwise_cos_mean = nan
        self.adapter_pairwise_cos_max = nan
        self.adapter_pairwise_cos_below_band = nan
        self.adapter_pairwise_cos_above_band = nan
        self.adapter_diversity_mean_component = nan
        self.adapter_diversity_worst_component = nan
        for idx in range(self.visual_residual_adapter_count):
            setattr(self, f"adapter_output_norm_{idx}", nan)
            setattr(self, f"adapter_contribution_norm_{idx}", nan)
            for other_idx in range(idx + 1, self.visual_residual_adapter_count):
                setattr(self, f"adapter_pairwise_cos_{idx}_{other_idx}", nan)

        pooled = self._pool_adapter_outputs_by_image_mean(adapter_outputs, cu_seqlens)
        if pooled is None or pooled.numel() == 0 or pooled.shape[1] <= 1:
            return zero

        pooled = pooled.float()
        unit = F.normalize(pooled, dim=-1)
        pairwise = torch.matmul(unit, unit.transpose(1, 2))
        tri = torch.triu_indices(pairwise.shape[1], pairwise.shape[2], offset=1, device=pairwise.device)
        vals = pairwise[:, tri[0], tri[1]].reshape(-1)
        if vals.numel() == 0:
            return zero

        low = float(self.adapter_diversity_target_low)
        high = float(self.adapter_diversity_target_high)
        if high < low:
            low, high = high, low
        lower_violation = torch.relu(vals.new_tensor(low) - vals).pow(2)
        upper_violation = torch.relu(vals - vals.new_tensor(high)).pow(2)
        weighted_violation = (
            lower_violation
            + float(self.adapter_diversity_upper_weight) * upper_violation
        )
        mean_component = weighted_violation.mean()
        worst_component = weighted_violation.max()
        loss = (
            mean_component
            + float(self.adapter_diversity_worst_pair_weight) * worst_component
        )
        with torch.no_grad():
            self.adapter_pairwise_cos_mean = vals.detach().mean().to(device=device, dtype=dtype)
            self.adapter_pairwise_cos_max = vals.detach().max().to(device=device, dtype=dtype)
            self.adapter_pairwise_cos_below_band = (
                (vals.detach() < low).float().mean().to(device=device, dtype=dtype)
            )
            self.adapter_pairwise_cos_above_band = (
                (vals.detach() > high).float().mean().to(device=device, dtype=dtype)
            )
            self.adapter_diversity_mean_component = mean_component.detach().to(device=device, dtype=dtype)
            self.adapter_diversity_worst_component = worst_component.detach().to(device=device, dtype=dtype)

            pooled_detached = pooled.detach()
            output_norms = pooled_detached.norm(dim=-1).mean(dim=0)
            for idx, value in enumerate(output_norms):
                setattr(
                    self,
                    f"adapter_output_norm_{idx}",
                    value.to(device=device, dtype=dtype),
                )

            routes = self.route_probs
            if torch.is_tensor(routes) and routes.numel() > 0:
                routes = routes.detach().float()
                n = min(pooled_detached.shape[0], routes.shape[0])
                if n > 0:
                    contribution_norms = (
                        pooled_detached[:n] * routes[:n].unsqueeze(-1)
                    ).norm(dim=-1).mean(dim=0)
                    for idx, value in enumerate(contribution_norms):
                        setattr(
                            self,
                            f"adapter_contribution_norm_{idx}",
                            value.to(device=device, dtype=dtype),
                        )

            pairwise_mean = pairwise.detach().mean(dim=0)
            for idx in range(pairwise_mean.shape[0]):
                for other_idx in range(idx + 1, pairwise_mean.shape[1]):
                    setattr(
                        self,
                        f"adapter_pairwise_cos_{idx}_{other_idx}",
                        pairwise_mean[idx, other_idx].to(device=device, dtype=dtype),
                    )
        return loss.to(device=device, dtype=dtype)

    def _compute_router_aux_losses(
        self,
        final_delta,
        cu_seqlens,
        org_hidden_states=None,
        pooled_vision_states=None,
        adapter_outputs=None,
    ):
        device = final_delta.device
        zero = final_delta.new_tensor(0.0)
        nan = final_delta.new_tensor(float("nan"))
        self.adapter_effective_delta_ratio_mean = nan
        self.adapter_effective_delta_ratio_min = nan
        self.adapter_effective_delta_ratio_max = nan
        self.prototype_anchor_loss = zero
        self.adapter_diversity_loss = zero
        self.route_proto_kl = nan
        self.route_proto_agreement = nan
        self.prototype_usage_max = nan
        self.prototype_usage_min = nan
        self.adapter_pairwise_cos_mean = nan
        self.adapter_pairwise_cos_max = nan
        self.adapter_pairwise_cos_below_band = nan
        self.adapter_pairwise_cos_above_band = nan
        self.adapter_diversity_mean_component = nan
        self.adapter_diversity_worst_component = nan
        for idx in range(self.visual_residual_adapter_count):
            setattr(self, f"prototype_usage_{idx}", nan)
        route_probs = self.route_probs
        if not torch.is_tensor(route_probs) or route_probs.numel() == 0:
            return zero, zero, zero, zero, zero, zero

        route_probs = route_probs.float()
        adapter_count = route_probs.shape[-1]

        usage = route_probs.mean(dim=0)
        usage = usage / usage.sum().clamp_min(1e-8)
        uniform = torch.full_like(usage, 1.0 / max(adapter_count, 1))
        usage_balance_loss = (usage * (usage.clamp_min(1e-8).log() - uniform.log())).sum()

        if adapter_count > 1:
            sample_entropy = -(route_probs * route_probs.clamp_min(1e-8).log()).sum(dim=-1)
            sample_entropy_norm = sample_entropy / torch.log(route_probs.new_tensor(float(adapter_count)))
            sample_entropy_loss = torch.relu(
                route_probs.new_tensor(float(self.adapter_sample_entropy_target)) - sample_entropy_norm
            ).pow(2).mean()
        else:
            sample_entropy_loss = zero

        pooled_delta = self._pool_tokens_by_image_mean(final_delta.float(), cu_seqlens)
        if pooled_delta is None or pooled_delta.numel() == 0:
            common_mode_loss = zero
        else:
            pooled_delta_norm = pooled_delta.norm(dim=-1)
            pooled_delta_mean_norm = pooled_delta_norm.mean().clamp_min(1e-8)
            common_ratio = pooled_delta.mean(dim=0).norm() / pooled_delta_mean_norm
            common_mode_loss = torch.relu(
                common_ratio - float(self.adapter_common_mode_target)
            ).pow(2)

        effective_delta_loss = zero
        if org_hidden_states is not None:
            pooled_org = self._pool_tokens_by_image_mean(org_hidden_states.detach().float(), cu_seqlens)
            if pooled_delta is not None and pooled_org is not None and pooled_delta.numel() > 0:
                n = min(pooled_delta.shape[0], pooled_org.shape[0], route_probs.shape[0])
                if n > 0:
                    delta_ratio = (
                        pooled_delta[:n].norm(dim=-1)
                        / pooled_org[:n].norm(dim=-1).clamp_min(1e-8)
                    )
                    confidence = route_probs[:n].max(dim=-1).values.detach().to(delta_ratio.device)
                    floor_loss = confidence * torch.relu(
                        delta_ratio.new_tensor(float(self.adapter_effective_delta_target_low)) - delta_ratio
                    ).pow(2)
                    ceiling_loss = torch.relu(
                        delta_ratio - delta_ratio.new_tensor(float(self.adapter_effective_delta_target_high))
                    ).pow(2)
                    effective_delta_loss = floor_loss.mean() + ceiling_loss.mean()
                    self.adapter_effective_delta_ratio_mean = delta_ratio.detach().mean().to(device=device, dtype=final_delta.dtype)
                    self.adapter_effective_delta_ratio_min = delta_ratio.detach().min().to(device=device, dtype=final_delta.dtype)
                    self.adapter_effective_delta_ratio_max = delta_ratio.detach().max().to(device=device, dtype=final_delta.dtype)

        prototype_anchor_loss = self._compute_prototype_anchor_loss(
            pooled_vision_states,
            route_probs,
            final_delta.dtype,
            device,
        )
        adapter_diversity_loss = self._compute_adapter_diversity_loss(
            adapter_outputs,
            cu_seqlens,
            final_delta.dtype,
            device,
        )

        return (
            usage_balance_loss.to(device=device, dtype=final_delta.dtype),
            sample_entropy_loss.to(device=device, dtype=final_delta.dtype),
            common_mode_loss.to(device=device, dtype=final_delta.dtype),
            effective_delta_loss.to(device=device, dtype=final_delta.dtype),
            prototype_anchor_loss.to(device=device, dtype=final_delta.dtype),
            adapter_diversity_loss.to(device=device, dtype=final_delta.dtype),
        )

    def forward(self,
                hidden_states: torch.Tensor,
                grid_thw: torch.Tensor,
                v_r_token_list: Optional[list[torch.Tensor]] = None,
                embedding: Optional[torch.Tensor] = None,
                text_pooling_mask: Optional[torch.Tensor] = None,
                gating_temperature_override: float = None,
                images_per_sample: Optional[list[int]] = None,
                **kwargs):
        assert len(images_per_sample) == embedding.shape[0]
        batch_size = embedding.shape[0]
        pic_seqlens = [0] + list(accumulate(images_per_sample))

        self.alpha_list = None 
        self.G_list = []
        self.route_probs = None
        pooled_vision_states = None


        rotary_pos_emb_with_rep = None
        embedding_after_pooling = self.embedding_pooling(embedding, mask=text_pooling_mask)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        original_seq_lens_list = (grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0]).tolist()

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        first_insert = True
        current_num_r_token = self.cfg.RP_SPACE_LENGTH

        deepstack_feature_lists = []
        deepstack_delta_norm_values = []
        deepstack_delta_ratio_values = []
        deepstack_residual_layer_count = 0
        cu_seqlens_with_rep, position_embeddings_with_rep = None, None
        hidden_states_with_rep, org_hidden_states= None, None
        total_pic_num = cu_seqlens.size(0) - 1
        run_mmrl_branch = True
        forward_hit_layers = []
        if not self._printed_forward_layer_entry_audit:
            print("[MMRL_FORWARD_ENTRY_AUDIT]")
            print(f"  v_r_token_list_is_none={v_r_token_list is None}")
            print(f"  planned_insert_layers_0based={self.insert_layers}")
            print(f"  planned_insert_layers_natural={[idx + 1 for idx in self.insert_layers]}")
            print(f"  deepstack_visual_indexes_0based={self.deepstack_visual_indexes}")
            print(f"  deepstack_visual_indexes_natural={[idx + 1 for idx in self.deepstack_visual_indexes]}")
            self._printed_forward_layer_entry_audit = True
        for layer_num, blk in enumerate(self.blocks):
            if layer_num not in self.insert_layers or v_r_token_list is None:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    rotary_pos_emb=rotary_pos_emb,
                    **kwargs,
                )
                org_hidden_states = hidden_states
            else:
                ############ N图切分+门控 ############
                org_input_states = hidden_states if first_insert else org_hidden_states
                if first_insert:
                    # cu_seqlens: [0, len0, len0+len1, ...]
                    img_seqlens = cu_seqlens[1:] - cu_seqlens[:-1]  # [Total_Images]
                    img_indices = torch.repeat_interleave(
                        torch.arange(total_pic_num, device=hidden_states.device),
                        img_seqlens
                    )
                    pooled_vision_states = self.hidden_state_pooling.forward_vectorized(
                        hidden_states,
                        img_indices,
                        total_pic_num
                    )  # [Total_Images, Dim]
                    images_per_sample_tensor = torch.tensor(images_per_sample, device=hidden_states.device)
                    expanded_text_embedding = torch.repeat_interleave(
                        embedding_after_pooling,
                        images_per_sample_tensor,
                        dim=0
                    )
                    if self.ablate_visual_gate:
                        self.alpha_list = torch.ones(
                            (total_pic_num, 1),
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                        self.G_list = torch.ones(
                            (total_pic_num, 1),
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                        run_mmrl_branch = True
                    else:
                        self.alpha_list = self.Task_classifier(pooled_vision_states, expanded_text_embedding)
                        if self.direct_mmrl_output:
                            self.route_probs = None
                        else:
                            self.route_probs = self.adapter_router(
                                pooled_vision_states,
                                expanded_text_embedding,
                                self.alpha_list,
                            ).to(dtype=hidden_states.dtype)
                        # G_list: [Total_Images, 1]
                        if self.training:
                            # 训练时保留 soft gate，保证梯度可过
                            self.G_list = self.visionGating(
                                self.alpha_list,
                                gating_temperature_override
                            ).to(dtype=hidden_states.dtype)
                        else:
                            raw_g = self.visionGating(
                                self.alpha_list,
                                gating_temperature_override
                            )
                            self.G_list = (raw_g > 0.5).to(dtype=hidden_states.dtype)
                            if self.G_list.sum() == 0:
                                run_mmrl_branch = False

                ############ N图切分+门控 ############
                idx = self.insert_layers.index(layer_num)
                forward_hit_layers.append(layer_num)
                assert v_r_token_list[idx] is not None
                r_tokens_input = v_r_token_list[idx]
                assert r_tokens_input is not None

                org_hidden_states = blk(hidden_states if first_insert else org_hidden_states,
                                        cu_seqlens=cu_seqlens,
                                        position_embeddings=position_embeddings,
                                        rotary_pos_emb=rotary_pos_emb,
                                        **kwargs)
                hidden_states = org_hidden_states
                if self.training or run_mmrl_branch:
                    if first_insert:
                        hidden_states_with_rep = self.blocks_with_rep[idx](
                            org_input_states,
                            cu_seqlens=cu_seqlens,
                            position_embeddings=position_embeddings,
                            r_token=r_tokens_input,
                            rotary_pos_emb=rotary_pos_emb,
                            first_insert=True,
                            **kwargs
                        )
                        position_embeddings_with_rep, cu_seqlens_with_rep, rotary_pos_emb_with_rep = \
                            _resize_cu_and_pos(r_tokens_input,
                                               cu_seqlens,
                                               position_embeddings,
                                               rotary_pos_emb)
                    else:
                        assert cu_seqlens_with_rep is not None
                        assert position_embeddings_with_rep is not None
                        hidden_states_with_rep = self.blocks_with_rep[idx](
                            hidden_states_with_rep,
                            cu_seqlens=cu_seqlens_with_rep,
                            position_embeddings=position_embeddings_with_rep,
                            r_token=r_tokens_input,
                            rotary_pos_emb=rotary_pos_emb_with_rep,
                            first_insert=False,
                            **kwargs
                        )
                    first_insert = False
                else:
                    pass
                # 规定这里输出的都是没移除rep的
                if first_insert:
                    first_insert = False    

            if layer_num in self.deepstack_visual_indexes:
                if layer_num not in self.insert_layers:
                    feature_to_save = hidden_states
                else:
                    feature_to_save = org_hidden_states
                    if (
                        self.enable_deepstack_mmrl_residual
                        and self.deepstack_mmrl_residual_scale != 0.0
                        and (self.training or run_mmrl_branch)
                        and hidden_states_with_rep is not None
                        and isinstance(self.G_list, torch.Tensor)
                    ):
                        deepstack_rep_states = _strip_r_token(
                            hidden_states_with_rep,
                            original_seq_lens_list,
                            current_num_r_token,
                        )
                        local_delta = deepstack_rep_states - org_hidden_states
                        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                        g_mask = torch.repeat_interleave(self.G_list, seqlens, dim=0).to(
                            device=local_delta.device,
                            dtype=local_delta.dtype,
                        )
                        local_delta = local_delta * g_mask
                        feature_to_save = org_hidden_states + local_delta * float(self.deepstack_mmrl_residual_scale)
                        with torch.no_grad():
                            local_delta_norm = local_delta.detach().float().norm(dim=-1).mean()
                            local_org_norm = org_hidden_states.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
                            deepstack_delta_norm_values.append(local_delta_norm)
                            deepstack_delta_ratio_values.append(local_delta_norm / local_org_norm)
                            deepstack_residual_layer_count += 1
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](feature_to_save)
                deepstack_feature_lists.append(deepstack_feature)

        if not self._printed_forward_layer_hit_audit:
            print("[MMRL_FORWARD_HIT_AUDIT]")
            print(f"  actual_hit_layers_0based={forward_hit_layers}")
            print(f"  actual_hit_layers_natural={[idx + 1 for idx in forward_hit_layers]}")
            print(f"  expected_insert_layers_0based={self.insert_layers if v_r_token_list is not None else []}")
            print(f"  expected_insert_layers_natural={[idx + 1 for idx in self.insert_layers] if v_r_token_list is not None else []}")
            self._printed_forward_layer_hit_audit = True

        if run_mmrl_branch and hidden_states_with_rep is not None:
            hidden_states_with_rep = _strip_r_token(hidden_states_with_rep,
                                                    original_seq_lens_list,
                                                    current_num_r_token)
            # for i in range(cu_seqlens.size(0) - 1):
            #     hidden_states_with_rep = hidden_states_with_rep[cu_seqlens[i]:cu_seqlens[i+1]] * self.G_list[i]
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            delta = hidden_states_with_rep - org_hidden_states
            G_mask = torch.repeat_interleave(self.G_list, seqlens, dim=0)
            gated_delta = delta * G_mask
            if self.direct_mmrl_output:
                adapter_outputs = None
                final_delta = gated_delta
                hidden_states = org_hidden_states + final_delta
                self._prepare_route_utility_metrics(hidden_states)
            else:
                if torch.is_tensor(self.route_probs) and self.route_probs.numel() > 0:
                    token_route_probs = torch.repeat_interleave(self.route_probs, seqlens, dim=0)
                else:
                    token_route_probs = gated_delta.new_ones(
                        (gated_delta.shape[0], self.visual_residual_adapter_count),
                        dtype=gated_delta.dtype,
                    ) / float(self.visual_residual_adapter_count)
                adapter_outputs = torch.stack(
                    [adapter(delta) for adapter in self.residual_adapters],
                    dim=1,
                )
                routed_delta = (adapter_outputs * token_route_probs.unsqueeze(-1)).sum(dim=1)
                if self.enable_adapter_router_identity_residual:
                    routed_delta = delta + routed_delta
                # Gate the complete residual so adapter biases cannot bypass G=0.
                final_delta = routed_delta * G_mask
                hidden_states = org_hidden_states + final_delta
                self._register_route_utility_monitor(
                    hidden_states,
                    adapter_outputs,
                    final_delta,
                    delta,
                    G_mask,
                    cu_seqlens,
                )
        else:
            hidden_states = org_hidden_states
            gated_delta = torch.zeros_like(hidden_states)
            final_delta = torch.zeros_like(hidden_states)
            adapter_outputs = None
            self._prepare_route_utility_metrics(hidden_states)

        (
            self.adapter_usage_balance_loss,
            self.adapter_sample_entropy_loss,
            self.adapter_common_mode_loss,
            self.adapter_effective_delta_loss,
            self.prototype_anchor_loss,
            self.adapter_diversity_loss,
        ) = self._compute_router_aux_losses(
            final_delta,
            cu_seqlens,
            org_hidden_states=org_hidden_states,
            pooled_vision_states=pooled_vision_states,
            adapter_outputs=adapter_outputs,
        )

        if deepstack_delta_norm_values:
            self.deepstack_delta_norm_mean = torch.stack(deepstack_delta_norm_values).mean().to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self.deepstack_delta_to_org_ratio = torch.stack(deepstack_delta_ratio_values).mean().to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            self.deepstack_delta_norm_mean = hidden_states.new_tensor(float("nan"))
            self.deepstack_delta_to_org_ratio = hidden_states.new_tensor(float("nan"))
        self.deepstack_residual_layers = hidden_states.new_tensor(float(deepstack_residual_layer_count))

        org_hidden_norm = org_hidden_states.detach().float().norm(dim=-1).mean()
        final_delta_norm = final_delta.detach().float().norm(dim=-1).mean()
        delta_to_org_ratio = final_delta_norm / org_hidden_norm.clamp_min(1e-8)
        alpha_mean = torch.sigmoid(self.alpha_list.detach().float()).mean() if self.alpha_list is not None else hidden_states.new_tensor(0.0)
        alpha_std = torch.sigmoid(self.alpha_list.detach().float()).std(unbiased=False) if self.alpha_list is not None and self.alpha_list.numel() > 1 else hidden_states.new_tensor(0.0)
        g_mean = self.G_list.detach().float().mean() if isinstance(self.G_list, torch.Tensor) else hidden_states.new_tensor(0.0)
        g_std = self.G_list.detach().float().std(unbiased=False) if isinstance(self.G_list, torch.Tensor) and self.G_list.numel() > 1 else hidden_states.new_tensor(0.0)
        residual_debug = self._compute_residual_debug_metrics(
            final_delta,
            gated_delta,
            org_hidden_states,
            cu_seqlens,
            route_probs=self.route_probs,
        )
        self.debug_context = {
            "alpha_prob_mean": alpha_mean,
            "alpha_prob_std": alpha_std,
            "G_mean": g_mean,
            "G_std": g_std,
            "final_delta_norm_mean": final_delta_norm,
            "org_hidden_norm_mean": org_hidden_norm,
            "delta_to_org_ratio": delta_to_org_ratio,
            "visual_adapter_count": hidden_states.new_tensor(float(self.visual_residual_adapter_count)),
            "adapter_weight_norm": self._adapter_weight_norm(hidden_states.device),
            "adapter_usage_balance_loss": self.adapter_usage_balance_loss.detach(),
            "adapter_sample_entropy_loss": self.adapter_sample_entropy_loss.detach(),
            "adapter_common_mode_loss": self.adapter_common_mode_loss.detach(),
            "adapter_effective_delta_loss": self.adapter_effective_delta_loss.detach(),
            "prototype_anchor_loss": self.prototype_anchor_loss.detach(),
            "adapter_diversity_loss": self.adapter_diversity_loss.detach(),
            "adapter_effective_delta_ratio_mean": self.adapter_effective_delta_ratio_mean.detach(),
            "adapter_effective_delta_ratio_min": self.adapter_effective_delta_ratio_min.detach(),
            "adapter_effective_delta_ratio_max": self.adapter_effective_delta_ratio_max.detach(),
            "route_proto_kl": self.route_proto_kl.detach(),
            "route_proto_agreement": self.route_proto_agreement.detach(),
            "prototype_usage_max": self.prototype_usage_max.detach(),
            "prototype_usage_min": self.prototype_usage_min.detach(),
            "adapter_pairwise_cos_mean": self.adapter_pairwise_cos_mean.detach(),
            "adapter_pairwise_cos_max": self.adapter_pairwise_cos_max.detach(),
            "adapter_pairwise_cos_below_band": self.adapter_pairwise_cos_below_band.detach(),
            "adapter_pairwise_cos_above_band": self.adapter_pairwise_cos_above_band.detach(),
            "adapter_diversity_mean_component": self.adapter_diversity_mean_component.detach(),
            "adapter_diversity_worst_component": self.adapter_diversity_worst_component.detach(),
            "deepstack_mmrl_residual_scale": hidden_states.new_tensor(float(self.deepstack_mmrl_residual_scale)),
            "deepstack_delta_norm_mean": self.deepstack_delta_norm_mean.detach(),
            "deepstack_delta_to_org_ratio": self.deepstack_delta_to_org_ratio.detach(),
            "deepstack_residual_layers": self.deepstack_residual_layers.detach(),
            **residual_debug,
            **self._route_utility_step_metrics,
        }
        for idx in range(self.visual_residual_adapter_count):
            value = getattr(
                self,
                f"prototype_usage_{idx}",
                hidden_states.new_tensor(float("nan")),
            )
            self.debug_context[f"prototype_usage_{idx}"] = value.detach() if torch.is_tensor(value) else hidden_states.new_tensor(float(value))
            for prefix in ("adapter_output_norm", "adapter_contribution_norm"):
                value = getattr(
                    self,
                    f"{prefix}_{idx}",
                    hidden_states.new_tensor(float("nan")),
                )
                self.debug_context[f"{prefix}_{idx}"] = value.detach()
            for other_idx in range(idx + 1, self.visual_residual_adapter_count):
                key = f"adapter_pairwise_cos_{idx}_{other_idx}"
                value = getattr(self, key, hidden_states.new_tensor(float("nan")))
                self.debug_context[key] = value.detach()

        hidden_states = self.merger(hidden_states)
        k_results = None
        self.k_results = None
        self.k_mask_results = None
        self.tax_loss = hidden_states.new_tensor(0.0)
        self.capacity_prior_loss = self.tax_loss
        return hidden_states, deepstack_feature_lists, k_results
