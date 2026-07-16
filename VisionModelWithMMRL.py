from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

import MMRLGating
import utils

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
        self.mmrl_residual_guard_loss = torch.tensor(0.0)
        self.mmrl_semantic_anchor_loss = torch.tensor(0.0)
        self.mmrl_semantic_cos_mean = torch.tensor(float("nan"))
        self.mmrl_semantic_cos_p05 = torch.tensor(float("nan"))
        self.adapter_sample_entropy_target = float(getattr(self.cfg, "ADAPTER_SAMPLE_ENTROPY_TARGET", 0.55))
        self.mmrl_residual_ratio_upper = float(getattr(self.cfg, "MMRL_RESIDUAL_RATIO_UPPER", 0.20))
        self.enable_router_kmeans_prior = bool(
            getattr(self.cfg, "ENABLE_ROUTER_KMEANS_PRIOR", False)
        )
        self.router_prior_temperature = float(getattr(self.cfg, "ROUTER_PRIOR_TEMPERATURE", 0.25))
        self.router_residual_start_fraction = float(getattr(self.cfg, "ROUTER_RESIDUAL_START_FRACTION", 0.30))
        self.router_residual_max_scale = float(getattr(self.cfg, "ROUTER_RESIDUAL_MAX_SCALE", 0.50))
        self.router_residual_logit_bound = float(getattr(self.cfg, "ROUTER_RESIDUAL_LOGIT_BOUND", 1.0))
        self.register_buffer(
            "router_prior_centroids",
            torch.zeros(self.visual_residual_adapter_count, self.cfg.vision_token_dim),
        )
        self.register_buffer(
            "router_prior_bias",
            torch.zeros(self.visual_residual_adapter_count),
        )
        self.register_buffer(
            "router_prior_temperature_value",
            torch.tensor(self.router_prior_temperature, dtype=torch.float32),
        )
        self.register_buffer("router_prior_ready", torch.tensor(False, dtype=torch.bool))
        self.alpha_list = []
        self.G_list = []
        self.route_probs = None
        self.router_prior_probs = None
        self.router_prior_logits = None
        self.router_residual_logits = None
        self.router_residual_scale_current = 0.0
        self.ablate_visual_gate = bool(getattr(self.cfg, "ABLATE_VISUAL_GATE", False))
        # Defaults preserve the complete trained path when a saved model is loaded for inference.
        self.current_stage_id = 4
        self.current_stage_progress = 1.0
        self.mmrl_warmup_fraction = 1.0 / 3.0
        self.experts_enabled = True
        self.debug_context = {}
        self._printed_stage_path_audits = set()

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

    def _router_features_from_final_tokens(
        self,
        token_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # The route prior must only see the untouched final vision branch.
        normalized_tokens = F.layer_norm(
            token_states.detach().float(),
            (token_states.shape[-1],),
        )
        pooled = self._pool_tokens_by_image_mean(normalized_tokens, cu_seqlens)
        if pooled is None:
            return token_states.new_empty((0, token_states.shape[-1]))
        pooled = F.normalize(pooled, dim=-1, eps=1e-8)
        return pooled.to(device=token_states.device, dtype=token_states.dtype)

    @torch.no_grad()
    def extract_router_calibration_features(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the untouched vision branch and return final per-image features."""
        hidden_states = self.patch_embed(hidden_states.type(self.dtype))
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                rotary_pos_emb=rotary_pos_emb,
            )
        return self._router_features_from_final_tokens(hidden_states, cu_seqlens)

    @torch.no_grad()
    def set_router_prior(
        self,
        centroids: torch.Tensor,
        bias: torch.Tensor,
        temperature: float,
    ):
        expected = (self.visual_residual_adapter_count, self.cfg.vision_token_dim)
        if tuple(centroids.shape) != expected:
            raise ValueError(
                f"router centroids must have shape {expected}, got {tuple(centroids.shape)}"
            )
        if tuple(bias.shape) != (self.visual_residual_adapter_count,):
            raise ValueError(
                "router prior bias must have shape "
                f"({self.visual_residual_adapter_count},), got {tuple(bias.shape)}"
            )
        self.router_prior_centroids.copy_(
            F.normalize(centroids.float(), dim=-1).to(self.router_prior_centroids)
        )
        self.router_prior_bias.copy_(bias.float().to(self.router_prior_bias))
        self.router_prior_temperature_value.fill_(max(float(temperature), 1e-6))
        self.router_prior_ready.fill_(True)

    def _router_residual_scale(self) -> float:
        if int(self.current_stage_id) < 4:
            return 0.0
        start = min(max(self.router_residual_start_fraction, 0.0), 0.99)
        progress = min(max(float(self.current_stage_progress), 0.0), 1.0)
        ramp = max(progress - start, 0.0) / max(1.0 - start, 1e-8)
        return float(self.router_residual_max_scale) * ramp

    def _compute_semantic_route_probs(
        self,
        router_vision_states: torch.Tensor,
        expanded_text_embedding: torch.Tensor,
        alpha_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.enable_router_kmeans_prior:
            return self.adapter_router(
                router_vision_states,
                expanded_text_embedding,
                alpha_logits,
            )

        if not bool(self.router_prior_ready.item()):
            raise RuntimeError(
                "router prior is not calibrated; Stage2 must run before Stage4"
            )
        if router_vision_states.shape[0] != expanded_text_embedding.shape[0]:
            raise ValueError(
                "router vision/text batch mismatch: "
                f"{router_vision_states.shape[0]} != {expanded_text_embedding.shape[0]}"
            )

        vision_f = router_vision_states.float()
        centroids = F.normalize(self.router_prior_centroids.float(), dim=-1)
        temperature = max(float(self.router_prior_temperature_value.item()), 1e-6)
        prior_logits = vision_f @ centroids.transpose(0, 1)
        prior_logits = prior_logits / temperature + self.router_prior_bias.float()

        raw_residual = self.adapter_router.compute_logits(
            router_vision_states,
            expanded_text_embedding,
            alpha_logits,
        ).float()
        raw_residual = raw_residual - raw_residual.mean(dim=-1, keepdim=True)
        bound = max(self.router_residual_logit_bound, 1e-6)
        bounded_residual = bound * torch.tanh(raw_residual / bound)
        residual_scale = self._router_residual_scale()
        final_logits = prior_logits + residual_scale * bounded_residual

        self.router_prior_logits = prior_logits.detach()
        self.router_prior_probs = torch.softmax(prior_logits, dim=-1).detach()
        self.router_residual_logits = bounded_residual.detach()
        self.router_residual_scale_current = residual_scale
        return torch.softmax(final_logits, dim=-1).to(dtype=router_vision_states.dtype)

    def _shared_strength(self):
        if int(self.current_stage_id) == 3:
            warmup = float(self.mmrl_warmup_fraction)
            if warmup <= 0.0:
                return 1.0
            return max(0.0, min(float(self.current_stage_progress) / warmup, 1.0))
        if int(self.current_stage_id) >= 4:
            return 1.0
        return 0.0

    def _bound_expert_delta(self, expert_delta, org_hidden_states, g_mask, cu_seqlens):
        bounded_chunks = []
        pre_cap_ratios = []
        cap_scales = []
        upper = float(self.mmrl_residual_ratio_upper)
        if upper <= 0.0:
            self.mmrl_residual_guard_loss = expert_delta.new_tensor(0.0)
            return expert_delta, expert_delta.new_empty(0), expert_delta.new_empty(0)

        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            start, end = int(start), int(end)
            if end <= start:
                continue
            delta_chunk = expert_delta[start:end]
            gated_chunk = delta_chunk * g_mask[start:end]
            org_chunk = org_hidden_states[start:end].detach()
            delta_rms = gated_chunk.float().square().mean().sqrt()
            org_rms = org_chunk.float().square().mean().sqrt().clamp_min(1e-8)
            ratio = delta_rms / org_rms
            scale = torch.clamp(
                ratio.new_tensor(upper) / ratio.clamp_min(1e-8),
                max=1.0,
            )
            bounded_chunks.append(delta_chunk * scale.to(dtype=delta_chunk.dtype))
            pre_cap_ratios.append(ratio)
            cap_scales.append(scale)

        if not bounded_chunks:
            zero = expert_delta.new_tensor(0.0)
            self.mmrl_residual_guard_loss = zero
            return expert_delta, expert_delta.new_empty(0), expert_delta.new_empty(0)

        ratios = torch.stack(pre_cap_ratios)
        scales = torch.stack(cap_scales)
        self.mmrl_residual_guard_loss = torch.relu(
            ratios - ratios.new_tensor(upper)
        ).pow(2).mean().to(dtype=expert_delta.dtype)
        return torch.cat(bounded_chunks, dim=0), ratios, scales

    def _compute_mmrl_semantic_anchor(self, prompted_states, original_states):
        prompted_norm = F.layer_norm(
            prompted_states.float(),
            (prompted_states.shape[-1],),
        )
        original_norm = F.layer_norm(
            original_states.detach().float(),
            (original_states.shape[-1],),
        )
        token_cos = F.cosine_similarity(
            prompted_norm,
            original_norm,
            dim=-1,
            eps=1e-6,
        ).clamp(min=-1.0, max=1.0)
        token_drift = 1.0 - token_cos
        semantic_anchor_loss = torch.sqrt(
            token_drift.square().mean() + 1e-8
        )
        self.mmrl_semantic_anchor_loss = (
            semantic_anchor_loss
            if int(self.current_stage_id) in (3, 4)
            else semantic_anchor_loss.detach()
        )
        with torch.no_grad():
            detached_cos = token_cos.detach()
            self.mmrl_semantic_cos_mean = detached_cos.mean()
            self.mmrl_semantic_cos_p05 = torch.quantile(detached_cos, 0.05)

    def _compute_router_aux_losses(self, reference_tensor):
        zero = reference_tensor.new_tensor(0.0)
        self.adapter_usage_balance_loss = zero
        self.adapter_sample_entropy_loss = zero
        route_probs = self.route_probs
        if not self.experts_enabled or not torch.is_tensor(route_probs) or route_probs.numel() == 0:
            return

        routes = route_probs.float()
        usage = routes.mean(dim=0)
        usage = usage / usage.sum().clamp_min(1e-8)
        uniform = torch.full_like(usage, 1.0 / routes.shape[-1])
        self.adapter_usage_balance_loss = (
            usage * (usage.clamp_min(1e-8).log() - uniform.log())
        ).sum().to(dtype=reference_tensor.dtype)

        if routes.shape[-1] > 1:
            entropy = -(routes * routes.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / torch.log(routes.new_tensor(float(routes.shape[-1])))
            self.adapter_sample_entropy_loss = torch.relu(
                routes.new_tensor(self.adapter_sample_entropy_target) - entropy
            ).pow(2).mean().to(dtype=reference_tensor.dtype)

    def _compute_debug_metrics(
        self,
        expert_delta,
        gated_expert_delta,
        final_delta,
        org_hidden_states,
        adapter_outputs,
        cu_seqlens,
        g_mask,
        pre_cap_ratios,
        cap_scales,
    ):
        device = org_hidden_states.device
        nan = torch.tensor(float("nan"), device=device)
        metrics = {
            "expert_branch_strength": torch.tensor(self._shared_strength(), device=device),
            "G_mean": nan,
            "expert_delta_raw_to_org_ratio": nan,
            "expert_delta_pre_cap_to_org_ratio": nan,
            "expert_delta_cap_scale_mean": nan,
            "expert_delta_cap_active_fraction": nan,
            "expert_delta_to_org_ratio": nan,
            "expert_delta_common_mode_ratio": nan,
            "expert_delta_specificity_ratio": nan,
            "expert_delta_token_specificity_ratio": nan,
            "final_delta_to_org_ratio": nan,
            "adapter_pairwise_cos_mean": nan,
            "adapter_route_entropy_norm": nan,
            "adapter_usage_max": nan,
            "adapter_usage_min": nan,
            "router_prior_entropy_norm": nan,
            "router_prior_usage_max": nan,
            "router_prior_usage_min": nan,
            "router_prior_margin": nan,
            "router_residual_scale": torch.tensor(
                float(self.router_residual_scale_current), device=device
            ),
            "router_residual_to_prior_ratio": nan,
            "route_prior_kl": nan,
        }

        with torch.no_grad():
            raw = expert_delta.detach().float()
            gated_expert_delta_f = gated_expert_delta.detach().float()
            final_delta_f = final_delta.detach().float()
            org = org_hidden_states.detach().float()
            metrics["G_mean"] = g_mask.detach().float().mean()
            if pre_cap_ratios.numel() > 0:
                ratios = pre_cap_ratios.detach().float()
                scales = cap_scales.detach().float()
                metrics["expert_delta_pre_cap_to_org_ratio"] = ratios.mean()
                metrics["expert_delta_cap_scale_mean"] = scales.mean()
                metrics["expert_delta_cap_active_fraction"] = (scales < 1.0).float().mean()

            org_norm = org.norm(dim=-1).mean().clamp_min(1e-8)
            metrics["expert_delta_raw_to_org_ratio"] = raw.norm(dim=-1).mean() / org_norm
            metrics["expert_delta_to_org_ratio"] = (
                gated_expert_delta_f.norm(dim=-1).mean() / org_norm
            )
            metrics["final_delta_to_org_ratio"] = final_delta_f.norm(dim=-1).mean() / org_norm

            pooled_expert_delta = self._pool_tokens_by_image_mean(gated_expert_delta_f, cu_seqlens)
            if pooled_expert_delta is not None:
                pooled_norm = pooled_expert_delta.norm(dim=-1)
                norm_mean = pooled_norm.mean().clamp_min(1e-8)
                common = pooled_expert_delta.mean(dim=0)
                specific = pooled_expert_delta - common
                metrics["expert_delta_common_mode_ratio"] = common.norm() / norm_mean
                metrics["expert_delta_specificity_ratio"] = specific.norm(dim=-1).mean() / norm_mean

                token_specific = []
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                    start, end = int(start), int(end)
                    if end > start:
                        image_tokens = gated_expert_delta_f[start:end]
                        token_specific.append(image_tokens - image_tokens.mean(dim=0, keepdim=True))
                if token_specific:
                    token_specific = torch.cat(token_specific, dim=0)
                    metrics["expert_delta_token_specificity_ratio"] = (
                        token_specific.norm(dim=-1).mean()
                        / gated_expert_delta_f.norm(dim=-1).mean().clamp_min(1e-8)
                    )

            if adapter_outputs is not None and adapter_outputs.numel() > 0:
                pooled_adapter = []
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                    start, end = int(start), int(end)
                    if end > start:
                        pooled_adapter.append(adapter_outputs[start:end].float().mean(dim=0))
                if pooled_adapter:
                    pooled_adapter = torch.stack(pooled_adapter, dim=0)
                    pairwise_values = []
                    for left in range(self.visual_residual_adapter_count):
                        for right in range(left + 1, self.visual_residual_adapter_count):
                            valid = (
                                (pooled_adapter[:, left].norm(dim=-1) > 1e-8)
                                & (pooled_adapter[:, right].norm(dim=-1) > 1e-8)
                            )
                            if valid.any():
                                pairwise_values.append(F.cosine_similarity(
                                    pooled_adapter[valid, left], pooled_adapter[valid, right], dim=-1
                                ))
                    if pairwise_values:
                        metrics["adapter_pairwise_cos_mean"] = torch.cat(pairwise_values).mean()

            routes = self.route_probs
            if torch.is_tensor(routes) and routes.numel() > 0:
                routes = routes.detach().float()
                usage = routes.mean(dim=0)
                usage = usage / usage.sum().clamp_min(1e-8)
                entropy = -(routes * routes.clamp_min(1e-8).log()).sum(dim=-1)
                if routes.shape[-1] > 1:
                    entropy = entropy / torch.log(routes.new_tensor(float(routes.shape[-1])))
                metrics["adapter_route_entropy_norm"] = entropy.mean()
                metrics["adapter_usage_max"] = usage.max()
                metrics["adapter_usage_min"] = usage.min()

                prior = self.router_prior_probs
                if torch.is_tensor(prior) and prior.shape == routes.shape:
                    prior = prior.detach().float()
                    prior_usage = prior.mean(dim=0)
                    prior_usage = prior_usage / prior_usage.sum().clamp_min(1e-8)
                    prior_entropy = -(
                        prior * prior.clamp_min(1e-8).log()
                    ).sum(dim=-1)
                    if prior.shape[-1] > 1:
                        prior_entropy = prior_entropy / torch.log(
                            prior.new_tensor(float(prior.shape[-1]))
                        )
                    metrics["router_prior_entropy_norm"] = prior_entropy.mean()
                    metrics["router_prior_usage_max"] = prior_usage.max()
                    metrics["router_prior_usage_min"] = prior_usage.min()
                    top2 = prior.topk(k=min(2, prior.shape[-1]), dim=-1).values
                    if top2.shape[-1] == 2:
                        metrics["router_prior_margin"] = (
                            top2[:, 0] - top2[:, 1]
                        ).mean()
                    metrics["route_prior_kl"] = (
                        prior
                        * (
                            prior.clamp_min(1e-8).log()
                            - routes.clamp_min(1e-8).log()
                        )
                    ).sum(dim=-1).mean()

                prior_logits = self.router_prior_logits
                residual_logits = self.router_residual_logits
                if (
                    torch.is_tensor(prior_logits)
                    and torch.is_tensor(residual_logits)
                    and prior_logits.shape == residual_logits.shape
                ):
                    prior_centered = prior_logits.float() - prior_logits.float().mean(
                        dim=-1, keepdim=True
                    )
                    residual_effective = (
                        residual_logits.float()
                        * float(self.router_residual_scale_current)
                    )
                    metrics["router_residual_to_prior_ratio"] = (
                        residual_effective.norm(dim=-1)
                        / prior_centered.norm(dim=-1).clamp_min(1e-8)
                    ).mean()

        return metrics


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
        self.alpha_list = None 
        self.G_list = []
        self.route_probs = None
        self.router_prior_probs = None
        self.router_prior_logits = None
        self.router_residual_logits = None
        self.router_residual_scale_current = 0.0
        pooled_vision_states = None
        expanded_text_embedding = None


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
                        # G_list: [Total_Images, 1]
                        if self.training:
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

        if self.experts_enabled:
            if expanded_text_embedding is None:
                raise RuntimeError("router text features were not built at the first insert layer")
            router_vision_states = self._router_features_from_final_tokens(
                org_hidden_states,
                cu_seqlens,
            )
            self.route_probs = self._compute_semantic_route_probs(
                router_vision_states,
                expanded_text_embedding,
                self.alpha_list,
            ).to(dtype=hidden_states.dtype)

        if run_mmrl_branch and hidden_states_with_rep is not None:
            hidden_states_with_rep = _strip_r_token(hidden_states_with_rep,
                                                    original_seq_lens_list,
                                                    current_num_r_token)
            self._compute_mmrl_semantic_anchor(
                hidden_states_with_rep,
                org_hidden_states,
            )
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            expert_delta = hidden_states_with_rep - org_hidden_states
            g_mask = torch.repeat_interleave(self.G_list, seqlens, dim=0).to(
                device=expert_delta.device,
                dtype=expert_delta.dtype,
            )
            bounded_expert_delta, pre_cap_ratios, cap_scales = self._bound_expert_delta(
                expert_delta,
                org_hidden_states,
                g_mask,
                cu_seqlens,
            )

            if self.experts_enabled:
                token_route_probs = torch.repeat_interleave(self.route_probs, seqlens, dim=0)
                adapter_outputs = torch.stack(
                    [adapter(bounded_expert_delta) for adapter in self.residual_adapters],
                    dim=1,
                )
                adapter_correction = (
                    adapter_outputs * token_route_probs.unsqueeze(-1)
                ).sum(dim=1)
            else:
                adapter_outputs = None
                adapter_correction = torch.zeros_like(bounded_expert_delta)

            expert_delta_after_adapter = bounded_expert_delta + adapter_correction
            branch_scale = g_mask * float(self._shared_strength())
            final_delta = expert_delta_after_adapter * branch_scale
            gated_expert_delta = bounded_expert_delta * branch_scale
            hidden_states = org_hidden_states + final_delta
        else:
            hidden_states = org_hidden_states
            expert_delta = torch.zeros_like(hidden_states)
            gated_expert_delta = torch.zeros_like(hidden_states)
            final_delta = torch.zeros_like(hidden_states)
            adapter_outputs = None
            g_mask = torch.zeros((hidden_states.shape[0], 1), device=hidden_states.device, dtype=hidden_states.dtype)
            pre_cap_ratios = hidden_states.new_empty(0)
            cap_scales = hidden_states.new_empty(0)
            self.mmrl_residual_guard_loss = hidden_states.new_tensor(0.0)
            self.mmrl_semantic_anchor_loss = hidden_states.new_tensor(0.0)
            self.mmrl_semantic_cos_mean = hidden_states.new_tensor(float("nan"))
            self.mmrl_semantic_cos_p05 = hidden_states.new_tensor(float("nan"))

        self._compute_router_aux_losses(hidden_states)
        stage_audit_key = (int(self.current_stage_id), bool(self.experts_enabled))
        if stage_audit_key not in self._printed_stage_path_audits:
            print(
                "[MMRL_STAGE_PATH_AUDIT] "
                f"stage={self.current_stage_id} "
                f"expert_branch_strength={self._shared_strength():.6f} "
                f"experts_enabled={self.experts_enabled} "
                f"router_active={torch.is_tensor(self.route_probs)} "
                f"router_prior_ready={bool(self.router_prior_ready.item())} "
                f"router_residual_scale={self.router_residual_scale_current:.6f}"
            )
            self._printed_stage_path_audits.add(stage_audit_key)

        residual_debug = self._compute_debug_metrics(
            expert_delta,
            gated_expert_delta,
            final_delta,
            org_hidden_states,
            adapter_outputs,
            cu_seqlens,
            g_mask,
            pre_cap_ratios,
            cap_scales,
        )
        self.debug_context = {
            "mmrl_residual_guard_loss": self.mmrl_residual_guard_loss.detach(),
            "mmrl_semantic_anchor_loss": self.mmrl_semantic_anchor_loss.detach(),
            "mmrl_semantic_cos_mean": self.mmrl_semantic_cos_mean.detach(),
            "mmrl_semantic_cos_p05": self.mmrl_semantic_cos_p05.detach(),
            "adapter_usage_balance_loss": self.adapter_usage_balance_loss.detach(),
            "adapter_sample_entropy_loss": self.adapter_sample_entropy_loss.detach(),
            **residual_debug,
        }

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists
