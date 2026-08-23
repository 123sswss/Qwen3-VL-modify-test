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
        if r_token is None or r_token.ndim != 3:
            raise ValueError(
                "MMRLVitBlock expects per-image r_token=[N,R,D], got "
                f"{None if r_token is None else tuple(r_token.shape)}"
            )
        num_r_tokens = r_token.shape[1]
        full_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
        if r_token.shape[0] != len(full_seq_lengths):
            raise ValueError(
                "per-image Rep batch does not match visual sequences: "
                f"rep_images={r_token.shape[0]} visual_images={len(full_seq_lengths)}"
            )
        if first_insert:
            hidden_states_split = torch.split(hidden_states, full_seq_lengths, dim=0)
            new_hidden_states_list = [
                torch.cat([image_rep, image_states], dim=0)
                for image_rep, image_states in zip(r_token, hidden_states_split)
            ]
            hidden_states = torch.cat(new_hidden_states_list, dim=0)
            position_embeddings, cu_seqlens, rotary_pos_emb = _resize_cu_and_pos(
                r_token, cu_seqlens, position_embeddings, rotary_pos_emb
            )
        else:
            hidden_states_split = torch.split(hidden_states, full_seq_lengths, dim=0)
            new_hidden_states_list = []
            for image_rep, seq_hidden in zip(r_token, hidden_states_split):
                if self.INSERT_METHOD == "replace":
                    updated_seq = torch.cat(
                        [image_rep, seq_hidden[num_r_tokens:]],
                        dim=0,
                    )
                elif self.INSERT_METHOD == "add":
                    prefix = seq_hidden[:num_r_tokens] + image_rep
                    suffix = seq_hidden[num_r_tokens:]
                    updated_seq = torch.cat([prefix, suffix], dim=0)
                else:
                    raise ValueError(
                        f"Unsupported MMRL INSERT_METHOD={self.INSERT_METHOD!r}"
                    )
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
    if r_token.ndim != 3:
        raise ValueError(
            f"position resize expects per-image Rep [N,R,D], got {tuple(r_token.shape)}"
        )
    num_r_tokens = r_token.shape[1]
    total_pic_num = cu_seqlens.shape[0] - 1
    if r_token.shape[0] != total_pic_num:
        raise ValueError(
            "position resize image count mismatch: "
            f"rep_images={r_token.shape[0]} visual_images={total_pic_num}"
        )
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
        self.mmrl_relation_loss = torch.tensor(0.0)
        self.mmrl_relation_gram_loss = torch.tensor(0.0)
        self.mmrl_variance_floor_loss = torch.tensor(0.0)
        self.mmrl_cross_relation_loss = torch.tensor(0.0)
        self.mmrl_cross_relation_loss_weight = float(
            getattr(self.cfg, "MMRL_CROSS_RELATION_LOSS_WEIGHT", 0.0)
        )
        self.mmrl_relation_max_tokens = int(getattr(self.cfg, "MMRL_RELATION_MAX_TOKENS", 64))
        self.mmrl_variance_floor_ratio = float(getattr(self.cfg, "MMRL_VARIANCE_FLOOR_RATIO", 0.50))
        self.mmrl_variance_floor_weight = float(getattr(self.cfg, "MMRL_VARIANCE_FLOOR_WEIGHT", 0.10))
        self.current_stage_id = 0
        self.current_stage_step = 0
        self.enable_deepstack_mmrl_residual = bool(getattr(self.cfg, "ENABLE_DEEPSTACK_MMRL_RESIDUAL", False))
        self.deepstack_mmrl_residual_scale = float(getattr(self.cfg, "DEEPSTACK_MMRL_RESIDUAL_SCALE", 0.0))
        self.mmrl_delta_to_org_ratio = torch.tensor(float("nan"))
        self.deepstack_delta_norm_mean = torch.tensor(float("nan"))
        self.deepstack_delta_to_org_ratio = torch.tensor(float("nan"))
        self.deepstack_residual_layers = torch.tensor(0.0)
        self.alpha_list = []
        self.G_list = []


        self.null_image_token = nn.Parameter(torch.zeros(1, self.cfg.vision_token_dim))
        nn.init.normal_(self.null_image_token, std=0.02)
        self.ablate_visual_gate = bool(getattr(self.cfg, "ABLATE_VISUAL_GATE", False))
        self.use_alpha_prob_train_gate = bool(
            getattr(self.cfg, "USE_ALPHA_PROB_TRAIN_GATE", False)
        )
        self.use_alpha_mean_train_gate = bool(
            getattr(self.cfg, "USE_ALPHA_MEAN_TRAIN_GATE", False)
        )
        self.force_open_train_gate = bool(
            getattr(self.cfg, "FORCE_OPEN_TRAIN_GATE", False)
        )
        self.use_alpha_weighted_relation = bool(
            getattr(self.cfg, "USE_ALPHA_WEIGHTED_RELATION", False)
        )
        self.alpha_relation_max_weight = float(
            getattr(self.cfg, "ALPHA_RELATION_MAX_WEIGHT", 2.0)
        )
        active_train_gates = sum((
            self.use_alpha_prob_train_gate,
            self.use_alpha_mean_train_gate,
            self.force_open_train_gate,
        ))
        if active_train_gates > 1:
            raise ValueError(
                "Alpha probability, Alpha batch-mean, and force-open training gates "
                "are mutually exclusive"
            )
        if self.use_alpha_weighted_relation and not self.force_open_train_gate:
            raise ValueError(
                "Alpha-weighted Relation requires the force-open training gate"
            )
        if self.alpha_relation_max_weight < 1.0:
            raise ValueError(
                "ALPHA_RELATION_MAX_WEIGHT must be at least 1, got "
                f"{self.alpha_relation_max_weight}"
            )
        self.alpha_relation_weights = None
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

    def _compute_mmrl_relation_loss(
        self,
        adapted_states: torch.Tensor,
        org_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        relation_weights: Optional[torch.Tensor] = None,
    ):
        zero = adapted_states.new_tensor(0.0, dtype=torch.float32)
        if cu_seqlens is None or cu_seqlens.numel() <= 1:
            return zero, zero, zero, zero

        relation_losses = []
        variance_losses = []
        cross_relation_losses = []
        active_relation_weights = []
        max_tokens = max(int(self.mmrl_relation_max_tokens), 2)
        starts = cu_seqlens[:-1].tolist()
        ends = cu_seqlens[1:].tolist()
        if relation_weights is not None and relation_weights.numel() != len(starts):
            raise RuntimeError(
                "Relation weight count does not match visual image count: "
                f"weights={relation_weights.numel()} images={len(starts)}"
            )
        for image_index, (start, end) in enumerate(zip(starts, ends)):
            start, end = int(start), int(end)
            token_count = end - start
            if token_count < 2:
                continue
            if token_count > max_tokens:
                sample_idx = torch.linspace(
                    start,
                    end - 1,
                    steps=max_tokens,
                    device=adapted_states.device,
                ).round().to(torch.long)
                adapted = adapted_states.index_select(0, sample_idx).float()
                original = org_states.index_select(0, sample_idx).detach().float()
            else:
                adapted = adapted_states[start:end].float()
                original = org_states[start:end].detach().float()

            adapted_unit = F.normalize(adapted, dim=-1, eps=1e-6)
            original_unit = F.normalize(original, dim=-1, eps=1e-6)
            adapted_gram = adapted_unit @ adapted_unit.transpose(0, 1)
            original_gram = original_unit @ original_unit.transpose(0, 1)
            relation_losses.append(F.mse_loss(adapted_gram, original_gram))
            if relation_weights is not None:
                active_relation_weights.append(
                    relation_weights.reshape(-1)[image_index]
                )
            if (
                self.mmrl_cross_relation_loss_weight > 0.0
                and torch.is_grad_enabled()
            ):
                cross_gram = adapted_unit @ original_unit.transpose(0, 1)
                cross_relation_losses.append(
                    F.mse_loss(cross_gram, original_gram)
                )

            adapted_centered = adapted_unit - adapted_unit.mean(dim=0, keepdim=True)
            original_centered = original_unit - original_unit.mean(dim=0, keepdim=True)
            adapted_dispersion = adapted_centered.square().sum(dim=-1).mean().clamp_min(1e-12).sqrt()
            original_dispersion = original_centered.square().sum(dim=-1).mean().clamp_min(1e-12).sqrt()
            variance_losses.append(torch.relu(
                original_dispersion.detach() * self.mmrl_variance_floor_ratio - adapted_dispersion
            ).pow(2))

        if not relation_losses:
            return zero, zero, zero, zero
        def reduce_losses(losses):
            stacked = torch.stack(losses)
            if relation_weights is None:
                return stacked.mean()
            weights = torch.stack(active_relation_weights).to(
                device=stacked.device,
                dtype=stacked.dtype,
            )
            return (stacked * weights).sum() / weights.sum().clamp_min(1e-8)

        gram_loss = reduce_losses(relation_losses)
        variance_loss = reduce_losses(variance_losses)
        cross_relation_loss = (
            reduce_losses(cross_relation_losses)
            if cross_relation_losses
            else zero
        )
        total = gram_loss + self.mmrl_variance_floor_weight * variance_loss
        return total, gram_loss, variance_loss, cross_relation_loss

    def _compute_absolute_alignment_metrics(
        self,
        candidate_states: torch.Tensor,
        org_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        prefix: str,
        include_gram: bool = False,
    ):
        device = org_states.device
        nan = torch.tensor(float("nan"), device=device)
        metrics = {
            f"{prefix}_token_cos_mean": nan,
            f"{prefix}_token_cos_p05": nan,
            f"{prefix}_pooled_cos_mean": nan,
        }
        if include_gram:
            metrics[f"{prefix}_gram_loss"] = nan
        if candidate_states is None or candidate_states.numel() == 0 or org_states.numel() == 0:
            return metrics

        with torch.no_grad():
            count = min(candidate_states.shape[0], org_states.shape[0])
            candidate = candidate_states[:count].detach().float()
            original = org_states[:count].detach().float()
            token_cos = F.cosine_similarity(candidate, original, dim=-1, eps=1e-6)
            metrics[f"{prefix}_token_cos_mean"] = token_cos.mean().to(device)
            metrics[f"{prefix}_token_cos_p05"] = torch.quantile(token_cos, 0.05).to(device)

            pooled_candidate = self._pool_tokens_by_image_mean(candidate, cu_seqlens)
            pooled_original = self._pool_tokens_by_image_mean(original, cu_seqlens)
            if pooled_candidate is not None and pooled_original is not None:
                pooled_count = min(pooled_candidate.shape[0], pooled_original.shape[0])
                pooled_cos = F.cosine_similarity(
                    pooled_candidate[:pooled_count],
                    pooled_original[:pooled_count],
                    dim=-1,
                    eps=1e-6,
                )
                metrics[f"{prefix}_pooled_cos_mean"] = pooled_cos.mean().to(device)

            if include_gram:
                _, gram_loss, _, _ = self._compute_mmrl_relation_loss(
                    candidate,
                    original,
                    cu_seqlens,
                )
                metrics[f"{prefix}_gram_loss"] = gram_loss.detach().to(device)
        return metrics

    def _compute_residual_debug_metrics(
        self,
        final_delta: torch.Tensor,
        org_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ):
        device = final_delta.device
        nan = torch.tensor(float("nan"), device=device)
        metrics = {
            "final_delta_norm_std": nan,
            "final_delta_norm_p95": nan,
            "final_delta_norm_max": nan,
            "final_token_ratio_mean": nan,
            "final_token_ratio_p50": nan,
            "final_token_ratio_p90": nan,
            "final_token_ratio_p95": nan,
            "final_token_ratio_p99": nan,
            "final_token_ratio_max": nan,
            "final_token_ratio_over_1_fraction": nan,
            "final_token_ratio_over_1_5_fraction": nan,
            "final_token_ratio_over_2_fraction": nan,
            "delta_pool_norm_mean": nan,
            "delta_pool_norm_std": nan,
            "delta_pool_pairwise_cos_mean": nan,
            "delta_pool_pairwise_cos_std": nan,
            "delta_pool_common_mode_ratio": nan,
            "delta_pool_specificity_ratio": nan,
            "delta_org_pool_cos_mean": nan,
            "delta_org_pool_cos_std": nan,
            "delta_token_norm_entropy": nan,
            "delta_token_norm_entropy_norm": nan,
            "delta_token_top10_mass": nan,
        }

        with torch.no_grad():
            final_delta = final_delta.detach().float()
            org_hidden_states = org_hidden_states.detach().float()

            final_token_norm = final_delta.norm(dim=-1)
            org_token_norm = org_hidden_states.norm(dim=-1).clamp_min(1e-8)
            final_token_ratio = final_token_norm / org_token_norm

            metrics["final_delta_norm_std"] = final_token_norm.std(unbiased=False)
            metrics["final_delta_norm_p95"] = torch.quantile(final_token_norm, 0.95) if final_token_norm.numel() > 0 else nan
            metrics["final_delta_norm_max"] = final_token_norm.max() if final_token_norm.numel() > 0 else nan
            if final_token_ratio.numel() > 0:
                quantiles = final_token_ratio.new_tensor([0.50, 0.90, 0.95, 0.99])
                final_q = torch.quantile(final_token_ratio, quantiles)
                for idx, suffix in enumerate(("p50", "p90", "p95", "p99")):
                    metrics[f"final_token_ratio_{suffix}"] = final_q[idx]
                metrics["final_token_ratio_mean"] = final_token_ratio.mean()
                metrics["final_token_ratio_max"] = final_token_ratio.max()
                for threshold, suffix in ((1.0, "1"), (1.5, "1_5"), (2.0, "2")):
                    metrics[f"final_token_ratio_over_{suffix}_fraction"] = (
                        final_token_ratio > threshold
                    ).float().mean()

            pooled_delta = self._pool_tokens_by_image_mean(final_delta, cu_seqlens)
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

    def _prepare_base_visual_tokens(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)
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
        return hidden_states, cu_seqlens, position_embeddings, rotary_pos_emb

    def extract_gate_pooled_vision(self, pixel_values, grid_thw):
        """Reproduce the exact main-branch features seen by Task_classifier."""
        if not self.insert_layers:
            raise RuntimeError("Task classifier requires at least one MMRL insertion layer")
        first_insert_layer = min(self.insert_layers)

        # The backbone is frozen in Stage 1; keep only the pooler differentiable.
        with torch.no_grad():
            (
                hidden_states,
                cu_seqlens,
                position_embeddings,
                rotary_pos_emb,
            ) = self._prepare_base_visual_tokens(
                pixel_values.type(self.dtype),
                grid_thw,
            )
            for layer_num in range(first_insert_layer):
                hidden_states = self.blocks[layer_num](
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    rotary_pos_emb=rotary_pos_emb,
                )

        image_token_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        image_indices = torch.repeat_interleave(
            torch.arange(
                image_token_lengths.numel(),
                device=hidden_states.device,
            ),
            image_token_lengths,
        )
        return self.hidden_state_pooling.forward_vectorized(
            hidden_states.detach(),
            image_indices,
            image_token_lengths.numel(),
        )

    def forward(self,
                hidden_states: torch.Tensor,
                grid_thw: torch.Tensor,
                rep_generator: Optional[nn.Module] = None,
                embedding: Optional[torch.Tensor] = None,
                text_pooling_mask: Optional[torch.Tensor] = None,
                gating_temperature_override: float = None,
                images_per_sample: Optional[list[int]] = None,
                **kwargs):
        assert len(images_per_sample) == embedding.shape[0]
        self.alpha_list = None 
        self.G_list = []
        self.alpha_relation_weights = None


        rotary_pos_emb_with_rep = None
        gate_embedding_after_pooling = None
        if not self.ablate_visual_gate:
            gate_embedding_after_pooling = self.embedding_pooling(
                embedding,
                mask=text_pooling_mask,
            )
        (
            hidden_states,
            cu_seqlens,
            position_embeddings,
            rotary_pos_emb,
        ) = self._prepare_base_visual_tokens(hidden_states, grid_thw)

        original_seq_lens_list = (grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0]).tolist()

        first_insert = True
        current_num_r_token = self.cfg.RP_SPACE_LENGTH

        deepstack_feature_lists = []
        deepstack_delta_norm_values = []
        deepstack_delta_ratio_values = []
        deepstack_residual_layer_count = 0
        cu_seqlens_with_rep, position_embeddings_with_rep = None, None
        hidden_states_with_rep, org_hidden_states = None, None
        total_pic_num = cu_seqlens.size(0) - 1
        run_mmrl_branch = rep_generator is not None
        dynamic_rep_tokens = None
        forward_hit_layers = []
        if not self._printed_forward_layer_entry_audit:
            print("[MMRL_FORWARD_ENTRY_AUDIT]")
            print(f"  rep_generator_is_none={rep_generator is None}")
            print(f"  planned_insert_layers_0based={self.insert_layers}")
            print(f"  planned_insert_layers_natural={[idx + 1 for idx in self.insert_layers]}")
            print(f"  deepstack_visual_indexes_0based={self.deepstack_visual_indexes}")
            print(f"  deepstack_visual_indexes_natural={[idx + 1 for idx in self.deepstack_visual_indexes]}")
            self._printed_forward_layer_entry_audit = True
        for layer_num, blk in enumerate(self.blocks):
            if (
                layer_num not in self.insert_layers
                or rep_generator is None
            ):
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
                        run_mmrl_branch = rep_generator is not None
                    else:
                        img_indices = torch.repeat_interleave(
                            torch.arange(total_pic_num, device=hidden_states.device),
                            img_seqlens
                        )
                        gate_pooled_vision_states = self.hidden_state_pooling.forward_vectorized(
                            hidden_states,
                            img_indices,
                            total_pic_num
                        )  # [Total_Images, Dim]
                        images_per_sample_tensor = torch.tensor(
                            images_per_sample,
                            device=hidden_states.device,
                        )
                        expanded_gate_text_embedding = torch.repeat_interleave(
                            gate_embedding_after_pooling,
                            images_per_sample_tensor,
                            dim=0
                        )
                        self.alpha_list = self.Task_classifier(
                            gate_pooled_vision_states,
                            expanded_gate_text_embedding,
                        )
                        # G_list: [Total_Images, 1]
                        if self.training:
                            if self.force_open_train_gate:
                                self.G_list = torch.ones_like(
                                    self.alpha_list,
                                    dtype=hidden_states.dtype,
                                )
                            elif self.use_alpha_mean_train_gate:
                                self.G_list = self.visionGating.batch_mean_probability(
                                    self.alpha_list
                                ).to(dtype=hidden_states.dtype)
                            elif self.use_alpha_prob_train_gate:
                                # Deterministic classifier confidence directly scales
                                # the MMRL residual and its CE gradient.
                                self.G_list = self.visionGating.probability(
                                    self.alpha_list
                                ).to(
                                    dtype=hidden_states.dtype
                                )
                            else:
                                self.G_list = self.visionGating(
                                    self.alpha_list,
                                    gating_temperature_override
                                ).to(dtype=hidden_states.dtype)
                            if self.use_alpha_weighted_relation:
                                self.alpha_relation_weights = (
                                    self.visionGating.inverse_probability_weights(
                                        self.alpha_list,
                                        max_weight=self.alpha_relation_max_weight,
                                    )
                                )
                        else:
                            raw_g = self.visionGating(
                                self.alpha_list,
                                gating_temperature_override
                            )
                            self.G_list = (raw_g > 0.5).to(dtype=hidden_states.dtype)
                            if self.G_list.sum() == 0:
                                run_mmrl_branch = False

                    if self.training or run_mmrl_branch:
                        dynamic_rep_tokens = rep_generator(
                            visual_states=hidden_states,
                            cu_seqlens=cu_seqlens,
                            text_states=embedding,
                            text_mask=text_pooling_mask,
                            images_per_sample=images_per_sample,
                        )
                        if len(dynamic_rep_tokens) != len(self.insert_layers):
                            raise RuntimeError(
                                "dynamic Rep layer count mismatch: "
                                f"generated={len(dynamic_rep_tokens)} "
                                f"expected={len(self.insert_layers)}"
                            )
                        expected_shape = (
                            total_pic_num,
                            current_num_r_token,
                            self.cfg.vision_token_dim,
                        )
                        invalid_shapes = [
                            (index, tuple(tokens.shape))
                            for index, tokens in enumerate(dynamic_rep_tokens)
                            if tuple(tokens.shape) != expected_shape
                        ]
                        if invalid_shapes:
                            raise RuntimeError(
                                "dynamic Rep tensor shape mismatch: "
                                f"expected={expected_shape} actual={invalid_shapes}"
                            )

                ############ N图切分+门控 ############
                idx = self.insert_layers.index(layer_num)
                forward_hit_layers.append(layer_num)

                org_hidden_states = blk(hidden_states if first_insert else org_hidden_states,
                                        cu_seqlens=cu_seqlens,
                                        position_embeddings=position_embeddings,
                                        rotary_pos_emb=rotary_pos_emb,
                                        **kwargs)
                hidden_states = org_hidden_states
                if self.training or run_mmrl_branch:
                    if dynamic_rep_tokens is None or dynamic_rep_tokens[idx] is None:
                        raise RuntimeError(
                            f"dynamic Rep tokens missing for insertion layer {layer_num}"
                        )
                    r_tokens_input = dynamic_rep_tokens[idx]
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
                        if self.training:
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
            expected_insert_layers = (
                self.insert_layers
                if rep_generator is not None
                else []
            )
            print(f"  expected_insert_layers_0based={expected_insert_layers}")
            print(f"  expected_insert_layers_natural={[idx + 1 for idx in expected_insert_layers]}")
            self._printed_forward_layer_hit_audit = True

        if run_mmrl_branch and hidden_states_with_rep is not None:
            hidden_states_with_rep = _strip_r_token(hidden_states_with_rep,
                                                    original_seq_lens_list,
                                                    current_num_r_token)
            # for i in range(cu_seqlens.size(0) - 1):
            #     hidden_states_with_rep = hidden_states_with_rep[cu_seqlens[i]:cu_seqlens[i+1]] * self.G_list[i]
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            if self.training:
                (
                    self.mmrl_relation_loss,
                    self.mmrl_relation_gram_loss,
                    self.mmrl_variance_floor_loss,
                    self.mmrl_cross_relation_loss,
                ) = self._compute_mmrl_relation_loss(
                    hidden_states_with_rep,
                    org_hidden_states,
                    cu_seqlens,
                    relation_weights=self.alpha_relation_weights,
                )
            else:
                self.mmrl_relation_loss = hidden_states.new_tensor(0.0)
                self.mmrl_relation_gram_loss = hidden_states.new_tensor(0.0)
                self.mmrl_variance_floor_loss = hidden_states.new_tensor(0.0)
                self.mmrl_cross_relation_loss = hidden_states.new_tensor(0.0)
            delta = hidden_states_with_rep - org_hidden_states
            G_mask = torch.repeat_interleave(self.G_list, seqlens, dim=0)
            final_delta = delta * G_mask
            if self.training:
                raw_mmrl_norm = final_delta.float().norm(dim=-1).mean()
                raw_org_norm = org_hidden_states.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
                raw_mmrl_ratio = raw_mmrl_norm / raw_org_norm
                self.mmrl_delta_to_org_ratio = raw_mmrl_ratio.detach().to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            else:
                self.mmrl_delta_to_org_ratio = hidden_states.new_tensor(float("nan"))
            hidden_states = org_hidden_states + final_delta
        else:
            hidden_states = org_hidden_states
            final_delta = torch.zeros_like(hidden_states)
            self.mmrl_delta_to_org_ratio = hidden_states.new_tensor(float("nan"))
            self.mmrl_relation_loss = hidden_states.new_tensor(0.0)
            self.mmrl_relation_gram_loss = hidden_states.new_tensor(0.0)
            self.mmrl_variance_floor_loss = hidden_states.new_tensor(0.0)
            self.mmrl_cross_relation_loss = hidden_states.new_tensor(0.0)

        if self.training:
            mmrl_state_alignment = self._compute_absolute_alignment_metrics(
                hidden_states_with_rep if run_mmrl_branch else None,
                org_hidden_states,
                cu_seqlens,
                prefix="mmrl_state",
            )
            final_state_alignment = self._compute_absolute_alignment_metrics(
                hidden_states,
                org_hidden_states,
                cu_seqlens,
                prefix="final_state",
                include_gram=True,
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
                org_hidden_states,
                cu_seqlens,
            )
            relation_weight_mean = (
                self.alpha_relation_weights.detach().float().mean()
                if self.alpha_relation_weights is not None
                else hidden_states.new_tensor(1.0)
            )
            relation_weight_std = (
                self.alpha_relation_weights.detach().float().std(unbiased=False)
                if self.alpha_relation_weights is not None
                and self.alpha_relation_weights.numel() > 1
                else hidden_states.new_tensor(0.0)
            )
            relation_weight_min = (
                self.alpha_relation_weights.detach().float().min()
                if self.alpha_relation_weights is not None
                else hidden_states.new_tensor(1.0)
            )
            relation_weight_max = (
                self.alpha_relation_weights.detach().float().max()
                if self.alpha_relation_weights is not None
                else hidden_states.new_tensor(1.0)
            )
            self.debug_context = {
                "alpha_prob_mean": alpha_mean,
                "alpha_prob_std": alpha_std,
                "G_mean": g_mean,
                "G_std": g_std,
                "alpha_relation_weight_mean": relation_weight_mean,
                "alpha_relation_weight_std": relation_weight_std,
                "alpha_relation_weight_min": relation_weight_min,
                "alpha_relation_weight_max": relation_weight_max,
                "final_delta_norm_mean": final_delta_norm,
                "org_hidden_norm_mean": org_hidden_norm,
                "delta_to_org_ratio": delta_to_org_ratio,
                "mmrl_delta_to_org_ratio": self.mmrl_delta_to_org_ratio.detach(),
                "mmrl_relation_loss": self.mmrl_relation_loss.detach(),
                "mmrl_relation_gram_loss": self.mmrl_relation_gram_loss.detach(),
                "mmrl_variance_floor_loss": self.mmrl_variance_floor_loss.detach(),
                "mmrl_cross_relation_loss": (
                    self.mmrl_cross_relation_loss.detach()
                ),
                "deepstack_mmrl_residual_scale": hidden_states.new_tensor(float(self.deepstack_mmrl_residual_scale)),
                "deepstack_delta_norm_mean": self.deepstack_delta_norm_mean.detach(),
                "deepstack_delta_to_org_ratio": self.deepstack_delta_to_org_ratio.detach(),
                "deepstack_residual_layers": self.deepstack_residual_layers.detach(),
                **mmrl_state_alignment,
                **final_state_alignment,
                **residual_debug,
            }
        else:
            self.debug_context = {}
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists
