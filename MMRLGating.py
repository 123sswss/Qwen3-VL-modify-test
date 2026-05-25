from typing import Optional

import torch
from torch import nn

from utils import attention_pooling


def _cfg_get(config, key, default):
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


class Task_classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_proj = nn.Linear(config.vision_token_dim, config.GATING_MID_DIM)
        self.text_proj = nn.Linear(config.text_token_dim, config.GATING_MID_DIM)
        self.fc_fusion = nn.Linear(config.GATING_MID_DIM * 2, config.GATING_MID_DIM)
        self.output_head = nn.Linear(config.GATING_MID_DIM, 1)
        self.relu = nn.ReLU()

        nn.init.constant_(self.output_head.bias, -0.5)

    def forward(self,
                vision_token_after_pooling: torch.Tensor,
                text_embedding_after_pooling: torch.Tensor):
        v_feat = self.relu(self.vision_proj(vision_token_after_pooling))
        t_feat = self.relu(self.text_proj(text_embedding_after_pooling))
        combined = torch.cat((v_feat, t_feat), dim=-1)
        combined = self.relu(self.fc_fusion(combined))
        alpha = self.output_head(combined)
        return alpha


class HardConcreteGate(nn.Module):
    def __init__(self,
                 temperature: float,
                 stretching_length: float = 0.1,
                 eps: float = 1e-7):
        super().__init__()
        self.temperature = temperature
        self.upper_bound = 1 + stretching_length
        self.lower_bound = 0 - stretching_length
        self.eps = eps

    # 传入 HardConcreteGate 的都必须是未经归一化的值
    def forward(self,
                logits: torch.Tensor,
                temperature_override: Optional[float] = None) -> torch.Tensor:
        temp = temperature_override if temperature_override is not None else self.temperature
        if self.training:
            u = torch.rand_like(logits)
            u = torch.clamp(u, min=self.eps, max=1 - self.eps)
            noise = torch.log(u) - torch.log(1 - u)
        else:
            noise = 0.0
        y = torch.sigmoid((noise + logits) / temp)
        y_stretched = y * (self.upper_bound - self.lower_bound) + self.lower_bound
        gate = torch.clamp(y_stretched, min=0, max=1)
        return gate


class textGating(nn.Module):
    """Group-threshold text gate with a minimal capacity prior.

    40 个文本 rep-token 固定切成 8 组，每组 5 个。router 根据视觉/文本上下文、
    专业模式概率 batch_alpha 与 has_image 输出 8 个 group logits。前向用 hard
    threshold，反向用 sigmoid straight-through；最终再乘专业模式总开关 G。
    """

    def __init__(self,
                 config,
                 epsilon: float = 0.1,
                 temperature: float = 1.0,
                 lambda_: float = 0.2):
        super().__init__()
        self.total_rep_num = config.RP_SPACE_LENGTH * 8
        self.group_count = int(_cfg_get(config, "TEXT_GATE_GROUP_COUNT", 8))
        if self.group_count <= 0 or self.total_rep_num % self.group_count != 0:
            raise ValueError(
                f"TEXT_GATE_GROUP_COUNT={self.group_count} must divide total_rep_num={self.total_rep_num}"
            )
        self.group_size = self.total_rep_num // self.group_count
        self.selection_mode = str(_cfg_get(config, "TEXT_GATE_SELECTION_MODE", "group_threshold_prior"))
        self.capacity_prior_sigma = float(_cfg_get(config, "TEXT_GATE_CAPACITY_PRIOR_SIGMA", 0.40))
        energy = _cfg_get(
            config,
            "TEXT_GATE_CAPACITY_PRIOR_ENERGY",
            [1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0],
        )
        if len(energy) != self.group_count + 1:
            raise ValueError(
                f"TEXT_GATE_CAPACITY_PRIOR_ENERGY length must be group_count+1={self.group_count + 1}, got {len(energy)}"
            )
        self.register_buffer(
            "capacity_prior_energy",
            torch.tensor([float(x) for x in energy], dtype=torch.float32),
            persistent=False,
        )
        self.attention_pooling = attention_pooling(config.vision_token_dim, config.GATING_MID_DIM)

        self.group_score_head = nn.Sequential(
            nn.Linear(config.vision_token_dim + config.text_token_dim + 2, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, self.group_count),
        )
        nn.init.constant_(self.group_score_head[-1].bias, 0.0)

        self.temperature = float(temperature)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.debug_context = {}
        self.live_context = {}

    def _tensor_stats(self, x: torch.Tensor):
        x = x.detach().float()
        return {
            "mean": x.mean(),
            "std": x.std(unbiased=False) if x.numel() > 1 else x.new_tensor(0.0),
            "min": x.min(),
            "max": x.max(),
        }

    def _compute_entropy(self, probs: torch.Tensor, eps: float = 1e-8):
        probs = probs.detach().float()
        probs = probs / probs.sum().clamp_min(eps)
        return -(probs * (probs.clamp_min(eps).log())).sum()

    def _build_group_threshold_gate(self,
                                    pooled_vision: torch.Tensor,
                                    text_embedding: torch.Tensor,
                                    batch_alpha: torch.Tensor,
                                    has_image: torch.Tensor,
                                    temperature_override: Optional[float] = None):
        feat = torch.cat([pooled_vision, text_embedding, batch_alpha, has_image], dim=-1)
        group_logits = self.group_score_head(feat)  # [B, 8]
        temp = float(temperature_override) if temperature_override is not None else self.temperature
        temp = max(temp, 1e-4)
        soft_group_gate = torch.sigmoid(group_logits / temp)
        hard_group_gate = (group_logits > 0).to(dtype=soft_group_gate.dtype)
        st_group_gate = hard_group_gate.detach() - soft_group_gate.detach() + soft_group_gate

        batch_G = (batch_alpha > 0.5).to(dtype=st_group_gate.dtype)
        gated_group_gate = st_group_gate * batch_G
        slot_gate = gated_group_gate.repeat_interleave(self.group_size, dim=-1)
        return slot_gate, group_logits, soft_group_gate, hard_group_gate, gated_group_gate, batch_G

    def _compute_capacity_prior_loss(self,
                                     soft_group_gate: torch.Tensor,
                                     batch_G: torch.Tensor):
        if soft_group_gate.numel() == 0:
            return soft_group_gate.new_tensor(0.0), soft_group_gate.new_empty(0)
        soft_count = soft_group_gate.sum(dim=-1)  # [B]
        ks = torch.arange(
            self.group_count + 1,
            device=soft_group_gate.device,
            dtype=soft_group_gate.dtype,
        ).view(1, -1)
        sigma = max(float(self.capacity_prior_sigma), 1e-4)
        weights = torch.exp(-0.5 * ((soft_count.unsqueeze(-1) - ks) / sigma).pow(2))
        energy = self.capacity_prior_energy.to(device=soft_group_gate.device, dtype=soft_group_gate.dtype).view(1, -1)
        per_sample = (weights * energy).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-8)

        expert_weight = batch_G.detach().squeeze(-1).to(dtype=per_sample.dtype)
        denom = expert_weight.sum().clamp_min(1.0)
        loss = (per_sample * expert_weight).sum() / denom
        return loss, per_sample

    def _count_ratio(self, counts: torch.Tensor, k: int):
        if counts.numel() == 0:
            return counts.new_tensor(0.0)
        return (counts == k).float().mean()

    def forward(self,
                delta_vision_token: torch.Tensor,
                alpha: torch.Tensor,
                batch_indices_token: torch.Tensor,
                batch_indices_img: torch.Tensor,
                batch_size: int,
                text_embedding: torch.Tensor,
                has_image: Optional[torch.Tensor] = None,
                temperature_override: Optional[float] = None):
        pooled_vision = self.attention_pooling.forward_vectorized(
            delta_vision_token, batch_indices_token, batch_size,
        )  # [B, Dv]

        alpha_prob = torch.sigmoid(alpha)
        batch_alpha = torch.zeros(batch_size, 1, device=alpha_prob.device, dtype=alpha_prob.dtype)
        batch_alpha.index_reduce_(0, batch_indices_img, alpha_prob, reduce="amax", include_self=False)

        if has_image is None:
            has_image = torch.ones(batch_size, 1, device=text_embedding.device, dtype=text_embedding.dtype)

        slot_gate, group_logits, soft_group_gate, hard_group_gate, gated_group_gate, batch_G = self._build_group_threshold_gate(
            pooled_vision=pooled_vision,
            text_embedding=text_embedding,
            batch_alpha=batch_alpha,
            has_image=has_image,
            temperature_override=temperature_override,
        )

        slot_usage_live = slot_gate.mean(dim=0)
        group_usage_live = gated_group_gate.mean(dim=0)
        active_counts_live = slot_gate.sum(dim=-1)
        soft_group_counts_live = soft_group_gate.sum(dim=-1)
        hard_group_counts_pre_G_live = hard_group_gate.sum(dim=-1)
        hard_group_counts_post_G_live = gated_group_gate.detach().sum(dim=-1)
        self.live_context = {
            "hard_k_logits_live": slot_gate,
            "slot_usage_live": slot_usage_live,
            "group_logits_live": group_logits,
            "soft_group_gate_live": soft_group_gate,
            "hard_group_gate_live": hard_group_gate,
            "gated_group_gate_live": gated_group_gate,
            "group_usage_live": group_usage_live,
            "active_counts_live": active_counts_live,
            "soft_group_counts_live": soft_group_counts_live,
            "hard_group_counts_pre_G_live": hard_group_counts_pre_G_live,
            "hard_group_counts_post_G_live": hard_group_counts_post_G_live,
            "batch_alpha_live": batch_alpha,
            "batch_G_live": batch_G,
        }

        slot_usage = slot_usage_live.detach().float()
        dead_slot_ratio = (slot_usage < 0.01).float().mean()
        group_usage = group_usage_live.detach().float()
        active_counts = active_counts_live.detach().float()
        active_stats = self._tensor_stats(active_counts)
        batch_alpha_stats = self._tensor_stats(batch_alpha)
        group_logit_stats = self._tensor_stats(group_logits)
        group_prob_stats = self._tensor_stats(soft_group_gate)
        soft_count_stats = self._tensor_stats(soft_group_counts_live)
        preG_count_stats = self._tensor_stats(hard_group_counts_pre_G_live.detach().float())
        postG_count_stats = self._tensor_stats(hard_group_counts_post_G_live.detach().float())
        group_entropy = self._compute_entropy(group_usage)
        if group_usage.numel() > 1:
            group_entropy_norm = group_entropy / torch.log(group_usage.new_tensor(float(group_usage.numel())))
        else:
            group_entropy_norm = group_usage.new_tensor(0.0)

        capacity_prior_loss = slot_gate.new_tensor(0.0)
        capacity_prior_energy_live = slot_gate.new_zeros(batch_size)
        if self.training:
            capacity_prior_loss, capacity_prior_energy_live = self._compute_capacity_prior_loss(
                soft_group_gate=soft_group_gate,
                batch_G=batch_G,
            )

        self.debug_context = {
            "soft_group_count_mean": soft_count_stats["mean"],
            "soft_group_count_std": soft_count_stats["std"],
            "hard_group_count_pre_G_mean": preG_count_stats["mean"],
            "hard_group_count_pre_G_std": preG_count_stats["std"],
            "hard_group_count_post_G_mean": postG_count_stats["mean"],
            "hard_group_count_post_G_std": postG_count_stats["std"],
            "active_token_count_mean": active_stats["mean"],
            "active_token_count_std": active_stats["std"],
            "batch_alpha_mean": batch_alpha_stats["mean"],
            "batch_alpha_std": batch_alpha_stats["std"],
            "batch_G_on_ratio": batch_G.detach().float().mean(),
            "batch_alpha_boundary_ratio": ((batch_alpha.detach().float() > 0.45) & (batch_alpha.detach().float() < 0.55)).float().mean(),
            "group_logit_mean": group_logit_stats["mean"],
            "group_logit_std": group_logit_stats["std"],
            "group_logit_abs_mean": group_logits.detach().float().abs().mean(),
            "group_prob_mean": group_prob_stats["mean"],
            "group_prob_std": group_prob_stats["std"],
            "group_boundary_ratio": (group_logits.detach().float().abs() < 0.2).float().mean(),
            "dead_slot_ratio": dead_slot_ratio,
            "group_entropy": group_entropy,
            "group_entropy_norm": group_entropy_norm,
            "raw_capacity_prior_loss": capacity_prior_loss.detach().float(),
            "capacity_prior_energy_mean": capacity_prior_energy_live.detach().float().mean() if capacity_prior_energy_live.numel() > 0 else slot_gate.new_tensor(0.0),
        }
        for idx, value in enumerate(group_usage.tolist()):
            self.debug_context[f"group_usage_{idx}"] = float(value)
        pre_counts = hard_group_counts_pre_G_live.detach().float().round().to(torch.long)
        post_counts = hard_group_counts_post_G_live.detach().float().round().to(torch.long)
        for k in range(self.group_count + 1):
            self.debug_context[f"preG_group_count_ratio_{k}"] = self._count_ratio(pre_counts, k)
            self.debug_context[f"postG_group_count_ratio_{k}"] = self._count_ratio(post_counts, k)

        self.capacity_prior_loss = capacity_prior_loss
        self.tax_loss = capacity_prior_loss  # compatibility alias during migration
        return (slot_gate, capacity_prior_loss) if self.training else slot_gate
