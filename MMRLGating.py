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
    """Text-side group router.

    top4_group: 40 个 text rep token 切成 8 组，每组 5 个；router 输出 8 个
    group score，每个样本固定选择 top-4 组，按组 id 顺序展开为 20 个 token。
    """

    def __init__(self,
                 config,
                 epsilon: float = 0.1,
                 temperature: float = 1.0,
                 lambda_: float = 0.2):
        super().__init__()
        self.total_rep_num = int(_cfg_get(config, "RP_SPACE_LENGTH", 40))
        self.group_count = int(_cfg_get(config, "TEXT_GATE_GROUP_COUNT", 8))
        if self.group_count <= 0 or self.total_rep_num % self.group_count != 0:
            raise ValueError(
                f"TEXT_GATE_GROUP_COUNT={self.group_count} must divide total_rep_num={self.total_rep_num}"
            )
        self.group_size = self.total_rep_num // self.group_count
        self.selection_mode = str(_cfg_get(config, "TEXT_GATE_SELECTION_MODE", "top4_group"))
        self.top4_group_count = int(_cfg_get(config, "TOP4_GROUP_COUNT", 4))
        self.top4_group_count = max(1, min(self.top4_group_count, self.group_count))
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

    def _build_top4_group_gate(self,
                               pooled_vision: torch.Tensor,
                               text_embedding: torch.Tensor,
                               batch_alpha: torch.Tensor,
                               has_image: torch.Tensor):
        feat = torch.cat([pooled_vision, text_embedding, batch_alpha, has_image], dim=-1)
        group_logits = self.group_score_head(feat)  # [B, 8]
        _, selected_group_ids = torch.topk(group_logits, k=self.top4_group_count, dim=-1)
        selected_group_ids = torch.sort(selected_group_ids, dim=-1).values
        group_mask = torch.zeros_like(group_logits)
        group_mask.scatter_(1, selected_group_ids, 1.0)
        soft_group_mask = torch.softmax(group_logits, dim=-1) * float(self.top4_group_count)
        st_group_mask = group_mask.detach() - soft_group_mask.detach() + soft_group_mask

        token_offsets = torch.arange(self.group_size, device=group_logits.device, dtype=selected_group_ids.dtype).view(1, 1, -1)
        selected_token_indices = (selected_group_ids.unsqueeze(-1) * self.group_size + token_offsets).reshape(group_logits.shape[0], -1)
        slot_gate = group_mask.repeat_interleave(self.group_size, dim=-1)
        return slot_gate, group_logits, group_mask, st_group_mask, selected_group_ids, selected_token_indices

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

        slot_gate, group_logits, group_mask, st_group_mask, selected_group_ids, selected_token_indices = self._build_top4_group_gate(
            pooled_vision=pooled_vision,
            text_embedding=text_embedding,
            batch_alpha=batch_alpha,
            has_image=has_image,
        )

        slot_usage_live = slot_gate.mean(dim=0)
        group_usage_live = group_mask.mean(dim=0)
        active_counts_live = slot_gate.sum(dim=-1)
        selected_group_counts_live = group_mask.sum(dim=-1)
        top4_values = group_logits.gather(1, selected_group_ids)
        kth_top4_score = top4_values[:, -1]
        unselected_scores = group_logits.masked_fill(group_mask.bool(), float("-inf"))
        best_unselected_score = unselected_scores.max(dim=-1).values
        group_score_margin = kth_top4_score - best_unselected_score
        self.live_context = {
            "hard_k_logits_live": slot_gate,
            "slot_usage_live": slot_usage_live,
            "group_logits_live": group_logits,
            "top4_group_scores_live": group_logits,
            "top4_group_mask_live": group_mask,
            "top4_group_mask_st_live": st_group_mask,
            "top4_selected_group_ids_live": selected_group_ids,
            "top4_selected_token_indices_live": selected_token_indices,
            "group_usage_live": group_usage_live,
            "active_counts_live": active_counts_live,
            "top4_selected_group_counts_live": selected_group_counts_live,
            "top4_group_score_margin_live": group_score_margin,
            "batch_alpha_live": batch_alpha,
        }

        slot_usage = slot_usage_live.detach().float()
        dead_slot_ratio = (slot_usage < 0.01).float().mean()
        group_usage = group_usage_live.detach().float()
        active_counts = active_counts_live.detach().float()
        active_stats = self._tensor_stats(active_counts)
        batch_alpha_stats = self._tensor_stats(batch_alpha)
        group_logit_stats = self._tensor_stats(group_logits)
        selected_count_stats = self._tensor_stats(selected_group_counts_live.detach().float())
        margin_stats = self._tensor_stats(group_score_margin.detach().float())
        group_entropy = self._compute_entropy(group_usage)
        if group_usage.numel() > 1:
            group_entropy_norm = group_entropy / torch.log(group_usage.new_tensor(float(group_usage.numel())))
        else:
            group_entropy_norm = group_usage.new_tensor(0.0)

        self.debug_context = {
            "top4_selected_group_count_mean": selected_count_stats["mean"],
            "top4_selected_group_count_std": selected_count_stats["std"],
            "active_token_count_mean": active_stats["mean"],
            "active_token_count_std": active_stats["std"],
            "batch_alpha_mean": batch_alpha_stats["mean"],
            "batch_alpha_std": batch_alpha_stats["std"],
            "batch_alpha_boundary_ratio": ((batch_alpha.detach().float() > 0.45) & (batch_alpha.detach().float() < 0.55)).float().mean(),
            "top4_group_score_mean": group_logit_stats["mean"],
            "top4_group_score_std": group_logit_stats["std"],
            "top4_group_score_margin_mean": margin_stats["mean"],
            "dead_slot_ratio": dead_slot_ratio,
            "top4_group_usage_max": group_usage.max(),
            "top4_group_usage_min": group_usage.min(),
            "top4_group_usage_std": group_usage.std(unbiased=False) if group_usage.numel() > 1 else group_usage.new_tensor(0.0),
            "top4_group_usage_entropy": group_entropy,
            "top4_group_usage_entropy_norm": group_entropy_norm,
            "group_usage_max": group_usage.max(),
            "group_usage_min": group_usage.min(),
            "group_usage_entropy": group_entropy,
            "group_usage_entropy_norm": group_entropy_norm,
        }
        for idx, value in enumerate(group_usage.tolist()):
            self.debug_context[f"top4_group_usage_{idx}"] = float(value)
            self.debug_context[f"group_usage_{idx}"] = float(value)

        self.capacity_prior_loss = slot_gate.new_tensor(0.0)
        self.tax_loss = self.capacity_prior_loss
        return slot_gate
