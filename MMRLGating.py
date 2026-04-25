# MMRLGating.PY
from typing import Optional

import torch
from torch import nn
from utils import attention_pooling

class Task_classifier(nn.Module):
    def __init__(self,config):
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
        # if self.training:
        #     ss = random.random()
        #     if ss < 0.1:
        #         t_feat = torch.zeros_like(t_feat)
        #     elif ss < 0.2 and ss >= 0.1:
        #         v_feat = torch.zeros_like(v_feat)
        #     else:
        #         pass

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
    ################## 传入HardConcreteGate的都必须是未经归一化的值 ##################
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
    def __init__(self,
                 config,
                 epsilon: float = 0.1,
                 temperature: float = 1.0,
                 lambda_=0.2):
        super().__init__()
        self.total_rep_num = config.RP_SPACE_LENGTH * 8
        # 4/23修改，修改原因：文本门控从“前缀式 K 控制”升级为“预算控制 + slot 内容选择”，
        # 需要显式区分视觉语义主体与残差强度统计两类证据。
        self.attention_pooling = attention_pooling(config.vision_token_dim, config.GATING_MID_DIM)
        self.visual_semantic_proj = nn.Sequential(
            nn.Linear(config.vision_token_dim * 2, config.GATING_MID_DIM),
            nn.ReLU(),
        )
        self.confidence_proj = nn.Sequential(
            nn.Linear(8, config.GATING_MID_DIM // 2),
            nn.ReLU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
            nn.ReLU(),
        )

        self.text_relevance_head = nn.Sequential(
            nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, 1)
        )
        nn.init.constant_(self.text_relevance_head[-1].bias, 0.0)

        self.k_budget_head = nn.Sequential(
            nn.Linear(config.GATING_MID_DIM * 2 + config.GATING_MID_DIM // 2 + 2, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, 1)
        )
        self.slot_score_head = nn.Sequential(
            nn.Linear(config.GATING_MID_DIM * 2 + config.GATING_MID_DIM // 2 + 2, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, self.total_rep_num)
        )

        self.hard_concrete = HardConcreteGate(temperature)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.debug_context = {}

        nn.init.constant_(self.k_budget_head[-1].bias, -1.5)  # 原来 -0.5 太激进
        self.lambda_general = 1.0
        self.lambda_expert = 0.25
        self.k_cap_expert = 14.0
        self.k_min_expert = 2.0
        self.infer_alpha_threshold = 0.5

    def _build_hard_topk_mask(self, slot_logits: torch.Tensor, k_budget: torch.Tensor) -> torch.Tensor:
        # 4/23修改，修改原因：推理阶段必须从 40 个 slot 中按分数选择 Top-K，而不是再按前缀放行。
        batch_size = slot_logits.size(0)
        hard_mask = torch.zeros_like(slot_logits)
        k_hard = torch.clamp(torch.round(k_budget.squeeze(-1)), min=0, max=self.total_rep_num).to(torch.long)
        for i in range(batch_size):
            current_k = int(k_hard[i].item())
            if current_k <= 0:
                continue
            topk_idx = torch.topk(slot_logits[i], k=current_k, dim=-1).indices
            hard_mask[i, topk_idx] = 1.0
        return hard_mask

    def _build_soft_topk_mask(self, slot_logits: torch.Tensor, k_budget: torch.Tensor) -> torch.Tensor:
        slot_probs = torch.softmax(slot_logits, dim=-1)
        scaled = slot_probs * k_budget
        return torch.clamp(scaled, min=0.0, max=1.0)

    def forward(self,
                delta_vision_token: torch.Tensor,
                alpha: torch.Tensor,
                batch_indices_token: torch.Tensor,
                batch_indices_img: torch.Tensor,
                batch_size: int,
                text_embedding: torch.Tensor,
                pooled_semantic_mean: Optional[torch.Tensor] = None,
                pooled_semantic_max: Optional[torch.Tensor] = None,
                confidence_stats: Optional[torch.Tensor] = None,
                has_image: Optional[torch.Tensor] = None,
                temperature_override: Optional[float] = None):
        pooled_vision = self.attention_pooling.forward_vectorized(
            delta_vision_token, batch_indices_token, batch_size
        )  # [B, Dv]

        alpha_prob = torch.sigmoid(alpha)
        batch_alpha = torch.zeros(batch_size, 1, device=alpha_prob.device, dtype=alpha_prob.dtype)
        batch_alpha.index_reduce_(0, batch_indices_img, alpha_prob, reduce="amax", include_self=False)

        if has_image is None:
            has_image = torch.ones(batch_size, 1, device=text_embedding.device, dtype=text_embedding.dtype)
        if pooled_semantic_mean is None:
            pooled_semantic_mean = pooled_vision
        if pooled_semantic_max is None:
            pooled_semantic_max = pooled_vision
        if confidence_stats is None:
            confidence_stats = torch.zeros(batch_size, 8, device=text_embedding.device, dtype=text_embedding.dtype)

        text_relevance_prob = torch.sigmoid(self.text_relevance_head(text_embedding))  # [B, 1]
        visual_semantic = torch.cat([pooled_semantic_mean, pooled_semantic_max], dim=-1)
        visual_semantic = self.visual_semantic_proj(visual_semantic)
        confidence_feat = self.confidence_proj(confidence_stats)
        text_feat = self.text_proj(text_embedding)

        # 4/23修改，修改原因：预算头仍保留总量控制，但改为由“语义主体 + 强度统计 + 文本语义”共同决定。
        feat = torch.cat([visual_semantic, text_feat, confidence_feat, batch_alpha, has_image], dim=-1)
        raw_budget = self.total_rep_num * torch.sigmoid(self.k_budget_head(feat))     # [B, 1]

        # 4/23修改，修改原因：保留 text-only expert 能力，但在无图时显式缩小预算上限。
        image_scale = 0.5 + 0.5 * has_image                                            # 无图=0.5，有图=1.0
        relevance_scale = (0.25 + 0.75 * text_relevance_prob)

        base_budget = raw_budget * image_scale * relevance_scale

        # 修复策略改为“负样本强关断、正样本尽量保持原行为”：
        # - 训练时仍用连续 alpha 调制预算，帮助预算头和 alpha 对齐；
        # - 推理时若 alpha 未通过阈值，直接把预算清零；若通过，则完整保留原预算。
        # 这样能抑制通用样本的软提示泄露，同时避免破坏原本已学好的专家 prompt 结构。
        if self.training:
            alpha_scale = batch_alpha.clamp(0.0, 1.0)
            k_budget = base_budget * alpha_scale
        else:
            alpha_scale = (batch_alpha > self.infer_alpha_threshold).to(dtype=raw_budget.dtype)
            k_budget = torch.where(alpha_scale > 0, base_budget, torch.zeros_like(base_budget))

        # 4/23修改，修改原因：新增 slot 打分头，对每个文本专家 token 单独评分，支持 token-wise Top-K 内容选择。
        slot_logits = self.slot_score_head(feat)
        hard_topk_mask = self._build_hard_topk_mask(slot_logits, k_budget)

        # alpha 只负责放行/抑制
        if self.training:
            soft_topk_mask = self._build_soft_topk_mask(slot_logits, k_budget)
            selected_mask = hard_topk_mask + soft_topk_mask - soft_topk_mask.detach()
            selected_mask = selected_mask * batch_alpha
        else:
            alpha_hard_mask = (batch_alpha > self.infer_alpha_threshold).to(dtype=hard_topk_mask.dtype)
            selected_mask = hard_topk_mask * alpha_hard_mask

        if self.training:
            batch_alpha_prob = batch_alpha.detach()
            penalty_mask = torch.where(
                batch_alpha_prob > 0.5,
                torch.zeros_like(batch_alpha_prob),
                1.0 - batch_alpha_prob
            )
            dynamic_lambda = self.lambda_ * penalty_mask

            k_soft = selected_mask.sum(dim=-1)  # [B]
            target_var = selected_mask.new_tensor(4.0)
            k_var = k_soft.var(unbiased=False)
            collapse_loss = 0.05 * torch.relu(target_var - k_var)

            slot_usage = selected_mask.mean(dim=0)
            slot_collapse_loss = 0.01 * torch.sum(slot_usage.pow(2))

            raw_loss = dynamic_lambda.squeeze(-1) * (k_soft / self.total_rep_num)
            tax_loss = raw_loss.mean() + collapse_loss + slot_collapse_loss
        else:
            tax_loss = None

        self.debug_context = {
            "batch_alpha": batch_alpha.detach(),
            "k_budget": k_budget.detach(),
            "raw_budget": raw_budget.detach(),
            "base_budget": base_budget.detach(),
            "alpha_scale": alpha_scale.detach(),
            "slot_logits": slot_logits.detach(),
            "selected_mask": selected_mask.detach(),
        }

        return {
            "selected_mask": selected_mask,
            "hard_topk_mask": hard_topk_mask,
            "slot_logits": slot_logits,
            "k_budget": k_budget.squeeze(-1),
            "k_selected": selected_mask.sum(dim=-1),
            "tax_loss": tax_loss,
        }


