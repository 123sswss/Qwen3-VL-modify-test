from typing import Optional

import torch
from torch import nn
from utils import attention_pooling
import random


class Task_classifier(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.vision_proj = nn.Linear(config.vision_token_dim, config.GATING_MID_DIM)
        self.text_proj = nn.Linear(config.text_token_dim, config.GATING_MID_DIM)
        self.fc_fusion = nn.Linear(config.GATING_MID_DIM * 2, config.GATING_MID_DIM)
        self.output_head = nn.Linear(config.GATING_MID_DIM, 1)
        self.relu = nn.ReLU()
        nn.init.constant_(self.output_head.bias, -2.0)

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

        # [Batch, MID_DIM*2]
        combined = torch.cat((v_feat, t_feat), dim=-1)
        combined = self.relu(self.fc_fusion(combined))
        alpha = self.output_head(combined)
        # alpha = torch.sigmoid(alpha)
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
        self.attention_pooling = attention_pooling(config.vision_token_dim, config.GATING_MID_DIM)

        self.text_relevance_head = nn.Sequential(
            nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, 1)
        )
        nn.init.constant_(self.text_relevance_head[-1].bias, -1.0)

        # K 主体由 vision/text 特征预测；alpha 只做放行/抑制，不再主导 K 本体
        self.k_budget_head = nn.Sequential(
            nn.Linear(config.vision_token_dim + config.text_token_dim + 2, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, 1)
        )
        nn.init.constant_(self.k_budget_head[-1].bias, -2.0)

        self.hard_concrete = HardConcreteGate(temperature)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.debug_context = {}

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
            delta_vision_token, batch_indices_token, batch_size
        )  # [B, Dv]

        alpha_prob = torch.sigmoid(alpha)
        batch_alpha = torch.zeros(batch_size, 1, device=alpha_prob.device, dtype=alpha_prob.dtype)
        batch_alpha.index_reduce_(0, batch_indices_img, alpha_prob, reduce="amax", include_self=False)

        if has_image is None:
            has_image = torch.ones(batch_size, 1, device=text_embedding.device, dtype=text_embedding.dtype)

        text_relevance_prob = torch.sigmoid(self.text_relevance_head(text_embedding))  # [B, 1]

        # K 预算主体：由 vision/text/alpha/has_image 联合决定
        feat = torch.cat([pooled_vision, text_embedding, batch_alpha, has_image], dim=-1)
        raw_budget = self.total_rep_num * torch.sigmoid(self.k_budget_head(feat))     # [B, 1]

        # text-only 显式抑制；但不归零，保留 text-only expert 能力
        image_scale = 0.5 + 0.5 * has_image                                            # 无图=0.5，有图=1.0
        k_budget = raw_budget * image_scale * (0.25 + 0.75 * text_relevance_prob)      # [B, 1]

        # 由连续 budget 构造 slot gate
        slot_idx = torch.arange(
            self.total_rep_num, device=k_budget.device, dtype=k_budget.dtype
        ).view(1, -1)  # [1, K]

        K_logits = k_budget - slot_idx                                                  # [B, K]
        hard_k_logits = self.hard_concrete(K_logits, temperature_override)

        # alpha 只负责放行/抑制
        if self.training:
            hard_k_logits = hard_k_logits * batch_alpha
        else:
            alpha_hard_mask = (batch_alpha > 0.5).to(dtype=hard_k_logits.dtype)
            hard_k_logits = hard_k_logits * alpha_hard_mask

        if self.training:
            batch_alpha_prob = batch_alpha.detach()
            penalty_mask = torch.where(
                batch_alpha_prob > 0.5,
                torch.zeros_like(batch_alpha_prob),
                1.0 - batch_alpha_prob
            )
            dynamic_lambda = self.lambda_ * penalty_mask

            k_soft = hard_k_logits.sum(dim=-1)  # [B]
            target_var = hard_k_logits.new_tensor(4.0)
            k_var = k_soft.var(unbiased=False)
            collapse_loss = 0.05 * torch.relu(target_var - k_var)

            raw_loss = dynamic_lambda.squeeze(-1) * (k_soft / self.total_rep_num)
            tax_loss = raw_loss.mean() + collapse_loss
            return hard_k_logits, tax_loss

        return hard_k_logits


