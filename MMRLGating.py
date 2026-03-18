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
        # 默认让 alpha 趋近于 0，只有看到强烈的专业特征才激活
        nn.init.constant_(self.output_head.bias, 0.0)

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
        self.intensity_mlp = nn.Sequential(
            nn.Linear(config.vision_token_dim, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, self.total_rep_num)
        )
        
        nn.init.constant_(self.intensity_mlp[-1].bias, 2.0) 
        nn.init.normal_(self.intensity_mlp[-1].weight, std=0.01)
        
        self.threshold_root = nn.Parameter(torch.zeros([1, self.total_rep_num]))
        nn.init.constant_(self.threshold_root, -2.0)
        self.epsilon = epsilon
        self.softplus = nn.Softplus()
        self.hard_concrete = HardConcreteGate(temperature)
        self.lambda_ = lambda_
        self.text_relevance_head = nn.Sequential(
            nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, 1)
        )
        nn.init.constant_(self.text_relevance_head[-1].bias, 0.0)
        self.debug_context = {}
        self.th_scale_mlp = nn.Sequential(
            nn.Linear(config.vision_token_dim + config.text_token_dim + 1, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, 1)
        )
        nn.init.constant_(self.th_scale_mlp[-1].bias, 0.0)

    def forward(self,
                delta_vision_token: torch.Tensor,
                alpha: torch.Tensor,
                batch_indices_token: torch.Tensor,
                batch_indices_img: torch.Tensor,
                batch_size: int,
                text_embedding: torch.Tensor,
                temperature_override: Optional[float] = None):
        pooled_vision = self.attention_pooling.forward_vectorized(
            delta_vision_token, batch_indices_token, batch_size
        )
        alpha_logits = torch.sigmoid(alpha)
        batch_alpha = torch.zeros(batch_size, 1, device=alpha_logits.device, dtype=alpha_logits.dtype)
        batch_alpha.index_reduce_(0, batch_indices_img, alpha_logits, reduce="amax", include_self=False)
        
        # text relevance
        text_relevance_logits = self.text_relevance_head(text_embedding)
        text_relevance_prob = torch.sigmoid(text_relevance_logits)  # [B, 1]
        # master gate
        master_gate = batch_alpha * text_relevance_prob             # [B, 1]
        # --- intensity: softplus (avoid sigmoid saturation) ---
        raw_intensity = self.softplus(self.intensity_mlp(pooled_vision))     # [B, K]
        modulated_intensity = raw_intensity * master_gate                    # [B, K]
        total_intensity = modulated_intensity.sum(dim=-1, keepdim=True)      # [B, 1]
        # --- context-adaptive threshold scaling ---
        feat = torch.cat([pooled_vision, text_embedding, batch_alpha], dim=-1)  # [B, Dv+Dt+1]
        scale = torch.sigmoid(self.th_scale_mlp(feat))                          # [B, 1]
        scale = 0.5 + scale                                                     # [B, 1] in [0.5, 1.5]
        threshold_base = self.softplus(self.threshold_root) + self.epsilon      # [1, K]
        threshold = threshold_base * scale                                       # [B, K] 关键：不要 unsqueeze(-1)
        K_logits = total_intensity - torch.cumsum(threshold, dim=-1)            # [B, K]
        hard_k_logits = self.hard_concrete(K_logits, temperature_override)      # [B, K]
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
            
            # hard_k_logits: [B, total_rep_num]  (soft/hard gate)
            k_soft = hard_k_logits.sum(dim=-1)  # [B], soft K 值
            # 防塌缩正则：如果 K 方差太小，就惩罚
            target_var = 4.0  # 可调，建议 2~6 之间
            k_var = k_soft.var(unbiased=False)
            collapse_loss = torch.relu(target_var - k_var)  # 方差太小就惩罚
            collapse_loss = 0.05 * collapse_loss            # 乘一个很小的权重
            
            raw_loss = (dynamic_lambda * total_intensity / self.total_rep_num).squeeze(-1)
            tax_loss = raw_loss.mean()
            tax_loss = tax_loss + collapse_loss
            return hard_k_logits, tax_loss
        return hard_k_logits


