from typing import Optional

import torch
from torch import nn
import config as cfg
from utils import attention_pooling


class Task_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_proj = nn.Linear(cfg.vision_token_dim, cfg.GATING_MID_DIM)
        self.text_proj = nn.Linear(cfg.text_token_dim, cfg.GATING_MID_DIM)
        self.fc_fusion = nn.Linear(cfg.GATING_MID_DIM * 2, cfg.GATING_MID_DIM)
        self.output_head = nn.Linear(cfg.GATING_MID_DIM, 1)
        self.relu = nn.ReLU()
        # 默认让 alpha 趋近于 0，只有看到强烈的专业特征才激活
        nn.init.constant_(self.output_head.bias, -2.0)

    def forward(self,
                vision_token_after_pooling: torch.Tensor,
                text_embedding_after_pooling: torch.Tensor):
        v_feat = self.relu(self.vision_proj(vision_token_after_pooling))
        t_feat = self.relu(self.text_proj(text_embedding_after_pooling))
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
                 epsilon: float = 0.1,
                 temperature: float = 1.0,
                 lambda_=0.01):
        super().__init__()
        self.total_rep_num = cfg.RP_SPACE_LENGTH * 8
        self.attention_pooling = attention_pooling(cfg.vision_token_dim, cfg.GATING_MID_DIM)

        self.intensity_mlp = nn.Sequential(nn.Linear(cfg.vision_token_dim + 1, cfg.GATING_MID_DIM),  # 注意维度变化
                                           nn.ReLU(),
                                           nn.Linear(cfg.GATING_MID_DIM, self.total_rep_num))
        self.threshold_mlp = nn.Sequential(nn.Linear(cfg.vision_token_dim + 1, cfg.GATING_MID_DIM),  # 注意维度变化
                                           nn.ReLU(),
                                           nn.Linear(cfg.GATING_MID_DIM, self.total_rep_num))
        self.epsilon = epsilon
        self.softplus = nn.Softplus()
        self.hard_concrete = HardConcreteGate(temperature)
        self.lambda_ = lambda_

    def forward(self,
                delta_vision_token: torch.Tensor,  # [Total_Tokens, Dim]
                alpha: torch.Tensor,  # [Total_Images, 1]
                batch_indices_token: torch.Tensor,  # [Total_Tokens]
                batch_indices_img: torch.Tensor,  # [Total_Images]
                batch_size: int,
                temperature_overide: Optional[float] = None):
        # [Batch_Size, Dim]
        pooled_vision = self.attention_pooling.forward_vectorized(
            delta_vision_token, batch_indices_token, batch_size
        )

        alpha_logits = torch.sigmoid(alpha)
        batch_alpha = torch.zeros(batch_size, 1, device=alpha_logits.device, dtype=alpha_logits.dtype)
        batch_alpha.index_reduce_(0, batch_indices_img, alpha_logits, reduce="amax", include_self=False)
        hidden_state = torch.cat([pooled_vision, batch_alpha], dim=-1)
        intensity = torch.sigmoid(self.intensity_mlp(hidden_state)).sum(dim=-1, keepdim=True)
        threshold = self.threshold_mlp(hidden_state)
        threshold = self.softplus(threshold) + self.epsilon
        K_logits = intensity - torch.cumsum(threshold, -1)
        hard_k_logits = self.hard_concrete(K_logits, temperature_overide)
        if self.training:
            batch_alpha_prob = batch_alpha
            dynamic_lambda = self.lambda_ * (1.0 - batch_alpha_prob)
            raw_loss = dynamic_lambda * intensity.sum(dim=-1) / self.total_rep_num
            tax_loss = raw_loss.mean()
            return hard_k_logits, tax_loss
        return hard_k_logits


