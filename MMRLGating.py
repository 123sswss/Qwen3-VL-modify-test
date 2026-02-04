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
        nn.init.constant_(self.output_head.bias, -2.0)

    def forward(self,
                vision_token_after_pooling: torch.Tensor,
                text_embedding_after_pooling: torch.Tensor):
        v_feat = self.relu(self.vision_proj(vision_token_after_pooling))
        t_feat = self.relu(self.text_proj(text_embedding_after_pooling))
        if self.training:
            ss = random.random()
            if ss < 0.1:
                t_feat = torch.zeros_like(t_feat)
            elif ss < 0.2 and ss >= 0.1:
                v_feat = torch.zeros_like(v_feat)
            else:
                pass

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

        self.intensity_mlp = nn.Sequential(nn.Linear(config.vision_token_dim + 1, config.GATING_MID_DIM),  # 注意维度变化
                                           nn.ReLU(),
                                           nn.Linear(config.GATING_MID_DIM, self.total_rep_num))
        self.threshold_mlp = nn.Sequential(nn.Linear(config.vision_token_dim + 1, config.GATING_MID_DIM),  # 注意维度变化
                                           nn.ReLU(),
                                           nn.Linear(config.GATING_MID_DIM, self.total_rep_num))
        
        nn.init.constant_(self.intensity_mlp[-1].bias, -3.0)
        nn.init.normal_(self.intensity_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.threshold_mlp[-1].bias, 3.0)
        nn.init.normal_(self.threshold_mlp[-1].weight, std=0.01)
        
        self.epsilon = epsilon
        self.softplus = nn.Softplus()
        self.hard_concrete = HardConcreteGate(temperature)
        self.lambda_ = lambda_

        self.text_relevance_head = nn.Sequential(nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
                                                 nn.ReLU(),
                                                 nn.Linear(config.GATING_MID_DIM, 1))
        nn.init.constant_(self.text_relevance_head[-1].bias, -2.0)

    def forward(self,
                delta_vision_token: torch.Tensor,  # [Total_Tokens, Dim]
                alpha: torch.Tensor,               # [Total_Images, 1]
                batch_indices_token: torch.Tensor, # [Total_Tokens]
                batch_indices_img: torch.Tensor,   # [Total_Images]
                batch_size: int,
                text_embedding: torch.Tensor,      # [Batch_Size, Text_Dim]
                temperature_overide: Optional[float] = None):
        # [Batch_Size, Dim]
        pooled_vision = self.attention_pooling.forward_vectorized(
            delta_vision_token, batch_indices_token, batch_size
        )

        alpha_logits = torch.sigmoid(alpha)
        batch_alpha = torch.zeros(batch_size, 1, device=alpha_logits.device, dtype=alpha_logits.dtype)
        batch_alpha.index_reduce_(0, batch_indices_img, alpha_logits, reduce="amax", include_self=False)
        
        text_relevance_logits = self.text_relevance_head(text_embedding)  # [Batch, 1]
        text_relevance_prob = torch.sigmoid(text_relevance_logits)  # [0, 1]
        
        hidden_state = torch.cat([pooled_vision, batch_alpha], dim=-1)
        intensity = torch.sigmoid(self.intensity_mlp(hidden_state)).sum(dim=-1, keepdim=True)
        threshold = self.threshold_mlp(hidden_state)
        threshold = self.softplus(threshold) + self.epsilon
        modulated_intensity = intensity * text_relevance_prob
        K_logits = modulated_intensity - torch.cumsum(threshold, -1)
        hard_k_logits = self.hard_concrete(K_logits, temperature_overide)

        if self.training:
            batch_alpha_prob = batch_alpha
            dynamic_lambda = self.lambda_ * (1.0 - batch_alpha_prob)
            raw_loss = dynamic_lambda * intensity.sum(dim=-1) / self.total_rep_num
            text_sparsity_loss = 0.05 * text_relevance_prob.mean()
            tax_loss = (raw_loss + text_sparsity_loss).mean()
            return hard_k_logits, tax_loss
        return hard_k_logits


