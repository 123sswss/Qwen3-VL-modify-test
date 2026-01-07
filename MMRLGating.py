from typing import Optional

import torch
from torch import nn
import config as cfg

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
        #todo:将alpha经过sigmoid之后进行loss计算
        return alpha

class Gating(nn.Module):
    def __init__(self,):
        super().__init__()
        self.gating_temperature = cfg.gating_temperature
        self.upper_bounds = 1 + cfg.stretching_length
        self.lower_bounds = 0 - cfg.stretching_length

    def forward(self,
                alpha: torch.Tensor,
                temperature_overide: Optional[float] = None):

        if temperature_overide is not None:
            temperature = temperature_overide
        else:
            temperature = self.gating_temperature

        if self.training:
            u = torch.rand_like(alpha)
            u = torch.clamp(u, min=1e-7, max=1 - 1e-7)
            epsilon = torch.log(u) - torch.log(1 - u)
        else:
            epsilon = 0
        # 此处alpha未经sigmoid
        s = torch.sigmoid((epsilon + alpha) / temperature)

        ss = s * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        G = torch.clip(ss, 0, 1)
        return G

