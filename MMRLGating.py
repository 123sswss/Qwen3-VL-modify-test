#todo:给这个模块设计一个独特的LOSS
import torch
from torch import nn
import config as cfg

class MMRL_gating(nn.Module):
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
        alpha = torch.sigmoid(self.output_head(combined))
        return alpha