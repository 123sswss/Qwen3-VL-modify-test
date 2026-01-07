import math

import torch
from torch import nn

class attention_pooling(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, input_dim))
        self.projector_q = nn.Linear(input_dim, proj_dim)
        self.projector_e = nn.Linear(input_dim, proj_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.vpatch_proj_dim = proj_dim

    def forward(self, input_embeds):
        # input_embeds: [B, S, D] 或 [S, D]
        qq = self.projector_q(self.query)
        kk = self.projector_e(input_embeds)
        d_k = self.vpatch_proj_dim

        score = torch.matmul(qq, kk.transpose(-1, -2)) / math.sqrt(d_k)
        score = torch.nn.functional.softmax(score, dim=-1)

        g = torch.matmul(score, input_embeds)

        return self.ln(g.squeeze(-2))