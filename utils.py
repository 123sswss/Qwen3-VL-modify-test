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
        self.proj_dim = proj_dim

    def forward(self, input_embeds):
        # input_embeds: [B, S, D] 或 [S, D]
        qq = self.projector_q(self.query)
        kk = self.projector_e(input_embeds)
        d_k = self.proj_dim

        score = torch.matmul(qq, kk.transpose(-1, -2)) / math.sqrt(d_k)
        score = torch.nn.functional.softmax(score, dim=-1)

        g = torch.matmul(score, input_embeds)

        return self.ln(g.squeeze(-2))

    def forward_vectorized(self, x, batch_indices, batch_size):
        # qq: [1, Proj_Dim]
        qq = self.projector_q(self.query)
        # kk: [Total_Tokens, Proj_Dim]
        kk = self.projector_e(x)
        # logits: [Total_Tokens]
        logits = (kk * qq).sum(dim=-1) / math.sqrt(self.proj_dim)
        logits = logits.to(dtype=x.dtype)
        max_logits = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        max_logits.fill_(-1e9)
        max_logits.index_reduce_(0, batch_indices, logits, reduce="amax", include_self=True)
        gathered_max = max_logits[batch_indices]
        exp_logits = torch.exp(logits - gathered_max)
        sum_exp = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        exp_logits = exp_logits.to(dtype=x.dtype)
        sum_exp.index_add_(0, batch_indices, exp_logits)
        gathered_sum = sum_exp[batch_indices]
        attn_weights = exp_logits / (gathered_sum + 1e-6)  # [Total_Tokens]
        weighted_input = x * attn_weights.unsqueeze(-1)
        output = torch.zeros(batch_size, x.shape[-1], device=x.device, dtype=x.dtype)
        output.index_add_(0, batch_indices, weighted_input)
        return self.ln(output)