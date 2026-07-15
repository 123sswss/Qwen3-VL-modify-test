import math
from typing import Optional

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

    def _masked_softmax_dense(self, score: torch.Tensor, mask: Optional[torch.Tensor]):
        # score: [B, 1, S]
        if mask is None:
            return torch.softmax(score, dim=-1)

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)  # [B, 1, S]
        elif mask.dim() != 3:
            raise ValueError(f"mask dim must be 2 or 3, got {mask.dim()}")

        mask = mask.to(device=score.device, dtype=torch.bool)

        neg = torch.finfo(score.dtype).min
        score = score.masked_fill(~mask, neg)

        prob = torch.softmax(score, dim=-1)
        prob = prob * mask.to(prob.dtype)
        prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-6)
        return prob

    def forward(self, input_embeds, mask: Optional[torch.Tensor] = None):
        # input_embeds: [B, S, D] 或 [S, D]
        squeeze_output = False
        if input_embeds.dim() == 2:
            input_embeds = input_embeds.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
            squeeze_output = True

        B, S, D = input_embeds.shape

        qq_raw = self.projector_q(self.query.to(dtype=input_embeds.dtype))
        qq = qq_raw.expand(B, 1, -1)

        kk = self.projector_e(input_embeds)

        score = torch.bmm(qq, kk.transpose(1, 2)) / math.sqrt(self.proj_dim)
        score = self._masked_softmax_dense(score, mask)

        g = torch.bmm(score, input_embeds)
        g = g.view(B, D)
        g = self.ln(g)

        if squeeze_output:
            g = g.squeeze(0)
        return g
    # 4/17 增加valid_mask
    def forward_vectorized(
        self,
        x: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        # x: [Total_Tokens, D]
        # batch_indices: [Total_Tokens]
        qq = self.projector_q(self.query.to(dtype=x.dtype))          # [1, P]
        kk = self.projector_e(x)                                     # [Total_Tokens, P]
        logits = (kk * qq).sum(dim=-1) / math.sqrt(self.proj_dim)    # [Total_Tokens]
        logits = logits.to(dtype=x.dtype)

        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

        neg = torch.finfo(logits.dtype).min
        masked_logits = torch.where(valid_mask, logits, torch.full_like(logits, neg))

        max_logits = torch.full((batch_size,), neg, device=x.device, dtype=x.dtype)
        max_logits.index_reduce_(0, batch_indices, masked_logits, reduce="amax", include_self=True)
        gathered_max = max_logits[batch_indices]

        exp_logits = (torch.exp(masked_logits - gathered_max) * valid_mask.to(x.dtype)).to(x.dtype)

        sum_exp = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        sum_exp.index_add_(0, batch_indices, exp_logits)
        gathered_sum = sum_exp[batch_indices]

        attn_weights = exp_logits / (gathered_sum + 1e-6)  # [Total_Tokens]

        weighted_input = x * attn_weights.unsqueeze(-1)
        output = torch.zeros(batch_size, x.shape[-1], device=x.device, dtype=x.dtype)
        output.index_add_(0, batch_indices, weighted_input)

        return self.ln(output)