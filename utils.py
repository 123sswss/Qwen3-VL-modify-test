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
        if x.dim() != 2:
            raise ValueError(f"x must have shape [tokens, dim], got {tuple(x.shape)}")
        if batch_indices.dim() != 1 or batch_indices.shape[0] != x.shape[0]:
            raise ValueError(
                "batch_indices must have one entry per token, got "
                f"{tuple(batch_indices.shape)} for x={tuple(x.shape)}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if valid_mask is None:
            valid_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        else:
            if valid_mask.dim() != 1 or valid_mask.shape[0] != x.shape[0]:
                raise ValueError(
                    "valid_mask must have one entry per token, got "
                    f"{tuple(valid_mask.shape)} for x={tuple(x.shape)}"
                )
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

        qq = self.projector_q(self.query.to(dtype=x.dtype)).squeeze(0)
        pooled = []
        # CUDA index_add_/index_reduce_ use unordered atomics for repeated indices.
        # A fixed per-sample reduction keeps the same attention formula reproducible.
        for sample_index in range(batch_size):
            sample_mask = (batch_indices == sample_index) & valid_mask
            sample_tokens = x[sample_mask]
            if sample_tokens.shape[0] == 0:
                pooled.append(x.new_zeros(x.shape[-1]))
                continue

            keys = self.projector_e(sample_tokens)
            logits = (keys * qq).sum(dim=-1) / math.sqrt(self.proj_dim)
            weights = torch.softmax(logits.to(dtype=x.dtype), dim=0)
            weights = weights / (weights.sum() + 1e-6)
            pooled.append((sample_tokens * weights.unsqueeze(-1)).sum(dim=0))

        return self.ln(torch.stack(pooled, dim=0))
