# utils.py
import math
from typing import Optional

import torch
from torch import nn


def segment_mean(x: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    # 4/23修改，修改原因：为多图样本聚合提供稳定的分组 mean 归约，支撑新的 Top-K 文本门控视觉证据汇总。
    if x.dim() != 2:
        raise ValueError(f"x must be [N, D], got shape={tuple(x.shape)}")
    if batch_indices.dim() != 1 or batch_indices.size(0) != x.size(0):
        raise ValueError("batch_indices must be [N] and aligned with x")

    out = torch.zeros(batch_size, x.size(-1), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch_indices, x)

    counts = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    counts.index_add_(0, batch_indices, torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype))
    return out / counts.clamp_min(1.0)


def segment_max(x: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    # 4/23修改，修改原因：保留多图场景下单张关键图像的强信号，避免仅均值聚合导致尖峰证据被抹平。
    if x.dim() != 2:
        raise ValueError(f"x must be [N, D], got shape={tuple(x.shape)}")
    if batch_indices.dim() != 1 or batch_indices.size(0) != x.size(0):
        raise ValueError("batch_indices must be [N] and aligned with x")

    init = torch.finfo(x.dtype).min
    out = torch.full((batch_size, x.size(-1)), init, device=x.device, dtype=x.dtype)
    out.index_reduce_(0, batch_indices, x, reduce="amax", include_self=True)

    counts = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    counts.index_add_(0, batch_indices, torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype))
    return torch.where(counts > 0, out, torch.zeros_like(out))


class attention_pooling(nn.Module):
    """
    兼容原版接口：
        - __init__(input_dim, proj_dim)
        - forward(input_embeds, mask=None)
        - forward_vectorized(x, batch_indices, batch_size, valid_mask=None)

    新增可选参数：
        - num_heads: 多头数，默认 1；为 1 时行为退化为原单头版本
    """
    def __init__(self, input_dim, proj_dim, num_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_heads = max(1, int(num_heads))

        # 与原版保持风格一致：query 是可学习参数
        # 单头时形状 [1, D]，多头时形状 [H, D]
        self.query = nn.Parameter(torch.randn(self.num_heads, input_dim))

        # 保持接口和语义兼容：仍然是 q / e 两个投影器
        # 这里采用“共享 key/value 投影 + 多 query”的实现，
        # num_heads=1 时与原版几乎同构。
        self.projector_q = nn.Linear(input_dim, proj_dim)
        self.projector_e = nn.Linear(input_dim, proj_dim)

        # 多头输出融合到 input_dim；单头时就是 identity
        if self.num_heads == 1:
            self.out_proj = nn.Identity()
        else:
            self.out_proj = nn.Linear(self.num_heads * input_dim, input_dim, bias=False)
            self._init_out_proj_as_head_average()

        self.ln = nn.LayerNorm(input_dim)

    def _init_out_proj_as_head_average(self):
        """
        把多头融合层初始化成“各头平均”，这样初始行为更稳，
        不至于一上来就因为随机融合把池化输出打散。
        """
        with torch.no_grad():
            self.out_proj.weight.zero_()
            eye = torch.eye(self.input_dim, device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
            for h in range(self.num_heads):
                start = h * self.input_dim
                end = start + self.input_dim
                self.out_proj.weight[:, start:end] += eye / self.num_heads

    def _masked_softmax_dense(self, score: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        score:
            - 单头: [B, 1, S]
            - 多头: [B, H, S]
        mask:
            - [B, S] 或 [B, 1, S]
        """
        if mask is None:
            return torch.softmax(score, dim=-1)

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)  # [B, 1, S]
        elif mask.dim() != 3:
            raise ValueError(f"mask dim must be 2 or 3, got {mask.dim()}")

        mask = mask.to(device=score.device, dtype=torch.bool)
        # 广播到 [B, H, S]
        if mask.size(1) == 1 and score.size(1) != 1:
            mask = mask.expand(-1, score.size(1), -1)

        neg = torch.finfo(score.dtype).min
        score = score.masked_fill(~mask, neg)

        prob = torch.softmax(score, dim=-1)
        prob = prob * mask.to(prob.dtype)
        prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-6)
        return prob

    def forward(self, input_embeds, mask: Optional[torch.Tensor] = None):
        """
        input_embeds:
            - [B, S, D] 或 [S, D]
        返回:
            - [B, D] 或 [D]
        """
        squeeze_output = False
        if input_embeds.dim() == 2:
            input_embeds = input_embeds.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
            squeeze_output = True

        B, S, D = input_embeds.shape

        # q: [H, P]
        qq = self.projector_q(self.query.to(dtype=input_embeds.dtype))
        # k: [B, S, P]
        kk = self.projector_e(input_embeds)

        # score: [B, H, S]
        score = torch.einsum("hp,bsp->bhs", qq, kk) / math.sqrt(self.proj_dim)
        attn = self._masked_softmax_dense(score, mask)

        # 多头聚合原始输入，而不是投影后的 kk，保持与原版语义一致
        # g: [B, H, D]
        g = torch.einsum("bhs,bsd->bhd", attn, input_embeds)

        # [B, H*D] -> [B, D]
        g = g.reshape(B, self.num_heads * D)
        g = self.out_proj(g)
        g = self.ln(g)

        if squeeze_output:
            g = g.squeeze(0)
        return g

    def forward_vectorized(
        self,
        x: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """
        x: [Total_Tokens, D]
        batch_indices: [Total_Tokens]
        batch_size: int
        valid_mask: [Total_Tokens] bool，可选

        返回:
            [batch_size, D]
        """
        if x.dim() != 2:
            raise ValueError(f"x must be [N, D], got shape={tuple(x.shape)}")
        if batch_indices.dim() != 1:
            raise ValueError(f"batch_indices must be [N], got shape={tuple(batch_indices.shape)}")
        if x.size(0) != batch_indices.size(0):
            raise ValueError("x.size(0) must equal batch_indices.size(0)")

        N, D = x.shape
        device = x.device
        dtype = x.dtype

        if valid_mask is None:
            valid_mask = torch.ones(N, device=device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            if valid_mask.dim() != 1 or valid_mask.size(0) != N:
                raise ValueError("valid_mask must be [N]")

        # q: [H, P]
        qq = self.projector_q(self.query.to(dtype=dtype))  # [H, P]
        # k: [N, P]
        kk = self.projector_e(x)                           # [N, P]

        # logits: [N, H]
        logits = torch.einsum("np,hp->nh", kk, qq) / math.sqrt(self.proj_dim)
        logits = logits.to(dtype=dtype)

        neg = torch.finfo(dtype).min
        head_outputs = []

        # H 通常不大（如 4/8），循环头数是可接受的
        for h in range(self.num_heads):
            logits_h = logits[:, h]  # [N]
            masked_logits = torch.where(valid_mask, logits_h, torch.full_like(logits_h, neg))

            max_logits = torch.full((batch_size,), neg, device=device, dtype=dtype)
            max_logits.index_reduce_(0, batch_indices, masked_logits, reduce="amax", include_self=True)
            gathered_max = max_logits[batch_indices]

            fp32_masked_logits = masked_logits.float()
            fp32_gathered_max = gathered_max.float()
            exp_logits = torch.exp(fp32_masked_logits - fp32_gathered_max) * valid_mask.float()

            sum_exp = torch.zeros(batch_size, device=device, dtype=torch.float32)
            sum_exp.index_add_(0, batch_indices, exp_logits)
            gathered_sum = sum_exp[batch_indices]

            attn_weights = (exp_logits / (gathered_sum + 1e-6)).to(dtype)  # [N]

            weighted_input = x * attn_weights.unsqueeze(-1)    # [N, D]
            output_h = torch.zeros(batch_size, D, device=device, dtype=dtype)
            output_h.index_add_(0, batch_indices, weighted_input)  # [B, D]

            head_outputs.append(output_h)

        # [B, H, D]
        output = torch.stack(head_outputs, dim=1)
        # [B, H*D]
        output = output.reshape(batch_size, self.num_heads * D)
        output = self.out_proj(output)
        output = self.ln(output)
        return output