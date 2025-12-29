import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import config as cfg

class _Vsimilarity(nn.Module):
    def __init__(self,):
        super().__init__()
        if cfg.VPATCH_EMBEDDING_COMPRESS_METHOD == "attention pooling":
            self.pooling = attention_pooling()
        elif cfg.VPATCH_EMBEDDING_COMPRESS_METHOD == "average":
            self.pooling = lambda x: x.mean(dim=1)
        elif cfg.VPATCH_EMBEDDING_COMPRESS_METHOD == "FALCON":
            self.pooling = FALCON_embedding_compress()
        else:
            raise NotImplementedError

class attention_pooling(nn.Module):
    def __init__(self,):
        super().__init__()
        #todo:query_cache

        self.query = nn.Parameter(torch.randn(1, cfg.text_token_dim))
        self.projector_q = nn.Linear(cfg.text_token_dim, cfg.VPATCH_RATING_DIM)
        self.projector_e = nn.Linear(cfg.text_token_dim, cfg.VPATCH_RATING_DIM)
        self.ln = nn.LayerNorm(cfg.text_token_dim)

    def forward(self, input_embeds):
        # input_embeds: [B, S, D]
        batch_size = input_embeds.shape[0]
        qq = self.projector_q(self.query).unsqueeze(0).expand(batch_size, -1, -1) # [B, 1, R_DIM]
        kk = self.projector_e(input_embeds) # [B, S, R_DIM]
        score = torch.matmul(qq, kk.transpose(-1, -2))/math.sqrt(cfg.VPATCH_RATING_DIM)
        score = torch.nn.functional.softmax(score, dim=-1)
        g = torch.matmul(score, input_embeds) # [B, 1, D]
        return self.ln(g.squeeze(1))

class FALCON_embedding_compress(nn.Module):
    def __init__(self,):
        super().__init__()
        self.register = nn.Parameter(torch.randn(1, 1, cfg.text_token_dim))
        self.attention_head = nn.Linear(cfg.text_token_dim, cfg.text_token_dim * 3)
        self.ln = nn.LayerNorm(cfg.text_token_dim)

    def forward(self, input_embeds):
        # input_embeds: [B, S, D]
        batch_size = input_embeds.shape[0]
        reg = self.register.expand(batch_size, -1, -1)  # [B, 1, D]
        combined = torch.cat([input_embeds, reg], dim=1)  # [B, S+1, D]
        qkv = self.attention_head(combined).chunk(3, dim=-1)
        q, k, v = qkv
        output = F.scaled_dot_product_attention(q, k, v)
        output = output[:, -1, :]  # [B, D]
        return self.ln(output)

class IVTP_similarity(_Vsimilarity):
    def __init__(self,):
        super().__init__()
        self.v_projector = nn.Linear(cfg.text_token_dim, cfg.VPATCH_RATING_DIM)
        self.t_projector = nn.Linear(cfg.text_token_dim, cfg.VPATCH_RATING_DIM)
        # ln(1/0.07) 约等于 2.65
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.65)

    def forward(self, roi_hidden_states, input_embeds):
        pic_hs = F.normalize(self.v_projector(roi_hidden_states), p=2, dim=-1)  # [N, R_DIM]
        text_feat = self.pooling(input_embeds)  # [B, D]
        text_hs = F.normalize(self.t_projector(text_feat), p=2, dim=-1)  # [B, R_DIM]
        if text_hs.dim() > 1 and text_hs.size(0) > 1:
            text_hs = text_hs.mean(dim=0, keepdim=True)  # [1, R_DIM]
        elif text_hs.dim() == 1:
            text_hs = text_hs.unsqueeze(0)  # [1, R_DIM]
        scale = self.logit_scale.exp()
        score = torch.matmul(text_hs, pic_hs.transpose(-1, -2)) * scale

        return score.view(-1)  # [N]
