import torch
import torch.nn as nn
import math

import config as cfg

class attention_pooling(nn.Module):
    def __init__(self,):
        super().__init__()
        #todo:query_cache
        self.query = nn.Parameter(torch.randn(1, cfg.text_token_dim))
        self.projector_q = nn.Linear(cfg.text_token_dim, cfg.VPATCH_RATING_DIM)
        self.projector_e = nn.Linear(cfg.text_token_dim, cfg.VPATCH_RATING_DIM)

    def forward(self, input_embeds):
        qq = self.projector_q(self.query)
        kk = self.projector_e(input_embeds)
        score = torch.matmul(qq, kk.T)/math.sqrt(cfg.VPATCH_RATING_DIM)
        score = torch.nn.functional.softmax(score, dim=-1)
        g = torch.matmul(score, input_embeds)
        return g






def IVTP_similarity(roi_hidden_states, input_embeds):
    pass