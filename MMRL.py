import torch
import torch.nn as nn
from types import SimpleNamespace

class MMRL(nn.Module):
    def __init__(self,config):
        super().__init__()
        cfg = SimpleNamespace(**config.mmrl_config)
        self.use_direct_learnable_rep = bool(getattr(cfg, "ABLATE_DIRECT_LEARNABLE_REP", False))
        self.insert_layer_count = len(cfg.INSERT_LAYER)
        self.rp_space_length = cfg.RP_SPACE_LENGTH
        self.vision_token_dim = cfg.vision_token_dim
        self.text_token_dim = cfg.text_token_dim
        self.shared_represent_space = nn.Parameter(torch.empty(cfg.RP_SPACE_LENGTH, cfg.RP_SPACE_DIM))
        nn.init.normal_(self.shared_represent_space, std=0.02)
        self.v_r_token_projector = nn.ModuleList([nn.Linear(cfg.RP_SPACE_DIM, cfg.vision_token_dim) for _ in range(len(cfg.INSERT_LAYER))])
        self.t_r_token_projector = nn.ModuleList([nn.Linear(cfg.RP_SPACE_DIM, cfg.text_token_dim) for _ in range(len(cfg.INSERT_LAYER))])
        self.direct_v_tokens = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.RP_SPACE_LENGTH, cfg.vision_token_dim)) for _ in range(self.insert_layer_count)
        ])
        self.direct_t_tokens = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.RP_SPACE_LENGTH, cfg.text_token_dim)) for _ in range(self.insert_layer_count)
        ])
        for param in self.direct_v_tokens:
            nn.init.normal_(param, std=0.02)
        for param in self.direct_t_tokens:
            nn.init.normal_(param, std=0.02)

        self.cached_v_tokens = None
        self.cached_t_tokens = None

    def _compute_tokens(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.use_direct_learnable_rep:
            v_list = [param for param in self.direct_v_tokens]
            t_list = [param for param in self.direct_t_tokens]
            return v_list, t_list
        v_list = [vp(self.shared_represent_space) for vp in self.v_r_token_projector]
        t_list = [tp(self.shared_represent_space) for tp in self.t_r_token_projector]
        return v_list, t_list

    def forward(self):
        if self.training:
            self.cached_v_tokens = None
            self.cached_t_tokens = None
            return self._compute_tokens()
        else:
            if self.cached_v_tokens is None:
                with torch.no_grad():
                    v_out, t_out = self._compute_tokens()
                    self.cached_v_tokens = [t.detach() for t in v_out]
                    self.cached_t_tokens = [t.detach() for t in t_out]
            return self.cached_v_tokens, self.cached_t_tokens