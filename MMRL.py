import torch
import torch.nn as nn
from types import SimpleNamespace

class MMRL(nn.Module):
    def __init__(self,config):
        super().__init__()
        cfg = SimpleNamespace(**config.mmrl_config)
        self.shared_represent_space = nn.Parameter(torch.empty(cfg.RP_SPACE_LENGTH, cfg.RP_SPACE_DIM))
        nn.init.normal_(self.shared_represent_space, std=0.02)
        self.v_r_token_projector = nn.ModuleList([nn.Linear(cfg.RP_SPACE_DIM, cfg.vision_token_dim) for _ in range(len(cfg.INSERT_LAYER))])
        self.t_r_token_projector = nn.ModuleList([nn.Linear(cfg.RP_SPACE_DIM, cfg.text_token_dim) for _ in range(len(cfg.INSERT_LAYER))])

        self.cached_v_tokens = None
        self.cached_t_tokens = None

    def _compute_tokens(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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