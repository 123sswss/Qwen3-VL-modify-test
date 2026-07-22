import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import List, Optional

class MMRL(nn.Module):
    def __init__(self,config):
        super().__init__()
        cfg = SimpleNamespace(**config.mmrl_config)
        self.use_direct_learnable_rep = bool(getattr(cfg, "ABLATE_DIRECT_LEARNABLE_REP", False))
        self.use_visual_conditioning = bool(getattr(cfg, "ENABLE_VISUAL_CONDITIONED_MMRL", False))
        self.insert_layer_count = len(cfg.INSERT_LAYER)
        self.rp_space_length = cfg.RP_SPACE_LENGTH
        self.rp_space_dim = cfg.RP_SPACE_DIM
        self.vision_token_dim = cfg.vision_token_dim
        self.shared_represent_space = nn.Parameter(torch.empty(cfg.RP_SPACE_LENGTH, cfg.RP_SPACE_DIM))
        nn.init.normal_(self.shared_represent_space, std=0.02)
        self.v_r_token_projector = nn.ModuleList([nn.Linear(cfg.RP_SPACE_DIM, cfg.vision_token_dim) for _ in range(len(cfg.INSERT_LAYER))])
        self.direct_v_tokens = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.RP_SPACE_LENGTH, cfg.vision_token_dim)) for _ in range(self.insert_layer_count)
        ])
        for param in self.direct_v_tokens:
            nn.init.normal_(param, std=0.02)

        if self.use_visual_conditioning:
            if self.use_direct_learnable_rep:
                raise ValueError(
                    "ENABLE_VISUAL_CONDITIONED_MMRL is incompatible with "
                    "ABLATE_DIRECT_LEARNABLE_REP"
                )
            condition_hidden_dim = int(getattr(cfg, "MMRL_CONDITION_HIDDEN_DIM", 128))
            if condition_hidden_dim <= 0:
                raise ValueError("MMRL_CONDITION_HIDDEN_DIM must be positive")
            self.visual_condition_norm = nn.LayerNorm(cfg.vision_token_dim)
            self.visual_condition_projector = nn.Sequential(
                nn.Linear(cfg.vision_token_dim, condition_hidden_dim),
                nn.SiLU(),
                nn.Linear(
                    condition_hidden_dim,
                    cfg.RP_SPACE_LENGTH * cfg.RP_SPACE_DIM,
                ),
            )
            # Preserve the exact shared-token baseline at initialization.
            nn.init.zeros_(self.visual_condition_projector[-1].weight)
            nn.init.zeros_(self.visual_condition_projector[-1].bias)
        else:
            self.visual_condition_norm = None
            self.visual_condition_projector = None

        self.cached_v_tokens = None
        self.condition_delta_ratio = torch.tensor(float("nan"))

    def _compute_tokens(
        self,
        visual_condition: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if self.use_direct_learnable_rep:
            return [param for param in self.direct_v_tokens]

        represent_space = self.shared_represent_space
        if visual_condition is not None:
            if not self.use_visual_conditioning:
                raise ValueError("visual_condition was provided while conditioning is disabled")
            if visual_condition.ndim != 2 or visual_condition.shape[-1] != self.vision_token_dim:
                raise ValueError(
                    "visual_condition must be [N, vision_token_dim], got "
                    f"{tuple(visual_condition.shape)}"
                )
            condition_delta = self.visual_condition_projector(
                self.visual_condition_norm(visual_condition)
            ).reshape(
                visual_condition.shape[0],
                self.rp_space_length,
                self.rp_space_dim,
            )
            represent_space = self.shared_represent_space.unsqueeze(0) + condition_delta
            with torch.no_grad():
                base_norm = self.shared_represent_space.detach().float().norm(dim=-1).mean()
                delta_norm = condition_delta.detach().float().norm(dim=-1).mean()
                self.condition_delta_ratio = (
                    delta_norm / base_norm.clamp_min(1e-8)
                ).to(device=visual_condition.device, dtype=visual_condition.dtype)
        else:
            self.condition_delta_ratio = self.shared_represent_space.new_tensor(float("nan"))

        return [vp(represent_space) for vp in self.v_r_token_projector]

    def forward(self, visual_condition: Optional[torch.Tensor] = None):
        if visual_condition is not None:
            self.cached_v_tokens = None
            return self._compute_tokens(visual_condition)
        if self.training:
            self.cached_v_tokens = None
            return self._compute_tokens()
        else:
            if self.cached_v_tokens is None:
                with torch.no_grad():
                    v_out = self._compute_tokens()
                    self.cached_v_tokens = [t.detach() for t in v_out]
            return self.cached_v_tokens
