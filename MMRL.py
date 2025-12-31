import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from collections import OrderedDict
from typing import Tuple, List

import config as cfg

def create_lora_configs(task_definitions, base_config):
    lora_configs = {}
    for vision_adapter_name, overrides in task_definitions.items():
        current_config_dict = base_config.copy()
        if overrides:
            current_config_dict.update(overrides)
        lora_configs[vision_adapter_name] = LoraConfig(**current_config_dict)
    return lora_configs


def _create_peft_projector(lora_definitions, lora_config, in_features, out_features):
    # base_projector = nn.Linear(in_features, out_features)
    base_projector = nn.Sequential(
        OrderedDict([
            ("linear", nn.Linear(in_features, out_features))
        ])
    )
    lora_config_map = create_lora_configs(lora_definitions, lora_config)
    peft_projector = None
    for name, lconfig in lora_config_map.items():
        if peft_projector is None:
            peft_projector = get_peft_model(base_projector, lconfig, adapter_name=name)
        else:
            peft_projector.add_adapter(name, lconfig)

    return peft_projector


class MMRL(nn.Module):
    def __init__(self,
                 insert_layer_num: int,
                 vision_token_dim: int,
                 text_token_dim: int):
        super().__init__()
        assert insert_layer_num == len(cfg.MMRL_VISION_LORA_DEFINITIONS.keys()), \
            "insert_layer_num must be equal to the number of vision adapters"
        self.shared_represent_space = nn.Parameter(torch.empty(cfg.RP_SPACE_LENGTH, cfg.RP_SPACE_DIM))
        nn.init.normal_(self.shared_represent_space, std=0.02)
        self.r2v_backbone_projector = _create_peft_projector(
            cfg.MMRL_VISION_LORA_DEFINITIONS,
            cfg.MMRL_VISION_LORA_CONFIG,
            cfg.RP_SPACE_DIM,
            vision_token_dim
        )
        self.r2t_backbone_projector = _create_peft_projector(
            cfg.MMRL_TEXT_LORA_DEFINITIONS,
            cfg.MMRL_TEXT_LORA_CONFIG,
            cfg.RP_SPACE_DIM,
            text_token_dim
        )
        self.v_adapter_keys = list(cfg.MMRL_VISION_LORA_DEFINITIONS.keys())
        self.t_adapter_keys = list(cfg.MMRL_TEXT_LORA_DEFINITIONS.keys())
        self.cached_v_tokens = None
        self.cached_t_tokens = None

    def _compute_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        v_list = []
        for name in self.v_adapter_keys:
            self.r2v_backbone_projector.set_adapter(name)
            out = self.r2v_backbone_projector(self.shared_represent_space)
            v_list.append(out)
        t_list = []
        for name in self.t_adapter_keys:
            self.r2t_backbone_projector.set_adapter(name)
            out = self.r2t_backbone_projector(self.shared_represent_space)
            t_list.append(out)
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

            # 直接返回缓存
            return self.cached_v_tokens, self.cached_t_tokens








