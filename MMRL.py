import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

import config as cfg

def create_lora_configs(task_definitions, base_config):
    """
    一个工厂函数，用于批量创建 LoraConfig 对象。

    Args:
        task_definitions (dict): 一个字典，key是任务名，value是该任务的特定配置字典。
        base_config (dict): 包含所有任务共享参数的基础配置字典。

    Returns:
        dict: 一个字典，key是任务名，value是生成的 LoraConfig 对象。
    """
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
    def __init__(self, insert_layer_num:int, vision_token_dim:int, text_token_dim:int):
        super().__init__()
        assert insert_layer_num == len(cfg.MMRL_VISION_LORA_DEFINITIONS.keys()), \
            "insert_layer_num must be equal to the number of vision adapters"
        assert insert_layer_num == len(cfg.MMRL_TEXT_LORA_DEFINITIONS.keys()), \
            "insert_layer_num must be equal to the number of text adapters"

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

        self.v_lora_config_map = cfg.MMRL_VISION_LORA_DEFINITIONS
        self.t_lora_config_map = cfg.MMRL_TEXT_LORA_DEFINITIONS

    def forward(self):
        v_r_token_list = []
        ########################## vision part ##########################
        for vision_adapter_name in self.v_lora_config_map.keys():
            with self.r2v_backbone_projector.adapter_config(vision_adapter_name):
                projected_output = self.r2v_backbone_projector(self.shared_represent_space)
                v_r_token_list.append(projected_output)

        ########################## text part ##########################
        t_r_token_list = []
        for text_adapter_name in self.t_lora_config_map.keys():
            with self.r2t_backbone_projector.adapter_config(text_adapter_name):
                projected_output = self.r2t_backbone_projector(self.shared_represent_space)
                t_r_token_list.append(projected_output)

        return v_r_token_list, t_r_token_list








