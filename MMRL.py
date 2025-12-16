import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from typing import Optional
from collections import OrderedDict
from safetensors import safe_open
from safetensors.torch import save_file

import config as cfg

def create_lora_configs(task_definitions, base_config):
    """
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
    def __init__(self,
                 insert_layer_num: int,
                 vision_token_dim: int,
                 text_token_dim: int,
                 mode: str,
                 precomputed_path: Optional[str] = None):
        """
        初始化 MMRL 模块。

        Args:
            insert_layer_num (int): 要插入的层数，必须与LoRA定义数量一致。
            vision_token_dim (int): 视觉分支输出token的维度。
            text_token_dim (int): 文本分支输出token的维度。
            mode (str, optional): 模块的运行模式，'train' 或 'inference'。默认为 'train'。
            precomputed_path (Optional[str], optional): 在 'inference' 模式下，
                                                       预计算张量文件的路径。默认为 None。
        """
        super().__init__()
        self.mode = mode

        if self.mode == 'train':
            # --- 训练模式：初始化所有组件 ---
            print("MMRL is initialized in 'train' mode.")
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

        elif self.mode == 'inference':
            if precomputed_path is None:
                raise ValueError("In 'inference' mode, 'precomputed_path' must be provided.")

            loaded_tensors = {}
            metadata = {}
            with safe_open(precomputed_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata is None:
                    raise RuntimeError("Metadata not found in safetensors file. Cannot reconstruct token lists.")
                for key in f.keys():
                    loaded_tensors[key] = f.get_tensor(key)
            num_vision_tokens = int(metadata.get('num_vision_tokens', '0'))
            num_text_tokens = int(metadata.get('num_text_tokens', '0'))

            self.v_r_token_list = [loaded_tensors[f'vision_{i}'] for i in range(num_vision_tokens)]
            self.t_r_token_list = [loaded_tensors[f'text_{i}'] for i in range(num_text_tokens)]
            print(f"Successfully loaded {len(self.v_r_token_list)} vision tokens and {len(self.t_r_token_list)} text tokens.")

        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from 'train' or 'inference'.")

    def forward(self):
        """
        前向传播。
        在训练模式下，执行计算。
        在推理模式下，直接返回预加载的张量。
        """
        if self.mode == 'train':
            # --- 训练模式的计算逻辑 ---
            v_r_token_list = []
            for vision_adapter_name in self.v_lora_config_map.keys():
                with self.r2v_backbone_projector.adapter_config(vision_adapter_name):
                    projected_output = self.r2v_backbone_projector(self.shared_represent_space)
                    v_r_token_list.append(projected_output)

            t_r_token_list = []
            for text_adapter_name in self.t_lora_config_map.keys():
                with self.r2t_backbone_projector.adapter_config(text_adapter_name):
                    projected_output = self.r2t_backbone_projector(self.shared_represent_space)
                    t_r_token_list.append(projected_output)

            return v_r_token_list, t_r_token_list

        elif self.mode == 'inference':
            # --- 推理模式直接返回结果 ---
            # 用户可以按需将张量移动到特定设备
            # e.g., v_tokens = [t.to('cuda') for t in self.v_r_token_list]
            return self.v_r_token_list, self.t_r_token_list

        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from 'train' or 'inference'.")

    def save_precomputed_tensors(self, file_path: str):
        """
        计算并以 safetensors 格式保存输出的张量列表。
        此方法只能在 'train' 模式下调用。

        Args:
            file_path (str): 保存 .safetensors 文件的路径。
        """
        if self.mode != 'train':
            raise RuntimeError("This method can only be called when the module is in 'train' mode.")

        print("Generating precomputed tensors for saving...")
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            v_r_token_list, t_r_token_list = self.forward()

        # 创建一个扁平的字典以便 safetensors 保存
        flat_token_dict = {}
        for i, tensor in enumerate(v_r_token_list):
            # 保存前最好将张量移到CPU，以保证通用性
            flat_token_dict[f'vision_{i}'] = tensor.cpu()
        for i, tensor in enumerate(t_r_token_list):
            flat_token_dict[f'text_{i}'] = tensor.cpu()

        # 保存元数据，用于在加载时恢复列表结构
        metadata = {
            'num_vision_tokens': str(len(v_r_token_list)),
            'num_text_tokens': str(len(t_r_token_list))
        }

        # 保存到文件
        save_file(flat_token_dict, file_path, metadata=metadata)
        print(f"Precomputed tokens have been successfully saved to {file_path}")








