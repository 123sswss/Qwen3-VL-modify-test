from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

class QWen3WithMMRL(qwen3_vl.Qwen3VLModel):
    def __init__(self,
                 config,
                 ):
        super().__init__(config)
        self.MMRL =