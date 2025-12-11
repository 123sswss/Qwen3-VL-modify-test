import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3VLForConditionalGeneration

class modified_visual(nn.Module):
    def __init__(self, qwen):
        super().__init__()
        self.qwen = qwen
