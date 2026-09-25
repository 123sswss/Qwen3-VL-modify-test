"""PathVQA inference interface for the V2 position-specific conditional P20."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from slake.visual_selection_layer_mix import CONFIG_NAME, VisualSelectionLayerMixModel
from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface


class VisualSelectionLayerMixInterface(VisualSelectionPrefixInterface):
    """Reuse V1's exact question source, chat template, and decode protocol."""

    def __init__(self, checkpoint_path: str, base_model_path: str) -> None:
        checkpoint = Path(checkpoint_path)
        with (checkpoint / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        self.model = VisualSelectionLayerMixModel(base, init_seed=int(config["init_seed"]))
        self.model.load_v2(checkpoint)
        self.model.eval()
        self.device = next(base.parameters()).device
        self.last_generation_timing = None
        print(f"[V2_INTERFACE] checkpoint={checkpoint} parameters={self.model._audit_parameters()}")
