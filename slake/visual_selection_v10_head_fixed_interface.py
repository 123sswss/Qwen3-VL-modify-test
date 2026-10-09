"""Independent head-fixed loader, same P20 prefill/cache semantics."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from diagnostics.v10_head_fixed_protocol import METHOD, CONFIG_NAME
from slake.visual_selection_v10_head_fixed import VisualSelectionV10HeadFixedModel
from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface


class VisualSelectionV10HeadFixedInterface(VisualSelectionPrefixInterface):
    def __init__(self,checkpoint_path,base_model_path):
        checkpoint = Path(checkpoint_path)
        config = json.loads((checkpoint/CONFIG_NAME).read_text(encoding="utf-8"))
        if config["method"] != METHOD:
            raise ValueError("Wrong head-fixed method")
        self.processor = AutoProcessor.from_pretrained(base_model_path,trust_remote_code=True)
        base = AutoModelForImageTextToText.from_pretrained(base_model_path,torch_dtype=torch.bfloat16,
                                                        device_map="auto",trust_remote_code=True)
        self.model = VisualSelectionV10HeadFixedModel(base,init_seed=config["init_seed"])
        self.model.load_v10(checkpoint)
        self.model.eval()
        self.prefix_tokens = 20
        self.device = next(base.parameters()).device
        self.last_generation_timing = None
