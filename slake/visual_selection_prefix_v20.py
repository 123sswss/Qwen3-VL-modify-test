"""Exploratory V20: V1 unchanged except selection reads blocks7/15/23."""
from slake.visual_selection_prefix import VisualSelectionPrefixModel

class VisualSelectionPrefixV20Model(VisualSelectionPrefixModel):
    method_name = "visual_selection_prefix_v20_deeper_three"
    selection_layers = (7,15,23)

    def __init__(self,base_model,init_seed=44):
        if len(base_model.model.visual.blocks)!=24:
            raise ValueError("V20 requires exactly 24 ViT blocks")
        super().__init__(base_model,init_seed=init_seed)
        print("[V20_LAYER_AUDIT] selection_indices=7,15,23 natural_layers=8,16,24 visual_prompt_index=17",flush=True)
