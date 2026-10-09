"""RSVQA-LR, original five-epoch V10 head-fixed configuration, seed44."""
from slake.train_v10 import main
from slake.visual_selection_v10_head_fixed import VisualSelectionV10HeadFixedModel
from diagnostics.v10_head_fixed_protocol import EXPERIMENT, METHOD, TOTAL, GROUP_LRS

if __name__ == "__main__":
    main(model_class=VisualSelectionV10HeadFixedModel,
         experiment_override=EXPERIMENT.replace("pathvqa_", "rsvqa_lr_", 1),
         method=METHOD, expected_trainable=TOTAL, group_lrs=GROUP_LRS)
