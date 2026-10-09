"""Five-epoch head-fixed V10: only P20 base LR changes from 0.3 to 0.1."""
from slake.train_v10 import main
from slake.visual_selection_v10_head_fixed import VisualSelectionV10HeadFixedModel
from diagnostics.v10_head_fixed_protocol import EXPERIMENT, METHOD, TOTAL, GROUP_LRS

if __name__ == "__main__":
    main(model_class=VisualSelectionV10HeadFixedModel,
         experiment_override=EXPERIMENT.replace("_norm_fixed_", "_p20lr01_norm_fixed_"),
         method=METHOD, expected_trainable=TOTAL, group_lrs={**GROUP_LRS, "p20":0.1})
