"""Independent seven-epoch budget; same model/init/optimizer, save5/6/7."""
from slake.train_v10 import main
from slake.visual_selection_v10_head_fixed import VisualSelectionV10HeadFixedModel
from diagnostics.v10_head_fixed_protocol import EXPERIMENT, METHOD, TOTAL, GROUP_LRS

if __name__ == "__main__":
    main(model_class=VisualSelectionV10HeadFixedModel,
         experiment_override=EXPERIMENT.replace("_5ep_","_7ep_"),
         method=METHOD,expected_trainable=TOTAL,group_lrs=GROUP_LRS,
         default_epochs=7,default_save_epochs=(5,6,7))
