"""Only the authorized PathVQA44 LN/default-init/Meta-Net-LR combination."""
from diagnostics.v10_protocol import EXPECTED_GROUP_COUNTS as ORIGINAL_COUNTS, GROUP_LRS as ORIGINAL_LRS

EXPERIMENT = "pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44"
METHOD = "visual_selection_v10_condition_head_fixed"
CONFIG_NAME = "v10_head_fixed_config.json"
WEIGHTS_NAME = "v10_head_fixed.pt"
GROUP_COUNTS = {**ORIGINAL_COUNTS, "summary_norm":5120}
TOTAL = sum(GROUP_COUNTS.values())
GROUP_LRS = {**ORIGINAL_LRS,"meta_net":3e-4,"summary_norm":1e-4}
