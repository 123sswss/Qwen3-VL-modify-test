##################################### init #####################################
USE_MMRL = True
POOLING_DIM = 128

##################################### MMRL #####################################
RP_SPACE_LENGTH = 40
RP_SPACE_DIM = 512
MMRL_MEMORY_QUERY_COUNT = 128
MMRL_MEMORY_ATTENTION_DIM = 128
MMRL_MEMORY_POOLING_MODE = "independent"
MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT = 0.0
MMRL_MEMORY_SLOT_COSINE_MAX = 0.995
MMRL_PROJECTOR_HIDDEN_DIM = 1024
MMRL_CROSS_ATTENTION_HEADS = 8
MMRL_QUERY_ARCHITECTURE = "layer_mlp_post_cross"
MMRL_REP_UPDATE_MODE = "replace"
MMRL_LAYER_LORA_RANK = 0
MMRL_CA_LAYER_LORA_TARGET = "none"
MMRL_CA_LAYER_LORA_RANK = 0
MMRL_CA_LAYER_LORA_ALPHA = 1.0
# V:5*DIM -(DIM*1024)> 5*1024
# T:5*DIM -(DIM*2560)> 5*2560

# 0-based vision block indexes. Python range is left-closed/right-open.
# range(16, 24) means natural-language layers 17..24.
INSERT_LAYER = range(16, 24)

GATING_MID_DIM = 512
##################################### Gating #####################################
stretching_length = 0.1
gating_temperature = 2/3

