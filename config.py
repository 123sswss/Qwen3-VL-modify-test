##################################### init #####################################
USE_MMRL = True
POOLING_DIM = 128

##################################### MMRL #####################################
RP_SPACE_LENGTH = 5
RP_SPACE_DIM = 512 # 一千二百万参数左右
# V:5*DIM -(DIM*1024)> 5*1024
# T:5*DIM -(DIM*2560)> 5*2560

# 0-based vision block indexes. Python range is left-closed/right-open.
# range(16, 24) means natural-language layers 17..24.
INSERT_LAYER = range(16, 24)

INSERT_METHOD = "replace"
# INSERT_METHOD = "add"

GATING_MID_DIM = 512
##################################### Gating #####################################
stretching_length = 0.1
gating_temperature = 2/3

