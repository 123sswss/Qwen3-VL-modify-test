##################################### init #####################################
USE_MMRL = True

SPECIAL_TOKENS = {"additional_special_tokens":
                      [f"<|REP_placeholder{i}|>" for i in range(40)]}
POOLING_DIM = 128

##################################### MMRL #####################################
RP_SPACE_LENGTH = 5
RP_SPACE_DIM = 512 # 一千二百万参数左右
# V:5*DIM -(DIM*1024)> 5*1024
# T:5*DIM -(DIM*2560)> 5*2560

INSERT_LAYER = range(17,25)

INSERT_METHOD = "replace"
# INSERT_METHOD = "add"

GATING_MID_DIM = 512
##################################### Gating #####################################
stretching_length = 0.1
gating_temperature = 2/3
text_gating_epsilon = 0.1


