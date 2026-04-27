import os

##################################### init #####################################
USE_MMRL = True
POOLING_DIM = 128

##################################### MMRL #####################################
RP_SPACE_LENGTH = 5
RP_SPACE_DIM = 512 # 一千二百万参数左右
# V:5*DIM -(DIM*1024)> 5*1024
# T:5*DIM -(DIM*2560)> 5*2560

# 用 tuple 保证长度稳定、可安全 len()，同时便于序列化/打印
INSERT_LAYER = tuple(range(17, 25))

INSERT_METHOD = "replace"
# INSERT_METHOD = "add"

NUM_INSERT_LAYERS = len(INSERT_LAYER)

# 控制变量实验：视觉侧 rep token 数保持不变；仅扩展文本侧 soft prompt 总量。
# 默认 1 表示与视觉保持一一对应；设为 2 时，文本侧总量从 40 扩到 80。
TEXT_REP_EXPAND_FACTOR = int(os.getenv("MMRL_TEXT_REP_EXPAND_FACTOR", "1"))
TOTAL_REP_TOKENS = RP_SPACE_LENGTH * NUM_INSERT_LAYERS
TOTAL_TEXT_REP_TOKENS = TOTAL_REP_TOKENS * TEXT_REP_EXPAND_FACTOR

SPECIAL_TOKENS = {
    "additional_special_tokens": [f"<|REP_placeholder{i}|>" for i in range(TOTAL_TEXT_REP_TOKENS)]
}

GATING_MID_DIM = 512
##################################### Gating #####################################
stretching_length = 0.1
gating_temperature = 2/3
text_gating_epsilon = 0.1


