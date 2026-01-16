##################################### init #####################################
USE_MMRL = False

SPECIAL_TOKENS = {"additional_special_tokens":
                      [f"<|REP_placeholder{i}|>" for i in range(40)]}
POOLING_DIM = 128

##################################### MMRL #####################################
RP_SPACE_LENGTH = 5
RP_SPACE_DIM = 512 # 一千二百万参数左右
# V:5*DIM -(DIM*1024)> 5*1024
# T:5*DIM -(DIM*2560)> 5*2560

INSERT_LAYER = range(17,23)

INSERT_METHOD = "replace"
# INSERT_METHOD = "add"

GATING_MID_DIM = 512
##################################### Gating #####################################
stretching_length = 0.1
gating_temperature = 2/3
text_gating_epsilon = 0.1
##################################### VPATCH #####################################
'''
# 协议：全局图总是放在所有图的最前面
VPATCH_SIMILARITY_METHOD = "IVTP"
# VPATCH_SIMILARITY_METHOD = "FALCON"
# VPATCH_SIMILARITY_METHOD = "cross attention"

VPATCH_COMPRESS_RATIO = 0.02

VPATCH_EMBEDDING_COMPRESS_METHOD = "attention pooling"
# VPATCH_EMBEDDING_COMPRESS_METHOD = "average pooling"
# VPATCH_EMBEDDING_COMPRESS_METHOD = "FALCON"
# 可以为全体VPATCH分配的token总量
VPATCH_TOTAL_NUM = 576
'''

