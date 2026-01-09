##################################### init #####################################
USE_MMRL = False
vision_token_dim=1024
text_token_dim=2560

SPECIAL_TOKENS = {
        "additional_special_tokens": ["<|text_R_token_start|>",
                                      "<|text_R_token_end|>",
                                      "<|text_R_token_placeholder|>",
                                      "<|request_zoom|>",
                                      "<|detail_start|>",
                                      "<|detail_end|>",
                                      "<|detail_placeholder|>",]
    }
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
##################################### MMRL Gating #####################################
stretching_length = 0.1
gating_temperature = 2/3
##################################### VPATCH #####################################
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

