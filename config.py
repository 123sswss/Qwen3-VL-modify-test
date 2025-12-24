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
##################################### MMRL #####################################
RP_SPACE_LENGTH = 5
RP_SPACE_DIM = 512 # 一千二百万参数左右
# V:5*DIM -(DIM*1024)> 5*1024
# T:5*DIM -(DIM*2560)> 5*2560

MMRL_VISION_LORA_CONFIG = {
    "lora_dropout": 0.05,
    "bias": "none",
    "modules_to_save": ["linear"],
}
MMRL_VISION_LORA_DEFINITIONS = {
    "v_to_layer1":{"r": 8, "lora_alpha": 16},
    "v_to_layer2":{"r": 8, "lora_alpha": 16},
    "v_to_layer3":{"r": 8, "lora_alpha": 16},
    "v_to_layer4":{"r": 8, "lora_alpha": 16},
    "v_to_layer5":{"r": 8, "lora_alpha": 16},
    "v_to_layer6":{"r": 8, "lora_alpha": 16},
    "v_to_layer7":{"r": 8, "lora_alpha": 16},
    "v_to_layer8":{"r": 8, "lora_alpha": 16},
}

MMRL_TEXT_LORA_CONFIG = {
    "lora_dropout": 0.05,
    "bias": "none",
    "modules_to_save": ["linear"],
}
MMRL_TEXT_LORA_DEFINITIONS = {
    "t_to_layer1":{"r": 8, "lora_alpha": 16},
    "t_to_layer2":{"r": 8, "lora_alpha": 16},
    "t_to_layer3":{"r": 8, "lora_alpha": 16},
    "t_to_layer4":{"r": 8, "lora_alpha": 16},
    "t_to_layer5":{"r": 8, "lora_alpha": 16},
    "t_to_layer6":{"r": 8, "lora_alpha": 16},
    "t_to_layer7":{"r": 8, "lora_alpha": 16},
    "t_to_layer8":{"r": 8, "lora_alpha": 16},
}

INSERT_LAYER = range(17,23)

INSERT_METHOD = "replace"
# INSERT_METHOD = "add"

##################################### VPATCH #####################################
VPATCH_SIMILARITY_METHOD = "IVTP"
# VPATCH_SIMILARITY_METHOD = "FALCON"
# VPATCH_SIMILARITY_METHOD = "cross attention"

VPATCH_COMPRESS_RATIO = 0.02

VPATCH_RATING_DIM = 128

