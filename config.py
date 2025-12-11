RP_SPACE_LENGTH = 128
RP_SPACE_DIM = 128

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