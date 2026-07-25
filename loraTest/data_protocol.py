"""Shared training and evaluation data protocol for every PEFT baseline."""


TRAIN_EXPERT_JSONS = (
    "/root/autodl-tmp/dataset/1json.json",
    "/root/autodl-tmp/dataset/1conv_c.json",
    "/root/autodl-tmp/dataset/14json.json",
)

TRAIN_EXPERT_IMAGE_DIRS = (
    "/root/autodl-tmp/dataset/1/train",
    "/root/autodl-tmp/dataset/14",
)

EVAL_JSON_PATHS = (
    "/root/autodl-tmp/dataset/test/1_test.json",
    "/root/autodl-tmp/dataset/test/14_test.json",
)

EVAL_IMAGE_DIRS = (
    "/root/autodl-tmp/dataset/test/1",
    "/root/autodl-tmp/dataset/test/14",
)

EVAL_MAX_NEW_TOKENS = 64


def build_training_data_config():
    return {
        "expert_json": list(TRAIN_EXPERT_JSONS),
        "expert_img_dir": list(TRAIN_EXPERT_IMAGE_DIRS),
        "general_json": [],
        "general_img_dir": [],
        "total_limit": 999999,
        "enable_views": ("expert-mm",),
        "mode": "peft_ce",
        "ce_enabled": True,
        "deterministic_sampling": True,
        "assistant_turn_policy": "joint",
    }
