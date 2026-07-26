"""Shared training and evaluation data protocol for every PEFT baseline."""


TRAIN_EXPERT_JSONS = (
    "/root/autodl-tmp/dataset/1json.json",
    "/root/autodl-tmp/dataset/2conv_c.json",
    "/root/autodl-tmp/dataset/1conv_c.json",
    "/root/autodl-tmp/dataset/4conv_c.json",
    "/root/autodl-tmp/dataset/14json.json",
    "/root/autodl-tmp/dataset/prof_test.json",
    "/root/autodl-tmp/dataset/test2_train.json",
    "/root/autodl-tmp/dataset/test7_train.json",
)

TRAIN_EXPERT_IMAGE_DIRS = (
    "/root/autodl-tmp/dataset/1/train",
    "/root/autodl-tmp/dataset/2/train",
    "/root/autodl-tmp/dataset/4/train",
    "/root/autodl-tmp/dataset/14",
)

TRAIN_GENERAL_JSONS = (
    "/root/autodl-tmp/dataset/llava_instruct_150k.json",
    "/root/autodl-tmp/dataset/gen_test.json",
    "/root/autodl-tmp/dataset/conversation_58k.json",
)

TRAIN_GENERAL_IMAGE_DIRS = (
    "/root/autodl-tmp/dataset/gen/train2017",
    "/root/autodl-tmp/dataset/gen/val2017",
)

EVAL_JSON_PATHS = (
    "/root/autodl-tmp/dataset/seen_simple/llava_test.json",
    "/root/autodl-tmp/dataset/test/14_test.json",
)

EVAL_IMAGE_DIRS = (
    "/root/autodl-tmp/dataset/seen_simple/image",
    "/root/autodl-tmp/dataset/test/14",
)

EVAL_MAX_NEW_TOKENS = 256


def _build_training_data_config():
    return {
        "expert_json": list(TRAIN_EXPERT_JSONS),
        "expert_img_dir": list(TRAIN_EXPERT_IMAGE_DIRS),
        "general_json": list(TRAIN_GENERAL_JSONS),
        "general_img_dir": list(TRAIN_GENERAL_IMAGE_DIRS),
        "total_limit": 20000,
        "enable_views": ("expert-mm",),
        "mode": "peft_ce",
        "ce_enabled": True,
        "sampling_seed": 42,
    }


def build_training_data_config():
    config = _build_training_data_config()
    config.update({
        "deterministic_sampling": True,
        "assistant_turn_policy": "joint",
    })
    return config


def build_legacy_training_data_config():
    config = _build_training_data_config()
    config.update({
        "deterministic_sampling": False,
        "assistant_turn_policy": "all",
    })
    return config
