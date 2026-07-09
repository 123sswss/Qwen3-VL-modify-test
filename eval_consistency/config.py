"""Shared configuration for the output-disruption experiments."""

BASE_MODEL_PATH = "/root/autodl-tmp/model"
DATASET_JSON = "/root/autodl-tmp/dataset/llava_instruct_150k.json"
IMAGE_DIR = "/root/autodl-tmp/dataset/gen/train2017"

DEFAULT_NUM_SAMPLES = 500
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_SEED = 114514
DEFAULT_OUTPUT_ROOT = "./eval_consistency/results"

MODEL_KWARGS = {
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "attn_implementation": "sdpa",
}
