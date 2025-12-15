from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import MMRL
import config

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
)

# 中层网络还是深层网络插入？暂且设定为深层网络插入（17~23层）
MMRL_model = MMRL.MMRL(
    insert_layer_num=len(config.INSERT_LAYER),
    vision_token_dim=1024,
    text_token_dim=2560)

v_r_token_list, t_r_token_list = MMRL_model()

