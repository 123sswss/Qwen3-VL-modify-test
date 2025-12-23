from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os

import QWen3WithMMRL as qwen3
import visual as qwen3Vison
import addSpecialToken
import config as cfg

model_id = os.path.abspath("../model/qwen3vl")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

model_config = model.model.language_model.config

######################### MMRL #########################
addSpecialToken.add_special_token(model, tokenizer, cfg.MMRL_SPECIAL_TOKENS)
# 主干模块
QWen3WithMMRL = qwen3.QWen3WithMMRL(config=model_config,
                            mode="train",
                            precomputed_path=os.path.abspath("./Rcache"))
# 视觉模块
VisionWithMMRL = qwen3Vison.VisionWithMMRL(model_config)
VisionBlockWithMMRL = qwen3Vison.VisionBlockWithMMRL(model_config)

model.model = QWen3WithMMRL
model.model.visual = VisionWithMMRL
model.model.visual.blocks = VisionBlockWithMMRL
######################### Vpatch #########################

