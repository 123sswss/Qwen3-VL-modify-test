from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import config
import visual as V

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
)
model_config = model.model.language_model.config

VisionWithMMRL = V.VisionWithMMRL(model_config)
VisionBlockWithMMRL = V.VisionBlockWithMMRL(model_config)

