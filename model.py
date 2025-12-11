from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import MMRL

qwen = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
)

MMRL = MMRL.MMRL()
