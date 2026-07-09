from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image


class BaselineModelInterface:
    def __init__(self, model_path):
        print(f"加载基线模型: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype="auto", device_map="auto"
        )
        self.model.eval()
        print("基线模型就绪。")

    def infer(self, image: Image.Image, prompt_text: str, max_new_tokens=256, temperature=0.2, do_sample=False) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature
        )
        output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    import os
    import sys
    from test import (
        DEFAULT_IMAGE_DIRS,
        DEFAULT_JSON_PATHS,
        parse_cli_or_default,
        run_evaluation,
    )

    MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.getenv("QWEN3VL_MODEL_PATH", "/root/autodl-tmp/model")
    JSON_PATHS = parse_cli_or_default(sys.argv[2] if len(sys.argv) > 2 else None, DEFAULT_JSON_PATHS)
    IMAGE_DIRS = parse_cli_or_default(sys.argv[3] if len(sys.argv) > 3 else None, DEFAULT_IMAGE_DIRS)

    model = BaselineModelInterface(MODEL_PATH)
    run_evaluation(JSON_PATHS, model, image_dirs=IMAGE_DIRS)
