# infer_baseline_gguf.py
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
import io, base64


class BaselineModelInterface:
    def __init__(self, model_path, mmproj_path):
        print(f"加载GGUF模型: {model_path}")
        print(f"加载MMProj: {mmproj_path}")
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        self.model = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=-1,
            n_ctx=4096,
            n_batch=2048,
            verbose=False,
        )
        print("基线模型就绪。")

    @staticmethod
    def _pil_to_data_uri(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def infer(self, image: Image.Image, prompt_text: str, max_new_tokens=256, temperature=0.2, do_sample=True) -> str:
        image_uri = self._pil_to_data_uri(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        resp = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=0.9,
            stream=False,
        )
        return resp["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    import sys
    from test import run_evaluation

    # 用法:
    # python infer_gguf.py /path/to/model.gguf /path/to/mmproj-BF16.gguf /path/test.json /path/images
    MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/model/gemma-3-4b-it-UD-Q6_K_XL.gguf"
    MMPROJ_PATH = sys.argv[2] if len(sys.argv) > 2 else "/root/model/mmproj-F16.gguf"
    JSON_PATH = sys.argv[3] if len(sys.argv) > 3 else "/root/autodl-tmp/dataset/test2_val.json"
    IMAGE_DIR = sys.argv[4] if len(sys.argv) > 4 else "/root/autodl-tmp/dataset/2/train"

    model = BaselineModelInterface(MODEL_PATH, MMPROJ_PATH)
    run_evaluation(JSON_PATH, model, image_dir=IMAGE_DIR)