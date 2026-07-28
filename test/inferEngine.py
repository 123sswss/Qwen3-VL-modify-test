import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LORATEST_DIR = PROJECT_ROOT / "loraTest"
for module_path in (PROJECT_ROOT, LORATEST_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    GenerationConfig
)
from safetensors.torch import load_file
import QWen3WithMMRL
import processingWithMMRL
from generation_timing import generate_with_timing


class MissingGateError(RuntimeError):
    fatal_evaluation_error = True


class Qwen3VLMMRLForGen(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.config = config
        current_vocab_size = len(tokenizer)
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, current_vocab_size, bias=False)
        self.generation_config = GenerationConfig.from_model_config(config)
        if tokenizer.pad_token_id is not None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id
        self.post_init()
        if self.model.visual.decouple_stage_pooling:
            self.model.visual.reset_router_pooling_from_gate()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class ModelInterface:
    """统一模型推理接口，方便对比实验"""

    def __init__(self, trained_model_path, base_model_path):
        self.last_gate_value = None
        self.last_generation_timing = None
        print("[1/3] 加载配置与构建模型架构...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print(f"    -> Tokenizer 词表大小: {len(self.tokenizer)}")

        try:
            config = AutoConfig.from_pretrained(trained_model_path, trust_remote_code=True)
            print("    -> 使用训练目录的 Config")
        except:
            print("    -> 回退使用基座 Config")
            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

        image_processor = AutoImageProcessor.from_pretrained(base_model_path, trust_remote_code=True)

        with torch.device("cuda"):
            self.model = Qwen3VLMMRLForGen(config, self.tokenizer)
            self.model.to(torch.bfloat16)

        print(f"[2/3] 加载训练权重: {trained_model_path} ...")
        safetensors_path = os.path.join(trained_model_path, "model.safetensors")
        bin_path = os.path.join(trained_model_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"在 {trained_model_path} 中未找到权重文件")

        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"    Missing keys: {msg.missing_keys}")
        print(f"    Unexpected keys: {msg.unexpected_keys}")
        self.model.eval()

        self.processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
            image_processor=image_processor, tokenizer=self.tokenizer
        )
        print("[3/3] 模型就绪。")

    def infer(self, image: Image.Image, prompt_text: str, max_new_tokens=256, temperature=0.0, do_sample=False) -> str:
        """单次推理，返回生成文本"""
        self.last_gate_value = None
        self.last_generation_timing = None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if hasattr(self.model.model, "rope_deltas"):
            self.model.model.rope_deltas = None

        inputs = self.processor(
            text=[text_prompt],
            images=image,
            padding=False,
            max_length=False,
            truncation=False,
            return_tensors="pt"
        ).to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            generated_ids, self.last_generation_timing = generate_with_timing(
                self.model,
                inputs,
                generate_kwargs,
            )
            gate = getattr(self.model.model.visual, "G_list", None)
            if not torch.is_tensor(gate) or gate.numel() == 0:
                raise MissingGateError(
                    "Checkpoint evaluation did not produce a valid visual G_list"
                )
            gate_values = gate.detach().float().reshape(-1).cpu().tolist()
            self.last_gate_value = (
                gate_values[0] if len(gate_values) == 1 else gate_values
            )
            input_len = inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_len:]
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
