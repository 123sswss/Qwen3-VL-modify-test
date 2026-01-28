import os
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    GenerationConfig
)

# 导入你项目中的自定义模块
import config as cfg
import QWen3WithMMRL
import processingWithMMRL
# 复用 train.py 中的 Wrapper 类定义，确保结构一致以便加载权重
from transformers import Qwen3VLForConditionalGeneration

# ==============================================================================
# 1. 必须重定义 Wrapper 类 (与 train.py 保持一致以匹配权重 Key)
# ==============================================================================
class Qwen3VLMMRLForTrain(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.config = config
        current_vocab_size = len(tokenizer)

        # 初始化魔改 Base Model
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)

        # 初始化 LM Head
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, current_vocab_size, bias=False)

        # 生成配置
        self.generation_config = GenerationConfig.from_model_config(config)
        if tokenizer.pad_token_id is not None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, images_per_sample=None, **kwargs):
        # 推理时主要依赖父类的 generate -> forward 流程
        # 这里的 forward 主要是为了兼容训练时的调用接口
        # 实际 generate 会调用 self.model.forward
        return super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)


# ==============================================================================
# 2. 推理加载函数
# ==============================================================================
def load_mmrl_model(base_model_path, checkpoint_dir, device="cuda"):
    print(f"-> 正在加载基础配置: {base_model_path}")
    
    # 1. 加载基础配置和工具
    # 注意：通常 checkpoint_dir 里会有 config.json，优先用那个，否则用 base 的
    config_path = checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "config.json")) else base_model_path
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS) # 别忘了加特殊 Token，否则 embedding 维度对不上
    
    image_processor = AutoImageProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 2. 初始化空模型架构
    print("-> 正在构建自定义 MMRL 模型架构...")
    # 必须在 device 上初始化，或者初始化后搬运，推荐 bfloat16
    with torch.device(device):
        model = Qwen3VLMMRLForTrain(config, tokenizer)
        model.to(dtype=torch.bfloat16)

    # 3. 加载训练好的权重
    print(f"-> 正在加载训练权重: {checkpoint_dir}")
    # 寻找 bin 或 safetensors
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        # 如果是大模型分片存储，或者是 safetensors，需要做相应处理
        # 这里假设是 Trainer 默认生成的单个 bin 文件，或者是 safetensors
        safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        else:
            # 尝试加载分片索引（如果只是简单验证脚本，建议先 merge 权重或只用 latest checkpoint）
            raise FileNotFoundError(f"在 {checkpoint_dir} 中找不到权重文件 (pytorch_model.bin 或 model.safetensors)")
    else:
        state_dict = torch.load(bin_path, map_location="cpu")

    # 4. 载入 State Dict
    # strict=False 是为了容错，但对于自定义模型，最好看下 missing keys
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"-> 权重加载结果: missing={len(keys.missing_keys)}, unexpected={len(keys.unexpected_keys)}")
    
    # 确保 MMRL 相关的键没有丢失
    mmrl_missing = [k for k in keys.missing_keys if "MMRL" in k or "visual.blocks_with_rep" in k]
    if mmrl_missing:
        print("[Warning] ⚠️ MMRL 关键参数似乎丢失了，请检查权重文件！")
        print(mmrl_missing[:5])
    
    model.eval()
    return model, tokenizer, image_processor


# ==============================================================================
# 3. 推理主函数
# ==============================================================================
def inference(model, tokenizer, image_processor, image_path, prompt, device="cuda"):
    # 1. 准备数据
    image = Image.open(image_path).convert("RGB")
    
    # 构造 Qwen 风格的对话 Prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 2. 使用 Processor 处理
    # 如果你有 processingWithMMRL，建议使用它，因为可能包含特殊的 grid 处理
    # 这里模拟 train.py 中的用法
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )
    
    text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        images=image,
        text=text_inputs,
        return_tensors="pt",
        padding=True
    )
    
    # 移至 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 这里的 images_per_sample 很重要，影响 VisionModelWithMMRL 中的切分
    # 单图推理手动指定为 [1]
    inputs["images_per_sample"] = [1] 

    # 3. 生成
    print("-> 正在生成...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,      # 验证时可以关掉采样看确定性结果
            temperature=0.7,
            top_p=0.9
        )

    # 4. 解码
    # generated_ids 包含了 input_ids，需要切掉
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    # ==========================================================================
    # 5. [核心] 获取门控状态 (验证可行性)
    # ==========================================================================
    # 注意：model.model 是 QWen3WithMMRL
    # model.model.visual 是 VisionWithMMRL
    # 在 forward 过程中，visual 模块保存了 self.alpha_list
    
    visual_module = model.model.visual
    alpha_val = None
    if hasattr(visual_module, "alpha_list") and len(visual_module.alpha_list) > 0:
        # alpha_list 是 list 或 tensor，取最后一个值（如果是多轮对话可能存了多个）
        # 或者因为是 generate，forward 跑了很多次，通常存的是最后一次 forward 的状态
        raw_alpha = visual_module.alpha_list
        if isinstance(raw_alpha, torch.Tensor):
            alpha_val = torch.sigmoid(raw_alpha).mean().item()
        elif isinstance(raw_alpha, list) and len(raw_alpha) > 0:
            # 取决于 Task_classifier 返回的是 logits 还是 list
            try:
                alpha_val = torch.sigmoid(raw_alpha[-1]).mean().item()
            except:
                alpha_val = "Unknown format"
    
    print("-" * 40)
    print(f"【输入图片】: {image_path}")
    print(f"【Prompt】: {prompt}")
    print(f"【Gate Alpha (Expert Probability)】: {alpha_val} (越接近1越倾向于专家模式)")
    print(f"【输出结果】: {output_text}")
    print("-" * 40)


if __name__ == "__main__":
    # 配置路径
    BASE_MODEL_PATH = "/root/autodl-tmp/model"     # 原始模型路径
    CHECKPOINT_DIR = "/root/autodl-tmp/Qwen3-VL-modify-test/mmrl_gating_output"  # 你的 train.py 输出目录
    
    # 测试图片和 Prompt
    TEST_IMAGE = "/root/autodl-tmp/dataset/prof/DJI_20230926075232_0092_V_JPG.rf.48fe81f65bb5cb95a7d4d33d8e6c9655.jpg"
    TEST_PROMPT = "详细描述这张图片中的电力设备缺陷。" # 针对你训练数据的 Prompt

    # 加载模型
    model, tokenizer, image_processor = load_mmrl_model(BASE_MODEL_PATH, CHECKPOINT_DIR)
    
    # 推理
    inference(model, tokenizer, image_processor, TEST_IMAGE, TEST_PROMPT)