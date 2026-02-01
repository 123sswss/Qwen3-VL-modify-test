import os
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
import config as cfg
import QWen3WithMMRL
import processingWithMMRL

# ==============================================================================
# 1. 模型包装器 (保持与 overfit.py 一致，确保结构对齐)
# ==============================================================================
class Qwen3VLMMRLForGen(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer):
        import torch.nn as nn
        nn.Module.__init__(self) 
        self.config = config
        
        # 1. 获取准确的词表大小
        current_vocab_size = len(tokenizer)
        
        # 2. 初始化你的魔改 Base Model
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)
        
        # 3. 初始化 LM Head
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, current_vocab_size, bias=False)
        
        # 4. 生成配置修复
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

# ==============================================================================
# 2. 推理主函数
# ==============================================================================
def inference():
    # --------------------------------------------------------------------------
    # 配置路径 (请确保这些路径真实存在)
    # --------------------------------------------------------------------------
    # 1. 训练好的模型权重目录
    TRAINED_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-modify-test/mmrl_output"  
    
    # 2. 原始基座模型路径 (用于读取 Config 和 Processor，防止训练后 Config 缺失)
    BASE_MODEL_PATH = "/root/autodl-tmp/model" 
    
    # 3. 输入图片和文本
    IMAGE_PATH = "/root/autodl-tmp/dataset/prof/DJI_20230926081916_0007_V_JPG.rf.1fa4fe9940fad62f8aa9221adeb3739a.jpg"
    PROMPT_TEXT = "\n分析设备状态并输出JSON。"
    # IMAGE_PATH = "/root/autodl-tmp/Qwen3-VL-modify-test/test.png"
    # PROMPT_TEXT = "描述一下这张图片。"
    
    # --------------------------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------------------------
    print("[1/3] 加载配置与构建模型架构...")
    
    # 【修复步骤 A】: 强制从“基座路径”加载 Tokenizer，确保包含完整词表
    print(f"    -> 正在从基座路径 {BASE_MODEL_PATH} 加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"无法从 {BASE_MODEL_PATH} 加载 Tokenizer，请检查路径是否正确。错误: {e}")

    # 【修复步骤 B】: 重新添加特殊 Token (必须与训练时完全一致)
    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
    print(f"    -> Tokenizer 加载完毕，当前词表大小: {len(tokenizer)} (应约为 151709)")

    # 【修复步骤 C】: Config 优先尝试从训练目录加载 (为了获取 mmrl_config)，失败则用基座
    try:
        config = AutoConfig.from_pretrained(TRAINED_MODEL_PATH, trust_remote_code=True)
        print("    -> 使用训练目录的 Config")
    except:
        print("    -> 训练目录 Config 缺失，回退使用基座 Config")
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # Image Processor 从基座加载
    image_processor = AutoImageProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 在 GPU 上构建模型
    # 此时 len(tokenizer) 正常，QWen3WithMMRL 内部的 resize_token_embeddings 就会创建正确大小的层
    with torch.device("cuda"):
        model = Qwen3VLMMRLForGen(config, tokenizer)
        model.to(torch.bfloat16)

    print(f"[2/3] 加载训练权重: {TRAINED_MODEL_PATH} ...")
    
    # 自动判断权重格式
    safetensors_path = os.path.join(TRAINED_MODEL_PATH, "model.safetensors")
    bin_path = os.path.join(TRAINED_MODEL_PATH, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"在 {TRAINED_MODEL_PATH} 中未找到权重文件")

    # 加载权重
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"    -> 权重加载完成。")
    
    model.eval()

    # --------------------------------------------------------------------------
    # 下面保持不变
    # --------------------------------------------------------------------------
    print("[3/3] 正在推理...")
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )

    if not os.path.exists(IMAGE_PATH):
        # 如果没有图，生成一张纯色图防止报错
        Image.new('RGB', (100, 100), color='red').save(IMAGE_PATH)
    
    image = Image.open(IMAGE_PATH).convert("RGB")

    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    if hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None

    # 2. 处理输入时加入 Padding，规避 Mask 长度不匹配的 Bug
    # QWen3WithMMRL 的 forward 逻辑在处理变长 mask 时可能有缺陷
    # 模仿 overfit.py，给定一个足够长的 max_length (比如 2048 或 4096)
    inputs = processor(
        text=[text_prompt], 
        images=image,
        padding="max_length",  # 【关键修改】强制填充
        max_length=4096,       # 【关键修改】设定一个足够大的长度(需 > 图片token数 + 文本数)
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    # 3. 生成
    print(f"    -> 开始生成 (Input shape: {inputs.input_ids.shape})...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=True,
            temperature=0.1,
            # Qwen-VL 处理 pad token 的习惯
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False 
        )
        
        # 解码时要注意去除 padding 和 input 部分
        # generated_ids 的长度可能等于 max_length + new_tokens，或者提前停止
        
        # 获取实际生成的 token (去掉 input 部分)
        # 注意：由于使用了 padding，generated_ids 前面部分包含了大量的 pad token 或者 input
        # 标准做法是截断 input_ids 的长度
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print("\n" + "="*40)
        print(f"Prompt: {PROMPT_TEXT}")
        print("-" * 40)
        print(f"Output: {output_text}")
        print("="*40)

        if hasattr(model.model.visual, "alpha_list") and model.model.visual.alpha_list is not None:
            alpha_logits = model.model.visual.alpha_list  # [Total_Images, 1]
            alpha_probs = torch.sigmoid(alpha_logits)
            
            print(f"[Debug] 门控状态:")
            print(f"  ├─ Alpha Logits (原始): {alpha_logits.squeeze().detach().cpu().tolist()}")
            print(f"  ├─ Alpha Probs (sigmoid): {alpha_probs.squeeze().detach().cpu().tolist()}")
            print(f"  └─ 平均激活值: {alpha_probs.mean().item():.4f}")
            print(f"     (>0.5=专家模式, <0.5=通用模式)")

if __name__ == "__main__":
    inference()