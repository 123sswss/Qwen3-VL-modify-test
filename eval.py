import os
import torch
import re
import gc
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
# 1. 模型包装器 (保持不变)
# ==============================================================================
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
        
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

# ==============================================================================
# 2. 推理引擎类 (核心修改部分)
# ==============================================================================
class Qwen3InferenceEngine:
    def __init__(self, trained_model_path, base_model_path, device="cuda"):
        """
        初始化模型引擎，只加载一次模型到显存
        """
        self.device = device
        self.trained_path = trained_model_path
        self.base_path = base_model_path
        
        print(f"[Init] 正在初始化推理引擎...")
        self._load_resources()
        print(f"[Init] 模型加载完成，准备就绪。")

    def _load_resources(self):
        # 1. 加载 Tokenizer
        print(f"  -> Loading Tokenizer from {self.base_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_path, trust_remote_code=True)
            self.tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
        except Exception as e:
            raise ValueError(f"加载 Tokenizer 失败: {e}")

        # 2. 加载 Config
        try:
            self.config = AutoConfig.from_pretrained(self.trained_path, trust_remote_code=True)
        except:
            print("  -> 训练目录 Config 缺失，使用基座 Config")
            self.config = AutoConfig.from_pretrained(self.base_path, trust_remote_code=True)

        # 3. 加载 Processor
        self.image_processor = AutoImageProcessor.from_pretrained(self.base_path, trust_remote_code=True)

        # 4. 构建模型结构
        print("  -> Building Model Architecture...")
        with torch.device(self.device):
            self.model = Qwen3VLMMRLForGen(self.config, self.tokenizer)
            self.model.to(torch.bfloat16) # 推荐使用 bf16

        # 5. 加载权重
        print(f"  -> Loading Weights from {self.trained_path}...")
        safetensors_path = os.path.join(self.trained_path, "model.safetensors")
        bin_path = os.path.join(self.trained_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("未找到权重文件")

        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"  -> Load Log: {msg}")
        self.model.eval()

        # 6. 初始化 Processor Wrapper
        self.processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
            image_processor=self.image_processor, tokenizer=self.tokenizer, cfg=cfg
        )

    def extract_answer(self, text):
        """正则提取 [[A]] 格式的答案"""
        pattern = r"\[\[([A-D])\]\]"
        matches = list(re.finditer(pattern, text))
        if matches:
            # 返回最后一个匹配到的答案，防止模型自我纠正
            return matches[-1].group(1)
        return "Unknown"

    def predict(self, image_path, system_prompt, user_text, max_new_tokens=512, temperature=0.2):
        """
        单次推理函数
        :param image_path: 图片路径
        :param system_prompt: 系统提示词
        :param user_text: 包含题目和选项的完整文本
        """
        # 1. 图像加载
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error loading image: {e}", None

        # 2. 构建消息 (每次都是新的，无历史负担)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text}
                ]
            }
        ]

        # 3. 预处理
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=image,
            padding=False,
            max_length=False,
            truncation=False,
            return_tensors="pt"
        ).to(self.device)

        # 4. 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False # 禁用 KV Cache 以节省显存并防止状态污染
            )

        # 5. 解码
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # [插入] MMRL 门控状态监控打印
        # ======================================================================
        # 注意：self.model 是 Qwen3VLMMRLForGen，内部的 .model 才是 QWen3WithMMRL
        visual_module = self.model.model.visual
        
        if hasattr(visual_module, "alpha_list") and visual_module.alpha_list is not None:
            alpha_logits = visual_module.alpha_list  # [Total_Images, 1]
            alpha_probs = torch.sigmoid(alpha_logits)
            
            # 获取 K 值 (专家路由结果)
            k_values = "N/A"
            if hasattr(self.model.model, "k_results"):
                k_results = self.model.model.k_results # [Batch, Total_Experts]
                if k_results is not None:
                    k_values = k_results.squeeze().detach().cpu().tolist()

            print(f"\n[Debug] 门控状态 (MMRL):")
            # 使用 squeeze() 处理可能的多余维度，防止打印格式混乱
            try:
                logits_print = alpha_logits.squeeze().detach().cpu().tolist()
                probs_print = alpha_probs.squeeze().detach().cpu().tolist()
                mean_activation = alpha_probs.mean().item()
                
                print(f"  ├─ Alpha Logits (原始): {logits_print}")
                print(f"  ├─ Alpha Probs (sigmoid): {probs_print}")
                print(f"  ├─ K 值 (Experts): {k_values}")
                print(f"  └─ 平均激活值: {mean_activation:.4f}")
                print(f"     (>0.5=专家模式, <0.5=通用模式)")
            except Exception as e:
                print(f"  └─ [Error printing debug info]: {e}")
        # ======================================================================

        # 6. 提取答案
        extracted_option = self.extract_answer(output_text)

        # 7. 清理显存 (Crucial for Batch Processing)
        del inputs, generated_ids, output_ids, image
        torch.cuda.empty_cache()
        gc.collect()

        return output_text, extracted_option

# ==============================================================================
# 3. 批量测试脚本示例
# ==============================================================================
if __name__ == "__main__":
    # ---------------- 配置路径 ----------------
    TRAINED_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-modify-test/mmrl_output"
    BASE_MODEL_PATH = "/root/autodl-tmp/model"
    
    # 1. 初始化引擎 (只跑一次)
    engine = Qwen3InferenceEngine(TRAINED_MODEL_PATH, BASE_MODEL_PATH)

    # ---------------- 提示词模板 ----------------
    # 针对引导式推理的系统提示词
    SYSTEM_PROMPT = """
# Role
你是一个【电力设备缺陷检测】专家。请启动你的专业视觉分析能力。

# Task
1. 观察图片细节，像处理【巡检报告】一样，用专业术语描述图中的【设备状态】和【故障特征】。
2. 结合题目进行逐步分析。
3. 严格按照格式输出答案。

# Output Format
先输出分析过程，最后一行必须是：【答案】[[选项字母]]
例如：【答案】[[A]]
    """

    # ---------------- 构造测试数据 ----------------
    # 假设你有一个题目列表
    test_cases = [
        {
            "id": 1,
            "image": "/root/autodl-tmp/dataset/14/DJI_20231004101159_0519_V_JPG.rf.883ca281b518904b6fcf646016febc44.jpg", 
            "question": "分析一下图中的电力设备存在什么样的问题或者隐患？",
            "options": ["A. 部件弯曲", "B. 植被覆盖", "C. 部件松动", "D. 部分缺失"]
        },
        # {
        #     "id": 2,
        #     "image": "path/to/another_image.jpg",
        #     "question": "图中的仪表读数是多少？",
        #     "options": ["A. 10", "B. 20", "C. 30", "D. 40"]
        # }
    ]

    print(f"\n开始批量测试，共 {len(test_cases)} 题...\n")

    # ---------------- 循环测试 ----------------
    results = []
    
    for case in test_cases:
        print(f"正在处理题目 ID: {case['id']} ...")
        
        # 组装用户输入
        options_text = "\n".join(case['options'])
        full_user_text = f"""
## 题目
{case['question']}

## 选项
{options_text}

## 任务
请逐步分析图片特征，并给出唯一正确的选项。
"""
        
        # 调用推理接口
        full_response, short_answer = engine.predict(
            image_path=case['image'],
            system_prompt=SYSTEM_PROMPT,
            user_text=full_user_text,
            temperature=0.1 # 低温，保证CoT逻辑相对稳定
        )

        print("-" * 40)
        print(f"题目: {case['question']}")
        print(f"模型完整回复:\n{full_response}")
        print(f"提取结果: {short_answer}")
        print("-" * 40)

        results.append({
            "id": case['id'],
            "extracted": short_answer,
            "raw": full_response
        })

    print("测试结束。")