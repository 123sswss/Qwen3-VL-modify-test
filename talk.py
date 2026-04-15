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
# 2. 推理主函数
# ==============================================================================
def inference():
    # --------------------------------------------------------------------------
    # 配置路径
    # --------------------------------------------------------------------------
    TRAINED_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-modify-test/mmrl_output"  
    BASE_MODEL_PATH = "/root/autodl-tmp/model" 
    
    # --------------------------------------------------------------------------
    # 加载模型（与原来完全一致）
    # --------------------------------------------------------------------------
    print("[1/3] 加载配置与构建模型架构...")
    
    print(f"    -> 正在从基座路径 {BASE_MODEL_PATH} 加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"无法从 {BASE_MODEL_PATH} 加载 Tokenizer，请检查路径是否正确。错误: {e}")

    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
    print(f"    -> Tokenizer 加载完毕，当前词表大小: {len(tokenizer)} (应约为 151709)")

    try:
        config = AutoConfig.from_pretrained(TRAINED_MODEL_PATH, trust_remote_code=True)
        print("    -> 使用训练目录的 Config")
    except:
        print("    -> 训练目录 Config 缺失，回退使用基座 Config")
        config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    image_processor = AutoImageProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    with torch.device("cuda"):
        model = Qwen3VLMMRLForGen(config, tokenizer)
        model.to(torch.bfloat16)

    print(f"[2/3] 加载训练权重: {TRAINED_MODEL_PATH} ...")
    
    safetensors_path = os.path.join(TRAINED_MODEL_PATH, "model.safetensors")
    bin_path = os.path.join(TRAINED_MODEL_PATH, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"在 {TRAINED_MODEL_PATH} 中未找到权重文件")

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")
    print(f"    -> 权重加载完成。")
    
    model.eval()

    # --------------------------------------------------------------------------
    # 构建 Processor（与原来一致）
    # --------------------------------------------------------------------------
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )

    # --------------------------------------------------------------------------
    # 交互式对话循环
    # --------------------------------------------------------------------------
    messages = []           # 多轮对话历史
    current_image = None    # 当前加载的 PIL Image
    image_pending = False   # 标记：下一条文本消息是否要附带图片

    BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                   交互式对话模式已启动                        ║
╠══════════════════════════════════════════════════════════════╣
║  指令:                                                       ║
║    /image <路径>   加载图片（支持相对路径），附到下一条消息      ║
║    /clear          清除全部对话上下文和已加载的图片             ║
║    /exit           退出程序                                   ║
║                                                              ║
║  用法示例:                                                    ║
║    /image ./test.png                                         ║
║    描述一下这张图片                                           ║
║    图中有几个人？          （追问，自动携带上文图片上下文）     ║
║    /image ./other.jpg      （切换新图片）                     ║
║    这张图是什么？                                             ║
║    /clear                  （重新开始）                       ║
╚══════════════════════════════════════════════════════════════╝"""
    print(BANNER)

    while True:
        # ---- 读取用户输入 ----
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if not user_input:
            continue

        # ---- /exit ----
        if user_input.lower() == "/exit":
            print("再见！")
            break

        # ---- /clear ----
        if user_input.lower() == "/clear":
            messages = []
            current_image = None
            image_pending = False
            print("[系统] 对话上下文、图片已全部清除。")
            continue

        # ---- /image <path> ----
        if user_input.lower().startswith("/image"):
            raw_path = user_input[6:].strip()   # "/image" 刚好6个字符，直接切片取后面的路径
            if not raw_path:
                print("[系统] 用法: /image <图片路径>")
                continue
            # 支持相对路径 & ~ 展开
            abs_path = os.path.abspath(os.path.expanduser(raw_path))
            try:
                current_image = Image.open(abs_path).convert("RGB")
                image_pending = True
                print(f"[系统] 图片已加载: {abs_path}")
                print("[系统] 请输入你的问题，图片将附到下一条消息中。")
            except Exception as e:
                print(f"[系统] 图片加载失败: {e}")
            continue

        # ---- 正常文本消息 ----
        # 构造当前轮的 user message
        content = []
        if image_pending and current_image is not None:
            content.append({"type": "image", "image": current_image})
            image_pending = False   # 图片已消费，后续追问靠对话历史
        content.append({"type": "text", "text": user_input})

        messages.append({"role": "user", "content": content})

        # ---- 构造 prompt ----
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if isinstance(text_prompt, list):
            text_prompt = text_prompt[0]

        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None

        # 收集对话历史中所有图片（按出现顺序）
        all_images = []
        for msg in messages:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        all_images.append(item["image"])

        # 决定传入 processor 的 images 参数
        if len(all_images) == 0:
            images_input = None
        elif len(all_images) == 1:
            images_input = all_images[0]
        else:
            images_input = all_images

        # ---- 编码输入 ----
        try:
            inputs = processor(
                text=[text_prompt],
                images=images_input,
                padding=False,
                max_length=False,
                truncation=False,
                return_tensors="pt"
            ).to(model.device)

            print(f"[系统] 正在生成 (输入 token 数: {inputs.input_ids.shape[1]}) ...")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
                input_len = inputs.input_ids.shape[1]
                output_ids = generated_ids[:, input_len:]
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print(f"\nAssistant> {output_text}")
            messages.append({"role": "assistant", "content": output_text})

            # Debug 门控状态 (保持不变)
            if hasattr(model.model, "visual") and hasattr(model.model.visual, "alpha_list") \
                    and model.model.visual.alpha_list is not None:
                alpha_logits = model.model.visual.alpha_list
                alpha_probs = torch.sigmoid(alpha_logits)
                k = model.model.k_results
                print(f"\n[Debug] 门控状态:")
                print(f"  ├─ Alpha Logits (原始): {alpha_logits.squeeze().detach().cpu().tolist()}")
                print(f"  ├─ Alpha Probs (sigmoid): {alpha_probs.squeeze().detach().cpu().tolist()}")
                print(f"  ├─ K 值: {k.squeeze().detach().cpu().tolist()}")
                print(f"  └─ 平均激活值: {alpha_probs.mean().item():.4f}")

        except Exception as e:
            print(f"[系统] 生成出错: {e}")
            # 回滚刚才追加的 user message，防止脏上下文越滚越崩
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            continue


if __name__ == "__main__":
    inference()