import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor

# 引入你的自定义模块
import config as cfg
import QWen3WithMMRL
import processingWithMMRL


def test_training_pipeline():
    print("=" * 20 + " 开始 MMRL 训练模式冒烟测试 " + "=" * 20)

    # 1. 路径设置 (请根据实际情况修改模型路径)
    # 如果没有权重，我们尝试加载 config/tokenizer 即可，模型权重将随机初始化
    MODEL_PATH = "../model/qwen3vl"

    # 检查路径是否存在，不存在则提示
    if not os.path.exists(MODEL_PATH):
        print(f"[警告] 路径 {MODEL_PATH} 不存在。")
        print("请确保你有 Qwen3-VL 的 config.json 和 tokenizer 文件。")
        # 这里为了演示，假设必须有这个路径，或者你可以修改为指向 Qwen2-VL 的路径
        return

    # 2. 加载基础配置和分词器
    print(f"[1/6] 加载配置和分词器 from {MODEL_PATH}...")
    try:
        # 加载 Config
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # 加载 ImageProcessor (用于传给你的自定义 Processor)
        image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"[错误] 加载基础组件失败: {e}")
        return

    # 3. 添加特殊 Token (参考 addSpecialToken.py 的逻辑)
    print("[2/6] 向 Tokenizer 添加特殊 Tokens...")
    special_tokens_dict = cfg.SPECIAL_TOKENS
    # 你的代码中 processor 初始化里也会用到 rep_tokens，这里先加到 tokenizer 避免 id 越界
    tokenizer.add_special_tokens(special_tokens_dict)
    print(f"    - 新词表大小: {len(tokenizer)}")

    # 4. 初始化你的自定义模型 (QWen3WithMMRL)
    print("[3/6] 初始化 QWen3WithMMRL 模型 (随机初始化权重)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with torch.device(device):
            # 建议加上 dtype=torch.bfloat16，因为 float32 初始化会占用双倍显存
            # 如果你的显卡不支持 bf16，改用 float16
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            # 设置默认数据类型，防止部分层默认初始化为 fp32 撑爆显存
            with torch.autocast(device_type=device, dtype=target_dtype):
                model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)

        # 确保模型转换到正确的精度 (虽然上面 autocast 有帮助，但显式转换更稳)
        model.to(dtype=target_dtype)

        # 强制切换到训练模式
        model.train()

        print(f"    - 模型已加载至: {model.device} | 精度: {model.dtype}")
        print(f"    - 是否开启 MMRL: {model.use_mmrl}")

    except Exception as e:
        print(f"[错误] 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 初始化你的自定义 Processor
    print("[4/6] 初始化 Qwen3ProcessorWithMMRL...")
    try:
        processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
            image_processor=image_processor,
            tokenizer=tokenizer,
            cfg=cfg
        )
    except Exception as e:
        print(f"[错误] Processor 初始化失败: {e}")
        return

    # 6. 构造 Dummy 数据 (图片 + 文本)
    print("[5/6] 构造测试数据...")
    # 创建一张 256x256 的纯色图片
    dummy_image = Image.new('RGB', (256, 256), color=(255, 0, 0))
    dummy_text = "Identify the color of this image."

    # 构造对话格式 (参考你的 processingWithMMRL 逻辑，它似乎处理 list[dict] 或 str)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": dummy_text},
            ],
        }
    ]

    try:
        # 使用自定义 processor 处理输入
        # 注意：apply_chat_template 在你的 processor 代码中被重写了部分逻辑
        # 这里直接调用 processor 的 __call__，它内部会处理 chat template
        inputs = processor(
            text=messages,
            images=dummy_image,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        print(f"    - Input IDs shape: {inputs.input_ids.shape}")
        if 'pixel_values' in inputs:
            print(f"    - Pixel Values shape: {inputs.pixel_values.shape}")
            print(f"    - Grid THW: {inputs.image_grid_thw}")

        # 简单检查 input_ids 里是否有 rep_token_ids
        rep_token_start_id = tokenizer.convert_tokens_to_ids("<|REP_placeholder0|>")
        has_rep = (inputs.input_ids == rep_token_start_id).any()
        print(f"    - Input_ids 中是否包含 REP placeholder: {has_rep.item()}")

    except Exception as e:
        print(f"[错误] 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. 运行 Forward 和 Backward
    print("[6/6] 运行 Forward 和 Backward 测试...")
    try:
        # Forward
        outputs = model(**inputs)

        # 检查输出
        # QWen3WithMMRL 继承自 Model，通常返回 last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
            print(f"    - Forward 成功. Last Hidden State Shape: {hidden_states.shape}")
        else:
            # 如果它是 CausalLM 结构，可能有 logits
            print(f"    - Forward 成功. Output keys: {outputs.keys()}")
            hidden_states = outputs[0]  # Fallback

        # 获取自定义 Loss
        # 在你的 QWen3WithMMRL 代码中，tax_loss 和 alpha_loss 被保存在 model 实例属性中
        tax_loss = model.tax_loss
        alpha_loss = model.alpha_loss

        print(f"    - Tax Loss: {tax_loss}")
        print(f"    - Alpha Loss: {alpha_loss}")

        # 构造一个伪造的总 Loss 进行反向传播测试
        # 因为没有 Labels，我们随便算一个 hidden states 的 sum 作为主 loss
        dummy_main_loss = hidden_states.sum() * 0.0001

        total_loss = dummy_main_loss
        if tax_loss is not None:
            total_loss += tax_loss
        if alpha_loss is not None:
            total_loss += alpha_loss

        print(f"    - Total Dummy Loss: {total_loss.item()}")

        # Backward
        total_loss.backward()
        print("    - Backward Pass 成功! 梯度已计算。")
        print("\n[成功] 所有测试通过！MMRL 模块集成看起来正常。")

    except Exception as e:
        print(f"[错误] 模型运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 确保不计算梯度时报错 (虽然默认是开启的，但在某些推理脚本上下文中可能被关掉)
    torch.set_grad_enabled(True)
    test_training_pipeline()