from transformers import AutoProcessor
import addSpecialToken

# 1. 初始化 Processor (确保路径正确)
model_path = "../model/qwen3vl"
processor = AutoProcessor.from_pretrained(model_path)

# 2. 模拟你的魔改操作：在文本前加上 40 个占位符
rep_tokens = [f"<|REP_placeholder{i}|>" for i in range(40)]
rep_str = "".join(rep_tokens)

# 模拟输入信息
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "E:\AAA24105033\imgtest\PixPin_2025-11-11_12-48-13.png",  # 随便给个路径
            },
            # 模拟你提到的：text = [rep_str + t for t in text]
            {"type": "text", "text": rep_str + "帮我找找图中又有哪些动物，然后告诉我他们的坐标是什么。"},
        ],
    }
]

# 3. 使用 Processor 处理数据
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

input_ids = inputs.input_ids[0]  # 取第一条数据

# 4. 打印分析报告
print(f"{'Index':<10} | {'Token ID':<10} | {'Decoded Token'}")
print("-" * 40)

for i, token_id in enumerate(input_ids):
    decoded_token = processor.tokenizer.decode([token_id])
    # 为了方便观察，给占位符加个标记
    if "REP_placeholder" in decoded_token:
        mark = " <--- YOUR SOFT PROMPT"
    elif "vision" in decoded_token or token_id == 151655:  # Qwen的特殊token ID
        mark = " <--- VISION TOKEN"
    else:
        mark = ""

    print(f"{i:<10} | {token_id.item():<10} | {repr(decoded_token)}{mark}")

# 5. 验证你的切片逻辑
print("\n" + "=" * 50)
print(f"当前 input_ids 总长度: {len(input_ids)}")
print(f"如果你使用 inputs_embeds[:, :40, :]，你实际覆盖的是索引 0 到 39 的内容。")
print(f"请对照上方列表，看索引 0-39 到底存的是什么。")
print("=" * 50)