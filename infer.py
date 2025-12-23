from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "../model/qwen3vl", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("../model/qwen3vl")

vocab = processor.tokenizer.get_vocab()

# 常见的坐标相关关键词：box, point, ref, coord, object
# 以及 Qwen 系列常用的特殊标识符开头 <|
target_keywords = ["box", "point", "ref", "coord", "object", "rect"]
special_coords = {k: v for k, v in vocab.items() if any(key in k.lower() for key in target_keywords) and "<|" in k}

print("找到的可能相关的特殊 Token:")
for token, token_id in sorted(special_coords.items(), key=lambda x: x[1]):
    print(f"ID: {token_id} \t Token: {token}")

# 另外，检查一下是否有专门的数字表示方式（有的模型会将 0-1000 编码为特殊 token）
# 看看有没有类似 <|0|> 到 <|1000|> 的东西
numeric_tokens = [k for k in vocab.keys() if k.startswith("<|") and k[2:-2].isdigit()]
if numeric_tokens:
    print(f"\n检测到数字型特殊 Token (共{len(numeric_tokens)}个)，例如: {numeric_tokens[:5]}")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "E:\AAA24105033\imgtest\PixPin_2025-11-11_12-48-13.png",
            },
            {"type": "text", "text": "帮我找找图中又有哪些动物，然后告诉我他们的坐标是什么。"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print(output_text)
