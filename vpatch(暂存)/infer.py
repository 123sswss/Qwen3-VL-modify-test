from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "../../model/qwen3vl", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("../../model/qwen3vl")

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
