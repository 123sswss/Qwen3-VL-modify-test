import config as cfg
import os

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def add_special_token(model_instance, tokenizer_instance, special_tokens):
    new_special_tokens = special_tokens
    _ = tokenizer_instance.add_special_tokens(new_special_tokens)
    model_instance.resize_token_embeddings(len(tokenizer_instance))

if __name__ == '__main__':
    model_id = os.path.abspath("../model/qwen3vl")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    print(f"原始词汇表大小: {len(tokenizer)}")
    add_special_token(model, tokenizer, cfg.SPECIAL_TOKENS)
    print(f"新词汇表大小: {len(tokenizer)}")
    for _, value in cfg.SPECIAL_TOKENS.items():
        for token in value:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"新 token '{token}' 的 ID: {token_id}")
    embedding_matrix_shape = model.get_input_embeddings().weight.shape
    print(f"模型输入嵌入矩阵的形状: {embedding_matrix_shape}")
    assert embedding_matrix_shape[0] == len(tokenizer), "模型嵌入层大小与分词器词汇表大小不匹配！"
