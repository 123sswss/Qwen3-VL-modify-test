import config as cfg
import os

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def add_MMRL_token(model, tokenizer):
    new_special_tokens = cfg.MMRL_SPECIAL_TOKENS
    _ = tokenizer.add_special_tokens(new_special_tokens)
    model.resize_token_embeddings(len(tokenizer))

if __name__ == '__main__':
    model_id = os.path.abspath("../model/qwen3vl")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    print(f"原始词汇表大小: {len(tokenizer)}")
    add_MMRL_token(model, tokenizer)
    print(f"新词汇表大小: {len(tokenizer)}")
    start_token_id = tokenizer.convert_tokens_to_ids("<|text_R_token|>")
    end_token_id = tokenizer.convert_tokens_to_ids("<|text_R_token_end|>")
    print(f"新 token '<|text_R_token|>' 的 ID: {start_token_id}")
    print(f"新 token '<|text_R_token_end|>' 的 ID: {end_token_id}")
    embedding_matrix_shape = model.get_input_embeddings().weight.shape
    print(f"模型输入嵌入矩阵的形状: {embedding_matrix_shape}")
    assert embedding_matrix_shape[0] == len(tokenizer), "模型嵌入层大小与分词器词汇表大小不匹配！"
