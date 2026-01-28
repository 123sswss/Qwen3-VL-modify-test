import os
import json
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any
from dataclasses import dataclass, field

from transformers import (
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor
)
from transformers import GenerationConfig

import config as cfg
import QWen3WithMMRL
import processingWithMMRL


# ==============================================================================
# 1. 定义支持 动态 Alpha Loss 的模型包装器
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

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, **kwargs):
        # 注意：Trainer 会把 dataset 返回的 extra fields 传到 kwargs 或者直接传参
        # 这里我们需要显式接收 alpha_labels

        outputs = super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)

        # 1. 获取 Tax Loss (稀疏性惩罚)
        mmrl_tax_loss = self.model.tax_loss

        # 2. 【核心修改】计算动态 Alpha Guide Loss
        # alpha_list shape: [Total_Images_In_Batch, 1]
        alpha_logits = self.model.visual.alpha_list

        alpha_guide_loss = 0.0

        if alpha_logits is not None and alpha_labels is not None:
            # alpha_labels shape: [Batch_Size]
            # images_per_sample: List[int] or Tensor, 记录每个样本包含几张图

            # 我们需要把 Batch 粒度的 label 对齐到 Image 粒度
            # 例如 Batch=2, Sample1有2图(label=1), Sample2有1图(label=0)
            # alpha_logits 有 3 个值
            # 扩展后的 labels 应该是 [1, 1, 0]

            # 确保 alpha_labels 是 Tensor
            if not isinstance(alpha_labels, torch.Tensor):
                alpha_labels = torch.tensor(alpha_labels, device=alpha_logits.device)
            else:
                alpha_labels = alpha_labels.to(alpha_logits.device)

            # 这里的 images_per_sample 通常在 kwargs 或者 processor 处理后会传进来
            # QWen3WithMMRL 的 forward 内部处理了 images_per_sample，这里我们需要拿到它
            # 如果 input_ids 存在，我们可以重新计算一下，或者信任 dataset/collator 传进来的

            if images_per_sample is None:
                # 兜底策略：假设每条数据只有1张图 (学术验证阶段通常如此)
                # 此时 Batch_Size == Total_Images
                expanded_labels = alpha_labels.view(-1, 1)
            else:
                # 如果有 images_per_sample (list 或 tensor)
                expanded_labels_list = []
                # 假设 images_per_sample 是一个列表，长度等于 batch_size
                for idx, count in enumerate(images_per_sample):
                    label = alpha_labels[idx]
                    expanded_labels_list.append(label.repeat(count))
                expanded_labels = torch.cat(expanded_labels_list).view(-1, 1)

            # 计算概率
            alpha_probs = torch.sigmoid(alpha_logits)

            # 确保维度匹配
            if alpha_probs.shape[0] != expanded_labels.shape[0]:
                # 极端情况防崩：截断或Padding，或者只取第一维
                min_len = min(alpha_probs.shape[0], expanded_labels.shape[0])
                alpha_probs = alpha_probs[:min_len]
                expanded_labels = expanded_labels[:min_len]

            # MSE Loss: 让预测的概率逼近 Target (0 或 1)
            # 权重设为 5.0，强力引导
            loss_fct = torch.nn.MSELoss()
            # 都要转成 float32/bfloat16
            alpha_guide_loss = loss_fct(alpha_probs, expanded_labels.to(alpha_probs.dtype)) * 2


        # 3. 合并 Loss
        if outputs.loss is not None:
            total_loss = outputs.loss + mmrl_tax_loss + alpha_guide_loss
            outputs.loss = total_loss
        if self.training and torch.rand(1).item() < 0.01:  # 1%概率打印
            print(f"\n[Debug] CE Loss: {outputs.loss.item():.4f} | "
                f"Tax: {mmrl_tax_loss:.4f} | "
                f"Alpha Guide: {alpha_guide_loss:.4f} | "
                f"Alpha Mean: {alpha_probs.mean().item():.4f}")

        return outputs


# ==============================================================================
# 2. 混合数据集 (专业 + 通用)
# ==============================================================================
class MixedMMRLDataset(Dataset):
    def __init__(self, processor,
                 expert_json, expert_img_dir,
                 general_json, general_img_dir,
                 general_ratio_limit=1.0):
        """
        general_ratio_limit: 限制通用数据的数量相对于专业数据的比例。
                             例如 1.0 表示通用数据最多和专业数据一样多。
        """
        self.processor = processor
        self.data_list = []

        # 1. 加载专业数据 (Target Alpha = 1.0)
        with open(expert_json, 'r', encoding='utf-8') as f:
            expert_data = json.load(f)

        print(f"[Dataset] 加载专业数据: {len(expert_data)} 条")
        for item in expert_data:
            image_file = item.get("image", "")
            full_path = os.path.join(expert_img_dir, image_file)
            if not os.path.exists(full_path):
                print(f"[Warning] 专业图片丢失跳过: {image_file}")
                continue
            self.data_list.append({
                "data": item,
                "img_root": expert_img_dir,
                "alpha_label": 1.0,  # 专家模式：全开
                "type": "expert"
            })

        # 2. 加载通用数据 (Target Alpha = 0.0)
        with open(general_json, 'r', encoding='utf-8') as f:
            general_data = json.load(f)

        # 根据比例裁剪通用数据
        max_general = int(len(expert_data) * general_ratio_limit)
        if len(general_data) > max_general:
            general_data = random.sample(general_data, max_general)

        print(f"[Dataset] 加载通用数据: {len(general_data)} 条 (Target Alpha=0)")
        for item in general_data:
            image_file = item.get("image", "")
            full_path = os.path.join(general_img_dir, image_file)
            if not os.path.exists(full_path):
                print(f"[Warning] 通用图片丢失跳过: {image_file}")
                continue
            self.data_list.append({
                "data": item,
                "img_root": general_img_dir,
                "alpha_label": 0.0,  # 通用模式：关闭
                "type": "general"
            })

        # 打乱
        random.shuffle(self.data_list)
        print(f"[Dataset] 总数据量: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_wrapper = self.data_list[idx]
        item = item_wrapper["data"]
        img_root = item_wrapper["img_root"]
        alpha_label = item_wrapper["alpha_label"]

        # 解析 LLaVA 格式
        # item: {"image": "xxx.jpg", "conversations": [...]}
        image_file = item.get("image")
        conversations = item.get("conversations")

        image_path = os.path.join(img_root, image_file)
        image = Image.open(image_path).convert("RGB")

        # 构造 Prompt
        # 假设 conversations[0] 是 human, [1] 是 gpt
        # Qwen 格式处理
        qwen_conv = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = turn["value"]

            # 强制截断“检测到：”之后的内容，防止答案泄露
            if role == "user" and "检测到：" in content:
                content = content.split("检测到：")[0]

            # 处理图片标记
            if "<image>" in content:
                content = content.replace("<image>", "")  # 移除标记，Processor 会加
                msg_content = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": content.strip()}
                ]
            else:
                msg_content = [{"type": "text", "text": content}]

            qwen_conv.append({"role": role, "content": msg_content})

        # Apply Template
        text_inputs = self.processor.apply_chat_template(
            qwen_conv, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            images=image,
            text=text_inputs,
            padding="max_length",
            max_length=1024,  # 根据显存调整
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)
        # 【修复】如果只有一张图，squeeze可能会把它压成一维 [3]，需要恢复成 [1, 3]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        # 简单构造 Labels (全量训练，或者你自己加 masking 逻辑)
        labels = input_ids.clone()
        # 找到assistant开始的位置
        # Qwen格式: <|im_start|>assistant\n
        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
        
        if len(assistant_positions) >= 2:  # 第二个<|im_start|>是assistant
            assistant_start = assistant_positions[1].item() + 2  # +2跳过"assistant\n"
            labels[:assistant_start] = -100  # mask掉user部分
        
        labels[attention_mask == 0] = -100  # mask掉padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "alpha_labels": float(alpha_label),
            "images_per_sample": 1
        }


# ==============================================================================
# 3. 自定义 Collator (处理 alpha_labels)
# ==============================================================================
@dataclass
class MMRLDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 提取 alpha_labels 和 images_per_sample
        alpha_labels = [f.pop("alpha_labels") for f in features]
        images_per_sample = [f.pop("images_per_sample") for f in features]

        # 使用 transformer 默认的 pad 逻辑处理 tensors
        # 因为 features 里的 tensors 长度可能不一样 (如果 padding=False)
        # 但我们在 dataset 里已经 padding="max_length" 了，所以可以直接 stack

        batch = {}
        for key in features[0].keys():
            # 【修复点】pixel_values 和 image_grid_thw 是变长的，必须拼接(cat)不能堆叠(stack)
            if key in ["pixel_values", "image_grid_thw"]:
                batch[key] = torch.cat([f[key] for f in features], dim=0)
            elif torch.is_tensor(features[0][key]):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = [f[key] for f in features]

        # 补充自定义字段
        batch["alpha_labels"] = torch.tensor(alpha_labels, dtype=torch.float32)
        batch["images_per_sample"] = images_per_sample

        return batch


# ==============================================================================
# 4. 训练主流程
# ==============================================================================
def train_gating(
        expert_json: str,
        expert_img_dir: str,
        general_json: str,
        general_img_dir: str,
        output_dir: str = "./mmrl_output"
):
    print("=" * 20 + " 启动 MMRL 门控混合训练 " + "=" * 20)
    MODEL_PATH = "/root/autodl-tmp/model"  # 修改为你的路径

    # 1. 配置加载
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)

    # 2. 模型初始化 (GPU 加速)
    print("-> 正在初始化模型...")
    original_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    original_model.resize_token_embeddings(len(tokenizer))

    with torch.device("cuda"):
        model = Qwen3VLMMRLForTrain(config, tokenizer)
        model.to(torch.bfloat16)

    print("-> 权重迁移...")
    model.model.load_state_dict(original_model.model.state_dict(), strict=False)
    model.lm_head.load_state_dict(original_model.lm_head.state_dict())
    del original_model
    torch.cuda.empty_cache()

    # 3. 冻结参数 & 激活 MMRL
    # 逻辑同 overfit.py，略微简化
    for param in model.parameters():
        param.requires_grad = False

    modules_to_train = [
        model.model.MMRL,
        model.model.visual.blocks_with_rep,
        model.model.visual.embedding_pooling,
        model.model.visual.Task_classifier,  # 重点训练这个
        model.model.visual.visionGating,
        model.model.visual.text_gating,
        model.model.visual.zero_init_layer
    ]

    print("-> 激活梯度...")
    trainable_num = 0
    for module in modules_to_train:
        for param in module.parameters():
            param.requires_grad = True
            trainable_num += param.numel()

    print(f"-> 可训练参数量: {trainable_num}")

    # 4. 数据准备
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )

    # general_ratio_limit=0.8 表示通用数据量最多是专业数据的 80%
    dataset = MixedMMRLDataset(
        processor,
        expert_json, expert_img_dir,
        general_json, general_img_dir,
        general_ratio_limit=1.0  # 建议 1:1 用于训练判别器
    )

    collator = MMRLDataCollator(processor)

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # 根据数据量调整，数据少可以多跑几个 epoch
        per_device_train_batch_size=2,  # 显存允许的话尽量大一点，保证Batch里同时有正负样本
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        save_strategy="no",
        logging_steps=5,
        remove_unused_columns=False,  # 必须 False，否则 alpha_labels 会被过滤
        bf16=True,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    print("-> 开始训练...")
    trainer.train()
    trainer.save_model(output_dir)
    print("-> 训练完成。")


if __name__ == "__main__":
    # 请替换为真实路径
    # general_data 可以直接用 COCO 的 caption 数据，或者 LLaVA 的 instruct 数据
    train_gating(
        expert_json="/root/autodl-tmp/dataset/generated_json.json",
        expert_img_dir="/root/autodl-tmp/dataset/prof",
        general_json="/root/autodl-tmp/dataset/llava_instruct_150k.json",
        general_img_dir="/root/autodl-tmp/dataset/gen/train2017"
    )