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
    def __init__(self, config, tokenizer, alpha_loss_weight=1.0):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.config = config
        self.alpha_loss_weight = alpha_loss_weight  # 保存 Alpha Loss 权重
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
        self.training_step_counter = 0

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)

        mmrl_tax_loss = self.model.tax_loss
        alpha_logits = self.model.visual.alpha_list
        alpha_guide_loss = 0.0

        if alpha_logits is not None and alpha_labels is not None:
            if isinstance(alpha_logits, list):
                if len(alpha_logits) == 0:
                    print("[Warning] alpha_list is empty, skipping alpha loss")
                    alpha_guide_loss = torch.tensor(0.0, device=outputs.loss.device)
                else:
                    alpha_logits = torch.stack(alpha_logits)

            if images_per_sample is None:
                expanded_labels = alpha_labels.view(-1, 1)
            else:
                expanded_labels_list = []
                for idx, count in enumerate(images_per_sample):
                    label = alpha_labels[idx]
                    expanded_labels_list.append(label.repeat(count))
                expanded_labels = torch.cat(expanded_labels_list).view(-1, 1)

            # 计算概率
            alpha_probs = torch.sigmoid(alpha_logits)

            # 确保维度匹配
            if alpha_probs.shape[0] != expanded_labels.shape[0]:
                min_len = min(alpha_probs.shape[0], expanded_labels.shape[0])
                alpha_probs = alpha_probs[:min_len]
                expanded_labels = expanded_labels[:min_len]

            # MSE Loss: 让预测的概率逼近 Target (0 或 1)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # 使用传入的权重参数
            alpha_guide_loss = loss_fct(alpha_logits, expanded_labels.to(alpha_logits.dtype)) * self.alpha_loss_weight

        # 3. 合并 Loss
        if outputs.loss is not None:
            total_loss = outputs.loss + mmrl_tax_loss + alpha_guide_loss
            outputs.loss = total_loss
        
        if self.training:
            self.training_step_counter += 1
            if self.training_step_counter % 50 == 0:
                print("\n" + "="*60)
                print(f"[Training Step {self.training_step_counter}] Loss Breakdown:")
                print(f"  ├─ CE Loss (Language):    {(outputs.loss - mmrl_tax_loss - alpha_guide_loss).item():>8.4f}")
                print(f"  ├─ Tax Loss (Sparsity):   {mmrl_tax_loss.item():>8.4f}")
                print(f"  ├─ Alpha Guide Loss:      {alpha_guide_loss.item():>8.4f} (Weight: {self.alpha_loss_weight})")
                print(f"  └─ Total Loss:            {outputs.loss.item():>8.4f}")
                print(f"[Alpha Statistics]")
                print(f"  ├─ Mean Probability:      {alpha_probs.mean().item():>8.4f}")
                print(f"  ├─ Target Mean:           {expanded_labels.float().mean().item():>8.4f}")
                print(f"  └─ Alpha Logits:          {torch.sigmoid(alpha_logits).squeeze().detach().cpu().tolist()}")
                print("="*60 + "\n")
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

        image_file = item.get("image")
        conversations = item.get("conversations")

        image_path = os.path.join(img_root, image_file)
        image = Image.open(image_path).convert("RGB")

        # Qwen 格式处理
        qwen_conv = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = turn["value"]

            if role == "user" and "检测到：" in content:
                content = content.split("检测到：")[0]

            if "<image>" in content:
                content = content.replace("<image>", "")
                msg_content = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": content.strip()}
                ]
            else:
                msg_content = [{"type": "text", "text": content}]

            qwen_conv.append({"role": role, "content": msg_content})

        text_inputs = self.processor.apply_chat_template(
            qwen_conv, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            images=image,
            text=text_inputs,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)
        
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        labels = input_ids.clone()
        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
        
        if len(assistant_positions) >= 2:
            assistant_start = assistant_positions[1].item() + 2
            labels[:assistant_start] = -100
        
        labels[attention_mask == 0] = -100
        
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
        alpha_labels = [f.pop("alpha_labels") for f in features]
        images_per_sample = [f.pop("images_per_sample") for f in features]

        batch = {}
        for key in features[0].keys():
            if key in ["pixel_values", "image_grid_thw"]:
                batch[key] = torch.cat([f[key] for f in features], dim=0)
            elif torch.is_tensor(features[0][key]):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = [f[key] for f in features]

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
        output_dir: str = "./mmrl_output",
        learning_rate: float = 2e-4,          # 参数化：学习率
        num_train_epochs: int = 5,            # 参数化：训练轮数
        general_ratio_limit: float = 0.4,     # 参数化：通用数据比例限制
        alpha_loss_weight: float = 8.0        # 参数化：Alpha Guide Loss 的权重
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
        # 传入 alpha_loss_weight 参数
        model = Qwen3VLMMRLForTrain(config, tokenizer, alpha_loss_weight=alpha_loss_weight)
        model.to(torch.bfloat16)

    print("-> 权重迁移...")
    model.model.load_state_dict(original_model.model.state_dict(), strict=False)
    model.lm_head.load_state_dict(original_model.lm_head.state_dict())
    del original_model
    torch.cuda.empty_cache()

    # 3. 冻结参数 & 激活 MMRL
    for param in model.parameters():
        param.requires_grad = False

    modules_to_train = [
        model.model.MMRL,
        model.model.visual.blocks_with_rep,
        model.model.visual.embedding_pooling,
        model.model.visual.Task_classifier,
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

    # 使用传入的 general_ratio_limit 参数
    dataset = MixedMMRLDataset(
        processor,
        expert_json, expert_img_dir,
        general_json, general_img_dir,
        general_ratio_limit=general_ratio_limit
    )

    collator = MMRLDataCollator(processor)

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,  # 使用参数
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,        # 使用参数
        save_strategy="no",
        logging_steps=5,
        remove_unused_columns=False,
        bf16=True,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    print(f"Expert samples: {sum(1 for x in dataset.data_list if x['alpha_label']==1.0)}")
    print(f"General samples: {sum(1 for x in dataset.data_list if x['alpha_label']==0.0)}")
    print(f"Training Config: LR={learning_rate}, Epochs={num_train_epochs}, AlphaWeight={alpha_loss_weight}")
    
    print("-> 开始训练...")
    trainer.train()
    trainer.save_model(output_dir)
    print("-> 训练完成。")


if __name__ == "__main__":
    train_gating(
        expert_json="/root/autodl-tmp/dataset/generated_json.json",
        expert_img_dir="/root/autodl-tmp/dataset/prof",
        general_json="/root/autodl-tmp/dataset/llava_instruct_150k.json",
        general_img_dir="/root/autodl-tmp/dataset/gen/train2017",
        learning_rate=2e-4,       
        num_train_epochs=3,       
        general_ratio_limit=1.0,  
        alpha_loss_weight=4.0     
    )