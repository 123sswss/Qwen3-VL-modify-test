from operator import is_
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
from transformers import TrainerCallback


class MMRLTrainingCallback(TrainerCallback):
    def __init__(self, dataset, initial_temp=1.0, final_temp=0.1,
                 total_epochs=5, enable_temp_annealing=True,target_tax=5.0,tax_warmup_epochs=1):
        self.dataset = dataset
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.enable_temp_annealing = enable_temp_annealing
        self.target_tax = target_tax
        self.tax_warmup_epochs = tax_warmup_epochs
        self.current_epoch = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        """在每个 step 开始时平滑更新参数"""
        model = kwargs['model']
        # 计算当前精确的 epoch（含小数部分）
        current_epoch = state.epoch 
        
        # --- 1. Temperature 退火逻辑 (保持原有) ---
        if self.enable_temp_annealing:
            progress = min(current_epoch / self.total_epochs, 1.0)
            current_temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress
            model.temperature_override = current_temp

        # --- 2. 动态加税逻辑 (新增强调) ---
        if current_epoch < self.tax_warmup_epochs:
            # 免税期
            model.tax_loss_weight = 0.0
        else:
            # 从免税期结束到训练结束，线性加税到 target_tax
            tax_progress = min((current_epoch - self.tax_warmup_epochs) / 
                               (self.total_epochs - self.tax_warmup_epochs), 1.0)
            model.tax_loss_weight = self.target_tax * tax_progress

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Epoch开始时的操作"""
        self.current_epoch = state.epoch

        # 1. 重新采样通用数据
        print(f"\n{'=' * 60}")
        print(f"[Epoch {int(self.current_epoch)}] Resampling general dataset...")
        self.dataset.resample_general_data()

        model = kwargs['model']
        tax_w = getattr(model, 'tax_loss_weight', 0.0)
        temp = getattr(model, 'temperature_override', 1.0)
        temp = temp if temp is not None else -1.0
        print(f"[Schedule Update] Current Tax Weight: {tax_w:.4f}, Temperature: {temp:.4f}")
        print(f"{'=' * 60}\n")

        # 2. 更新temperature
        if self.enable_temp_annealing:
            progress = min(self.current_epoch / self.total_epochs, 1.0)
            current_temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress
            kwargs['model'].temperature_override = current_temp
            print(f"[Temperature Annealing] Current: {current_temp:.4f} "
                  f"(Progress: {progress * 100:.1f}%)")
        print(f"{'=' * 60}\n")


# 1. 定义支持 动态 Alpha Loss 的模型包装器
class Qwen3VLMMRLForTrain(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer, tax_loss_weight=1.0, alpha_loss_weight=1.0):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.config = config
        self.tax_loss_weight = tax_loss_weight  # 保存 Tax Loss 权重
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
        self.temperature_override = None  # 添加temperature_override属性
        self.k_sums_history = []  # 记录k值历史

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, **kwargs):
        # 传递temperature_override到模型
        if hasattr(self, 'temperature_override') and self.temperature_override is not None:
            kwargs['gating_temperature_overied'] = self.temperature_override

        labels = kwargs.get('labels', None)
        gating_mask = None
        # 逻辑：
        # labels == -100 的位置是 Prompt (指令+图片)，这对门控是可见的。
        # labels != -100 的位置是 Answer (答案)，这对门控必须通过抹零来隐藏。
        # 注意：attention_mask 为 0 的位置 (Padding) 也要处理
        
        # 1. 找出 Prompt 部分 (labels 为 -100)
        is_prompt = (labels == -100)
        
        # 2. 确保不是 Padding (attention_mask 必须为 1)
        if 'attention_mask' in kwargs:
            att_mask = kwargs['attention_mask']
            # 保持维度对齐
            if att_mask.dim() == 2 and is_prompt.dim() == 2:
                is_prompt = is_prompt & (att_mask == 1)
        
        # 3. 转换为 float mask (1.0 = 保留, 0.0 = 抹除)
        gating_mask = is_prompt.to(dtype=self.model.dtype)
        
        # 将这个 mask 放入 kwargs 传给底层模型
        kwargs['mmrl_gating_mask'] = gating_mask

        outputs = super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)

        mmrl_tax_loss = self.model.tax_loss * self.tax_loss_weight
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

        # 获取 k_sums (从 model.visual 中)
        k_sums_value = None
        k_results = self.model.visual.k_results
        internal_raw_tax_loss = getattr(self.model, 'tax_loss', torch.tensor(0.0, device=outputs.loss.device))
        assert k_results is not None, f"Expected k_results to be not None, got {k_results}"
        if isinstance(k_results, tuple):
            k_sums_value = k_results[0]  # training模式
        else:
            k_sums_value = k_results  # eval模式

        # 1. 计算 Alpha Guide Loss (保持不变)
        alpha_logits = self.model.visual.alpha_list
        alpha_guide_loss = 0.0

        if alpha_logits is not None and alpha_labels is not None:
            if isinstance(alpha_logits, list):
                if len(alpha_logits) == 0:
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
            
            # 对齐维度
            min_len = min(alpha_logits.shape[0], expanded_labels.shape[0])
            alpha_logits = alpha_logits[:min_len]
            expanded_labels = expanded_labels[:min_len]
            
            alpha_probs = torch.sigmoid(alpha_logits)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            alpha_guide_loss = loss_fct(alpha_logits, expanded_labels.to(alpha_logits.dtype)) * self.alpha_loss_weight

        # 2. 新的 K Loss 调度策略 (核心修改)
        k_general_loss = 0.0
        k_expert_loss = 0.0
        
        # 统计变量初始化
        mean_k_general = 0.0
        mean_k_expert = 0.0

        if k_sums_value is not None and alpha_labels is not None:
            # 展平 labels 以对齐每张图片的 K 值
            if images_per_sample is None:
                k_expanded_labels = alpha_labels
            else:
                k_expanded_labels_list = []
                for idx, count in enumerate(images_per_sample):
                    label = alpha_labels[idx]
                    k_expanded_labels_list.append(label.repeat(count))
                k_expanded_labels = torch.cat(k_expanded_labels_list)

            # 确保维度对齐
            if k_sums_value.shape[0] == k_expanded_labels.shape[0]:
                # --- A. 区分通用数据与专业数据 ---
                is_general = (k_expanded_labels < 0.1) # General (Alpha=0)
                is_expert = (k_expanded_labels > 0.9)  # Expert (Alpha=1)

                # --- B. 通用数据策略：强力镇压 (Strict Inhibition) ---
                if is_general.any():
                    general_k = k_sums_value[is_general]
                    mean_k_general = general_k.mean().item()
                    
                    # [关键点 1]: 通用数据必须严厉镇压，这里可以使用固定的高权重，或者跟随 tax_loss_weight (如果它很大的话)
                    # 建议：直接给一个固定的强惩罚，比如 5.0
                    suppression_weight = max(self.tax_loss_weight, 5.0) 
                    k_general_loss = torch.mean(general_k) * suppression_weight / 40.0 * 5.0 
                    # 注意：如果你的 K 是求和(sum)，值可能很大(0~40)，除以40归一化后再乘权重比较好控制

                # --- C. 专业数据策略：激励 vs 惩罚 (Dynamic Schedule) ---
                if is_expert.any():
                    expert_k = k_sums_value[is_expert]
                    mean_k_expert = expert_k.mean().item()

                    # 判断当前阶段
                    if self.tax_loss_weight < 1e-6:
                        # [阶段 1: 强力激励期 (Wake Up Phase)]
                        # 问题：如果 expert_k 已经是 0，普通的 -mean(expert_k) 会遭遇梯度死亡。
                        # 解决：使用内部的 internal_tax_loss (它基于 intensity，没有 clamp，梯度依然存在)。
                        # 我们给它一个巨大的负权重，强制模型增大 intensity，从而把门顶开。
                        
                        # 注意：internal_tax_loss 默认是正的（惩罚），我们取负号变成激励。
                        # 权重建议给大一点 (例如 2.0)，因为我们要从 0 把它拉回来。
                        k_expert_loss = -internal_raw_tax_loss * 5.0 
                        
                        # (可选) 也可以保留原来的 K Loss 作为辅助，万一 K 没死透呢
                        # k_expert_loss += -torch.mean(expert_k) * 0.1
                        
                    else:
                        # [阶段 2: 稀疏期]
                        # 此时门已经开了，我们收取微小的过路费
                        # 归一化 K 值 (假设总数是 40)
                        k_normalized = expert_k / 40.0
                        sparsity_cost = 0.1
                        k_expert_loss = torch.mean(k_normalized ** 2) * sparsity_cost
            else:
                print(f"[Warning] K values mismatch...")

        # 3. 合并 Loss
        # total_k_loss 包含了对通用的惩罚 和 对专业的(激励/惩罚)
        total_mmrl_loss = k_general_loss + k_expert_loss + alpha_guide_loss
        
        if outputs.loss is not None:
            total_loss = outputs.loss + total_mmrl_loss
            outputs.loss = total_loss

        # 4. 日志打印修改
        if self.training:
            self.training_step_counter += 1
            if self.training_step_counter % 50 == 0:
                print("\n" + "=" * 60)
                print(f"[Training Step {self.training_step_counter}] Loss Breakdown:")
                print(f"  ├─ CE Loss (Language):    {(outputs.loss - total_mmrl_loss).item():>8.4f}")
                print(f"  ├─ Alpha Guide Loss:      {alpha_guide_loss.item():>8.4f}")
                print(f"  ├─ K Loss (General):      {k_general_loss if isinstance(k_general_loss, float) else k_general_loss.item():>8.4f} (Strict Suppress)")
                # 根据当前阶段显示 Loss 说明
                k_exp_loss_val = k_expert_loss if isinstance(k_expert_loss, float) else k_expert_loss.item()
                loss_type = "Incentive" if self.tax_loss_weight < 1e-6 else f"Tax(W={self.tax_loss_weight:.2f})"
                print(f"  ├─ K Loss (Expert):       {k_exp_loss_val:>8.4f} ({loss_type})")
                print(f"  └─ Total Loss:            {outputs.loss.item():>8.4f}")

                if k_sums_value is not None:
                    print(f"[K Statistics]")
                    print(f"  ├─ General Mean K:        {mean_k_general:>8.2f} (Should -> 0)")
                    print(f"  ├─ Expert Mean K:         {mean_k_expert:>8.2f} (Should be Active)")
                    print(f"  ├─ Max K (Batch):         {k_sums_value.max().item():>8.2f}")
                    print(f"  └─ Min K (Batch):         {k_sums_value.min().item():>8.2f}")

                print(f"[Alpha Statistics]")
                print(f"  ├─ Mean Probability:      {alpha_probs.mean().item():>8.4f}")
                print(f"  ├─ Target Mean:           {expanded_labels.float().mean().item():>8.4f}")
                
                assert hasattr(self, 'temperature_override')
                print(f"[Temperature] Current:      {self.temperature_override if self.temperature_override else -1:>8.4f}")

                print("=" * 60 + "\n")

        return outputs


# 2. 混合数据集 (专业 + 通用)
class MixedMMRLDataset(Dataset):
    def __init__(self, processor,
                 expert_json, expert_img_dir,
                 general_json, general_img_dir,
                 general_ratio_limit=1.0,
                 expert_limit=None):
        self.processor = processor
        # 处理expert_json列表合并
        if isinstance(expert_json, list):
            self.expert_data_raw = []
            for json_path in expert_json:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.expert_data_raw.extend(json.load(f))
            print(f"[Dataset] 合并了 {len(expert_json)} 个专业JSON文件，共 {len(self.expert_data_raw)} 条数据")
        else:
            with open(expert_json, 'r', encoding='utf-8') as f:
                self.expert_data_raw = json.load(f)
        # 处理expert_img_dir列表合并（创建文件名到路径的映射）
        if isinstance(expert_img_dir, list):
            self.img_path_mapping = {}  # 文件名 -> 完整路径
            seen_files = {}  # 用于检测冲突：文件名 -> 来源目录

            for img_dir in expert_img_dir:
                if not os.path.exists(img_dir):
                    print(f"[Warning] 图片目录不存在: {img_dir}")
                    continue

                files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
                for filename in files:
                    if filename in seen_files:
                        raise ValueError(
                            f"文件名冲突！'{filename}' 同时存在于:\n"
                            f"  - {seen_files[filename]}\n"
                            f"  - {img_dir}\n"
                            f"请重命名文件后重试。"
                        )
                    seen_files[filename] = img_dir
                    self.img_path_mapping[filename] = os.path.join(img_dir, filename)

            print(f"[Dataset] 合并了 {len(expert_img_dir)} 个专业图片文件夹，共 {len(self.img_path_mapping)} 张图片")
            self.use_img_mapping = True
        else:
            self.expert_img_dir = expert_img_dir
            self.use_img_mapping = False

        self.expert_img_dir = expert_img_dir
        self.general_img_dir = general_img_dir
        self.general_ratio_limit = general_ratio_limit
        if isinstance(expert_json, list):
            self.expert_data_raw = []
            for json_path in expert_json:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.expert_data_raw.extend(json.load(f))
            print(f"[Dataset] 合并了 {len(expert_json)} 个专业JSON文件，共 {len(self.expert_data_raw)} 条数据")
        else:
            with open(expert_json, 'r', encoding='utf-8') as f:
                self.expert_data_raw = json.load(f)
        if isinstance(expert_img_dir, list):
            self.img_path_mapping = {}
            seen_files = {}
            
            for img_dir in expert_img_dir:
                if not os.path.exists(img_dir):
                    print(f"[Warning] 图片目录不存在: {img_dir}")
                    continue
                
                files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
                for filename in files:
                    if filename in seen_files:
                        raise ValueError(
                            f"文件名冲突！'{filename}' 同时存在于:\n"
                            f"  - {seen_files[filename]}\n"
                            f"  - {img_dir}\n"
                            f"请重命名文件后重试。"
                        )
                    seen_files[filename] = img_dir
                    self.img_path_mapping[filename] = os.path.join(img_dir, filename)
            
            print(f"[Dataset] 合并了 {len(expert_img_dir)} 个专业图片文件夹，共 {len(self.img_path_mapping)} 张图片")
            self.use_img_mapping = True
            self.expert_img_dir = None  # 标记为使用映射模式
        else:
            self.expert_img_dir = expert_img_dir
            self.use_img_mapping = False
            self.img_path_mapping = None

        # 加载专业数据
        # with open(expert_json, 'r', encoding='utf-8') as f:
        #     self.expert_data_raw = json.load(f)

        # 加载通用数据（保存完整数据）
        with open(general_json, 'r', encoding='utf-8') as f:
            self.general_data_raw = json.load(f)

        if expert_limit is not None and expert_limit < len(self.expert_data_raw):
            self.expert_data_raw = random.sample(self.expert_data_raw, expert_limit)
            print(f"[Dataset] 专业数据集限制为 {expert_limit} 条（随机采样）")
        # 初始化数据列表
        self._build_data_list()

    def _build_data_list(self):
        """构建当前epoch的数据列表"""
        self.data_list = []

        # 添加专业数据
        for item in self.expert_data_raw:
            image_file = item.get("image", "")
            if self.use_img_mapping:
                if image_file not in self.img_path_mapping:
                    continue
                full_path = self.img_path_mapping[image_file]
            else:
                full_path = os.path.join(self.expert_img_dir, image_file)
                if not os.path.exists(full_path):
                    continue
            
            self.data_list.append({
                "data": item,
                "img_root": self.expert_img_dir,  # 这个字段在映射模式下不使用
                "alpha_label": 1.0,
                "type": "expert"
            })

        # 随机采样通用数据
        max_general = int(len(self.expert_data_raw) * self.general_ratio_limit)
        sampled_general = random.sample(self.general_data_raw,
                                        min(max_general, len(self.general_data_raw)))

        for item in sampled_general:
            image_file = item.get("image", "")
            full_path = os.path.join(self.general_img_dir, image_file)
            if not os.path.exists(full_path):
                continue
            self.data_list.append({
                "data": item,
                "img_root": self.general_img_dir,
                "alpha_label": 0.0,
                "type": "general"
            })

        random.shuffle(self.data_list)
        print(f"[Dataset Rebuilt] Expert: {sum(1 for x in self.data_list if x['alpha_label'] == 1.0)}, "
              f"General: {sum(1 for x in self.data_list if x['alpha_label'] == 0.0)}")

    def resample_general_data(self):
        """每个epoch调用此方法重新采样通用数据"""
        self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_wrapper = self.data_list[idx]
        item = item_wrapper["data"]
        img_root = item_wrapper["img_root"]
        alpha_label = item_wrapper["alpha_label"]

        image_file = item.get("image")
        conversations = item.get("conversations")

        if self.use_img_mapping and item_wrapper["type"] == "expert":
            image_path = self.img_path_mapping.get(image_file)
            if image_path is None:
                raise FileNotFoundError(f"图片文件 '{image_file}' 不在任何专业数据集文件夹中")
        else:
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


# 3. 自定义 Collator (处理 alpha_labels)
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


# 4. 训练主流程
def train_gating(
        expert_json: str,
        expert_img_dir: str,
        general_json: str,
        general_img_dir: str,
        output_dir: str = "./mmrl_output",
        learning_rate: float = 2e-4,       
        num_train_epochs: int = 5,         
        general_ratio_limit: float = 0.4,  
        alpha_loss_weight: float = 8.0,     
        tax_loss_weight: float = 1.0
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
        model = Qwen3VLMMRLForTrain(config, tokenizer, tax_loss_weight=tax_loss_weight, alpha_loss_weight=alpha_loss_weight)
        model.to(torch.bfloat16)

    print("-> 权重迁移...")
    model.model.load_state_dict(original_model.model.state_dict(), strict=False)
    model.lm_head.load_state_dict(original_model.lm_head.state_dict())
    del original_model
    torch.cuda.empty_cache()

    for param in model.parameters():
        param.requires_grad = False

    for idx, layer_num in enumerate(model.model.visual.cfg.INSERT_LAYER):
        if layer_num < len(model.model.visual.blocks):
            model.model.visual.blocks_with_rep[idx].load_state_dict(
                model.model.visual.blocks[layer_num].state_dict(),
                strict=False
            )
            print(f"[Init] Copied weights from blocks[{layer_num}] to blocks_with_rep[{idx}]")

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

    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )

    dataset = MixedMMRLDataset(
        processor,
        expert_json, expert_img_dir,
        general_json, general_img_dir,
        general_ratio_limit=general_ratio_limit
    )

    collator = MMRLDataCollator(processor)

    mmrl_callback = MMRLTrainingCallback(
    dataset=dataset,
    initial_temp=1.0,
    final_temp=0.1,
    target_tax=tax_loss_weight, # 这里传入你想要的目标值，比如 8.0
    tax_warmup_epochs=1,        # 设定前 1 个 epoch 免税
    total_epochs=num_train_epochs,
    enable_temp_annealing=True
    )

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        save_strategy="no",
        logging_steps=5,
        remove_unused_columns=False,
        bf16=True,
        dataloader_pin_memory=False,
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[mmrl_callback]
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
        expert_json=["/root/autodl-tmp/dataset/1json.json",
                     "/root/autodl-tmp/dataset/2conv_c.json",
                     "/root/autodl-tmp/dataset/1conv_c.json",
                     "/root/autodl-tmp/dataset/4conv_c.json",
                     "/root/autodl-tmp/dataset/14json.json"],
        expert_img_dir=["/root/autodl-tmp/dataset/1/train",
                        "/root/autodl-tmp/dataset/2/train",
                        "/root/autodl-tmp/dataset/4/train",
                        "/root/autodl-tmp/dataset/14"],
        general_json="/root/autodl-tmp/dataset/llava_instruct_150k.json",
        general_img_dir="/root/autodl-tmp/dataset/gen/train2017",
        learning_rate=5e-5,
        num_train_epochs=3,
        general_ratio_limit=1.0,  
        alpha_loss_weight=2.0,
        tax_loss_weight=8.0
    )