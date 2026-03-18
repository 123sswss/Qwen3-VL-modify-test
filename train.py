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

from logger import MetricsLogger

from dataset import MixedMMRLDataset


class MMRLTrainingCallback(TrainerCallback):
    def __init__(self, dataset, initial_temp=3.0, final_temp=0.3,
                 total_epochs=5, enable_temp_annealing=True,target_tax=5.0,tax_warmup_epochs=1):
        self.dataset = dataset
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.enable_temp_annealing = enable_temp_annealing
        self.target_tax = target_tax
        self.tax_warmup_epochs = tax_warmup_epochs
        self.current_epoch = 0

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时绘制曲线"""
        model = kwargs['model']
        if hasattr(model, 'metrics_logger') and model.metrics_logger is not None:
            model.metrics_logger.plot_curves(args.output_dir, 10000)
    
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


class Qwen3VLMMRLForTrain(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer, tax_loss_weight=1.0, alpha_loss_weight=1.0):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.config = config
        self.tax_loss_weight = tax_loss_weight
        self.alpha_loss_weight = alpha_loss_weight
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
        self.training_step_counter = 0
        self.temperature_override = None
        self.k_sums_history = []
        # 新增：K损失目标与权重
        self.k_target_expert = 6.0
        self.k_general_weight = 8.0
        self.k_expert_weight = 2.0
        self.metrics_logger = None
    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, **kwargs):
        # 修复：显式同步温度到内部模型，并传入兼容参数名
        if hasattr(self, 'temperature_override') and self.temperature_override is not None:
            self.model.temperature_override = self.temperature_override
            kwargs['gating_temperature_override'] = self.temperature_override
        labels = kwargs.get('labels', None)
        gating_mask = None
        is_prompt = (labels == -100)
        if 'attention_mask' in kwargs:
            att_mask = kwargs['attention_mask']
            if att_mask.dim() == 2 and is_prompt.dim() == 2:
                is_prompt = is_prompt & (att_mask == 1)
        
        gating_mask = is_prompt.to(dtype=self.model.dtype)
        kwargs['mmrl_gating_mask'] = gating_mask
        outputs = super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)
        raw_tax_loss = getattr(self.model, 'tax_loss', 0.0)
        if not torch.is_tensor(raw_tax_loss):
            raw_tax_loss = torch.tensor(raw_tax_loss, device=outputs.loss.device, dtype=outputs.loss.dtype)
        mmrl_tax_loss = raw_tax_loss * self.tax_loss_weight
        # B. Alpha Guide Loss
        alpha_logits = self.model.visual.alpha_list
        alpha_guide_loss = torch.tensor(0.0, device=outputs.loss.device)
        alpha_probs = None
        expanded_labels = None
        if alpha_logits is not None and alpha_labels is not None:
            if isinstance(alpha_logits, list) and len(alpha_logits) > 0:
                stacked_logits = torch.stack(alpha_logits)
            else:
                stacked_logits = alpha_logits
            if stacked_logits is not None:
                if images_per_sample is None:
                    expanded_labels = alpha_labels.view(-1, 1)
                else:
                    expanded_labels_list = []
                    for idx, count in enumerate(images_per_sample):
                        label = alpha_labels[idx]
                        expanded_labels_list.append(label.repeat(count))
                    expanded_labels = torch.cat(expanded_labels_list).view(-1, 1)
                min_len = min(stacked_logits.shape[0], expanded_labels.shape[0])
                sliced_logits = stacked_logits[:min_len]
                sliced_labels = expanded_labels[:min_len]
                
                loss_fct = torch.nn.BCEWithLogitsLoss()
                alpha_guide_loss = loss_fct(
                    sliced_logits,
                    sliced_labels.to(sliced_logits.dtype)
                ) * self.alpha_loss_weight
                
                alpha_probs = torch.sigmoid(sliced_logits).detach()
        # C. K Loss 策略（改为：General -> 0；Expert -> 至少达到目标）
        k_general_loss = torch.tensor(0.0, device=outputs.loss.device)
        k_expert_loss = torch.tensor(0.0, device=outputs.loss.device)
        mean_k_general = 0.0
        mean_k_expert = 0.0
        
        k_results = self.model.visual.k_results
        k_sums_value = k_results[0] if isinstance(k_results, tuple) else k_results
        if k_sums_value is not None and alpha_labels is not None:
            if images_per_sample is None:
                k_expanded_labels = alpha_labels
            else:
                k_expanded_labels_list = []
                for idx, count in enumerate(images_per_sample):
                    label = alpha_labels[idx]
                    k_expanded_labels_list.append(label.repeat(count))
                k_expanded_labels = torch.cat(k_expanded_labels_list)
            if k_sums_value.shape[0] == k_expanded_labels.shape[0]:
                is_general = (k_expanded_labels < 0.1)
                is_expert = (k_expanded_labels > 0.9)
                k_norm_factor = float(getattr(self.model.visual.text_gating, "total_rep_num", 40.0))
                if k_norm_factor <= 0:
                    k_norm_factor = 40.0
                if is_general.any():
                    general_k = k_sums_value[is_general]
                    mean_k_general = general_k.mean().item()
                    k_general_loss = ((general_k / k_norm_factor) ** 2).mean() * self.k_general_weight
                if is_expert.any():
                    expert_k = k_sums_value[is_expert]
                    mean_k_expert = expert_k.mean().item()
                    target_k = min(self.k_target_expert, k_norm_factor)
                    gap = torch.relu(target_k - expert_k)
                    k_expert_loss = ((gap / max(target_k, 1.0)) ** 2).mean() * self.k_expert_weight
        total_mmrl_loss = k_general_loss + k_expert_loss + alpha_guide_loss + mmrl_tax_loss
        
        if outputs.loss is not None:
            outputs.loss = outputs.loss + total_mmrl_loss
        if self.training:
            self.training_step_counter += 1
            if self.metrics_logger is not None:
                self.metrics_logger.log_step(
                    ce_loss=(outputs.loss - total_mmrl_loss).item(),
                    alpha_loss=alpha_guide_loss.item(),
                    k_general_loss=k_general_loss.item(),
                    k_expert_loss=k_expert_loss.item(),
                    tax_loss=mmrl_tax_loss.item(),
                    alpha_probs=alpha_probs,
                    alpha_labels=expanded_labels,
                    mean_k_general=mean_k_general,
                    mean_k_expert=mean_k_expert)
            if self.training_step_counter % 50 == 0:
                print("\n" + "=" * 60)
                print(f"[Training Step {self.training_step_counter}] Loss Breakdown:")
                print(f"  ├─ CE Loss (Language):    {(outputs.loss - total_mmrl_loss).item():>8.4f}")
                print(f"  ├─ Alpha Guide Loss:      {alpha_guide_loss.item():>8.4f}")
                print(f"  ├─ K Loss (General):      {k_general_loss.item():>8.4f} (Target K=0)")
                print(f"  ├─ K Loss (Expert):       {k_expert_loss.item():>8.4f} (Target K>={self.k_target_expert:.1f})")
                print(f"  ├─ Tax Loss:              {mmrl_tax_loss.item():>8.4f} (W={self.tax_loss_weight:.2f})")
                print(f"  └─ Total Loss:            {outputs.loss.item():>8.4f}")
                if k_sums_value is not None:
                    print(f"[K Statistics]")
                    print(f"  ├─ General Mean K:        {mean_k_general:>8.2f} (Should -> 0)")
                    print(f"  ├─ Expert Mean K:         {mean_k_expert:>8.2f} (Should >= {self.k_target_expert:.1f})")
                    print(f"  ├─ Max K (Batch):         {k_sums_value.max().item():>8.2f}")
                    print(f"  └─ Min K (Batch):         {k_sums_value.min().item():>8.2f}")
                if alpha_probs is not None and expanded_labels is not None:
                    print(f"[Alpha Statistics]")
                    print(f"  ├─ Mean Probability:      {alpha_probs.mean().item():>8.4f}")
                    print(f"  ├─ Target Mean:           {expanded_labels.float().mean().item():>8.4f}")
                
                curr_temp = self.temperature_override if self.temperature_override is not None else -1.0
                print(f"[Temperature] Current:      {curr_temp:>8.4f}")
                print("=" * 60 + "\n")
        return outputs

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
        resume_path: str = None, # 新增一个参数接收 checkpoint 路径
        learning_rate: float = 2e-4,       
        num_train_epochs: int = 5,         
        general_ratio_limit: float = 0.4,  
        alpha_loss_weight: float = 8.0,     
        tax_loss_weight: float = 1.0,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 16
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
    print(f"-> 正在从 {resume_path if resume_path else MODEL_PATH} 初始化模型...")
    
    if resume_path:
        # 如果有 checkpoint，直接加载训练好的模型
        # 注意：这里假设你的 Qwen3VLMMRLForTrain 实现了标准的 from_pretrained 或者能直接 load_state_dict
        model = Qwen3VLMMRLForTrain.from_pretrained(
            resume_path, 
            config=config, 
            tokenizer=tokenizer,
            tax_loss_weight=tax_loss_weight, 
            alpha_loss_weight=alpha_loss_weight,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        print("-> 已加载 checkpoint 权重，跳过权重迁移步骤。")
    else:
        # 原始的加载逻辑：从原始模型迁移权重
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
        model.model.visual.hidden_state_pooling,
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

    metrics_logger = MetricsLogger(
        batch_size=per_device_train_batch_size,  # 与 TrainingArguments 中的 per_device_train_batch_size 一致
        gradient_accumulation_steps=gradient_accumulation_steps  # 与 TrainingArguments 中的一致
    )
    model.metrics_logger = metrics_logger

    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )

    dataset = MixedMMRLDataset(
        processor,
        expert_json, expert_img_dir,
        general_json, general_img_dir,
        total_limit=10000, mode="mixed", general_ratio_limit=1.0
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
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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

    # print("-> 正在调试第一个 Batch 的数据...")
    # for batch in trainer.get_train_dataloader():
    #     input_ids = batch['input_ids']
    #     print(f"-> Batch input_ids 形状: {input_ids.shape}")
        
    #     # 假设你用来定位的 special token 是 cfg.SPECIAL_TOKENS 里的某个值
    #     # 这里打印出 input_ids 里是否有自定义的 token
    #     # （假设新加的 special token ID 都在 151643 之后）
    #     special_tokens_in_batch = (input_ids >= 151643).sum().item()
    #     print(f"-> 本批次中包含的大于 151643 的特殊 Token 数量: {special_tokens_in_batch}")
        
    #     if special_tokens_in_batch == 0:
    #         print("❌ 警告：在这个 Batch 中没有找到特殊 Token，这可能就是 t_feat 为 0 的原因！")
    #         # 打印解码后的文本，看看错在哪
    #     print("-> 解码后的文本: ", tokenizer.decode(input_ids[0][:])) 
    #     break # 只检查第一个 batch
        
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
                     "/root/autodl-tmp/dataset/14json.json",
                     "/root/autodl-tmp/dataset/prof_test.json"],
        expert_img_dir=["/root/autodl-tmp/dataset/1/train",
                        "/root/autodl-tmp/dataset/2/train",
                        "/root/autodl-tmp/dataset/4/train",
                        "/root/autodl-tmp/dataset/14"],
        general_json=["/root/autodl-tmp/dataset/llava_instruct_150k.json",
                      "/root/autodl-tmp/dataset/gen_test.json"],
        general_img_dir=["/root/autodl-tmp/dataset/gen/train2017",
                         "/root/autodl-tmp/dataset/gen/val2017"],
        learning_rate=1e-4,
        num_train_epochs=3,
        general_ratio_limit=1.0,  
        alpha_loss_weight=2.0,
        tax_loss_weight=4.0,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
    )# todo:训练策略大改 无需抑制通用任务的K  另外alpha辅助调值改成G值调制