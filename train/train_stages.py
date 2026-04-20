# train_stages.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Qwen3VLForConditionalGeneration, AutoConfig, AutoTokenizer, AutoImageProcessor,
    Trainer, TrainingArguments
)

import sys
sys.path.append("..")
import config as cfg
import processingWithMMRL
from data_pipeline import FourViewMMRLDataset, MMRLDataCollator
from train_utils import focal_bce_with_logits
import QWen3WithMMRL
from transformers import TrainerCallback

class StageScheduleCallback(TrainerCallback):
    def __init__(self, dataset, total_epochs, init_temp=1.0, final_temp=0.1, target_tax=4.0, tax_warmup_epochs=1):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.target_tax = target_tax
        self.tax_warmup_epochs = tax_warmup_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.resample_general_data()

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep = state.epoch if state.epoch is not None else 0.0

        # temp anneal
        prog = min(ep / self.total_epochs, 1.0)
        model.temperature_override = self.init_temp - (self.init_temp - self.final_temp) * prog

        # tax warmup
        if ep < self.tax_warmup_epochs:
            model.tax_loss_weight = 0.0
        else:
            p = min((ep - self.tax_warmup_epochs) / max(self.total_epochs - self.tax_warmup_epochs, 1e-6), 1.0)
            model.tax_loss_weight = self.target_tax * p

# =========================
#  Full model (Stage3/4用)
# =========================
class Qwen3VLMMRLForStages(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer):
        nn.Module.__init__(self)
        self.config = config
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, len(tokenizer), bias=False)
        self.post_init()

        # 补充 generation_config，避免 save_pretrained 时报错
        from transformers import GenerationConfig
        self.generation_config = GenerationConfig.from_model_config(config)

        self.ce_loss_weight = 1.0
        self.alpha_loss_weight = 0.5   # Stage3建议0.3~1.0，Stage4可0.5
        self.tax_loss_weight = 0.0     # Stage3=0, Stage4由调度升到目标
        self.k_general_weight = 0.0    # Stage3=0, Stage4可小值(如1.0~2.0)
        self.k_expert_weight = 0.0     # Stage3=0, Stage4可小值(如0.2~0.5)
        self.k_min_expert = 2.0
        self.use_tax = False
        self.use_k_loss = False
        self.temperature_override = None

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, **kwargs):
        if hasattr(self, "temperature_override") and self.temperature_override is not None:
            self.model.temperature_override = self.temperature_override
            kwargs["gating_temperature_override"] = self.temperature_override
        labels = kwargs.get("labels", None)
        if labels is not None and "attention_mask" in kwargs:
            is_prompt = (labels == -100)
            att_mask = kwargs["attention_mask"]
            if att_mask.dim() == 2 and is_prompt.dim() == 2:
                is_prompt = is_prompt & (att_mask == 1)
            kwargs["mmrl_gating_mask"] = is_prompt.to(dtype=self.model.dtype)
        
        outputs = super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)
        ce_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
        # 1) alpha guide
        alpha_guide_loss = torch.tensor(0.0, device=input_ids.device)
        alpha_logits = self.model.visual.alpha_list
        expanded_labels = None
        if alpha_logits is not None and alpha_labels is not None:
            if images_per_sample is None:
                expanded_labels = alpha_labels.view(-1, 1)
            else:
                tmp = []
                for i, c in enumerate(images_per_sample):
                    tmp.append(alpha_labels[i].repeat(c))
                expanded_labels = torch.cat(tmp).view(-1, 1)
            n = min(alpha_logits.shape[0], expanded_labels.shape[0])
            alpha_guide_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                alpha_logits[:n], expanded_labels[:n].to(alpha_logits.dtype)
            ) * self.alpha_loss_weight
        # 2) K loss (optional, stage4再开)
        k_general_loss = torch.tensor(0.0, device=input_ids.device)
        k_expert_loss = torch.tensor(0.0, device=input_ids.device)
        if self.use_k_loss and expanded_labels is not None:
            k_results = self.model.visual.k_results
            k_sums = k_results[0] if isinstance(k_results, tuple) else k_results
            if k_sums is not None and k_sums.shape[0] == expanded_labels.shape[0]:
                lbl = expanded_labels.squeeze(-1)
                is_general = (lbl < 0.1)
                is_expert = (lbl > 0.9)
                k_norm = float(getattr(self.model.visual.text_gating, "total_rep_num", 40.0))
                if is_general.any():
                    kg = k_sums[is_general]
                    k_general_loss = ((kg / k_norm) ** 2).mean() * self.k_general_weight
                if is_expert.any():
                    ke = k_sums[is_expert]
                    gap = torch.relu(self.k_min_expert - ke)
                    k_expert_loss = ((gap / max(self.k_min_expert, 1.0)) ** 2).mean() * self.k_expert_weight
        # 3) tax (optional, stage4再开)
        tax_raw = getattr(self.model, "tax_loss", torch.tensor(0.0, device=input_ids.device))
        tax_loss = tax_raw * self.tax_loss_weight if self.use_tax else torch.tensor(0.0, device=input_ids.device)
        outputs.loss = self.ce_loss_weight * ce_loss + alpha_guide_loss + k_general_loss + k_expert_loss + tax_loss
        return outputs


def build_model_and_processor(model_path):
    print("start to build_model_and_processor")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
    print("1. model components loaded")

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    base.resize_token_embeddings(len(tokenizer))
    print("2. base model loaded and resized")

    # 用 meta device 初始化骨架，跳过真实内存分配和随机初始化
    with torch.device("meta"):
        model = Qwen3VLMMRLForStages(config, tokenizer)
    # 从 base 拷贝共享权重
    model.model.load_state_dict(base.model.state_dict(), strict=False, assign=True)
    model.lm_head.load_state_dict(base.lm_head.state_dict(), strict=False, assign=True)
    del base
    torch.cuda.empty_cache()
    # 将仍在 meta 上的参数和buffer实际初始化到 cuda
    def _materialize_meta(module, prefix=""):
        for name, param in module.named_parameters(recurse=False):
            if param.device == torch.device("meta"):
                new_p = torch.empty(param.shape, dtype=torch.bfloat16, device="cuda")
                nn.init.normal_(new_p, std=0.02)
                setattr(module, name, nn.Parameter(new_p, requires_grad=param.requires_grad))
        for name, buf in module.named_buffers(recurse=False):
            if buf is not None and buf.device == torch.device("meta"):
                new_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cuda")
                module.register_buffer(name, new_buf)
        for child_name, child in module.named_children():
            _materialize_meta(child, prefix=f"{prefix}{child_name}.")
    _materialize_meta(model)
    model = model.to("cuda").to(torch.bfloat16)
    print("3. Qwen3VLMMRLForStages initialized with base weights")

    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )
    with torch.no_grad():
    for idx, layer_num in enumerate(model.model.visual.cfg.INSERT_LAYER):
        model.model.visual.blocks_with_rep[idx].load_state_dict(
            model.model.visual.blocks[layer_num].state_dict(),
            strict=False
        )
        print(f"[Init] Copied blocks[{layer_num}] -> blocks_with_rep[{idx}]")
    print("build_model_and_processor is done")
    return model, processor


# =========================
#  Freeze policy
# =========================
def set_trainable_stage(model, stage):
    for p in model.parameters():
        p.requires_grad = False

    v = model.model.visual
    if stage == 1:
        mods = [v.hidden_state_pooling, v.embedding_pooling, v.Task_classifier]
    elif stage == 2:
        mods = [v.hidden_state_pooling, v.embedding_pooling, v.Task_classifier, v.visionGating, v.text_gating]
    elif stage in [3, 4]:
        mods = [model.model.MMRL, v.blocks_with_rep, v.hidden_state_pooling, v.embedding_pooling,
                v.Task_classifier, v.visionGating, v.text_gating, v.zero_init_layer]
    else:
        raise ValueError("stage must be 1/2/3/4")

    for m in mods:
        for p in m.parameters():
            p.requires_grad = True


# =========================
#  轻前向工具（Stage1/2）
# =========================
def _build_text_pool_mask(attention_mask, input_ids, rep_placeholder_ids, device):
    placeholder_ids_tensor = torch.tensor(rep_placeholder_ids, device=device)
    is_placeholder = (input_ids.unsqueeze(-1) == placeholder_ids_tensor).any(dim=-1)
    text_pooling_mask = attention_mask.bool() & (~is_placeholder)
    return text_pooling_mask


@torch.no_grad()
def _extract_mm_pooled_vision(v, pixel_values, image_grid_thw):
    # 只跑视觉patch + pooling，不跑全主干
    hs = v.patch_embed(pixel_values.type(v.dtype))
    pos = v.fast_pos_embed_interpolate(image_grid_thw)
    hs = hs + pos
    seq_len, _ = hs.size()
    hs = hs.reshape(seq_len, -1)

    # 按图切分
    img_token_lens = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).to(torch.long)
    batch_indices = torch.repeat_interleave(
        torch.arange(image_grid_thw.shape[0], device=hs.device),
        img_token_lens
    )
    pooled = v.hidden_state_pooling.forward_vectorized(hs, batch_indices, image_grid_thw.shape[0])
    return pooled


def run_stage12_light(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    assert stage_id in [1, 2]
    set_trainable_stage(model, stage_id)
    model.train()

    # 数据：四视图都开，CE关
    ds = FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=data_cfg["total_limit"],
        enable_views=("expert-mm", "expert-text", "general-mm", "general-text"),
        mode=f"stage{stage_id}",
        ce_enabled=False,
        seed=train_cfg["seed"],
    )
    collator = MMRLDataCollator(processor)

    dl = DataLoader(
        ds,
        batch_size=train_cfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("dataloader_num_workers", 4),
        pin_memory=False,
        collate_fn=collator,
        drop_last=True,
    )

    # 仅优化可训练参数
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg["learning_rate"][stage_id], weight_decay=0.01)

    steps_per_update = train_cfg["gradient_accumulation_steps"]
    epochs = train_cfg["epochs"][stage_id]
    use_focal = (stage_id == 1)

    v = model.model.visual
    rep_placeholder_ids = model.model.rep_placeholder_ids

    global_step = 0
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            alpha_labels = batch["alpha_labels"].cuda(non_blocking=True).view(-1, 1).to(dtype=v.dtype)
            is_mm = batch["is_mm"].cuda(non_blocking=True)
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]

            # text embedding + pooling
            text_emb = model.model.get_input_embeddings()(input_ids).to(dtype=v.dtype)
            text_pool_mask = _build_text_pool_mask(attention_mask, input_ids, rep_placeholder_ids, input_ids.device)
            text_pooled = v.embedding_pooling(text_emb, mask=text_pool_mask)

            # vision pooled（仅mm样本有意义）
            vision_pooled = torch.zeros(
                text_pooled.shape[0], v.cfg.vision_token_dim, device=text_pooled.device, dtype=text_pooled.dtype
            )
            mm_idx = (is_mm == 1).nonzero(as_tuple=True)[0]
            if mm_idx.numel() > 0 and pixel_values is not None:
                mm_pixel = pixel_values.cuda(non_blocking=True)
                mm_grid = image_grid_thw.cuda(non_blocking=True)
                if mm_grid.dim() == 3:
                    mm_grid = mm_grid.squeeze(1)  # [N,1,3] -> [N,3]
                mm_vision_pooled = _extract_mm_pooled_vision(v, mm_pixel, mm_grid)
                vision_pooled[mm_idx] = mm_vision_pooled

            alpha_logits = v.Task_classifier(vision_pooled, text_pooled)

            if use_focal:
                cls_loss = focal_bce_with_logits(alpha_logits, alpha_labels)
            else:
                cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(alpha_logits, alpha_labels)

            # stage2：把分类结果传递给gate（轻量约束）
            gate_loss = torch.tensor(0.0, device=cls_loss.device)
            if stage_id == 2:
                g = v.visionGating(alpha_logits, getattr(model, "temperature_override", None))
                target = alpha_labels
                gate_loss = torch.nn.functional.mse_loss(g, target)

            loss = (cls_loss + gate_loss) / steps_per_update
            loss.backward()

            if (i + 1) % steps_per_update == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 20 == 0:
                    print(f"[Stage{stage_id}] ep={ep} step={global_step} cls={cls_loss.item():.4f} gate={gate_loss.item():.4f}")

    # save_path = f"{output_dir}/stage{stage_id}"
    # model.save_pretrained(save_path)
    # print(f"[Stage{stage_id}] saved to {save_path}")


# =========================
#  Stage3/4 (完整Trainer)
# =========================
def run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    assert stage_id in [3, 4]
    set_trainable_stage(model, stage_id)

    if stage_id == 3:
        model.ce_loss_weight = 1.0
        model.alpha_loss_weight = train_cfg.get("alpha_loss_weight_s3", 0.5)
        model.use_tax = False
        model.tax_loss_weight = 0.0
        model.use_k_loss = False
        model.k_general_weight = 0.0
        model.k_expert_weight = 0.0
    else:  # stage4
        model.ce_loss_weight = 1.0
        model.alpha_loss_weight = train_cfg.get("alpha_loss_weight_s4", 0.5)
        model.use_tax = True
        model.tax_loss_weight = 0.0  # 由callback warmup拉升
        model.use_k_loss = train_cfg.get("enable_k_loss_s4", True)
        model.k_general_weight = train_cfg.get("k_general_weight_s4", 1.5)
        model.k_expert_weight = train_cfg.get("k_expert_weight_s4", 0.3)

    ds = FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=data_cfg["total_limit"],
        enable_views=("expert-mm", "expert-text", "general-mm", "general-text"),
        mode=f"stage{stage_id}",
        ce_enabled=ce_enabled,
        seed=train_cfg["seed"],
    )
    collator = MMRLDataCollator(processor)
    cb = StageScheduleCallback(
        dataset=ds,
        total_epochs=train_cfg["epochs"][stage_id],
        init_temp=train_cfg.get("initial_temp", 1.0),
        final_temp=train_cfg.get("final_temp", 0.1),
        target_tax=(0.0 if stage_id == 3 else train_cfg["tax_loss_weight"]),
        tax_warmup_epochs=train_cfg.get("tax_warmup_epochs", 1),
    )
    args = TrainingArguments(
        output_dir=f"{output_dir}/stage{stage_id}",
        num_train_epochs=train_cfg["epochs"][stage_id],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"][stage_id],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator, callbacks=[cb])
    trainer.train()
    # trainer.save_model(f"{output_dir}/stage{stage_id}")


def run_stage(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    if stage_id in [1, 2]:
        run_stage12_light(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    elif stage_id in [3, 4]:
        run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    else:
        raise ValueError("stage_id must be 1/2/3/4")