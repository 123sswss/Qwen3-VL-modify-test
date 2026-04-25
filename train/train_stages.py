# train_stages.py
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.append("..")
import config as cfg
import processingWithMMRL
import QWen3WithMMRL
from data_pipeline import FourViewMMRLDataset, MMRLDataCollator, ID_TO_DATASET_GROUP
from logger import StageMetricLogger, TrainerMetricsCallback
from train_utils import focal_bce_with_logits


def print_stage_step_summary(stage_name, step, row):
    def _fmt(x, nd=4):
        try:
            if x is None:
                return "nan"
            if hasattr(x, "detach"):
                x = x.detach().float().item()
            x = float(x)
            if not np.isfinite(x):
                return "nan"
            return f"{x:.{nd}f}"
        except Exception:
            return "nan"

    print("\n" + "=" * 72)
    print(f"[{stage_name} | Training Step {step}] Loss Breakdown")

    if "total_loss" in row:
        print(f"  ├─ Total Loss:            {_fmt(row.get('total_loss')):>10}")
    if "cls_loss" in row:
        print(f"  ├─ Cls Loss:              {_fmt(row.get('cls_loss')):>10}")
    if "gate_loss" in row:
        print(f"  ├─ Gate Loss:             {_fmt(row.get('gate_loss')):>10}")
    if "ce_loss" in row:
        print(f"  ├─ CE Loss:               {_fmt(row.get('ce_loss')):>10}")
    if "alpha_guide_loss" in row:
        print(f"  ├─ Alpha Guide Loss:      {_fmt(row.get('alpha_guide_loss')):>10}")
    if "budget_loss" in row:
        print(f"  ├─ Budget Loss:           {_fmt(row.get('budget_loss')):>10}")
    if "tax_loss" in row:
        print(f"  ├─ Tax Loss:              {_fmt(row.get('tax_loss')):>10}")

    if "alpha_std" in row or "label_alpha_std" in row:
        print("[Alpha Statistics]")
        if "alpha_std" in row:
            print(f"  ├─ Alpha Std:             {_fmt(row.get('alpha_std')):>10}")
        if "label_alpha_std" in row:
            print(f"  └─ Label Alpha Std:       {_fmt(row.get('label_alpha_std')):>10}")

    if "k_selected_mean" in row:
        print("[Selector Statistics]")
        print(f"  ├─ Mean Selected K:       {_fmt(row.get('k_selected_mean'), 3):>10}")
        if "k_budget_mean" in row:
            print(f"  ├─ Mean Budget K:         {_fmt(row.get('k_budget_mean'), 3):>10}")
        if "alpha_open_rate" in row:
            print(f"  └─ Alpha Open Rate:       {_fmt(row.get('alpha_open_rate'), 4):>10}")

    print("[Schedule]")
    if "temperature" in row:
        print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), 4):>10}")
    if "budget_weight" in row:
        print(f"  ├─ Budget Weight:         {_fmt(row.get('budget_weight'), 4):>10}")
    if "tax_weight" in row:
        print(f"  ├─ Tax Weight:            {_fmt(row.get('tax_weight'), 4):>10}")
    if "learning_rate" in row:
        print(f"  └─ Learning Rate:         {_fmt(row.get('learning_rate'), 8):>10}")
    print("=" * 72 + "\n")


def _dbg_tensor_stats(name, x):
    if x is None:
        return f"{name}=None"
    if not torch.is_tensor(x):
        return f"{name}=<{type(x).__name__}>"
    with torch.no_grad():
        shape = tuple(x.shape)
        dtype = x.dtype
        device = x.device
        if x.numel() == 0:
            return f"{name}: shape={shape} dtype={dtype} device={device} numel=0"
        xf = x.detach().float()
        return (
            f"{name}: shape={shape} dtype={dtype} device={device} "
            f"min={xf.min().item():.4f} max={xf.max().item():.4f} mean={xf.mean().item():.4f}"
        )


class StageScheduleCallback(TrainerCallback):
    def __init__(
        self,
        dataset,
        total_epochs,
        init_temp=1.0,
        final_temp=0.1,
        target_budget=0.05,
        budget_warmup_epochs=1,
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.target_budget = target_budget
        self.budget_warmup_epochs = budget_warmup_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.resample_general_data()

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep = state.epoch if state.epoch is not None else 0.0

        progress = min(ep / max(self.total_epochs, 1e-6), 1.0)
        model.temperature_override = self.init_temp - (self.init_temp - self.final_temp) * progress

        if ep < self.budget_warmup_epochs:
            model.budget_loss_weight = 0.0
        else:
            budget_progress = min(
                (ep - self.budget_warmup_epochs) / max(self.total_epochs - self.budget_warmup_epochs, 1e-6),
                1.0,
            )
            model.budget_loss_weight = self.target_budget * budget_progress


class Qwen3VLMMRLForStages(Qwen3VLForConditionalGeneration):
    def __init__(self, config, tokenizer):
        nn.Module.__init__(self)
        self.config = config
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)
        hidden_size = config.text_config.hidden_size
        self.lm_head = nn.Linear(hidden_size, len(tokenizer), bias=False)
        self.post_init()

        from transformers import GenerationConfig

        self.generation_config = GenerationConfig.from_model_config(config)

        self.ce_loss_weight = 1.0
        self.alpha_loss_weight = 0.5
        self.budget_loss_weight = 0.0
        self.temperature_override = None
        self.enable_dataset_slot_loss_s4 = False
        self.slot_group_constraints_s4 = {}

        self.debug_mode = True
        self.debug_loss_threshold = 20.0

    def _expand_alpha_labels(self, alpha_labels, images_per_sample):
        if alpha_labels is None:
            return None
        if images_per_sample is None:
            return alpha_labels.view(-1, 1)

        chunks = []
        for i, c in enumerate(images_per_sample):
            c = int(c)
            if c > 0:
                chunks.append(alpha_labels[i].repeat(c))
        if len(chunks) == 0:
            return None
        return torch.cat(chunks).view(-1, 1)

    def _compute_alpha_guide_loss(self, alpha_logits, expanded_labels, device):
        if alpha_logits is None or expanded_labels is None or expanded_labels.numel() == 0:
            return alpha_logits, torch.tensor(0.0, device=device)

        if isinstance(alpha_logits, list):
            alpha_logits = torch.stack(alpha_logits) if len(alpha_logits) > 0 else None
        if alpha_logits is None:
            return alpha_logits, torch.tensor(0.0, device=device)
        if alpha_logits.dim() == 1:
            alpha_logits = alpha_logits.unsqueeze(-1)

        n = min(alpha_logits.shape[0], expanded_labels.shape[0])
        if n <= 0:
            return alpha_logits, torch.tensor(0.0, device=device)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            alpha_logits[:n], expanded_labels[:n].to(alpha_logits.dtype)
        ) * self.alpha_loss_weight
        return alpha_logits, loss

    def _collect_selector_metrics(self, device):
        k_selected_mean = torch.tensor(float("nan"), device=device)
        k_budget_mean = torch.tensor(float("nan"), device=device)
        alpha_open_rate = torch.tensor(float("nan"), device=device)

        selector = getattr(self.model.visual, "text_selector_outputs", None)
        if isinstance(selector, dict):
            k_selected = selector.get("k_selected")
            k_budget = selector.get("k_budget")
            if torch.is_tensor(k_selected) and k_selected.numel() > 0:
                k_selected_mean = k_selected.detach().float().mean()
            if torch.is_tensor(k_budget) and k_budget.numel() > 0:
                k_budget_mean = k_budget.detach().float().mean()

        alpha_logits = self.model.visual.alpha_list
        if isinstance(alpha_logits, list):
            alpha_logits = torch.stack(alpha_logits) if len(alpha_logits) > 0 else None
        if torch.is_tensor(alpha_logits) and alpha_logits.numel() > 0:
            alpha_open_rate = (torch.sigmoid(alpha_logits.detach().float()) > 0.5).float().mean()

        return k_selected_mean, k_budget_mean, alpha_open_rate

    def _compute_budget_loss(self, device):
        selector = getattr(self.model.visual, "text_selector_outputs", None)
        if not isinstance(selector, dict):
            return torch.tensor(0.0, device=device)

        k_budget = selector.get("k_budget")
        if not torch.is_tensor(k_budget) or k_budget.numel() == 0:
            return torch.tensor(0.0, device=device)

        total_rep_num = float(getattr(self.model.visual.text_gating, "total_rep_num", 1.0))
        total_rep_num = max(total_rep_num, 1.0)
        usage_ratio = (k_budget.float() / total_rep_num).clamp(min=0.0)

        # 预算头单独承担一个最小可用的“二次型使用惩罚”：
        # K 很小时惩罚近似可忽略，K 越大惩罚增长越快。
        return usage_ratio.pow(2).mean() * self.budget_loss_weight

    def _compute_dataset_slot_loss(self, dataset_group_ids, device):
        zero = torch.tensor(0.0, device=device)
        metrics = {}
        if not self.enable_dataset_slot_loss_s4:
            return zero, metrics
        if dataset_group_ids is None or not torch.is_tensor(dataset_group_ids):
            return zero, metrics

        selector = getattr(self.model.visual, "text_selector_outputs", None)
        if not isinstance(selector, dict):
            return zero, metrics

        selected_mask = selector.get("selected_mask")
        k_selected = selector.get("k_selected")
        if not torch.is_tensor(selected_mask) or selected_mask.numel() == 0:
            return zero, metrics

        dataset_group_ids = dataset_group_ids.to(device=device).view(-1)
        n = min(selected_mask.shape[0], dataset_group_ids.shape[0])
        if n <= 0:
            return zero, metrics

        selected_mask = selected_mask[:n].float()
        dataset_group_ids = dataset_group_ids[:n]
        if not torch.is_tensor(k_selected) or k_selected.numel() == 0:
            k_selected = selected_mask.sum(dim=-1)
        else:
            k_selected = k_selected[:n].float()

        slot_probs = selected_mask / selected_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_entropy = -(slot_probs * slot_probs.clamp_min(1e-6).log()).sum(dim=-1)

        loss_terms = []
        for gid, group_name in ID_TO_DATASET_GROUP.items():
            group_cfg = self.slot_group_constraints_s4.get(group_name)
            if not group_cfg:
                continue
            mask = dataset_group_ids == int(gid)
            if mask.sum().item() <= 0:
                continue

            ent = slot_entropy[mask]
            k_grp = k_selected[mask]
            ent_min = float(group_cfg.get("entropy_min", 0.0))
            ent_max = float(group_cfg.get("entropy_max", 1e9))
            weight = float(group_cfg.get("weight", 0.0))

            ent_range_loss = (torch.relu(ent_min - ent) + torch.relu(ent - ent_max)).mean()
            loss_terms.append(weight * ent_range_loss)

            metrics[f"slot_entropy_{group_name}"] = ent.mean().detach()
            metrics[f"k_mean_{group_name}"] = k_grp.mean().detach()

        if len(loss_terms) == 0:
            return zero, metrics
        return torch.stack(loss_terms).sum(), metrics

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, **kwargs):
        if self.temperature_override is not None:
            self.model.temperature_override = self.temperature_override
            kwargs["gating_temperature_override"] = self.temperature_override

        labels = kwargs.get("labels", None)
        if labels is not None and "attention_mask" in kwargs:
            is_prompt = labels == -100
            att_mask = kwargs["attention_mask"]
            if att_mask.dim() == 2 and is_prompt.dim() == 2:
                is_prompt = is_prompt & (att_mask == 1)
            kwargs["mmrl_gating_mask"] = is_prompt.to(dtype=self.model.dtype)

        outputs = super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)
        logits = outputs.logits
        device = logits.device if torch.is_tensor(logits) else input_ids.device

        if labels is None:
            ce_loss = torch.tensor(0.0, device=device)
        else:
            ce_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)

        expanded_labels = self._expand_alpha_labels(alpha_labels, images_per_sample)
        alpha_logits = self.model.visual.alpha_list
        alpha_logits, alpha_guide_loss = self._compute_alpha_guide_loss(alpha_logits, expanded_labels, device)

        budget_loss = self._compute_budget_loss(device)
        dataset_group_ids = kwargs.get("dataset_group_ids", None)
        slot_dataset_loss, slot_dataset_metrics = self._compute_dataset_slot_loss(dataset_group_ids, device)

        outputs.loss = self.ce_loss_weight * ce_loss + alpha_guide_loss + budget_loss + slot_dataset_loss

        with torch.no_grad():
            alpha_std = torch.tensor(float("nan"), device=device)
            label_alpha_std = torch.tensor(float("nan"), device=device)
            if torch.is_tensor(alpha_logits) and alpha_logits.numel() > 0:
                a = torch.sigmoid(alpha_logits.detach().float().view(-1))
                if a.numel() > 1:
                    alpha_std = a.std(unbiased=False)
            if torch.is_tensor(expanded_labels) and expanded_labels.numel() > 0:
                l = expanded_labels.detach().float().view(-1)
                if l.numel() > 1:
                    label_alpha_std = l.std(unbiased=False)

            k_selected_mean, k_budget_mean, alpha_open_rate = self._collect_selector_metrics(device)

            self._last_metrics = {
                "total_loss": outputs.loss.detach(),
                "ce_loss": ce_loss.detach(),
                "alpha_guide_loss": alpha_guide_loss.detach(),
                "budget_loss": budget_loss.detach(),
                "slot_dataset_loss": slot_dataset_loss.detach(),
                "alpha_std": alpha_std.detach(),
                "label_alpha_std": label_alpha_std.detach(),
                "k_selected_mean": k_selected_mean.detach(),
                "k_budget_mean": k_budget_mean.detach(),
                "alpha_open_rate": alpha_open_rate.detach(),
                "temperature": torch.tensor(
                    float(self.temperature_override) if self.temperature_override is not None else float("nan"),
                    device=device,
                ),
                "budget_weight": torch.tensor(float(self.budget_loss_weight), device=device),
            }
            self._last_metrics.update(slot_dataset_metrics)

        if self.debug_mode and torch.is_tensor(outputs.loss):
            loss_val = outputs.loss.detach().float().item()
            if loss_val > self.debug_loss_threshold or not torch.isfinite(outputs.loss):
                print("\n!!!! [HIGH-LOSS-DBG] abnormal loss detected !!!!")
                print(f"!!!! [HIGH-LOSS-DBG] final_loss={loss_val:.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] manual_ce={ce_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] alpha_loss={alpha_guide_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] budget_w={float(self.budget_loss_weight):.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] budget_loss={budget_loss.detach().float().item():.6f}")

                if labels is not None:
                    valid_per_sample = (labels != -100).sum(dim=1)
                    print(f"!!!! [HIGH-LOSS-DBG] valid_tokens_per_sample={valid_per_sample.tolist()}")
                    print(f"!!!! [HIGH-LOSS-DBG] total_valid_tokens={(labels != -100).sum().item()}/{labels.numel()}")

                print(f"!!!! [HIGH-LOSS-DBG] images_per_sample={images_per_sample}")
                print(f"!!!! [HIGH-LOSS-DBG] {_dbg_tensor_stats('alpha_labels', alpha_labels)}")
                print(f"!!!! [HIGH-LOSS-DBG] {_dbg_tensor_stats('alpha_logits', alpha_logits)}")
                print(f"!!!! [HIGH-LOSS-DBG] {_dbg_tensor_stats('expanded_labels', expanded_labels)}")
                print("!!!! [HIGH-LOSS-DBG] abnormal loss end !!!!\n")

        return outputs


def build_model_and_processor(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.add_special_tokens(cfg.SPECIAL_TOKENS)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print("Base model loaded. ")
    emb = base.get_input_embeddings().weight.shape[0]
    tok = len(tokenizer)
    if tok > emb:
        base.resize_token_embeddings(tok)
        print(f"[Resize] expand embedding: {emb} -> {tok}")
    else:
        print(f"[Resize] skip (tokenizer={tok}, embedding={emb})")
    print("Tokenizer resized. Now building MMRL model...")
    model = Qwen3VLMMRLForStages(config, tokenizer).to("cuda").to(torch.bfloat16)
    model.model.load_state_dict(base.model.state_dict(), strict=False)
    model.lm_head.load_state_dict(base.lm_head.state_dict(), strict=False)
    del base
    torch.cuda.empty_cache()
    print("MMRL model built and loaded with base weights.")
    print(
        f"[DBG] blocks={len(model.model.visual.blocks)}, "
        f"blocks_with_rep={len(model.model.visual.blocks_with_rep)}, "
        f"INSERT_LAYER={list(model.model.visual.cfg.INSERT_LAYER)}"
    )
    with torch.no_grad():
        for idx, layer_num in enumerate(model.model.visual.cfg.INSERT_LAYER):
            model.model.visual.blocks_with_rep[idx].load_state_dict(
                model.model.visual.blocks[layer_num - 1].state_dict(),
                strict=False,
            )
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )
    print("Processor built.")
    return model, processor


def set_trainable_stage(model, stage):
    for p in model.parameters():
        p.requires_grad = False

    v = model.model.visual
    if stage == 1:
        modules = [v.hidden_state_pooling, v.embedding_pooling, v.Task_classifier]
    elif stage == 2:
        modules = [v.hidden_state_pooling, v.embedding_pooling, v.Task_classifier, v.visionGating, v.text_gating]
    elif stage in [3, 4]:
        modules = [
            model.model.MMRL,
            v.blocks_with_rep,
            v.hidden_state_pooling,
            v.embedding_pooling,
            v.Task_classifier,
            v.visionGating,
            v.text_gating,
            v.zero_init_layer,
        ]
    else:
        raise ValueError("stage must be 1/2/3/4")

    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def _build_text_pool_mask(attention_mask, input_ids, rep_placeholder_ids, device):
    placeholder_ids_tensor = torch.tensor(rep_placeholder_ids, device=device)
    is_placeholder = (input_ids.unsqueeze(-1) == placeholder_ids_tensor).any(dim=-1)
    return attention_mask.bool() & (~is_placeholder)


@torch.no_grad()
def _extract_mm_pooled_vision(v, pixel_values, image_grid_thw):
    hs = v.patch_embed(pixel_values.type(v.dtype))
    pos = v.fast_pos_embed_interpolate(image_grid_thw)
    hs = hs + pos
    seq_len, _ = hs.size()
    hs = hs.reshape(seq_len, -1)

    img_token_lens = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).to(torch.long)
    batch_indices = torch.repeat_interleave(torch.arange(image_grid_thw.shape[0], device=hs.device), img_token_lens)
    return v.hidden_state_pooling.forward_vectorized(hs, batch_indices, image_grid_thw.shape[0])


def run_stage12_light(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    assert stage_id in [1, 2]
    set_trainable_stage(model, stage_id)
    model.train()

    metric_logger = StageMetricLogger(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        stage_name=f"stage{stage_id}",
        smooth_window=train_cfg.get("metric_smooth_window", 30),
        ema_alpha=train_cfg.get("metric_ema_alpha", 0.15),
        scatter_stride=train_cfg.get("metric_scatter_stride", 3),
        save_debug_figure=train_cfg.get("save_debug_figure", False),
    )

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

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg["learning_rate"][stage_id], weight_decay=0.01)

    steps_per_update = train_cfg["gradient_accumulation_steps"]
    epochs = train_cfg["epochs"][stage_id]
    use_focal = stage_id == 1

    v = model.model.visual
    rep_placeholder_ids = model.model.rep_placeholder_ids

    global_step = 0
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            alpha_labels = batch["alpha_labels"].cuda(non_blocking=True).view(-1, 1).to(dtype=v.dtype)
            is_mm = batch["is_mm"].cuda(non_blocking=True)
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]

            text_emb = model.model.get_input_embeddings()(input_ids).to(dtype=v.dtype)
            text_pool_mask = _build_text_pool_mask(attention_mask, input_ids, rep_placeholder_ids, input_ids.device)
            text_pooled = v.embedding_pooling(text_emb, mask=text_pool_mask)

            vision_pooled = torch.zeros(
                text_pooled.shape[0], v.cfg.vision_token_dim, device=text_pooled.device, dtype=text_pooled.dtype
            )
            mm_idx = (is_mm == 1).nonzero(as_tuple=True)[0]
            if mm_idx.numel() > 0 and pixel_values is not None:
                mm_pixel = pixel_values.cuda(non_blocking=True)
                mm_grid = image_grid_thw.cuda(non_blocking=True)
                if mm_grid.dim() == 3:
                    mm_grid = mm_grid.squeeze(1)
                vision_pooled[mm_idx] = _extract_mm_pooled_vision(v, mm_pixel, mm_grid)

            alpha_logits = v.Task_classifier(vision_pooled, text_pooled)
            cls_loss = (
                focal_bce_with_logits(alpha_logits, alpha_labels)
                if use_focal
                else torch.nn.functional.binary_cross_entropy_with_logits(alpha_logits, alpha_labels)
            )

            gate_loss = torch.tensor(0.0, device=cls_loss.device)
            if stage_id == 2:
                gate_output = v.visionGating(alpha_logits, getattr(model, "temperature_override", None))
                gate_loss = torch.nn.functional.mse_loss(gate_output, alpha_labels)

            ((cls_loss + gate_loss) / steps_per_update).backward()

            if (i + 1) % steps_per_update == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                with torch.no_grad():
                    alpha_prob = torch.sigmoid(alpha_logits.detach().float().view(-1))
                    alpha_std = alpha_prob.std(unbiased=False) if alpha_prob.numel() > 1 else torch.tensor(float("nan"), device=alpha_prob.device)
                    label_std = alpha_labels.detach().float().view(-1).std(unbiased=False) if alpha_labels.numel() > 1 else torch.tensor(float("nan"), device=alpha_labels.device)

                    metric_logger.log(
                        step=global_step,
                        total_loss=(cls_loss.detach() + gate_loss.detach()),
                        cls_loss=cls_loss.detach(),
                        gate_loss=gate_loss.detach(),
                        alpha_std=alpha_std,
                        label_alpha_std=label_std,
                        temperature=getattr(model, "temperature_override", float("nan")),
                        learning_rate=opt.param_groups[0]["lr"],
                    )

                if global_step % train_cfg.get("console_log_every", 50) == 0:
                    print_stage_step_summary(
                        f"stage{stage_id}",
                        global_step,
                        {
                            "total_loss": (cls_loss.detach() + gate_loss.detach()),
                            "cls_loss": cls_loss.detach(),
                            "gate_loss": gate_loss.detach(),
                            "alpha_std": alpha_std,
                            "label_alpha_std": label_std,
                            "temperature": getattr(model, "temperature_override", float("nan")),
                            "learning_rate": opt.param_groups[0]["lr"],
                        },
                    )

    metric_logger.finalize()


def run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    assert stage_id in [3, 4]
    set_trainable_stage(model, stage_id)

    model.ce_loss_weight = 1.0
    model.alpha_loss_weight = train_cfg.get(f"alpha_loss_weight_s{stage_id}", 0.5)
    model.budget_loss_weight = 0.0
    model.enable_dataset_slot_loss_s4 = bool(stage_id == 4 and train_cfg.get("enable_dataset_slot_loss_s4", False))
    model.slot_group_constraints_s4 = train_cfg.get("slot_group_constraints_s4", {}) if stage_id == 4 else {}

    ds = FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=data_cfg["total_limit"],
        enable_views=("expert-mm", "general-mm"),
        mode=f"stage{stage_id}",
        ce_enabled=True,
        seed=train_cfg["seed"],
    )
    collator = MMRLDataCollator(processor)
    schedule_cb = StageScheduleCallback(
        dataset=ds,
        total_epochs=train_cfg["epochs"][stage_id],
        init_temp=train_cfg.get("initial_temp", 1.0),
        final_temp=train_cfg.get("final_temp", 0.1),
        target_budget=(
            0.0
            if stage_id == 3
            else train_cfg.get("budget_loss_weight", train_cfg.get("tax_loss_weight", 0.05))
        ),
        budget_warmup_epochs=train_cfg.get("budget_warmup_epochs", train_cfg.get("tax_warmup_epochs", 1)),
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
    metric_logger = StageMetricLogger(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        stage_name=f"stage{stage_id}",
        smooth_window=train_cfg.get("metric_smooth_window", 30),
        ema_alpha=train_cfg.get("metric_ema_alpha", 0.15),
        scatter_stride=train_cfg.get("metric_scatter_stride", 3),
        save_debug_figure=train_cfg.get("save_debug_figure", False),
    )
    metrics_cb = TrainerMetricsCallback(
        metric_logger=metric_logger,
        print_every=train_cfg.get("console_log_every", 50),
        stage_name=f"stage{stage_id}",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[schedule_cb, metrics_cb],
    )
    trainer.train()


def run_stage(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    if stage_id in [1, 2]:
        run_stage12_light(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    elif stage_id in [3, 4]:
        run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    else:
        raise ValueError("stage_id must be 1/2/3/4")