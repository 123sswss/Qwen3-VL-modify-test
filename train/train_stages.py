# train_stages.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Qwen3VLForConditionalGeneration, AutoConfig, AutoTokenizer, AutoImageProcessor,
    Trainer, TrainingArguments
)
import numpy as np

import sys
sys.path.append("..")
import config as cfg
import processingWithMMRL
from data_pipeline import (
    FourViewMMRLDataset, MMRLDataCollator,
    DATASET_GROUP_TO_ID, ID_TO_DATASET_GROUP
)
from train_utils import focal_bce_with_logits
import QWen3WithMMRL
from transformers import TrainerCallback

from logger import StageMetricLogger, TrainerMetricsCallback

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
    if "k_general_loss" in row:
        print(f"  ├─ K Loss (General):      {_fmt(row.get('k_general_loss')):>10}")
    if "k_expert_loss" in row:
        print(f"  ├─ K Loss (Expert):       {_fmt(row.get('k_expert_loss')):>10}")
    if "tax_loss" in row:
        print(f"  ├─ Tax Loss:              {_fmt(row.get('tax_loss')):>10}")

    if "alpha_std" in row or "label_alpha_std" in row:
        print(f"[Alpha Statistics]")
        if "alpha_std" in row:
            print(f"  ├─ Alpha Std:             {_fmt(row.get('alpha_std')):>10}")
        if "label_alpha_std" in row:
            print(f"  └─ Label Alpha Std:       {_fmt(row.get('label_alpha_std')):>10}")

    if "k_general_mean" in row or "k_expert_mean" in row:
        print(f"[K Statistics]")
        if "k_general_mean" in row:
            print(f"  ├─ General Mean K:        {_fmt(row.get('k_general_mean'), 3):>10}")
        if "k_expert_mean" in row:
            print(f"  ├─ Expert Mean K:         {_fmt(row.get('k_expert_mean'), 3):>10}")
        if "dynamic_k_lambda_general" in row:
            print(f"  ├─ Lambda General:        {_fmt(row.get('dynamic_k_lambda_general'), 4):>10}")
        if "dynamic_k_lambda_expert" in row:
            print(f"  └─ Lambda Expert:         {_fmt(row.get('dynamic_k_lambda_expert'), 4):>10}")

    print(f"[Schedule]")
    if "temperature" in row:
        print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), 4):>10}")
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
        self, dataset, total_epochs,
        init_temp=1.0, final_temp=0.1,
        target_tax=4.0, tax_warmup_epochs=1,
        k_reg_start=0.25, k_reg_target=1.0,
        collapse_reg_start=0.1, collapse_reg_target=1.0
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.target_tax = target_tax
        self.tax_warmup_epochs = tax_warmup_epochs

        self.k_reg_start = k_reg_start
        self.k_reg_target = k_reg_target
        self.collapse_reg_start = collapse_reg_start
        self.collapse_reg_target = collapse_reg_target

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.resample_general_data()

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep = state.epoch if state.epoch is not None else 0.0
        prog = min(ep / max(self.total_epochs, 1e-6), 1.0)

        # temp anneal
        model.temperature_override = self.init_temp - (self.init_temp - self.final_temp) * prog

        # tax warmup
        if ep < self.tax_warmup_epochs:
            model.tax_loss_weight = 0.0
        else:
            p = min((ep - self.tax_warmup_epochs) / max(self.total_epochs - self.tax_warmup_epochs, 1e-6), 1.0)
            model.tax_loss_weight = self.target_tax * p

        # 新增：K 正则强度渐进
        if hasattr(model, "k_reg_scale"):
            model.k_reg_scale = self.k_reg_start + (self.k_reg_target - self.k_reg_start) * prog

        # 新增：防塌缩强度渐进
        if hasattr(model, "collapse_reg_scale"):
            model.collapse_reg_scale = self.collapse_reg_start + (self.collapse_reg_target - self.collapse_reg_start) * prog

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
        
        self.k_general_weight = 0.0
        self.k_expert_weight = 0.0
        # ===== dynamic lambda for K control =====
        self.use_dynamic_k_lambda = False
        self.k_general_target = 0.0
        self.k_expert_target = 8.0
        self.k_general_lambda = 0.0
        self.k_expert_lambda = 0.0
        self.k_lambda_lr_general = 0.02
        self.k_lambda_lr_expert = 0.01
        self.k_lambda_max_general = 5.0
        self.k_lambda_max_expert = 5.0

        self.use_tax = False
        self.use_k_loss = False
        self.temperature_override = None

        self.debug_mode = True
        self.debug_loss_threshold = 20.0

        # ===== dataset-conditioned K loss =====
        self.use_dataset_k_range_loss = False
        self.k_group_constraints = {}
        self.k_range_global_weight = 1.0
        self.k_reg_scale = 1.0
        self.dataset_group_to_id = dict(DATASET_GROUP_TO_ID)
        # ===== slot anti-collapse =====
        self.enable_slot_collapse_loss = False
        self.slot_collapse_weight = 0.0
        self.slot_neff_min = 6.0
        self.slot_top1_max = 0.35
        self.slot_top1_weight = 0.0
        self.collapse_reg_scale = 1.0
        self.slot_collapse_expert_only = True

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, **kwargs):
        if hasattr(self, "temperature_override") and self.temperature_override is not None:
            self.model.temperature_override = self.temperature_override
            kwargs["gating_temperature_override"] = self.temperature_override
        dataset_group_ids = kwargs.pop("dataset_group_ids", None)
        labels = kwargs.get("labels", None)
        if labels is not None and "attention_mask" in kwargs:
            is_prompt = (labels == -100)
            att_mask = kwargs["attention_mask"]
            if att_mask.dim() == 2 and is_prompt.dim() == 2:
                is_prompt = is_prompt & (att_mask == 1)
            kwargs["mmrl_gating_mask"] = is_prompt.to(dtype=self.model.dtype)
        
        outputs = super().forward(input_ids=input_ids, images_per_sample=images_per_sample, **kwargs)
        logits = outputs.logits
        labels = kwargs.get("labels", None)
        if labels is None:
            ce_loss = torch.tensor(0.0, device=logits.device)
        else:
            ce_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
        # 1) alpha guide
        alpha_guide_loss = torch.tensor(0.0, device=input_ids.device)
        alpha_logits = self.model.visual.alpha_list
        expanded_labels = None

        if isinstance(alpha_logits, list):
            if len(alpha_logits) > 0:
                alpha_logits = torch.stack(alpha_logits)
            else:
                alpha_logits = None

        if alpha_logits is not None and alpha_labels is not None:
            if images_per_sample is None:
                expanded_labels = alpha_labels.view(-1, 1)
            else:
                tmp = []
                for i, c in enumerate(images_per_sample):
                    c = int(c)
                    if c > 0:
                        tmp.append(alpha_labels[i].repeat(c))
                if len(tmp) > 0:
                    expanded_labels = torch.cat(tmp).view(-1, 1)
                else:
                    expanded_labels = None

            if expanded_labels is not None and expanded_labels.numel() > 0:
                if alpha_logits.dim() == 1:
                    alpha_logits = alpha_logits.unsqueeze(-1)

                n = min(alpha_logits.shape[0], expanded_labels.shape[0])
                if n > 0:
                    alpha_guide_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        alpha_logits[:n], expanded_labels[:n].to(alpha_logits.dtype)
                    ) * self.alpha_loss_weight

        # 2) K loss (dataset-conditioned) + slot anti-collapse
        device = logits.device
        nan_t = torch.tensor(float("nan"), device=device)

        k_total_loss = torch.tensor(0.0, device=device)
        k_general_loss = torch.tensor(0.0, device=device)
        k_expert_loss = torch.tensor(0.0, device=device)
        k_general_mean = nan_t
        k_expert_mean = nan_t

        k_loss_report = torch.tensor(0.0, device=device)
        k_loss_vqa = torch.tensor(0.0, device=device)
        k_loss_test = torch.tensor(0.0, device=device)
        k_mean_report = nan_t
        k_mean_vqa = nan_t
        k_mean_test = nan_t

        slot_collapse_loss = torch.tensor(0.0, device=device)
        slot_neff = nan_t
        slot_top1_share = nan_t

        if self.use_k_loss:
            k_out = self.model.visual.k_results
            k_sums = None
            selected_mask = None

            if isinstance(k_out, dict):
                k_sums = k_out.get("k_selected", None)
                selected_mask = k_out.get("selected_mask", None)
            elif torch.is_tensor(k_out):
                k_sums = k_out
            elif isinstance(k_out, (list, tuple)) and len(k_out) > 0 and torch.is_tensor(k_out[0]):
                k_sums = k_out[0]

            if k_sums is not None:
                k_sums = k_sums.float().view(-1)
                k_norm = float(getattr(self.model.visual.text_gating, "total_rep_num", 40.0))

                # 准备 group ids
                group_ids = None
                if dataset_group_ids is not None:
                    if torch.is_tensor(dataset_group_ids):
                        group_ids = dataset_group_ids.to(device=device, dtype=torch.long).view(-1)
                    else:
                        group_ids = torch.tensor(dataset_group_ids, device=device, dtype=torch.long).view(-1)

                    # 若长度不一致，尝试按 images_per_sample 展开
                    if group_ids.numel() != k_sums.numel() and images_per_sample is not None and len(images_per_sample) == group_ids.numel():
                        rep = []
                        for i, c in enumerate(images_per_sample):
                            c = int(c)
                            if c > 0:
                                rep.append(group_ids[i].repeat(c))
                        if len(rep) > 0:
                            group_ids = torch.cat(rep, dim=0)
                    if group_ids.numel() != k_sums.numel():
                        group_ids = None

                # ---- A) dataset-conditioned K range ----
                used_dataset_k_loss = False
                if self.use_dataset_k_range_loss and group_ids is not None and isinstance(self.k_group_constraints, dict) and len(self.k_group_constraints) > 0:
                    group_loss_cache = {}
                    group_mean_cache = {}
                    active_cnt = 0

                    for gname_raw, gcfg in self.k_group_constraints.items():
                        if isinstance(gname_raw, int):
                            gid = int(gname_raw)
                            gname = ID_TO_DATASET_GROUP.get(gid, str(gid))
                        else:
                            gname = str(gname_raw)
                            gid = self.dataset_group_to_id.get(gname, None)

                        if gid is None:
                            continue

                        mask = (group_ids == gid)
                        if not mask.any():
                            continue

                        kg = k_sums[mask]
                        kg_n = kg / k_norm

                        k_min = float(gcfg.get("k_min", gcfg.get("min", 0.0)))
                        k_max = float(gcfg.get("k_max", gcfg.get("max", k_norm)))
                        range_w = float(gcfg.get("range_weight", gcfg.get("weight", 1.0)))
                        mean_w = float(gcfg.get("mean_weight", 0.02))
                        anchor = float(gcfg.get("mean_anchor", gcfg.get("anchor", (k_min + k_max) * 0.5)))

                        k_min_n = k_min / k_norm
                        k_max_n = k_max / k_norm
                        anchor_n = anchor / k_norm

                        low = torch.relu(k_min_n - kg_n)
                        high = torch.relu(kg_n - k_max_n)

                        loss_range = range_w * (low.pow(2) + high.pow(2)).mean()
                        loss_mean = mean_w * (kg_n.mean() - anchor_n).pow(2)
                        g_loss = loss_range + loss_mean

                        group_loss_cache[gname] = g_loss
                        group_mean_cache[gname] = kg.mean()
                        active_cnt += 1

                    if active_cnt > 0:
                        k_dataset = torch.stack(list(group_loss_cache.values())).mean()
                        k_total_loss = k_dataset * float(self.k_range_global_weight) * float(self.k_reg_scale)
                        used_dataset_k_loss = True

                    # 兼容旧日志字段
                    if "general" in group_loss_cache:
                        k_general_loss = group_loss_cache["general"] * float(self.k_range_global_weight) * float(self.k_reg_scale)
                        k_general_mean = group_mean_cache.get("general", nan_t)

                    expert_losses = []
                    for n in ("report", "vqa", "test"):
                        if n in group_loss_cache:
                            expert_losses.append(group_loss_cache[n])
                    if len(expert_losses) > 0:
                        k_expert_loss = torch.stack(expert_losses).sum() * float(self.k_range_global_weight) * float(self.k_reg_scale)

                    expert_means = []
                    if "report" in group_mean_cache:
                        k_mean_report = group_mean_cache["report"]
                        k_loss_report = group_loss_cache["report"] * float(self.k_range_global_weight) * float(self.k_reg_scale)
                        expert_means.append(group_mean_cache["report"])
                    if "vqa" in group_mean_cache:
                        k_mean_vqa = group_mean_cache["vqa"]
                        k_loss_vqa = group_loss_cache["vqa"] * float(self.k_range_global_weight) * float(self.k_reg_scale)
                        expert_means.append(group_mean_cache["vqa"])
                    if "test" in group_mean_cache:
                        k_mean_test = group_mean_cache["test"]
                        k_loss_test = group_loss_cache["test"] * float(self.k_range_global_weight) * float(self.k_reg_scale)
                        expert_means.append(group_mean_cache["test"])

                    if len(expert_means) > 0:
                        k_expert_mean = torch.stack(expert_means).mean()

                # ---- B) fallback: 旧 general/expert 二元 K loss ----
                if (not used_dataset_k_loss) and (expanded_labels is not None) and (k_sums.shape[0] == expanded_labels.shape[0]):
                    lbl = expanded_labels.squeeze(-1)
                    is_general = (lbl < 0.1)
                    is_expert = (lbl > 0.9)

                    if is_general.any():
                        kg = k_sums[is_general].float()
                        k_general_mean = kg.mean()
                        if self.use_dynamic_k_lambda and self.training:
                            with torch.no_grad():
                                err_g = (k_general_mean.item() - self.k_general_target) / k_norm
                                self.k_general_lambda = self.k_general_lambda + self.k_lambda_lr_general * err_g
                                self.k_general_lambda = max(0.0, min(self.k_general_lambda, self.k_lambda_max_general))
                        current_lambda_g = self.k_general_lambda if self.use_dynamic_k_lambda else self.k_general_weight
                        k_general_loss = current_lambda_g * (kg / k_norm).mean()

                    if is_expert.any():
                        ke = k_sums[is_expert].float()
                        k_expert_mean = ke.mean()
                        if self.use_dynamic_k_lambda and self.training:
                            with torch.no_grad():
                                err_e = (k_expert_mean.item() - self.k_expert_target) / k_norm
                                self.k_expert_lambda = self.k_expert_lambda + self.k_lambda_lr_expert * err_e
                                self.k_expert_lambda = max(0.0, min(self.k_expert_lambda, self.k_lambda_max_expert))
                        current_lambda_e = self.k_expert_lambda if self.use_dynamic_k_lambda else self.k_expert_weight
                        k_expert_loss = current_lambda_e * ((ke - self.k_expert_target) / k_norm).pow(2).mean()

                    k_total_loss = k_general_loss + k_expert_loss

                # ---- C) slot anti-collapse ----
                if self.enable_slot_collapse_loss and (selected_mask is not None):
                    slot_mat = selected_mask.float()
                    if slot_mat.dim() == 1:
                        slot_mat = slot_mat.unsqueeze(0)

                    # expert-only 可选
                    if group_ids is not None and slot_mat.size(0) == group_ids.numel() and self.slot_collapse_expert_only:
                        gid_general = self.dataset_group_to_id.get("general", 0)
                        expert_mask = (group_ids != gid_general)
                        if expert_mask.any():
                            slot_mat = slot_mat[expert_mask]

                    if slot_mat.numel() > 0 and slot_mat.size(0) > 0:
                        usage = slot_mat.mean(dim=0)             # [S]
                        s = usage.sum()
                        if s > 1e-8:
                            p = usage / (s + 1e-8)
                            slot_neff = 1.0 / (p.pow(2).sum() + 1e-8)
                            slot_top1_share = p.max()

                            raw = torch.relu(slot_neff.new_tensor(float(self.slot_neff_min)) - slot_neff).pow(2)
                            if self.slot_top1_weight > 0:
                                raw = raw + float(self.slot_top1_weight) * torch.relu(
                                    slot_top1_share - slot_top1_share.new_tensor(float(self.slot_top1_max))
                                ).pow(2)

                            slot_collapse_loss = raw * float(self.slot_collapse_weight) * float(self.collapse_reg_scale)
        # 3) tax (optional, stage4再开)
        tax_raw = getattr(self.model, "tax_loss", torch.tensor(0.0, device=input_ids.device))
        tax_loss = tax_raw * self.tax_loss_weight if self.use_tax else torch.tensor(0.0, device=input_ids.device)
        outputs.loss = self.ce_loss_weight * ce_loss + alpha_guide_loss + k_total_loss + slot_collapse_loss + tax_loss

        # ---- cache metrics for external logger ----
        with torch.no_grad():
            alpha_std = torch.tensor(float("nan"), device=input_ids.device)
            label_alpha_std = torch.tensor(float("nan"), device=input_ids.device)
            cache_k_general_mean = torch.tensor(float("nan"), device=input_ids.device)
            cache_k_expert_mean = torch.tensor(float("nan"), device=input_ids.device)

            if alpha_logits is not None and torch.is_tensor(alpha_logits) and alpha_logits.numel() > 0:
                a = torch.sigmoid(alpha_logits.detach().float().view(-1))
                if a.numel() > 1:
                    alpha_std = a.std(unbiased=False)

            if expanded_labels is not None and torch.is_tensor(expanded_labels) and expanded_labels.numel() > 0:
                l = expanded_labels.detach().float().view(-1)
                if l.numel() > 1:
                    label_alpha_std = l.std(unbiased=False)

            if expanded_labels is not None and self.use_k_loss:
                k_results = self.model.visual.k_results
                k_sums = k_results.get("k_selected", None)
                if k_sums is not None and k_sums.shape[0] == expanded_labels.shape[0]:
                    lbl = expanded_labels.squeeze(-1)
                    is_general = (lbl < 0.1)
                    is_expert = (lbl > 0.9)
                    if is_general.any():
                        k_general_mean = k_sums[is_general].detach().float().mean()
                    if is_expert.any():
                        k_expert_mean = k_sums[is_expert].detach().float().mean()

            self._last_metrics = {
                "total_loss": outputs.loss.detach(),
                "ce_loss": ce_loss.detach(),
                "alpha_guide_loss": alpha_guide_loss.detach(),

                "k_dataset_loss": k_total_loss.detach(),
                "k_general_loss": k_general_loss.detach(),
                "k_expert_loss": k_expert_loss.detach(),
                "k_loss_report": k_loss_report.detach(),
                "k_loss_vqa": k_loss_vqa.detach(),
                "k_loss_test": k_loss_test.detach(),

                "k_general_mean": k_general_mean.detach(),
                "k_expert_mean": k_expert_mean.detach(),
                "k_mean_report": k_mean_report.detach(),
                "k_mean_vqa": k_mean_vqa.detach(),
                "k_mean_test": k_mean_test.detach(),

                "slot_collapse_loss": slot_collapse_loss.detach(),
                "slot_neff": slot_neff.detach(),
                "slot_top1_share": slot_top1_share.detach(),

                "tax_loss": tax_loss.detach(),
                "alpha_std": alpha_std.detach(),
                "label_alpha_std": label_alpha_std.detach(),
                "temperature": torch.tensor(
                    float(self.temperature_override) if self.temperature_override is not None else float("nan"),
                    device=device
                ),
                "tax_weight": torch.tensor(float(self.tax_loss_weight), device=device),
                "k_reg_scale": torch.tensor(float(getattr(self, "k_reg_scale", 1.0)), device=device),
                "collapse_reg_scale": torch.tensor(float(getattr(self, "collapse_reg_scale", 1.0)), device=device),
                "dynamic_k_lambda_general": torch.tensor(float(self.k_general_lambda), device=device),
                "dynamic_k_lambda_expert": torch.tensor(float(self.k_expert_lambda), device=device),
            }

        if self.debug_mode and torch.is_tensor(outputs.loss):
            loss_val = outputs.loss.detach().float().item()
            if loss_val > self.debug_loss_threshold or not torch.isfinite(outputs.loss):
                print("\n!!!! [HIGH-LOSS-DBG] abnormal loss detected !!!!")
                print(f"!!!! [HIGH-LOSS-DBG] final_loss={loss_val:.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] manual_ce={ce_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] alpha_loss={alpha_guide_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] k_general={k_general_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] k_expert={k_expert_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] use_tax={self.use_tax} tax_w={float(self.tax_loss_weight):.6f}")
                if torch.is_tensor(tax_raw):
                    print(f"!!!! [HIGH-LOSS-DBG] tax_raw={tax_raw.detach().float().item():.6f}")
                else:
                    print(f"!!!! [HIGH-LOSS-DBG] tax_raw={float(tax_raw):.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] tax_scaled={tax_loss.detach().float().item():.6f}")

                if labels is not None:
                    valid_per_sample = (labels != -100).sum(dim=1)
                    print(f"!!!! [HIGH-LOSS-DBG] valid_tokens_per_sample={valid_per_sample.tolist()}")
                    print(f"!!!! [HIGH-LOSS-DBG] total_valid_tokens={(labels != -100).sum().item()}/{labels.numel()}")

                print(f"!!!! [HIGH-LOSS-DBG] images_per_sample={images_per_sample}")
                print(f"!!!! [HIGH-LOSS-DBG] {_dbg_tensor_stats('alpha_labels', alpha_labels)}")
                print(f"!!!! [HIGH-LOSS-DBG] {_dbg_tensor_stats('alpha_logits', alpha_logits)}")
                print(f"!!!! [HIGH-LOSS-DBG] {_dbg_tensor_stats('expanded_labels', expanded_labels)}")

                if input_ids is not None:
                    print(f"!!!! [HIGH-LOSS-DBG] input_ids_shape={tuple(input_ids.shape)}")
                    print(f"!!!! [HIGH-LOSS-DBG] first_32_ids_sample0={input_ids[0, :32].detach().cpu().tolist()}")

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
    print(f"[DBG] blocks={len(model.model.visual.blocks)}, blocks_with_rep={len(model.model.visual.blocks_with_rep)}, INSERT_LAYER={list(model.model.visual.cfg.INSERT_LAYER)}")
    with torch.no_grad():
        for idx, layer_num in enumerate(model.model.visual.cfg.INSERT_LAYER):
            model.model.visual.blocks_with_rep[idx].load_state_dict(
                model.model.visual.blocks[layer_num-1].state_dict(),
                strict=False
            )
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )
    print("Processor built.")
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

    metric_logger = StageMetricLogger(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        stage_name=f"stage{stage_id}",
        smooth_window=train_cfg.get("metric_smooth_window", 30),
        ema_alpha=train_cfg.get("metric_ema_alpha", 0.15),
        scatter_stride=train_cfg.get("metric_scatter_stride", 3),
        save_debug_figure=train_cfg.get("save_debug_figure", False),
    )

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
                print_every = train_cfg.get("console_log_every", 50)
                if global_step % print_every == 0:
                    row = {
                        "total_loss": (cls_loss.detach() + gate_loss.detach()),
                        "cls_loss": cls_loss.detach(),
                        "gate_loss": gate_loss.detach(),
                        "alpha_std": alpha_std,
                        "label_alpha_std": label_std,
                        "temperature": getattr(model, "temperature_override", float("nan")),
                        "learning_rate": opt.param_groups[0]["lr"],
                    }
                    print_stage_step_summary(f"stage{stage_id}", global_step, row)
    metric_logger.finalize()

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
        model.tax_loss_weight = 0.0
        model.use_k_loss = train_cfg.get("enable_k_loss_s4", True)
        model.use_dynamic_k_lambda = True
        model.k_general_target = train_cfg.get("k_general_target_s4", 0.0)
        model.k_expert_target = train_cfg.get("k_expert_target_s4", 8.0)
        model.k_general_lambda = train_cfg.get("k_general_lambda_init_s4", 0.0)
        model.k_expert_lambda = train_cfg.get("k_expert_lambda_init_s4", 0.0)
        model.k_lambda_lr_general = train_cfg.get("k_lambda_lr_general_s4", 0.02)
        model.k_lambda_lr_expert = train_cfg.get("k_lambda_lr_expert_s4", 0.01)
        model.k_lambda_max_general = train_cfg.get("k_lambda_max_general_s4", 5.0)
        model.k_lambda_max_expert = train_cfg.get("k_lambda_max_expert_s4", 5.0)
        model.use_k_loss = train_cfg.get("enable_k_loss_s4", True)

        # 新：dataset-conditioned K
        model.use_dataset_k_range_loss = train_cfg.get("enable_dataset_k_loss_s4", True)
        model.k_group_constraints = train_cfg.get("k_group_constraints_s4", {})
        model.k_range_global_weight = train_cfg.get("k_range_global_weight_s4", 1.0)
        model.k_reg_scale = train_cfg.get("k_reg_start_scale_s4", 0.25)

        # 新：防塌缩
        model.enable_slot_collapse_loss = train_cfg.get("enable_slot_collapse_s4", True)
        model.slot_collapse_weight = train_cfg.get("slot_collapse_weight_s4", 0.02)
        model.slot_neff_min = train_cfg.get("slot_neff_min_s4", 6.0)
        model.slot_top1_max = train_cfg.get("slot_top1_max_s4", 0.35)
        model.slot_top1_weight = train_cfg.get("slot_top1_weight_s4", 0.2)
        model.collapse_reg_scale = train_cfg.get("collapse_reg_start_scale_s4", 0.1)
        model.slot_collapse_expert_only = train_cfg.get("slot_collapse_expert_only_s4", True)

        # 如果用了 dataset-conditioned K，就不再开旧动态 lambda
        if model.use_dataset_k_range_loss:
            model.use_dynamic_k_lambda = False
        else:
            model.use_dynamic_k_lambda = True
            model.k_general_target = train_cfg.get("k_general_target_s4", 0.0)
            model.k_expert_target = train_cfg.get("k_expert_target_s4", 8.0)
            model.k_general_lambda = train_cfg.get("k_general_lambda_init_s4", 0.0)
            model.k_expert_lambda = train_cfg.get("k_expert_lambda_init_s4", 0.0)
            model.k_lambda_lr_general = train_cfg.get("k_lambda_lr_general_s4", 0.02)
            model.k_lambda_lr_expert = train_cfg.get("k_lambda_lr_expert_s4", 0.01)
            model.k_lambda_max_general = train_cfg.get("k_lambda_max_general_s4", 5.0)
            model.k_lambda_max_expert = train_cfg.get("k_lambda_max_expert_s4", 5.0)

    ds = FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=data_cfg["total_limit"],
        enable_views=("expert-mm","general-mm"),
        mode=f"stage{stage_id}",
        ce_enabled=True,
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

        k_reg_start=(train_cfg.get("k_reg_start_scale_s4", 0.25) if stage_id == 4 else 0.0),
        k_reg_target=(train_cfg.get("k_reg_target_scale_s4", 1.0) if stage_id == 4 else 0.0),
        collapse_reg_start=(train_cfg.get("collapse_reg_start_scale_s4", 0.1) if stage_id == 4 else 0.0),
        collapse_reg_target=(train_cfg.get("collapse_reg_target_scale_s4", 1.0) if stage_id == 4 else 0.0),
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
        callbacks=[cb, metrics_cb]
    )
    trainer.train()
    # trainer.save_model(f"{output_dir}/stage{stage_id}")


def run_stage(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    if stage_id in [1, 2]:
        run_stage12_light(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    elif stage_id in [3, 4]:
        run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    else:
        raise ValueError("stage_id must be 1/2/3/4")