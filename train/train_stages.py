# train_stages.py
import os
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
from transformers import TrainerCallback
import json

from logger import StageMetricLogger, TrainerMetricsCallback
from train_utils import focal_bce_with_logits



def _build_experiment_context(train_cfg, output_dir, stage_id):
    experiment_name = train_cfg.get("experiment_name", "experiment")
    experiment_cfg = dict(train_cfg.get("experiment_cfg", {}))
    return {
        "experiment_name": experiment_name,
        "stage_name": f"stage{stage_id}",
        "stage_id": stage_id,
        "output_dir": output_dir,
        "stage_output_dir": f"{output_dir}/stage{stage_id}",
        "metrics_dir": f"{output_dir}/stage{stage_id}/metrics",
        "train_cfg": {
            "seed": train_cfg.get("seed"),
            "deterministic_sampling": train_cfg.get("deterministic_sampling", False),
            "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps"),
            "learning_rate": train_cfg.get("learning_rate", {}).get(stage_id),
            "epochs": train_cfg.get("epochs", {}).get(stage_id),
            "console_log_every": train_cfg.get("console_log_every"),
            "metric_smooth_window": train_cfg.get("metric_smooth_window"),
            "metric_ema_alpha": train_cfg.get("metric_ema_alpha"),
            "metric_scatter_stride": train_cfg.get("metric_scatter_stride"),
            "enable_gate_loss_stage2": train_cfg.get("enable_gate_loss_stage2"),
            "enable_alpha_guide_loss_s3": train_cfg.get("enable_alpha_guide_loss_s3"),
            "enable_alpha_guide_loss_s4": train_cfg.get("enable_alpha_guide_loss_s4"),
            "enable_tax_loss_s4": train_cfg.get("enable_tax_loss_s4"),
            "enable_k_loss_s4": train_cfg.get("enable_k_loss_s4"),
            "enable_group_anti_collapse_s4": train_cfg.get("enable_group_anti_collapse_s4"),
            "group_anti_collapse_weight_s4": train_cfg.get("group_anti_collapse_weight_s4"),
            "group_anti_collapse_warmup_epochs_s4": train_cfg.get("group_anti_collapse_warmup_epochs_s4"),
            "group_usage_ema_momentum": train_cfg.get("group_usage_ema_momentum"),
            "tax_loss_weight": train_cfg.get("tax_loss_weight"),
            "k_general_target_s4": train_cfg.get("k_general_target_s4"),
            "k_expert_target_s4": train_cfg.get("k_expert_target_s4"),
        },
        "experiment_cfg": experiment_cfg,
    }


def _write_experiment_manifest(output_dir, train_cfg):
    experiment_name = train_cfg.get("experiment_name", "experiment")
    manifest = {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "experiment_cfg": dict(train_cfg.get("experiment_cfg", {})),
        "train_cfg": {
            k: v for k, v in train_cfg.items()
            if k not in {"experiment_cfg"}
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{experiment_name}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[Experiment] manifest saved to {manifest_path}")

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

    if "alpha_mae" in row:
        print(f"[Alpha Statistics]")
        print(f"  └─ Alpha MAE:             {_fmt(row.get('alpha_mae')):>10}")

    if "k_selected_mean" in row:
        print("[Selector Statistics]")
        print(f"  ├─ Mean Selected K:       {_fmt(row.get('k_selected_mean'), 3):>10}")
        if "span_gates_mean" in row:
            print(f"  ├─ Spans Selected (avg):  {_fmt(row.get('span_gates_mean'), 3):>10}")
        if "span_budget_loss" in row:
            print(f"  ├─ Span Budget Loss:      {_fmt(row.get('span_budget_loss'), 4):>10}")
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
        target_tax=4.0,
        tax_warmup_epochs=1,
        target_group_anti_collapse_weight=0.0,
        group_anti_collapse_warmup_epochs=0,
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.target_tax = target_tax
        self.tax_warmup_epochs = tax_warmup_epochs
        self.target_group_anti_collapse_weight = float(target_group_anti_collapse_weight)
        self.group_anti_collapse_warmup_epochs = float(group_anti_collapse_warmup_epochs)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.resample_general_data()

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep = state.epoch if state.epoch is not None else 0.0

        progress = min(ep / max(self.total_epochs, 1e-6), 1.0)
        model.temperature_override = self.init_temp - (self.init_temp - self.final_temp) * progress

        if progress < self.selector_two_phase_split:
            explore_scale = self.selector_explore_init
        else:
            denom = max(1.0 - self.selector_two_phase_split, 1e-6)
            phase_progress = min(max((progress - self.selector_two_phase_split) / denom, 0.0), 1.0)
            explore_scale = self.selector_explore_init - (
                self.selector_explore_init - self.selector_explore_final
            ) * phase_progress
        model.selector_exploration_scale = float(explore_scale)
        model.model.visual.text_gating.start_exploration_scale = float(explore_scale)

        if ep < self.budget_warmup_epochs:
            model.budget_loss_weight = 0.0
        else:
            budget_progress = min(
                (ep - self.budget_warmup_epochs) / max(self.total_epochs - self.budget_warmup_epochs, 1e-6),
                1.0,
            )
            model.budget_loss_weight = self.target_budget * budget_progress


        text_gating = model._get_text_gating_module() if hasattr(model, "_get_text_gating_module") else None
        if text_gating is not None:
            if ep < self.group_anti_collapse_warmup_epochs:
                anti_collapse_p = 0.0
            else:
                anti_collapse_p = min(
                    (ep - self.group_anti_collapse_warmup_epochs) /
                    max(self.total_epochs - self.group_anti_collapse_warmup_epochs, 1e-6),
                    1.0,
                )
            text_gating.group_anti_collapse_weight = self.target_group_anti_collapse_weight * anti_collapse_p

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

        from transformers import GenerationConfig

        self.generation_config = GenerationConfig.from_model_config(config)

        self.ce_loss_weight = 1.0
        self.alpha_loss_weight = 0.5   # Stage3建议0.3~1.0，Stage4可0.5
        self.tax_loss_weight = 0.0     # Stage3=0, Stage4由调度升到目标
        self.enable_alpha_guide_loss = True
        self.enable_gate_loss_stage2 = True
        
        self.k_general_weight = 0.0
        self.k_expert_weight = 0.0
        # ===== dynamic lambda for K control =====
        self.use_dynamic_k_lambda = False
        self.k_general_target = 0.0
        self.k_expert_target = 0.0
        self.k_general_weight = 0.0
        self.k_expert_weight = 0.0
        self.k_general_lambda = 0.0
        self.k_expert_lambda = 0.0
        self.k_lambda_lr_general = 0.0
        self.k_lambda_lr_expert = 0.0
        self.k_lambda_max_general = 0.0
        self.k_lambda_max_expert = 0.0
        self.enable_dataset_slot_loss_s4 = False
        self.slot_group_constraints_s4 = {}

        self.debug_mode = True
        self.debug_loss_threshold = 20.0

    def _compute_group_anti_collapse_loss(self):
        text_gating = self._get_text_gating_module()
        zero = torch.tensor(0.0, device=self.lm_head.weight.device)
        if text_gating is None:
            return zero, {}
        if getattr(text_gating, "selection_mode", None) != "group_top4":
            return zero, {}
        if hasattr(text_gating, "compute_group_anti_collapse_loss"):
            return text_gating.compute_group_anti_collapse_loss()
        return zero, {}

    def _get_text_gating_module(self):
        visual = getattr(self.model, "visual", None)
        return getattr(visual, "text_gating", None)

    def _get_k_norm(self):
        text_gating = self._get_text_gating_module()
        return float(getattr(text_gating, "total_rep_num", 40.0))

    def _supports_text_gating_losses(self):
        text_gating = self._get_text_gating_module()
        if text_gating is None:
            return False
        return bool(getattr(text_gating, "enable_text_gating", True))

    def _expand_alpha_labels(self, alpha_labels, images_per_sample):
        if alpha_labels is None:
            return None
        if images_per_sample is None:
            return alpha_labels.view(-1, 1)

        tmp = []
        for i, c in enumerate(images_per_sample):
            c = int(c)
            if c > 0:
                tmp.append(alpha_labels[i].repeat(c))
        if len(tmp) == 0:
            return None
        return torch.cat(tmp).view(-1, 1)

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, task_type_ids=None, **kwargs):
        if hasattr(self, "temperature_override") and self.temperature_override is not None:
            self.model.temperature_override = self.temperature_override
            kwargs["gating_temperature_override"] = self.temperature_override
        self.model.visual.text_gating.start_exploration_scale = float(getattr(self, "selector_exploration_scale", 0.0))

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
        alpha_guide_loss = torch.tensor(0.0, device=device)
        budget_loss = self._compute_budget_loss(device)
        slot_dataset_loss, slot_dataset_metrics = self._compute_dataset_slot_loss(
            kwargs.get("dataset_group_ids", None), device
        )

        if isinstance(alpha_logits, list):
            if len(alpha_logits) > 0:
                alpha_logits = torch.stack(alpha_logits)
            else:
                alpha_logits = None

        expanded_labels = self._expand_alpha_labels(alpha_labels, images_per_sample)

        if self.enable_alpha_guide_loss and alpha_logits is not None and alpha_labels is not None:
            if expanded_labels is not None and expanded_labels.numel() > 0:
                if alpha_logits.dim() == 1:
                    alpha_logits = alpha_logits.unsqueeze(-1)

                n = min(alpha_logits.shape[0], expanded_labels.shape[0])
                if n > 0:
                    alpha_guide_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        alpha_logits[:n], expanded_labels[:n].to(alpha_logits.dtype)
                    ) * self.alpha_loss_weight

        # 2) K loss (optional, stage4再开)
        k_general_loss = torch.tensor(0.0, device=input_ids.device)
        k_expert_loss = torch.tensor(0.0, device=input_ids.device)
        k_general_mean = torch.tensor(float("nan"), device=input_ids.device)
        k_expert_mean = torch.tensor(float("nan"), device=input_ids.device)

        if self.use_k_loss and expanded_labels is not None:
            k_results = self.model.visual.k_results
            if isinstance(k_results, dict):
                k_sums = k_results.get("k_selected", None)
            elif isinstance(k_results, tuple):
                k_sums = k_results[0]
            else:
                k_sums = k_results

            if torch.is_tensor(k_sums) and k_sums.shape[0] == expanded_labels.shape[0]:
                lbl = expanded_labels.squeeze(-1)
                is_general = (lbl < 0.1)
                is_expert = (lbl > 0.9)
                k_norm = self._get_k_norm()

                # ===== general =====
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

                # ===== expert =====
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
        # 3) tax (optional, stage4再开)
        tax_raw = getattr(self.model, "tax_loss", torch.tensor(0.0, device=input_ids.device))
        if not torch.is_tensor(tax_raw):
            tax_raw = torch.tensor(float(tax_raw), device=input_ids.device, dtype=ce_loss.dtype)
        tax_loss = tax_raw * self.tax_loss_weight if self.use_tax else torch.tensor(0.0, device=input_ids.device)
        anti_collapse_loss, anti_collapse_debug = self._compute_group_anti_collapse_loss()
        outputs.loss = self.ce_loss_weight * ce_loss + alpha_guide_loss + k_general_loss + k_expert_loss + tax_loss + anti_collapse_loss

        # ---- cache metrics for external logger ----
        with torch.no_grad():
            alpha_mae = torch.tensor(float("nan"), device=input_ids.device)

            if alpha_logits is not None and torch.is_tensor(alpha_logits) and alpha_logits.numel() > 0:
                a = torch.sigmoid(alpha_logits.detach().float().view(-1))
                if expanded_labels is not None and torch.is_tensor(expanded_labels) and expanded_labels.numel() > 0:
                    l = expanded_labels.detach().float().view(-1)[:a.numel()].to(a.device)
                    n = min(a.numel(), l.numel())
                    if n > 0:
                        alpha_mae = (a[:n] - l[:n]).abs().mean()

            k_selected_mean, span_gates_mean, span_budget_loss_val, alpha_open_rate = self._collect_selector_metrics(device)

            self._last_metrics = {
                "total_loss": outputs.loss.detach(),
                "ce_loss": ce_loss.detach(),
                "alpha_guide_loss": alpha_guide_loss.detach(),
                "k_general_loss": k_general_loss.detach(),
                "k_expert_loss": k_expert_loss.detach(),
                "tax_loss": tax_loss.detach(),
                "anti_collapse_loss": anti_collapse_loss.detach(),
                "alpha_mae": alpha_mae.detach(),
                "k_general_mean": k_general_mean.detach(),
                "k_expert_mean": k_expert_mean.detach(),
                "alpha_open_rate": alpha_open_rate.detach(),
                "dynamic_k_lambda_general": torch.tensor(float(self.k_general_lambda), device=device),
                "dynamic_k_lambda_expert": torch.tensor(float(self.k_expert_lambda), device=device),
                "temperature": torch.tensor(
                    float(self.temperature_override) if self.temperature_override is not None else float("nan"),
                    device=device,
                ),
                "budget_weight": torch.tensor(float(self.budget_loss_weight), device=device),
                "tax_weight": torch.tensor(float(self.tax_loss_weight), device=device),
                "selector_explore": torch.tensor(float(self.selector_exploration_scale), device=device),
            }
            text_gating = self._get_text_gating_module()
            if text_gating is not None:
                self._last_metrics["anti_collapse_weight"] = torch.tensor(
                    float(getattr(text_gating, "group_anti_collapse_weight", 0.0)),
                    device=input_ids.device,
                )
                dbg = getattr(text_gating, "debug_context", {}) or {}
                selection_mode = getattr(text_gating, "selection_mode", None)
                if selection_mode is not None:
                    selection_mode_map = {
                        "dynamic_prefix": 0.0,
                        "fixed_prefix": 1.0,
                        "fixed_group20": 2.0,
                        "group_top4": 3.0,
                        "token_top20": 4.0,
                    }
                    self._last_metrics["selection_mode_id"] = torch.tensor(
                        float(selection_mode_map.get(str(selection_mode), -1.0)),
                        device=input_ids.device,
                    )

                for key in (
                    "group_count_loss",
                    "token_count_loss",
                    "token_balance_loss",
                    "raw_budget_mean",
                    "k_budget_mean",
                    "active_token_count_mean",
                    "batch_alpha_mean",
                    "text_rel_mean",
                ):
                    if key in dbg:
                        self._last_metrics[key] = dbg[key]

                hard_group_mask = getattr(text_gating, "last_hard_group_mask", None)
                if torch.is_tensor(hard_group_mask) and hard_group_mask.numel() > 0:
                    usage = hard_group_mask.detach().float().mean(dim=0)
                    usage = usage / usage.sum().clamp_min(1e-8)
                    for gi in range(int(usage.shape[0])):
                        self._last_metrics[f"group_usage_{gi}"] = usage[gi]
            self._last_metrics.update(anti_collapse_debug)

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


def build_model_and_processor(model_path, experiment_cfg=None):
    experiment_cfg = experiment_cfg or {}
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    active_rep_token_count = int(experiment_cfg.get("active_rep_token_count", getattr(cfg, "ACTIVE_REP_TOKEN_COUNT", 40)))
    cfg.ACTIVE_REP_TOKEN_COUNT = active_rep_token_count
    cfg.SPECIAL_TOKENS = cfg.build_special_tokens(active_rep_token_count)

    config.ACTIVE_REP_TOKEN_COUNT = active_rep_token_count
    config.ENABLE_TEXT_GATING = bool(experiment_cfg.get("enable_text_gating", os.getenv("MMRL_ENABLE_TEXT_GATING", "1") == "1"))
    config.FIXED_K_WHEN_TEXT_GATING_DISABLED = float(experiment_cfg.get(
        "fixed_k_when_text_gating_disabled",
        os.getenv("MMRL_FIXED_K_WHEN_TEXT_GATING_DISABLED", "40")
    ))
    config.ENABLE_TEXT_GATE_TAX_LOSS = bool(experiment_cfg.get(
        "enable_text_gate_tax_loss",
        os.getenv("MMRL_ENABLE_TEXT_GATE_TAX_LOSS", "1") == "1"
    ))
    config.ENABLE_TEXT_GATE_COLLAPSE_LOSS = bool(experiment_cfg.get(
        "enable_text_gate_collapse_loss",
        os.getenv("MMRL_ENABLE_TEXT_GATE_COLLAPSE_LOSS", "1") == "1"
    ))
    config.TEXT_GATE_SELECTION_MODE = str(experiment_cfg.get(
        "text_gate_selection_mode",
        os.getenv("MMRL_TEXT_GATE_SELECTION_MODE", "dynamic_prefix")
    ))
    config.TEXT_GATE_GROUP_SIZE = int(experiment_cfg.get(
        "text_gate_group_size",
        os.getenv("MMRL_TEXT_GATE_GROUP_SIZE", str(cfg.RP_SPACE_LENGTH))
    ))
    config.TEXT_GATE_NUM_GROUPS = int(experiment_cfg.get(
        "text_gate_num_groups",
        os.getenv("MMRL_TEXT_GATE_NUM_GROUPS", "8")
    ))
    config.TEXT_GATE_SELECTED_GROUP_COUNT = int(experiment_cfg.get(
        "text_gate_selected_group_count",
        os.getenv("MMRL_TEXT_GATE_SELECTED_GROUP_COUNT", "4")
    ))
    config.TEXT_GATE_SELECTED_GROUPS = experiment_cfg.get(
        "text_gate_selected_groups",
        os.getenv("MMRL_TEXT_GATE_SELECTED_GROUPS", "0,1,2,3")
    )
    config.TEXT_GATE_SHARED_GROUPS = experiment_cfg.get(
        "text_gate_shared_groups",
        [0, 1]
    )
    config.ENABLE_TEXT_GATE_GROUP_ANTI_COLLAPSE = bool(experiment_cfg.get(
        "enable_text_gate_group_anti_collapse",
        False
    ))
    config.TEXT_GATE_GROUP_ANTI_COLLAPSE_WEIGHT = float(experiment_cfg.get(
        "text_gate_group_anti_collapse_weight",
        0.0
    ))
    config.TEXT_GATE_GROUP_USAGE_EMA_MOMENTUM = float(experiment_cfg.get(
        "text_gate_group_usage_ema_momentum",
        0.98
    ))
    config.TEXT_GATE_GROUP_USAGE_EMA_EPS = float(experiment_cfg.get(
        "text_gate_group_usage_ema_eps",
        1e-6
    ))
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
    print(
        f"[Experiment] mode={config.TEXT_GATE_SELECTION_MODE} active_rep_token_count={config.ACTIVE_REP_TOKEN_COUNT} "
        f"group_size={config.TEXT_GATE_GROUP_SIZE} num_groups={config.TEXT_GATE_NUM_GROUPS} "
        f"selected_group_count={config.TEXT_GATE_SELECTED_GROUP_COUNT} selected_groups={config.TEXT_GATE_SELECTED_GROUPS}"
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

    experiment_context = _build_experiment_context(train_cfg, output_dir, stage_id)
    metric_logger = StageMetricLogger(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        stage_name=f"stage{stage_id}",
        experiment_name=experiment_context["experiment_name"],
        experiment_config=experiment_context,
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
        deterministic_sampling=train_cfg.get("deterministic_sampling", False),
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
            task_type_ids = batch["task_type_ids"].cuda(non_blocking=True)
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
            if stage_id == 2 and getattr(model, "enable_gate_loss_stage2", True):
                g = v.visionGating(alpha_logits, getattr(model, "temperature_override", None))
                target = alpha_labels
                gate_loss = torch.nn.functional.mse_loss(g, target)

            ((cls_loss + gate_loss) / steps_per_update).backward()

            if (i + 1) % steps_per_update == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                with torch.no_grad():
                    alpha_prob = torch.sigmoid(alpha_logits.detach().float().view(-1))
                    alpha_mae = (alpha_prob - alpha_labels.detach().float().view(-1)).abs().mean() if alpha_prob.numel() > 0 else torch.tensor(float("nan"), device=alpha_prob.device)

                    metric_logger.log(
                        step=global_step,
                        total_loss=(cls_loss.detach() + gate_loss.detach()),
                        cls_loss=cls_loss.detach(),
                        gate_loss=gate_loss.detach(),
                        alpha_mae=alpha_mae,
                        temperature=getattr(model, "temperature_override", float("nan")),
                        learning_rate=opt.param_groups[0]["lr"],
                    )
                print_every = train_cfg.get("console_log_every", 50)
                if global_step % print_every == 0:
                    row = {
                        "total_loss": (cls_loss.detach() + gate_loss.detach()),
                        "cls_loss": cls_loss.detach(),
                        "gate_loss": gate_loss.detach(),
                        "alpha_mae": alpha_mae,
                        "temperature": getattr(model, "temperature_override", float("nan")),
                        "learning_rate": opt.param_groups[0]["lr"],
                    }
                    print_stage_step_summary(f"stage{stage_id}", global_step, row)
    metric_logger.finalize()


def run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    assert stage_id in [3, 4]
    set_trainable_stage(model, stage_id)

    if stage_id == 3:
        model.ce_loss_weight = 1.0
        model.alpha_loss_weight = train_cfg.get("alpha_loss_weight_s3", 0.5)
        model.enable_alpha_guide_loss = train_cfg.get("enable_alpha_guide_loss_s3", True)
        model.use_tax = False
        model.tax_loss_weight = 0.0
        model.use_k_loss = False
        model.k_general_weight = 0.0
        model.k_expert_weight = 0.0
    else:  # stage4
        model.ce_loss_weight = 1.0
        model.alpha_loss_weight = train_cfg.get("alpha_loss_weight_s4", 0.5)
        model.enable_alpha_guide_loss = train_cfg.get("enable_alpha_guide_loss_s4", True)
        model.use_tax = train_cfg.get("enable_tax_loss_s4", True)
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

    model.enable_gate_loss_stage2 = train_cfg.get("enable_gate_loss_stage2", True)

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
        deterministic_sampling=train_cfg.get("deterministic_sampling", False),
    )
    collator = MMRLDataCollator(processor)
    schedule_cb = StageScheduleCallback(
        dataset=ds,
        total_epochs=train_cfg["epochs"][stage_id],
        init_temp=train_cfg.get("initial_temp", 1.0),
        final_temp=train_cfg.get("final_temp", 0.1),
        target_tax=(0.0 if stage_id == 3 else train_cfg["tax_loss_weight"]),
        tax_warmup_epochs=train_cfg.get("tax_warmup_epochs", 1),
        target_group_anti_collapse_weight=(
            0.0 if stage_id == 3 or not train_cfg.get("enable_group_anti_collapse_s4", True)
            else train_cfg.get("group_anti_collapse_weight_s4", 0.0)
        ),
        group_anti_collapse_warmup_epochs=train_cfg.get("group_anti_collapse_warmup_epochs_s4", 0),
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
    experiment_context = _build_experiment_context(train_cfg, output_dir, stage_id)
    metric_logger = StageMetricLogger(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        stage_name=f"stage{stage_id}",
        experiment_name=experiment_context["experiment_name"],
        experiment_config=experiment_context,
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
    if stage_id == 1:
        _write_experiment_manifest(output_dir, train_cfg)
    if stage_id in [1, 2]:
        run_stage12_light(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    elif stage_id in [3, 4]:
        run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    else:
        raise ValueError("stage_id must be 1/2/3/4")