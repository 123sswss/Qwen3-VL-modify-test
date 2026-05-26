# train_stages.py
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Qwen3VLForConditionalGeneration, AutoConfig, AutoTokenizer, AutoImageProcessor,
    Trainer, TrainingArguments, get_cosine_schedule_with_warmup
)
import numpy as np

import sys
sys.path.append("..")
import config as cfg
import processingWithMMRL
from data_pipeline import FourViewMMRLDataset, MMRLDataCollator
from train_utils import focal_bce_with_logits
import QWen3WithMMRL
from transformers import TrainerCallback
import json

from logger import StageMetricLogger, TrainerMetricsCallback


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
            "enable_alpha_guide_loss_s4": train_cfg.get("enable_alpha_guide_loss_s4"),
            "alpha_loss_weight_s4": train_cfg.get("alpha_loss_weight_s4"),
            "enable_capacity_prior_loss_s4": train_cfg.get("enable_capacity_prior_loss_s4"),
            "capacity_prior_loss_weight": train_cfg.get("capacity_prior_loss_weight"),
            "visual_residual_adapter_count": train_cfg.get("visual_residual_adapter_count"),
            "adapter_usage_balance_loss_weight": train_cfg.get("adapter_usage_balance_loss_weight"),
            "adapter_sample_entropy_loss_weight": train_cfg.get("adapter_sample_entropy_loss_weight"),
            "adapter_common_mode_loss_weight": train_cfg.get("adapter_common_mode_loss_weight"),
            "adapter_sample_entropy_target": train_cfg.get("adapter_sample_entropy_target"),
            "adapter_common_mode_target": train_cfg.get("adapter_common_mode_target"),
            "diag_every_steps": train_cfg.get("diag_every_steps"),
            "enable_expert_floor_loss_s4": train_cfg.get("enable_expert_floor_loss_s4"),
            "expert_floor_loss_weight": train_cfg.get("expert_floor_loss_weight"),
            "expert_min_active_tokens": train_cfg.get("expert_min_active_tokens"),
            "enable_collapse_loss_s4": train_cfg.get("enable_collapse_loss_s4"),
            "anti_collapse_weight": train_cfg.get("anti_collapse_weight"),
            "collapse_max_ratio": train_cfg.get("collapse_max_ratio"),
            "collapse_min_ratio": train_cfg.get("collapse_min_ratio"),
            "collapse_entropy_target": train_cfg.get("collapse_entropy_target"),
            "group_usage_dead_threshold": train_cfg.get("group_usage_dead_threshold"),
            "group_usage_ema_alpha": train_cfg.get("group_usage_ema_alpha"),
            "grad_spike_ema_alpha": train_cfg.get("grad_spike_ema_alpha"),
            "grad_spike_ratio_threshold": train_cfg.get("grad_spike_ratio_threshold"),
            "grad_spike_abs_threshold": train_cfg.get("grad_spike_abs_threshold"),
            "grad_spike_cooldown_steps": train_cfg.get("grad_spike_cooldown_steps"),
            "disable_general_mm_stage34": train_cfg.get("disable_general_mm_stage34"),
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
    if "capacity_prior_loss" in row:
        print(f"  ├─ Capacity Prior Loss:   {_fmt(row.get('capacity_prior_loss')):>10}")
    if "raw_capacity_prior_loss" in row:
        print(f"  ├─ Raw Capacity Prior:    {_fmt(row.get('raw_capacity_prior_loss')):>10}")
    if "alpha_mae" in row:
        print(f"[Alpha Statistics]")
        print(f"  └─ Alpha MAE:             {_fmt(row.get('alpha_mae')):>10}")

    if "k_general_mean" in row or "k_expert_mean" in row or "active_token_count_mean" in row:
        print(f"[Routing Statistics]")
        if "k_expert_mean" in row:
            print(f"  ├─ Expert Mean K:         {_fmt(row.get('k_expert_mean'), 3):>10}")
        if "active_token_count_mean" in row:
            print(f"  ├─ Active Tokens:         {_fmt(row.get('active_token_count_mean'), 3):>10}")
        if "k_budget_mean" in row:
            print(f"  ├─ K Budget:              {_fmt(row.get('k_budget_mean'), 3):>10}")
        if "raw_budget_mean" in row:
            print(f"  ├─ Raw Budget:            {_fmt(row.get('raw_budget_mean'), 3):>10}")
        if "k_general_mean" in row:
            print(f"  ├─ General Mean K:        {_fmt(row.get('k_general_mean'), 3):>10}")
        if "batch_alpha_mean" in row:
            print(f"  └─ Batch Alpha:           {_fmt(row.get('batch_alpha_mean'), 4):>10}")

    print(f"[Schedule]")
    if "temperature" in row:
        print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), 4):>10}")
    if "capacity_prior_weight" in row:
        print(f"  ├─ Capacity Prior Weight: {_fmt(row.get('capacity_prior_weight'), 4):>10}")
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
        target_capacity_prior_weight=0.03,
        capacity_prior_warmup_epochs=1,
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.target_capacity_prior_weight = target_capacity_prior_weight
        self.capacity_prior_warmup_epochs = capacity_prior_warmup_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.resample_data()

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep = state.epoch if state.epoch is not None else 0.0

        # temp anneal
        prog = min(ep / self.total_epochs, 1.0)
        model.temperature_override = self.init_temp - (self.init_temp - self.final_temp) * prog

        # capacity-prior warmup
        if ep < self.capacity_prior_warmup_epochs:
            model.capacity_prior_loss_weight = 0.0
        else:
            p = min(
                (ep - self.capacity_prior_warmup_epochs)
                / max(self.total_epochs - self.capacity_prior_warmup_epochs, 1e-6),
                1.0,
            )
            model.capacity_prior_loss_weight = self.target_capacity_prior_weight * p


class MMRLDiagnosticsCallback(TrainerCallback):
    def __init__(self, save_dir, every_steps=1000, stage_name="stage"):
        self.save_dir = save_dir
        self.every_steps = int(max(1, every_steps))
        self.stage_name = stage_name
        self.buffer = []
        self.jsonl_path = os.path.join(save_dir, "mmrl_diagnostics.jsonl")
        self.csv_path = os.path.join(save_dir, "mmrl_diagnostics.csv")
        self.csv_header_written = False
        os.makedirs(save_dir, exist_ok=True)

    def _to_python(self, value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            if value.numel() == 1:
                return float(value.item())
            return value.tolist()
        if isinstance(value, np.generic):
            return float(value)
        return value

    def _record_to_python(self, record):
        return {k: self._to_python(v) for k, v in record.items()}

    def _aggregate(self, rows):
        start_step = rows[0]["step"]
        end_step = rows[-1]["step"]
        scalar_stats = {}
        array_payload = {}
        for key in rows[0].keys():
            if key == "step":
                continue
            vals = [r[key] for r in rows if key in r]
            if not vals:
                continue
            first = vals[0]
            if isinstance(first, list):
                arr = np.array(vals, dtype=float)
                array_payload[key] = arr.mean(axis=0).tolist()
            elif isinstance(first, (int, float)):
                arr = np.array(vals, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                scalar_stats[f"{key}_mean"] = float(finite.mean())
                scalar_stats[f"{key}_std"] = float(finite.std())
                latest_finite = arr[np.isfinite(arr)][-1]
                scalar_stats[f"{key}_latest"] = float(latest_finite)
        return {
            "stage_name": self.stage_name,
            "step_start": int(start_step),
            "step_end": int(end_step),
            "window_size": len(rows),
            **scalar_stats,
            **array_payload,
        }

    def _append_csv(self, payload):
        flat_payload = {k: v for k, v in payload.items() if not isinstance(v, list)}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_payload.keys()))
            if not self.csv_header_written:
                writer.writeheader()
                self.csv_header_written = True
            writer.writerow(flat_payload)

    def _flush(self):
        if not self.buffer:
            return
        payload = self._aggregate(self.buffer)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._append_csv(payload)
        self.buffer.clear()

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0:
            return
        model = kwargs.get("model", None)
        if model is None:
            return
        metrics = getattr(model, "_last_metrics", None)
        if not isinstance(metrics, dict):
            return
        row = self._record_to_python({"step": step, **metrics})
        self.buffer.append(row)
        if len(self.buffer) >= self.every_steps:
            self._flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._flush()


class SharedRepGradMonitorCallback(TrainerCallback):
    def __init__(
        self,
        save_dir,
        stage_name="stage",
        ema_alpha=0.10,
        ratio_threshold=3.0,
        abs_threshold=0.5,
        cooldown_steps=20,
    ):
        self.save_dir = save_dir
        self.stage_name = stage_name
        self.ema_alpha = float(ema_alpha)
        self.ratio_threshold = float(ratio_threshold)
        self.abs_threshold = float(abs_threshold)
        self.cooldown_steps = int(max(0, cooldown_steps))
        self.grad_ema = None
        self.spike_count = 0
        self.last_spike_step = -10**9
        self.jsonl_path = os.path.join(save_dir, "shared_rep_grad_events.jsonl")
        self.csv_path = os.path.join(save_dir, "shared_rep_grad_events.csv")
        self.csv_header_written = False
        os.makedirs(save_dir, exist_ok=True)

    def _to_python(self, value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            if value.numel() == 1:
                return float(value.item())
            return value.tolist()
        if isinstance(value, np.generic):
            return float(value)
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        try:
            return float(value)
        except Exception:
            return str(value)

    def _append_csv(self, payload):
        flat_payload = {k: v for k, v in payload.items() if not isinstance(v, list)}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_payload.keys()))
            if not self.csv_header_written:
                writer.writeheader()
                self.csv_header_written = True
            writer.writerow(flat_payload)

    def _write_event(self, payload):
        payload = {k: self._to_python(v) for k, v in payload.items()}
        payload = {
            k: v for k, v in payload.items()
            if not (isinstance(v, float) and not np.isfinite(v))
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._append_csv(payload)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0:
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        grad_payload = getattr(model, "_last_shared_rep_grad", {}) or {}
        grad_norm = float(grad_payload.get("shared_rep_grad_norm", float("nan")))
        grad_mean_abs = float(grad_payload.get("shared_rep_grad_mean_abs", float("nan")))
        grad_max_abs = float(grad_payload.get("shared_rep_grad_max_abs", float("nan")))

        prev_ema = self.grad_ema
        if not np.isfinite(grad_norm):
            grad_ratio = float("nan")
            grad_spike = 0.0
            grad_ema = prev_ema if prev_ema is not None else float("nan")
        else:
            if prev_ema is None or (not np.isfinite(prev_ema)) or prev_ema <= 1e-8:
                prev_ema = grad_norm
            grad_ratio = grad_norm / max(prev_ema, 1e-8)
            grad_spike = float(
                grad_norm >= self.abs_threshold
                and grad_ratio >= self.ratio_threshold
                and (step - self.last_spike_step) > self.cooldown_steps
            )
            grad_ema = (1.0 - self.ema_alpha) * prev_ema + self.ema_alpha * grad_norm
            self.grad_ema = grad_ema

        metrics = getattr(model, "_last_metrics", None)
        if isinstance(metrics, dict):
            device = None
            try:
                device = next(model.parameters()).device
            except Exception:
                device = torch.device("cpu")

            metrics.update({
                "shared_rep_grad_norm": torch.tensor(grad_norm, device=device),
                "shared_rep_grad_mean_abs": torch.tensor(grad_mean_abs, device=device),
                "shared_rep_grad_max_abs": torch.tensor(grad_max_abs, device=device),
                "shared_rep_grad_ema": torch.tensor(float(grad_ema), device=device),
                "shared_rep_grad_ratio": torch.tensor(float(grad_ratio), device=device),
                "shared_rep_grad_spike_flag": torch.tensor(float(grad_spike), device=device),
                "shared_rep_grad_spike_count": torch.tensor(float(self.spike_count + int(grad_spike > 0.0)), device=device),
                "shared_rep_grad_steps_since_spike": torch.tensor(
                    float(step - self.last_spike_step) if self.last_spike_step > 0 else -1.0,
                    device=device,
                ),
            })

            if grad_spike > 0.0:
                self.spike_count += 1
                self.last_spike_step = step
                metrics["shared_rep_grad_spike_count"] = torch.tensor(float(self.spike_count), device=device)
                metrics["shared_rep_grad_steps_since_spike"] = torch.tensor(0.0, device=device)
                event_payload = {
                    "stage_name": self.stage_name,
                    "step": step,
                    "shared_rep_grad_norm": grad_norm,
                    "shared_rep_grad_mean_abs": grad_mean_abs,
                    "shared_rep_grad_max_abs": grad_max_abs,
                    "shared_rep_grad_ema": grad_ema,
                    "shared_rep_grad_ratio": grad_ratio,
                    "total_loss": metrics.get("total_loss"),
                    "ce_loss": metrics.get("ce_loss"),
                    "alpha_guide_loss": metrics.get("alpha_guide_loss"),
                    "expert_floor_loss": metrics.get("expert_floor_loss"),
                    "anti_collapse_loss": metrics.get("anti_collapse_loss"),
                    "capacity_prior_loss": metrics.get("capacity_prior_loss"),
                    "raw_capacity_prior_loss": metrics.get("raw_capacity_prior_loss"),
                    "k_expert_mean": metrics.get("k_expert_mean"),
                    "k_general_mean": metrics.get("k_general_mean"),
                    "expert_sample_count": metrics.get("expert_sample_count"),
                    "general_sample_count": metrics.get("general_sample_count"),
                    "active_token_count_mean": metrics.get("active_token_count_mean"),
                    "hard_group_count_pre_G_mean": metrics.get("hard_group_count_pre_G_mean"),
                    "hard_group_count_post_G_mean": metrics.get("hard_group_count_post_G_mean"),
                    "group_usage_max": metrics.get("group_usage_max"),
                    "group_usage_min": metrics.get("group_usage_min"),
                    "group_usage_entropy": metrics.get("group_usage_entropy"),
                    "batch_alpha_mean": metrics.get("batch_alpha_mean"),
                    "delta_to_org_ratio": metrics.get("delta_to_org_ratio"),
                }
                self._write_event(event_payload)

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
        self.alpha_loss_weight = 0.0
        self.capacity_prior_loss_weight = 0.0     # Stage3=0, Stage4由调度升到目标
        self.enable_alpha_guide_loss = False
        self.enable_gate_loss_stage2 = True
        self.adapter_usage_balance_loss_weight = 0.0
        self.adapter_sample_entropy_loss_weight = 0.0
        self.adapter_common_mode_loss_weight = 0.0
        self.enable_expert_floor_loss = False
        self.expert_floor_loss_weight = 0.0
        self.expert_min_active_tokens = 4.0
        self.enable_collapse_loss = False
        self.anti_collapse_weight = 0.0
        self.collapse_max_ratio = 0.45
        self.collapse_min_ratio = 0.05
        self.collapse_entropy_target = 0.75
        self.group_usage_dead_threshold = 0.03
        self.group_usage_ema_alpha = 0.10
        self.group_usage_ema = None
        
        self.use_capacity_prior_loss = False
        self.temperature_override = None

        self.debug_mode = True
        self.debug_loss_threshold = 20.0
        self._last_metrics = {}
        self._last_shared_rep_grad = {}
        self._shared_rep_grad_hook_handle = None
        self._shared_rep_grad_hook_param_id = None
        self._ensure_shared_rep_grad_hook()

    def _get_mmrl_module(self):
        return getattr(self.model, "MMRL", None)

    def _collect_rep_grad_metrics(self, device):
        result = {}
        mmrl = self._get_mmrl_module()
        if mmrl is None:
            return result

        shared = getattr(mmrl, "shared_represent_space", None)
        if shared is not None:
            result["shared_rep_norm_mean"] = shared.detach().float().norm(dim=-1).mean()
            if shared.grad is not None:
                result["shared_rep_grad_norm"] = shared.grad.detach().float().norm()

        def _grad_stats(modules, prefix):
            grads = []
            for m in modules:
                for p in m.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().float().norm())
            if grads:
                stacked = torch.stack(grads)
                result[f"{prefix}_grad_norm_mean"] = stacked.mean()
                result[f"{prefix}_grad_norm_max"] = stacked.max()

        _grad_stats(getattr(mmrl, "t_r_token_projector", []), "t_projector")
        _grad_stats(getattr(mmrl, "v_r_token_projector", []), "v_projector")
        result["text_rep_scale"] = self.model.text_rep_scale.detach().float().to(device)
        if getattr(self.model.text_rep_scale, "grad", None) is not None:
            result["text_rep_scale_grad_abs"] = self.model.text_rep_scale.grad.detach().float().abs().to(device)
        else:
            result["text_rep_scale_grad_abs"] = torch.tensor(0.0, device=device)

        visual = getattr(self.model, "visual", None)
        if visual is not None:
            def _module_grad_stats(module, prefix):
                grads = []
                for p in module.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().float().norm())
                if grads:
                    stacked = torch.stack(grads)
                    result[f"{prefix}_grad_norm_mean"] = stacked.mean()
                    result[f"{prefix}_grad_norm_max"] = stacked.max()
                else:
                    result[f"{prefix}_grad_norm_mean"] = torch.tensor(0.0, device=device)
                    result[f"{prefix}_grad_norm_max"] = torch.tensor(0.0, device=device)

            _module_grad_stats(getattr(visual, "adapter_router", nn.Module()), "adapter_router")
            _module_grad_stats(getattr(visual, "residual_adapters", nn.Module()), "visual_adapter")
            _module_grad_stats(getattr(visual, "visionGating", nn.Module()), "vision_gate")
            blocks_with_rep = getattr(visual, "blocks_with_rep", [])
            trainable_count = 0
            for block in blocks_with_rep:
                trainable_count += sum(p.numel() for p in block.parameters() if p.requires_grad)
            result["blocks_with_rep_trainable_param_count"] = torch.tensor(float(trainable_count), device=device)
        if self._last_shared_rep_grad:
            for key, value in self._last_shared_rep_grad.items():
                result[key] = torch.tensor(float(value), device=device)
        return result

    def _shared_rep_grad_hook(self, grad):
        grad_f = grad.detach().float()
        self._last_shared_rep_grad = {
            "shared_rep_grad_norm": float(grad_f.norm().item()),
            "shared_rep_grad_mean_abs": float(grad_f.abs().mean().item()),
            "shared_rep_grad_max_abs": float(grad_f.abs().max().item()),
        }
        return grad

    def _ensure_shared_rep_grad_hook(self):
        mmrl = self._get_mmrl_module()
        shared = getattr(mmrl, "shared_represent_space", None) if mmrl is not None else None
        if shared is None:
            return

        current_param_id = id(shared)
        if (
            self._shared_rep_grad_hook_handle is not None
            and self._shared_rep_grad_hook_param_id == current_param_id
        ):
            return

        if self._shared_rep_grad_hook_handle is not None:
            try:
                self._shared_rep_grad_hook_handle.remove()
            except Exception:
                pass

        self._shared_rep_grad_hook_handle = shared.register_hook(self._shared_rep_grad_hook)
        self._shared_rep_grad_hook_param_id = current_param_id

    def _get_text_gating_module(self):
        visual = getattr(self.model, "visual", None)
        return getattr(visual, "text_gating", None)

    def _get_k_norm(self):
        text_gating = self._get_text_gating_module()
        return float(getattr(text_gating, "total_rep_num", 40.0))

    def _compute_group_usage_stats(self, slot_usage, device):
        if slot_usage is None or slot_usage.numel() == 0:
            nan = torch.tensor(float("nan"), device=device)
            return {
                "group_usage_probs": None,
                "group_usage_max": nan,
                "group_usage_min": nan,
                "group_usage_entropy": nan,
                "group_usage_entropy_norm": nan,
                "group_usage_dead_ratio": nan,
            }

        group_count = 8
        total_slots = int(slot_usage.numel())
        if total_slots % group_count == 0:
            group_usage = slot_usage.view(group_count, total_slots // group_count).mean(dim=-1)
        else:
            group_usage = slot_usage
            group_count = int(group_usage.numel())

        group_usage = group_usage.clamp_min(0.0)
        usage_sum = group_usage.sum()
        if usage_sum <= 1e-8:
            group_probs = torch.full_like(group_usage, 1.0 / max(group_usage.numel(), 1))
        else:
            group_probs = group_usage / usage_sum

        entropy = -(group_probs * group_probs.clamp_min(1e-8).log()).sum()
        if group_probs.numel() > 1:
            entropy_norm = entropy / torch.log(group_probs.new_tensor(float(group_probs.numel())))
        else:
            entropy_norm = group_probs.new_tensor(0.0)

        return {
            "group_usage_probs": group_probs,
            "group_usage_max": group_probs.max(),
            "group_usage_min": group_probs.min(),
            "group_usage_entropy": entropy,
            "group_usage_entropy_norm": entropy_norm,
            "group_usage_dead_ratio": (group_probs < float(self.group_usage_dead_threshold)).float().mean(),
        }

    def _update_group_usage_ema(self, group_probs):
        if group_probs is None:
            return None
        probs = group_probs.detach().float()
        if self.group_usage_ema is None or self.group_usage_ema.shape != probs.shape:
            self.group_usage_ema = probs
        else:
            alpha = float(self.group_usage_ema_alpha)
            self.group_usage_ema = (1.0 - alpha) * self.group_usage_ema + alpha * probs
        return self.group_usage_ema

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
        self._last_shared_rep_grad = {}
        self._ensure_shared_rep_grad_hook()
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

        k_general_mean = torch.tensor(float("nan"), device=input_ids.device)
        k_expert_mean = torch.tensor(float("nan"), device=input_ids.device)
        k_results = self.model.visual.k_results
        k_sums = k_results[0] if isinstance(k_results, tuple) else k_results
        total_rep_num = self._get_k_norm()
        expert_floor_loss = torch.tensor(0.0, device=input_ids.device)
        anti_collapse_loss = torch.tensor(0.0, device=input_ids.device)
        raw_capacity_prior_loss = torch.tensor(0.0, device=input_ids.device)
        group_usage_max = torch.tensor(0.0, device=input_ids.device)
        group_usage_min = torch.tensor(0.0, device=input_ids.device)
        group_usage_entropy = torch.tensor(0.0, device=input_ids.device)
        group_usage_entropy_norm = torch.tensor(0.0, device=input_ids.device)
        group_usage_dead_ratio = torch.tensor(0.0, device=input_ids.device)
        expert_sample_count = torch.tensor(0.0, device=input_ids.device)
        general_sample_count = torch.tensor(0.0, device=input_ids.device)
        live_context = {}
        text_gating = self._get_text_gating_module()
        if text_gating is not None:
            live_context = getattr(text_gating, "live_context", {}) or {}

        raw_capacity_prior_loss = getattr(self.model, "capacity_prior_loss", raw_capacity_prior_loss)
        if raw_capacity_prior_loss is None:
            raw_capacity_prior_loss = getattr(self.model, "tax_loss", torch.tensor(0.0, device=input_ids.device))
        if not torch.is_tensor(raw_capacity_prior_loss):
            raw_capacity_prior_loss = torch.tensor(raw_capacity_prior_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            raw_capacity_prior_loss = raw_capacity_prior_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        capacity_prior_loss = (
            raw_capacity_prior_loss * self.capacity_prior_loss_weight
            if self.use_capacity_prior_loss
            else torch.tensor(0.0, device=input_ids.device, dtype=ce_loss.dtype)
        )
        adapter_usage_balance_loss = getattr(self.model.visual, "adapter_usage_balance_loss", torch.tensor(0.0, device=input_ids.device))
        adapter_sample_entropy_loss = getattr(self.model.visual, "adapter_sample_entropy_loss", torch.tensor(0.0, device=input_ids.device))
        adapter_common_mode_loss = getattr(self.model.visual, "adapter_common_mode_loss", torch.tensor(0.0, device=input_ids.device))
        if not torch.is_tensor(adapter_usage_balance_loss):
            adapter_usage_balance_loss = torch.tensor(adapter_usage_balance_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            adapter_usage_balance_loss = adapter_usage_balance_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        if not torch.is_tensor(adapter_sample_entropy_loss):
            adapter_sample_entropy_loss = torch.tensor(adapter_sample_entropy_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            adapter_sample_entropy_loss = adapter_sample_entropy_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        if not torch.is_tensor(adapter_common_mode_loss):
            adapter_common_mode_loss = torch.tensor(adapter_common_mode_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            adapter_common_mode_loss = adapter_common_mode_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        scaled_adapter_usage_balance_loss = adapter_usage_balance_loss * float(self.adapter_usage_balance_loss_weight)
        scaled_adapter_sample_entropy_loss = adapter_sample_entropy_loss * float(self.adapter_sample_entropy_loss_weight)
        scaled_adapter_common_mode_loss = adapter_common_mode_loss * float(self.adapter_common_mode_loss_weight)

        expert_mask = None
        if expanded_labels is not None and k_sums is not None and k_sums.shape[0] == expanded_labels.shape[0]:
            lbl = expanded_labels.squeeze(-1)
            is_general = (lbl < 0.1)
            expert_mask = (lbl > 0.9)
            general_sample_count = is_general.float().sum()
            expert_sample_count = expert_mask.float().sum()
            if is_general.any():
                k_general_mean = k_sums[is_general].detach().float().mean()
            else:
                k_general_mean = torch.tensor(0.0, device=input_ids.device)
            if expert_mask.any():
                k_expert_mean = k_sums[expert_mask].detach().float().mean()
            else:
                k_expert_mean = torch.tensor(0.0, device=input_ids.device)

        group_usage_live = live_context.get("group_usage_live", None)
        if torch.is_tensor(group_usage_live) and group_usage_live.numel() > 0:
            group_stats = self._compute_group_usage_stats(group_usage_live.detach().float(), input_ids.device)
            group_usage_max = group_stats["group_usage_max"]
            group_usage_min = group_stats["group_usage_min"]
            group_usage_entropy = group_stats["group_usage_entropy"]
            group_usage_entropy_norm = group_stats["group_usage_entropy_norm"]
            group_usage_dead_ratio = group_stats["group_usage_dead_ratio"]

        if self.enable_expert_floor_loss and expert_mask is not None and expert_mask.any() and k_sums is not None:
            expert_active = k_sums[expert_mask].float()
            min_active = min(float(self.expert_min_active_tokens), float(total_rep_num))
            gap = torch.relu(k_sums.new_tensor(min_active) - expert_active)
            expert_floor_loss = (gap / max(min_active, 1.0)).pow(2).mean() * float(self.expert_floor_loss_weight)

        if self.enable_collapse_loss and expert_mask is not None and expert_mask.any():
            hard_k_logits_live = live_context.get("hard_k_logits_live", None)
            if torch.is_tensor(hard_k_logits_live) and hard_k_logits_live.dim() == 2 and hard_k_logits_live.shape[0] == expert_mask.shape[0]:
                expert_gate = hard_k_logits_live[expert_mask]
                if expert_gate.numel() > 0:
                    slot_usage_live = expert_gate.mean(dim=0)
                    group_stats = self._compute_group_usage_stats(slot_usage_live, input_ids.device)
                    group_probs = group_stats["group_usage_probs"]
                    group_usage_max = group_stats["group_usage_max"]
                    group_usage_min = group_stats["group_usage_min"]
                    group_usage_entropy = group_stats["group_usage_entropy"]
                    group_usage_entropy_norm = group_stats["group_usage_entropy_norm"]
                    group_usage_dead_ratio = group_stats["group_usage_dead_ratio"]

                    max_penalty = torch.relu(group_usage_max - float(self.collapse_max_ratio)).pow(2)
                    min_penalty = torch.relu(float(self.collapse_min_ratio) - group_usage_min).pow(2)
                    entropy_penalty = torch.relu(float(self.collapse_entropy_target) - group_usage_entropy_norm).pow(2)
                    anti_collapse_loss = (max_penalty + min_penalty + entropy_penalty) * float(self.anti_collapse_weight)

                    ema_probs = self._update_group_usage_ema(group_probs)
                    _ = self._compute_group_usage_stats(ema_probs, input_ids.device)

        outputs.loss = (
            self.ce_loss_weight * ce_loss
            + alpha_guide_loss
            + capacity_prior_loss
            + scaled_adapter_usage_balance_loss
            + scaled_adapter_sample_entropy_loss
            + scaled_adapter_common_mode_loss
        )

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

            self._last_metrics = {
                "total_loss": outputs.loss.detach(),
                "ce_loss": ce_loss.detach(),
                "alpha_guide_loss": alpha_guide_loss.detach(),
                "expert_floor_loss": expert_floor_loss.detach(),
                "anti_collapse_loss": anti_collapse_loss.detach(),
                "k_general_loss": torch.tensor(0.0, device=input_ids.device),
                "k_expert_loss": expert_floor_loss.detach(),
                "raw_capacity_prior_loss": raw_capacity_prior_loss.detach(),
                "capacity_prior_loss": capacity_prior_loss.detach(),
                "adapter_usage_balance_loss": adapter_usage_balance_loss.detach(),
                "adapter_sample_entropy_loss": adapter_sample_entropy_loss.detach(),
                "adapter_common_mode_loss": adapter_common_mode_loss.detach(),
                "adapter_usage_balance_loss_scaled": scaled_adapter_usage_balance_loss.detach(),
                "adapter_sample_entropy_loss_scaled": scaled_adapter_sample_entropy_loss.detach(),
                "adapter_common_mode_loss_scaled": scaled_adapter_common_mode_loss.detach(),
                "alpha_mae": alpha_mae.detach(),
                "k_general_mean": k_general_mean.detach(),
                "k_expert_mean": k_expert_mean.detach(),
                "expert_sample_count": expert_sample_count.detach(),
                "general_sample_count": general_sample_count.detach(),
                "anti_collapse_weight": torch.tensor(float(self.anti_collapse_weight), device=input_ids.device),
                "group_usage_max": group_usage_max.detach(),
                "group_usage_min": group_usage_min.detach(),
                "group_usage_entropy": group_usage_entropy.detach(),
                "group_usage_entropy_norm": group_usage_entropy_norm.detach(),
                "group_usage_dead_ratio": group_usage_dead_ratio.detach(),
                "temperature": torch.tensor(float(self.temperature_override) if self.temperature_override is not None else float("nan"), device=input_ids.device),
                "capacity_prior_weight": torch.tensor(float(self.capacity_prior_loss_weight), device=input_ids.device),
                "adapter_usage_balance_weight": torch.tensor(float(self.adapter_usage_balance_loss_weight), device=input_ids.device),
                "adapter_sample_entropy_weight": torch.tensor(float(self.adapter_sample_entropy_loss_weight), device=input_ids.device),
                "adapter_common_mode_weight": torch.tensor(float(self.adapter_common_mode_loss_weight), device=input_ids.device),
            }
            if text_gating is not None:
                dbg = getattr(text_gating, "debug_context", {}) or {}
                for key, value in dbg.items():
                    self._last_metrics[key] = value
                if self.enable_collapse_loss and self.group_usage_ema is not None:
                    for idx, value in enumerate(self.group_usage_ema.detach().float().tolist()):
                        self._last_metrics[f"group_usage_ema_{idx}"] = float(value)
            visual_dbg = getattr(self.model.visual, "debug_context", {}) or {}
            for key, value in visual_dbg.items():
                self._last_metrics[key] = value
            model_dbg = getattr(self.model, "debug_context", {}) or {}
            for key, value in model_dbg.items():
                self._last_metrics[key] = value
            for key, value in self._collect_rep_grad_metrics(input_ids.device).items():
                self._last_metrics[key] = value

        if self.debug_mode and torch.is_tensor(outputs.loss):
            loss_val = outputs.loss.detach().float().item()
            if loss_val > self.debug_loss_threshold or not torch.isfinite(outputs.loss):
                print("\n!!!! [HIGH-LOSS-DBG] abnormal loss detected !!!!")
                print(f"!!!! [HIGH-LOSS-DBG] final_loss={loss_val:.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] manual_ce={ce_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] alpha_loss={alpha_guide_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] use_capacity_prior={self.use_capacity_prior_loss} capacity_w={float(self.capacity_prior_loss_weight):.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] raw_capacity_prior={raw_capacity_prior_loss.detach().float().item():.6f}")
                print(f"!!!! [HIGH-LOSS-DBG] capacity_prior_scaled={capacity_prior_loss.detach().float().item():.6f}")

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


def build_model_and_processor(model_path, experiment_cfg=None):
    experiment_cfg = experiment_cfg or {}
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    active_rep_token_count = int(experiment_cfg.get("active_rep_token_count", getattr(cfg, "ACTIVE_REP_TOKEN_COUNT", 40)))
    cfg.ACTIVE_REP_TOKEN_COUNT = active_rep_token_count
    cfg.SPECIAL_TOKENS = cfg.build_special_tokens(active_rep_token_count)

    config.ACTIVE_REP_TOKEN_COUNT = active_rep_token_count
    config.TEXT_GATE_SELECTION_MODE = str(experiment_cfg.get("text_gate_selection_mode", "group_threshold_prior"))
    config.TEXT_GATE_GROUP_COUNT = int(experiment_cfg.get("text_gate_group_count", 8))
    config.TEXT_GATE_CAPACITY_PRIOR_ENERGY = list(experiment_cfg.get(
        "text_gate_capacity_prior_energy",
        [1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0],
    ))
    config.TEXT_GATE_CAPACITY_PRIOR_SIGMA = float(experiment_cfg.get("text_gate_capacity_prior_sigma", 0.40))
    config.ABLATE_VISUAL_GATE = bool(experiment_cfg.get(
        "ablate_visual_gate",
        os.getenv("MMRL_ABLATE_VISUAL_GATE", "0") == "1"
    ))
    config.ABLATE_DIRECT_LEARNABLE_REP = bool(experiment_cfg.get(
        "ablate_direct_learnable_rep",
        os.getenv("MMRL_ABLATE_DIRECT_LEARNABLE_REP", "0") == "1"
    ))
    config.DISABLE_TEXT_GATE = bool(experiment_cfg.get(
        "disable_text_gate",
        os.getenv("MMRL_DISABLE_TEXT_GATE", "0") == "1"
    ))
    config.DISABLE_TEXT_PROMPT_INSERT = bool(experiment_cfg.get(
        "disable_text_prompt_insert",
        os.getenv("MMRL_DISABLE_TEXT_PROMPT_INSERT", "0") == "1"
    ))
    config.MASK_REP_PLACEHOLDERS_WHEN_TEXT_DISABLED = bool(experiment_cfg.get(
        "mask_rep_placeholders_when_text_disabled",
        os.getenv("MMRL_MASK_REP_PLACEHOLDERS_WHEN_TEXT_DISABLED", "1") == "1"
    ))
    config.TEXT_REP_INIT_SCALE = float(experiment_cfg.get("text_rep_init_scale", 1e-3))
    config.VISUAL_RESIDUAL_ADAPTER_COUNT = int(experiment_cfg.get(
        "visual_residual_adapter_count",
        os.getenv("MMRL_VISUAL_RESIDUAL_ADAPTER_COUNT", "4"),
    ))
    config.ADAPTER_USAGE_BALANCE_LOSS_WEIGHT = float(experiment_cfg.get(
        "adapter_usage_balance_loss_weight",
        os.getenv("MMRL_ADAPTER_USAGE_BALANCE_LOSS_WEIGHT", "0.0"),
    ))
    config.ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT = float(experiment_cfg.get(
        "adapter_sample_entropy_loss_weight",
        os.getenv("MMRL_ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT", "0.0"),
    ))
    config.ADAPTER_COMMON_MODE_LOSS_WEIGHT = float(experiment_cfg.get(
        "adapter_common_mode_loss_weight",
        os.getenv("MMRL_ADAPTER_COMMON_MODE_LOSS_WEIGHT", "0.0"),
    ))
    config.ADAPTER_SAMPLE_ENTROPY_TARGET = float(experiment_cfg.get(
        "adapter_sample_entropy_target",
        os.getenv("MMRL_ADAPTER_SAMPLE_ENTROPY_TARGET", "0.40"),
    ))
    config.ADAPTER_COMMON_MODE_TARGET = float(experiment_cfg.get(
        "adapter_common_mode_target",
        os.getenv("MMRL_ADAPTER_COMMON_MODE_TARGET", "0.85"),
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
    model._ensure_shared_rep_grad_hook()
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
    print(
        f"[Experiment] mode={config.TEXT_GATE_SELECTION_MODE} active_rep_token_count={config.ACTIVE_REP_TOKEN_COUNT} "
        f"group_count={config.TEXT_GATE_GROUP_COUNT} "
        f"ablate_visual_gate={config.ABLATE_VISUAL_GATE} "
        f"ablate_direct_learnable_rep={config.ABLATE_DIRECT_LEARNABLE_REP} "
        f"visual_residual_adapter_count={config.VISUAL_RESIDUAL_ADAPTER_COUNT} "
        f"adapter_usage_balance_loss_weight={config.ADAPTER_USAGE_BALANCE_LOSS_WEIGHT} "
        f"adapter_sample_entropy_loss_weight={config.ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT} "
        f"adapter_common_mode_loss_weight={config.ADAPTER_COMMON_MODE_LOSS_WEIGHT} "
        f"disable_text_gate={config.DISABLE_TEXT_GATE} "
        f"disable_text_prompt_insert={config.DISABLE_TEXT_PROMPT_INSERT} "
        f"mask_rep_placeholders_when_text_disabled={config.MASK_REP_PLACEHOLDERS_WHEN_TEXT_DISABLED}"
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
        mods = [v.hidden_state_pooling, v.embedding_pooling, v.Task_classifier, v.adapter_router]
    elif stage == 2:
        mods = [v.hidden_state_pooling, v.embedding_pooling, v.Task_classifier, v.adapter_router, v.visionGating, v.text_gating]
    elif stage in [3, 4]:
        mods = [model.model.MMRL,
                # v.blocks_with_rep, # ablation: keep rep-token branch forward, but freeze branch blocks
                v.hidden_state_pooling, 
                v.embedding_pooling,
                v.adapter_router,
                v.residual_adapters,
                v.visionGating, 
                v.text_gating, 
                ]
    else:
        raise ValueError("stage must be 1/2/3/4")

    for m in mods:
        for p in m.parameters():
            p.requires_grad = True

    if getattr(v, "disable_text_gate", False):
        for p in v.text_gating.parameters():
            p.requires_grad = False
    if getattr(model.model, "disable_text_prompt_insert", False):
        for p in model.model.MMRL.t_r_token_projector.parameters():
            p.requires_grad = False
        for p in getattr(model.model.MMRL, "direct_t_tokens", []):
            p.requires_grad = False
        model.model.text_rep_scale.requires_grad = False


def print_stage4_trainable_params_before_stage1(model):
    set_trainable_stage(model, 4)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in trainable)

    print("\n" + "!" * 120)
    # print("!" * 30 + " STAGE4 TRAINABLE PARAMETERS BEFORE STAGE1 " + "!" * 30)
    # print("!" * 120)
    # for name, p in trainable:
    #     print(f"[STAGE4-TRAINABLE] name={name} shape={tuple(p.shape)} numel={p.numel()}")
    print(f"[STAGE4-TRAINABLE][TOTAL] tensors={len(trainable)} total_numel={total}")
    print("!" * 120 + "\n")


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

    # 仅优化可训练参数
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg["learning_rate"][stage_id], weight_decay=0.01)

    steps_per_update = train_cfg["gradient_accumulation_steps"]
    epochs = train_cfg["epochs"][stage_id]
    use_focal = (stage_id == 1)
    updates_per_epoch = max((len(dl) + steps_per_update - 1) // steps_per_update, 1)
    total_optimizer_steps = max(epochs * updates_per_epoch, 1)
    warmup_steps = int(total_optimizer_steps * 0.05)
    if total_optimizer_steps > 1:
        warmup_steps = max(warmup_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    v = model.model.visual
    rep_placeholder_ids = model.model.rep_placeholder_ids

    global_step = 0
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            alpha_labels = batch["alpha_labels"].cuda(non_blocking=True).view(-1, 1).to(dtype=v.dtype)
            task_type_ids = batch["task_type_ids"].cuda(non_blocking=True)
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
            if stage_id == 2 and getattr(model, "enable_gate_loss_stage2", True):
                g = v.visionGating(alpha_logits, getattr(model, "temperature_override", None))
                target = alpha_labels
                gate_loss = torch.nn.functional.mse_loss(g, target)

            loss = (cls_loss + gate_loss) / steps_per_update
            loss.backward()

            if ((i + 1) % steps_per_update == 0) or (i + 1 == len(dl)):
                opt.step()
                scheduler.step()
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
        model.alpha_loss_weight = 0.0
        model.enable_alpha_guide_loss = False
        model.enable_expert_floor_loss = False
        model.expert_floor_loss_weight = 0.0
        model.enable_collapse_loss = False
        model.anti_collapse_weight = 0.0
        model.use_capacity_prior_loss = False
        model.capacity_prior_loss_weight = 0.0
    else:  # stage4
        model.ce_loss_weight = 1.0
        model.alpha_loss_weight = 0.0
        model.enable_alpha_guide_loss = False
        model.enable_expert_floor_loss = False
        model.expert_floor_loss_weight = 0.0
        model.expert_min_active_tokens = float(train_cfg.get("expert_min_active_tokens", 4.0))
        model.enable_collapse_loss = False
        model.anti_collapse_weight = 0.0
        model.collapse_max_ratio = float(train_cfg.get("collapse_max_ratio", 0.45))
        model.collapse_min_ratio = float(train_cfg.get("collapse_min_ratio", 0.05))
        model.collapse_entropy_target = float(train_cfg.get("collapse_entropy_target", 0.75))
        model.group_usage_dead_threshold = float(train_cfg.get("group_usage_dead_threshold", 0.03))
        model.group_usage_ema_alpha = float(train_cfg.get("group_usage_ema_alpha", 0.10))
        model.group_usage_ema = None
        model.use_capacity_prior_loss = train_cfg.get("enable_capacity_prior_loss_s4", True)
        model.capacity_prior_loss_weight = 0.0

    model.adapter_usage_balance_loss_weight = float(train_cfg.get("adapter_usage_balance_loss_weight", 0.0))
    model.adapter_sample_entropy_loss_weight = float(train_cfg.get("adapter_sample_entropy_loss_weight", 0.0))
    model.adapter_common_mode_loss_weight = float(train_cfg.get("adapter_common_mode_loss_weight", 0.0))
    model.enable_gate_loss_stage2 = train_cfg.get("enable_gate_loss_stage2", True)

    stage34_views = ("expert-mm",)
    print(f"[Stage{stage_id}] enable_views={stage34_views}")

    ds = FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=data_cfg["total_limit"],
        enable_views=stage34_views,
        mode=f"stage{stage_id}",
        ce_enabled=True,
        seed=train_cfg["seed"],
        deterministic_sampling=train_cfg.get("deterministic_sampling", False),
    )
    collator = MMRLDataCollator(processor)
    cb = StageScheduleCallback(
        dataset=ds,
        total_epochs=train_cfg["epochs"][stage_id],
        init_temp=train_cfg.get("initial_temp", 1.0),
        final_temp=train_cfg.get("final_temp", 0.1),
        target_capacity_prior_weight=(
            0.0
            if stage_id == 3 or (not train_cfg.get("enable_capacity_prior_loss_s4", True))
            else train_cfg["capacity_prior_loss_weight"]
        ),
        capacity_prior_warmup_epochs=train_cfg.get("capacity_prior_warmup_epochs", 1),
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
    diag_cb = MMRLDiagnosticsCallback(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        every_steps=train_cfg.get("diag_every_steps", 1000),
        stage_name=f"stage{stage_id}",
    )
    grad_monitor_cb = SharedRepGradMonitorCallback(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        stage_name=f"stage{stage_id}",
        ema_alpha=train_cfg.get("grad_spike_ema_alpha", 0.10),
        ratio_threshold=train_cfg.get("grad_spike_ratio_threshold", 3.0),
        abs_threshold=train_cfg.get("grad_spike_abs_threshold", 0.5),
        cooldown_steps=train_cfg.get("grad_spike_cooldown_steps", 20),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[cb, grad_monitor_cb, metrics_cb, diag_cb]
    )
    trainer.train()
    # trainer.save_model(f"{output_dir}/stage{stage_id}")


def run_stage(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    if stage_id == 1:
        _write_experiment_manifest(output_dir, train_cfg)
        print_stage4_trainable_params_before_stage1(model)
    if stage_id in [1, 2]:
        run_stage12_light(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    elif stage_id in [3, 4]:
        run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir)
    else:
        raise ValueError("stage_id must be 1/2/3/4")