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
from live_epoch_eval import run_live_epoch_evaluation

import numbers


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
            "data_sampling_seed": train_cfg.get("data_sampling_seed"),
            "data_order_seed": train_cfg.get("data_order_seed"),
            "deterministic_sampling": train_cfg.get("deterministic_sampling", False),
            "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps"),
            "learning_rate": train_cfg.get("learning_rate", {}).get(stage_id),
            "epochs": train_cfg.get("epochs", {}).get(stage_id),
            "console_log_every": train_cfg.get("console_log_every"),
            "metric_smooth_window": train_cfg.get("metric_smooth_window"),
            "metric_ema_alpha": train_cfg.get("metric_ema_alpha"),
            "metric_scatter_stride": train_cfg.get("metric_scatter_stride"),
            "visual_residual_adapter_count": train_cfg.get("visual_residual_adapter_count"),
            "adapter_usage_balance_loss_weight": train_cfg.get("adapter_usage_balance_loss_weight"),
            "adapter_sample_entropy_loss_weight": train_cfg.get("adapter_sample_entropy_loss_weight"),
            "adapter_common_mode_loss_weight": train_cfg.get("adapter_common_mode_loss_weight"),
            "adapter_effective_delta_loss_weight": train_cfg.get(
                "adapter_effective_delta_loss_weight",
                experiment_cfg.get("adapter_effective_delta_loss_weight"),
            ),
            "mmrl_relation_loss_weight": train_cfg.get(
                "mmrl_relation_loss_weight",
                experiment_cfg.get("mmrl_relation_loss_weight"),
            ),
            "mmrl_relation_max_tokens": train_cfg.get(
                "mmrl_relation_max_tokens",
                experiment_cfg.get("mmrl_relation_max_tokens"),
            ),
            "mmrl_variance_floor_ratio": train_cfg.get(
                "mmrl_variance_floor_ratio",
                experiment_cfg.get("mmrl_variance_floor_ratio"),
            ),
            "mmrl_variance_floor_weight": train_cfg.get(
                "mmrl_variance_floor_weight",
                experiment_cfg.get("mmrl_variance_floor_weight"),
            ),
            "adapter_sample_entropy_target": train_cfg.get("adapter_sample_entropy_target"),
            "adapter_common_mode_target": train_cfg.get("adapter_common_mode_target"),
            "adapter_effective_delta_target_low": train_cfg.get(
                "adapter_effective_delta_target_low",
                experiment_cfg.get("adapter_effective_delta_target_low"),
            ),
            "adapter_effective_delta_target_high": train_cfg.get(
                "adapter_effective_delta_target_high",
                experiment_cfg.get("adapter_effective_delta_target_high"),
            ),
            "adapter_diversity_loss_weight": train_cfg.get(
                "adapter_diversity_loss_weight",
                experiment_cfg.get("adapter_diversity_loss_weight"),
            ),
            "adapter_diversity_target_low": train_cfg.get(
                "adapter_diversity_target_low",
                experiment_cfg.get("adapter_diversity_target_low"),
            ),
            "adapter_diversity_target_high": train_cfg.get(
                "adapter_diversity_target_high",
                experiment_cfg.get("adapter_diversity_target_high"),
            ),
            "adapter_diversity_upper_weight": train_cfg.get(
                "adapter_diversity_upper_weight",
                experiment_cfg.get("adapter_diversity_upper_weight"),
            ),
            "adapter_diversity_worst_pair_weight": train_cfg.get(
                "adapter_diversity_worst_pair_weight",
                experiment_cfg.get("adapter_diversity_worst_pair_weight"),
            ),
            "enable_adapter_router_identity_residual": train_cfg.get(
                "enable_adapter_router_identity_residual",
                experiment_cfg.get("enable_adapter_router_identity_residual"),
            ),
            "raw_visual_adapter": train_cfg.get(
                "raw_visual_adapter",
                experiment_cfg.get("raw_visual_adapter"),
            ),
            "direct_mmrl_output": train_cfg.get(
                "direct_mmrl_output",
                experiment_cfg.get("direct_mmrl_output"),
            ),
            "enable_deepstack_mmrl_residual": train_cfg.get(
                "enable_deepstack_mmrl_residual",
                experiment_cfg.get("enable_deepstack_mmrl_residual"),
            ),
            "deepstack_mmrl_residual_scale": train_cfg.get(
                "deepstack_mmrl_residual_scale",
                experiment_cfg.get("deepstack_mmrl_residual_scale"),
            ),
            "diag_every_steps": train_cfg.get("diag_every_steps"),
            "grad_spike_ema_alpha": train_cfg.get("grad_spike_ema_alpha"),
            "grad_spike_ratio_threshold": train_cfg.get("grad_spike_ratio_threshold"),
            "grad_spike_abs_threshold": train_cfg.get("grad_spike_abs_threshold"),
            "grad_spike_cooldown_steps": train_cfg.get("grad_spike_cooldown_steps"),
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
    if "alpha_mae" in row:
        print(f"[Alpha Statistics]")
        print(f"  └─ Alpha MAE:             {_fmt(row.get('alpha_mae')):>10}")

    if "active_token_count_mean" in row:
        print(f"[Routing Statistics]")
        if "active_token_count_mean" in row:
            print(f"  ├─ Active Tokens:         {_fmt(row.get('active_token_count_mean'), 3):>10}")
        if "k_budget_mean" in row:
            print(f"  ├─ K Budget:              {_fmt(row.get('k_budget_mean'), 3):>10}")
        if "raw_budget_mean" in row:
            print(f"  ├─ Raw Budget:            {_fmt(row.get('raw_budget_mean'), 3):>10}")
        if "batch_alpha_mean" in row:
            print(f"  └─ Batch Alpha:           {_fmt(row.get('batch_alpha_mean'), 4):>10}")

    print(f"[Schedule]")
    if "temperature" in row:
        print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), 4):>10}")
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
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.init_temp = init_temp
        self.final_temp = final_temp

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.resample_data()

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ep = state.epoch if state.epoch is not None else 0.0

        # temp anneal
        total_epochs = max(float(self.total_epochs), 1e-8)
        prog = min(float(ep) / total_epochs, 1.0)
        model.temperature_override = self.init_temp - (self.init_temp - self.final_temp) * prog
        model.current_stage_progress = float(prog)
        model.current_stage_step = int(state.global_step) + 1


class StopAfterEpochCallback(TrainerCallback):
    def __init__(self, stop_after_epochs):
        self.stop_after_epochs = int(stop_after_epochs)

    def on_epoch_end(self, args, state, control, **kwargs):
        completed_epochs = int(round(float(state.epoch or 0.0)))
        if completed_epochs >= self.stop_after_epochs:
            print(
                f"[EARLY-STOP] completed_epochs={completed_epochs} "
                f"planned_epochs={int(args.num_train_epochs)}"
            )
            control.should_training_stop = True
        return control


class EpochEvaluationCallback(TrainerCallback):
    def __init__(self, processor, output_dir, stage_id, total_epochs):
        self.processor = processor
        self.output_dir = output_dir
        self.stage_id = int(stage_id)
        self.total_epochs = int(total_epochs)
        self.completed_epochs = set()

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        epoch_id = max(1, int(round(float(state.epoch or 0.0))))
        if epoch_id in self.completed_epochs:
            return
        self.completed_epochs.add(epoch_id)
        if self.stage_id == 3 and epoch_id >= self.total_epochs:
            print(
                f"[EPOCH-EVAL] stage=3 epoch={epoch_id} skipped; "
                "the normal final evaluation covers the identical weights"
            )
            return
        log_path = os.path.join(
            self.output_dir,
            "eval_epochs",
            f"stage{self.stage_id}_epoch{epoch_id}",
            "test.log",
        )
        print(f"[EPOCH-EVAL] stage={self.stage_id} epoch={epoch_id} begin")
        summary = run_live_epoch_evaluation(kwargs["model"], self.processor, log_path)
        print(
            f"[EPOCH-EVAL] stage={self.stage_id} epoch={epoch_id} "
            f"score={summary.get('score')} completed"
        )


class MMRLDiagnosticsCallback(TrainerCallback):
    DEFAULT_KEEP_KEYS = {
        "ce_loss",
        "active_token_count_mean",
        "delta_pool_common_mode_ratio",
        "delta_pool_specificity_ratio",
        "mmrl_delta_pool_common_mode_ratio",
        "mmrl_delta_pool_specificity_ratio",
        "adapter_route_entropy_norm",
    }
    KEEP_KEY_ALIASES = {
        "group_entropy_norm": ("group_entropy_norm", "group_usage_entropy_norm"),
        "group_usage_entropy_norm": ("group_entropy_norm", "group_usage_entropy_norm"),
    }

    def __init__(self, save_dir, every_steps=1000, stage_name="stage", keep_keys=None):
        self.save_dir = save_dir
        self.every_steps = int(max(1, every_steps))
        self.stage_name = stage_name
        self.keep_keys = self._expand_keep_keys(keep_keys or self.DEFAULT_KEEP_KEYS)
        self.buffer = []
        self.jsonl_path = os.path.join(save_dir, "mmrl_diagnostics.jsonl")
        self.csv_path = os.path.join(save_dir, "mmrl_diagnostics.csv")
        self.csv_header_written = False
        os.makedirs(save_dir, exist_ok=True)

    def _expand_keep_keys(self, keep_keys):
        expanded = set()
        for key in keep_keys:
            aliases = self.KEEP_KEY_ALIASES.get(key)
            if aliases is None:
                expanded.add(key)
            else:
                expanded.update(aliases)
        return expanded

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
            
    def _round_payload(self, value):
        if isinstance(value, numbers.Real):
            return round(float(value), 6)
        if isinstance(value, np.ndarray):
            return self._round_payload(value.tolist())
        if isinstance(value, list):
            return [self._round_payload(v) for v in value]
        if isinstance(value, dict):
            return {k: self._round_payload(v) for k, v in value.items()}
        return value

    def _flush(self):
        if not self.buffer:
            return
        payload = self._round_payload(self._aggregate(self.buffer))
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
        filtered_metrics = {k: v for k, v in metrics.items() if k in self.keep_keys}
        row = self._record_to_python({"step": step, **filtered_metrics})
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
                    "active_token_count_mean": metrics.get("active_token_count_mean"),
                    "batch_alpha_mean": metrics.get("batch_alpha_mean"),
                    "delta_to_org_ratio": metrics.get("delta_to_org_ratio"),
                }
                self._write_event(event_payload)

# =========================
#  Full model (Stage3)
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
        self.current_stage_id = 0
        self.current_stage_step = 0
        self.current_stage_progress = 0.0
        self.enable_alpha_guide_loss = False
        self.adapter_usage_balance_loss_weight = 0.0
        self.adapter_sample_entropy_loss_weight = 0.0
        self.adapter_common_mode_loss_weight = 0.0
        self.adapter_effective_delta_loss_weight = 0.0
        self.mmrl_relation_loss_weight = 0.0
        self.adapter_diversity_loss_weight = 0.0

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
            if modules is None:
                modules = []
            elif isinstance(modules, nn.Module):
                modules = [modules]

            grads = []
            for m in modules:
                for p in m.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().float().norm())
            if grads:
                stacked = torch.stack(grads)
                result[f"{prefix}_grad_norm_mean"] = stacked.mean()
                result[f"{prefix}_grad_norm_max"] = stacked.max()


        _grad_stats(getattr(mmrl, "v_r_token_projector", []), "v_projector")

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

        adapter_usage_balance_loss = getattr(self.model.visual, "adapter_usage_balance_loss", torch.tensor(0.0, device=input_ids.device))
        adapter_sample_entropy_loss = getattr(self.model.visual, "adapter_sample_entropy_loss", torch.tensor(0.0, device=input_ids.device))
        adapter_common_mode_loss = getattr(self.model.visual, "adapter_common_mode_loss", torch.tensor(0.0, device=input_ids.device))
        adapter_effective_delta_loss = getattr(self.model.visual, "adapter_effective_delta_loss", torch.tensor(0.0, device=input_ids.device))
        mmrl_relation_loss = getattr(self.model.visual, "mmrl_relation_loss", torch.tensor(0.0, device=input_ids.device))
        adapter_diversity_loss = getattr(self.model.visual, "adapter_diversity_loss", torch.tensor(0.0, device=input_ids.device))
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
        if not torch.is_tensor(adapter_effective_delta_loss):
            adapter_effective_delta_loss = torch.tensor(adapter_effective_delta_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            adapter_effective_delta_loss = adapter_effective_delta_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        if not torch.is_tensor(mmrl_relation_loss):
            mmrl_relation_loss = torch.tensor(mmrl_relation_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            mmrl_relation_loss = mmrl_relation_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        if not torch.is_tensor(adapter_diversity_loss):
            adapter_diversity_loss = torch.tensor(adapter_diversity_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            adapter_diversity_loss = adapter_diversity_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        effective_delta_weight = float(self.adapter_effective_delta_loss_weight)
        adapter_diversity_weight = float(self.adapter_diversity_loss_weight)
        usage_balance_weight = float(self.adapter_usage_balance_loss_weight)
        sample_entropy_weight = float(self.adapter_sample_entropy_loss_weight)
        scaled_adapter_usage_balance_loss = adapter_usage_balance_loss * usage_balance_weight
        scaled_adapter_sample_entropy_loss = adapter_sample_entropy_loss * sample_entropy_weight
        scaled_adapter_common_mode_loss = adapter_common_mode_loss * float(self.adapter_common_mode_loss_weight)
        scaled_adapter_effective_delta_loss = adapter_effective_delta_loss * float(effective_delta_weight)
        scaled_mmrl_relation_loss = mmrl_relation_loss * float(self.mmrl_relation_loss_weight)
        scaled_adapter_diversity_loss = adapter_diversity_loss * float(adapter_diversity_weight)

        outputs.loss = (
            self.ce_loss_weight * ce_loss
            + alpha_guide_loss
            + scaled_adapter_usage_balance_loss
            + scaled_adapter_sample_entropy_loss
            + scaled_adapter_common_mode_loss
            + scaled_adapter_effective_delta_loss
            + scaled_mmrl_relation_loss
            + scaled_adapter_diversity_loss
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
                "adapter_usage_balance_loss": adapter_usage_balance_loss.detach(),
                "adapter_sample_entropy_loss": adapter_sample_entropy_loss.detach(),
                "adapter_common_mode_loss": adapter_common_mode_loss.detach(),
                "adapter_effective_delta_loss": adapter_effective_delta_loss.detach(),
                "mmrl_relation_loss": mmrl_relation_loss.detach(),
                "adapter_diversity_loss": adapter_diversity_loss.detach(),
                "adapter_usage_balance_loss_scaled": scaled_adapter_usage_balance_loss.detach(),
                "adapter_sample_entropy_loss_scaled": scaled_adapter_sample_entropy_loss.detach(),
                "adapter_common_mode_loss_scaled": scaled_adapter_common_mode_loss.detach(),
                "adapter_effective_delta_loss_scaled": scaled_adapter_effective_delta_loss.detach(),
                "mmrl_relation_loss_scaled": scaled_mmrl_relation_loss.detach(),
                "adapter_diversity_loss_scaled": scaled_adapter_diversity_loss.detach(),
                "alpha_mae": alpha_mae.detach(),
                "temperature": torch.tensor(float(self.temperature_override) if self.temperature_override is not None else float("nan"), device=input_ids.device),
                "adapter_usage_balance_weight": torch.tensor(usage_balance_weight, device=input_ids.device),
                "adapter_sample_entropy_weight": torch.tensor(sample_entropy_weight, device=input_ids.device),
                "adapter_common_mode_weight": torch.tensor(float(self.adapter_common_mode_loss_weight), device=input_ids.device),
                "adapter_effective_delta_weight": torch.tensor(float(effective_delta_weight), device=input_ids.device),
                "adapter_diversity_weight": torch.tensor(float(adapter_diversity_weight), device=input_ids.device),
                "stage_progress": torch.tensor(float(getattr(self, "current_stage_progress", 0.0)), device=input_ids.device),
            }
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
    config.RP_SPACE_LENGTH = int(experiment_cfg.get("rp_space_length", 40))
    config.ABLATE_VISUAL_GATE = bool(experiment_cfg.get(
        "ablate_visual_gate",
        os.getenv("MMRL_ABLATE_VISUAL_GATE", "0") == "1"
    ))
    config.ABLATE_DIRECT_LEARNABLE_REP = bool(experiment_cfg.get(
        "ablate_direct_learnable_rep",
        os.getenv("MMRL_ABLATE_DIRECT_LEARNABLE_REP", "0") == "1"
    ))
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
    config.ADAPTER_EFFECTIVE_DELTA_LOSS_WEIGHT = float(experiment_cfg.get(
        "adapter_effective_delta_loss_weight",
        os.getenv("MMRL_ADAPTER_EFFECTIVE_DELTA_LOSS_WEIGHT", "0.0"),
    ))
    config.MMRL_RELATION_LOSS_WEIGHT = float(experiment_cfg.get(
        "mmrl_relation_loss_weight",
        os.getenv("MMRL_RELATION_LOSS_WEIGHT", "0.0"),
    ))
    config.MMRL_RELATION_MAX_TOKENS = int(experiment_cfg.get(
        "mmrl_relation_max_tokens",
        os.getenv("MMRL_RELATION_MAX_TOKENS", "64"),
    ))
    config.MMRL_VARIANCE_FLOOR_RATIO = float(experiment_cfg.get(
        "mmrl_variance_floor_ratio",
        os.getenv("MMRL_VARIANCE_FLOOR_RATIO", "0.50"),
    ))
    config.MMRL_VARIANCE_FLOOR_WEIGHT = float(experiment_cfg.get(
        "mmrl_variance_floor_weight",
        os.getenv("MMRL_VARIANCE_FLOOR_WEIGHT", "0.10"),
    ))
    config.ADAPTER_SAMPLE_ENTROPY_TARGET = float(experiment_cfg.get(
        "adapter_sample_entropy_target",
        os.getenv("MMRL_ADAPTER_SAMPLE_ENTROPY_TARGET", "0.40"),
    ))
    config.ADAPTER_COMMON_MODE_TARGET = float(experiment_cfg.get(
        "adapter_common_mode_target",
        os.getenv("MMRL_ADAPTER_COMMON_MODE_TARGET", "0.85"),
    ))
    config.ADAPTER_EFFECTIVE_DELTA_TARGET_LOW = float(experiment_cfg.get(
        "adapter_effective_delta_target_low",
        os.getenv("MMRL_ADAPTER_EFFECTIVE_DELTA_TARGET_LOW", "0.78"),
    ))
    config.ADAPTER_EFFECTIVE_DELTA_TARGET_HIGH = float(experiment_cfg.get(
        "adapter_effective_delta_target_high",
        os.getenv("MMRL_ADAPTER_EFFECTIVE_DELTA_TARGET_HIGH", "1.10"),
    ))
    config.ADAPTER_DIVERSITY_LOSS_WEIGHT = float(experiment_cfg.get(
        "adapter_diversity_loss_weight",
        os.getenv("MMRL_ADAPTER_DIVERSITY_LOSS_WEIGHT", "0.0"),
    ))
    config.ADAPTER_DIVERSITY_TARGET_LOW = float(experiment_cfg.get(
        "adapter_diversity_target_low",
        os.getenv("MMRL_ADAPTER_DIVERSITY_TARGET_LOW", "0.30"),
    ))
    config.ADAPTER_DIVERSITY_TARGET_HIGH = float(experiment_cfg.get(
        "adapter_diversity_target_high",
        os.getenv("MMRL_ADAPTER_DIVERSITY_TARGET_HIGH", "0.58"),
    ))
    config.ADAPTER_DIVERSITY_UPPER_WEIGHT = float(experiment_cfg.get(
        "adapter_diversity_upper_weight",
        os.getenv("MMRL_ADAPTER_DIVERSITY_UPPER_WEIGHT", "2.0"),
    ))
    config.ADAPTER_DIVERSITY_WORST_PAIR_WEIGHT = float(experiment_cfg.get(
        "adapter_diversity_worst_pair_weight",
        os.getenv("MMRL_ADAPTER_DIVERSITY_WORST_PAIR_WEIGHT", "1.0"),
    ))
    config.ENABLE_ADAPTER_ROUTER_IDENTITY_RESIDUAL = os.getenv(
        "MMRL_ENABLE_ADAPTER_ROUTER_IDENTITY_RESIDUAL",
        "1" if experiment_cfg.get("enable_adapter_router_identity_residual", False) else "0",
    ) == "1"
    config.DIRECT_MMRL_OUTPUT = os.getenv(
        "MMRL_DIRECT_MMRL_OUTPUT",
        "1" if experiment_cfg.get("direct_mmrl_output", False) else "0",
    ) == "1"
    config.RAW_VISUAL_ADAPTER = os.getenv(
        "MMRL_RAW_VISUAL_ADAPTER",
        "1" if experiment_cfg.get("raw_visual_adapter", False) else "0",
    ) == "1"
    config.ENABLE_DEEPSTACK_MMRL_RESIDUAL = os.getenv(
        "MMRL_ENABLE_DEEPSTACK_MMRL_RESIDUAL",
        "1" if experiment_cfg.get("enable_deepstack_mmrl_residual", False) else "0",
    ) == "1"
    config.DEEPSTACK_MMRL_RESIDUAL_SCALE = float(os.getenv(
        "MMRL_DEEPSTACK_MMRL_RESIDUAL_SCALE",
        str(experiment_cfg.get("deepstack_mmrl_residual_scale", 0.0)),
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print("Base model loaded. ")
    print("Tokenizer loaded. Now building MMRL model...")
    model = Qwen3VLMMRLForStages(config, tokenizer).to("cuda").to(torch.bfloat16)
    model._ensure_shared_rep_grad_hook()
    visual = model.model.visual
    if visual.direct_mmrl_output != config.DIRECT_MMRL_OUTPUT:
        raise RuntimeError(
            "DIRECT_MMRL_OUTPUT config propagation failed: "
            f"requested={config.DIRECT_MMRL_OUTPUT} actual={visual.direct_mmrl_output}"
        )
    if visual.raw_visual_adapter != config.RAW_VISUAL_ADAPTER:
        raise RuntimeError(
            "RAW_VISUAL_ADAPTER config propagation failed: "
            f"requested={config.RAW_VISUAL_ADAPTER} actual={visual.raw_visual_adapter}"
        )
    if visual.visual_residual_adapter_count != config.VISUAL_RESIDUAL_ADAPTER_COUNT:
        raise RuntimeError(
            "VISUAL_RESIDUAL_ADAPTER_COUNT config propagation failed: "
            f"requested={config.VISUAL_RESIDUAL_ADAPTER_COUNT} "
            f"actual={visual.visual_residual_adapter_count}"
        )
    propagation_checks = {
        "MMRL_RELATION_MAX_TOKENS": (
            config.MMRL_RELATION_MAX_TOKENS,
            visual.mmrl_relation_max_tokens,
        ),
        "MMRL_VARIANCE_FLOOR_RATIO": (
            config.MMRL_VARIANCE_FLOOR_RATIO,
            visual.mmrl_variance_floor_ratio,
        ),
        "MMRL_VARIANCE_FLOOR_WEIGHT": (
            config.MMRL_VARIANCE_FLOOR_WEIGHT,
            visual.mmrl_variance_floor_weight,
        ),
    }
    for name, (requested, actual) in propagation_checks.items():
        if abs(float(requested) - float(actual)) > 1e-9:
            raise RuntimeError(
                f"{name} config propagation failed: requested={requested} actual={actual}"
            )
    print(
        "[MMRL_STRUCTURE_AUDIT] "
        f"direct_mmrl_output={visual.direct_mmrl_output} "
        f"raw_visual_adapter={visual.raw_visual_adapter} "
        f"visual_residual_adapter_count={visual.visual_residual_adapter_count}"
    )
    model.model.load_state_dict(base.model.state_dict(), strict=False)
    model.lm_head.load_state_dict(base.lm_head.state_dict(), strict=False)
    del base
    torch.cuda.empty_cache()
    print("MMRL model built and loaded with base weights.")
    insert_layers = list(model.model.visual.insert_layers)
    print(
        "[MMRL_REP_INIT_AUDIT] "
        f"blocks={len(model.model.visual.blocks)} "
        f"blocks_with_rep={len(model.model.visual.blocks_with_rep)} "
        f"insert_layers_0based={insert_layers} "
        f"insert_layers_natural={[idx + 1 for idx in insert_layers]}"
    )
    if len(model.model.visual.blocks_with_rep) != len(insert_layers):
        raise ValueError(
            "blocks_with_rep count does not match insert layer count: "
            f"{len(model.model.visual.blocks_with_rep)} != {len(insert_layers)}"
        )
    with torch.no_grad():
        for idx, layer_num in enumerate(insert_layers):
            print(
                "[MMRL_REP_INIT] "
                f"rep_idx={idx} <- visual.blocks[{layer_num}] "
                f"natural_layer={layer_num + 1}"
            )
            model.model.visual.blocks_with_rep[idx].load_state_dict(
                model.model.visual.blocks[layer_num].state_dict(),
                strict=False
            )
    processor = processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor, tokenizer=tokenizer, cfg=cfg
    )
    print(
        f"ablate_visual_gate={config.ABLATE_VISUAL_GATE} "
        f"ablate_direct_learnable_rep={config.ABLATE_DIRECT_LEARNABLE_REP} "
        f"visual_residual_adapter_count={config.VISUAL_RESIDUAL_ADAPTER_COUNT} "
        f"adapter_usage_balance_loss_weight={config.ADAPTER_USAGE_BALANCE_LOSS_WEIGHT} "
        f"adapter_sample_entropy_loss_weight={config.ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT} "
        f"adapter_common_mode_loss_weight={config.ADAPTER_COMMON_MODE_LOSS_WEIGHT} "
        f"adapter_effective_delta_loss_weight={config.ADAPTER_EFFECTIVE_DELTA_LOSS_WEIGHT} "
        f"mmrl_relation_loss_weight={config.MMRL_RELATION_LOSS_WEIGHT} "
        f"mmrl_variance_floor_ratio={config.MMRL_VARIANCE_FLOOR_RATIO} "
        f"mmrl_variance_floor_weight={config.MMRL_VARIANCE_FLOOR_WEIGHT} "
        f"adapter_sample_entropy_target={config.ADAPTER_SAMPLE_ENTROPY_TARGET} "
        f"adapter_common_mode_target={config.ADAPTER_COMMON_MODE_TARGET} "
        f"adapter_effective_delta_target_low={config.ADAPTER_EFFECTIVE_DELTA_TARGET_LOW} "
        f"adapter_effective_delta_target_high={config.ADAPTER_EFFECTIVE_DELTA_TARGET_HIGH} "
        f"adapter_diversity_loss_weight={config.ADAPTER_DIVERSITY_LOSS_WEIGHT} "
        f"adapter_diversity_target_low={config.ADAPTER_DIVERSITY_TARGET_LOW} "
        f"adapter_diversity_target_high={config.ADAPTER_DIVERSITY_TARGET_HIGH} "
        f"adapter_diversity_upper_weight={config.ADAPTER_DIVERSITY_UPPER_WEIGHT} "
        f"adapter_diversity_worst_pair_weight={config.ADAPTER_DIVERSITY_WORST_PAIR_WEIGHT} "
        f"enable_adapter_router_identity_residual={config.ENABLE_ADAPTER_ROUTER_IDENTITY_RESIDUAL} "
        f"direct_mmrl_output={config.DIRECT_MMRL_OUTPUT} "
        f"raw_visual_adapter={config.RAW_VISUAL_ADAPTER} "
        f"enable_deepstack_mmrl_residual={config.ENABLE_DEEPSTACK_MMRL_RESIDUAL} "
        f"deepstack_mmrl_residual_scale={config.DEEPSTACK_MMRL_RESIDUAL_SCALE}"
    )
    print("Processor built.")
    return model, processor


# =========================
#  Freeze policy
# =========================
def set_trainable_stage(model, stage, train_cfg=None):
    for p in model.parameters():
        p.requires_grad = False

    v = model.model.visual
    if stage == 1:
        mods = [v.hidden_state_pooling, v.embedding_pooling, v.Task_classifier]
    elif stage == 3:
        mods = [v.hidden_state_pooling, v.embedding_pooling]
        if not v.raw_visual_adapter:
            mods.insert(0, model.model.MMRL)
            # v.blocks_with_rep stay frozen while the rep-token branch remains active.
        if not v.direct_mmrl_output:
            mods.extend([v.adapter_router, v.residual_adapters])
    else:
        raise ValueError("stage must be 1 or 3")

    for m in mods:
        if isinstance(m, nn.Parameter):
            m.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = True


def print_joint_trainable_params_before_stage1(model):
    set_trainable_stage(model, 3)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in trainable)

    print("\n" + "!" * 120)
    print(f"[JOINT-TRAINABLE][TOTAL] tensors={len(trainable)} total_numel={total}")
    print("!" * 120 + "\n")


# =========================
#  Stage1 lightweight classifier pretraining
# =========================
def _build_text_pool_mask(attention_mask):
    return attention_mask.bool()


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


def run_stage1_light(model, processor, data_cfg, train_cfg, output_dir):
    stage_id = 1
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
        seed=train_cfg.get("data_sampling_seed", 42),
        deterministic_sampling=train_cfg.get("deterministic_sampling", False),
    )
    collator = MMRLDataCollator(processor)
    data_generator = torch.Generator()
    data_generator.manual_seed(int(train_cfg.get("data_order_seed", 42)) + int(stage_id))

    dl = DataLoader(
        ds,
        batch_size=train_cfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("dataloader_num_workers", 4),
        pin_memory=False,
        collate_fn=collator,
        drop_last=True,
        generator=data_generator,
    )

    # 仅优化可训练参数
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg["learning_rate"][stage_id], weight_decay=0.01)

    steps_per_update = train_cfg["gradient_accumulation_steps"]
    epochs = train_cfg["epochs"][stage_id]
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
            text_pool_mask = _build_text_pool_mask(attention_mask)
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

            cls_loss = focal_bce_with_logits(alpha_logits, alpha_labels)
            loss = cls_loss / steps_per_update
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
                        total_loss=cls_loss.detach(),
                        cls_loss=cls_loss.detach(),
                        alpha_mae=alpha_mae,
                        learning_rate=opt.param_groups[0]["lr"],
                    )
                print_every = train_cfg.get("console_log_every", 50)
                if global_step % print_every == 0:
                    row = {
                        "total_loss": cls_loss.detach(),
                        "cls_loss": cls_loss.detach(),
                        "alpha_mae": alpha_mae,
                        "learning_rate": opt.param_groups[0]["lr"],
                    }
                    print_stage_step_summary(f"stage{stage_id}", global_step, row)
    metric_logger.finalize()

    # save_path = f"{output_dir}/stage{stage_id}"
    # model.save_pretrained(save_path)
    # print(f"[Stage{stage_id}] saved to {save_path}")


# =========================
#  Stage3 joint adaptation
# =========================
def run_stage3_full(model, processor, data_cfg, train_cfg, output_dir):
    stage_id = 3
    set_trainable_stage(model, stage_id)
    actual_epochs = int(train_cfg["epochs"][stage_id])
    schedule_epochs = int(
        train_cfg.get("schedule_epochs", {}).get(stage_id, actual_epochs)
    )
    if actual_epochs < 1 or schedule_epochs < actual_epochs:
        raise ValueError(
            "Stage3 requires 1 <= actual epochs <= schedule epochs, got "
            f"actual={actual_epochs}, schedule={schedule_epochs}"
        )
    print(
        f"[Stage3] actual_epochs={actual_epochs} "
        f"schedule_epochs={schedule_epochs}"
    )

    model.ce_loss_weight = 1.0
    model.alpha_loss_weight = 0.0
    model.enable_alpha_guide_loss = False

    model.current_stage_id = int(stage_id)
    model.current_stage_step = 0
    model.current_stage_progress = 0.0
    model.adapter_usage_balance_loss_weight = float(train_cfg.get("adapter_usage_balance_loss_weight", 0.0))
    model.adapter_sample_entropy_loss_weight = float(train_cfg.get("adapter_sample_entropy_loss_weight", 0.0))
    model.adapter_common_mode_loss_weight = float(train_cfg.get("adapter_common_mode_loss_weight", 0.0))
    experiment_cfg = train_cfg.get("experiment_cfg", {}) or {}
    model.adapter_effective_delta_loss_weight = float(train_cfg.get(
        "adapter_effective_delta_loss_weight",
        experiment_cfg.get("adapter_effective_delta_loss_weight", 0.0),
    ))
    model.mmrl_relation_loss_weight = float(train_cfg.get(
        "mmrl_relation_loss_weight",
        experiment_cfg.get("mmrl_relation_loss_weight", 0.0),
    ))
    print(
        "[ROUTE_REGULARIZATION] "
        f"usage_weight_start={model.adapter_usage_balance_loss_weight} "
        f"entropy_weight_start={model.adapter_sample_entropy_loss_weight}"
    )
    print(
        "[MMRL_RELATION] "
        f"weight={model.mmrl_relation_loss_weight} "
        f"max_tokens={model.model.visual.mmrl_relation_max_tokens} "
        f"variance_floor_ratio={model.model.visual.mmrl_variance_floor_ratio} "
        f"variance_floor_weight={model.model.visual.mmrl_variance_floor_weight}"
    )
    model.adapter_diversity_loss_weight = float(train_cfg.get(
        "adapter_diversity_loss_weight",
        experiment_cfg.get("adapter_diversity_loss_weight", 0.0),
    ))

    stage3_views = ("expert-mm",)
    print(f"[Stage{stage_id}] enable_views={stage3_views}")

    ds = FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=data_cfg["total_limit"],
        enable_views=stage3_views,
        mode=f"stage{stage_id}",
        ce_enabled=True,
        seed=train_cfg.get("data_sampling_seed", 42),
        deterministic_sampling=train_cfg.get("deterministic_sampling", False),
    )
    collator = MMRLDataCollator(processor)
    cb = StageScheduleCallback(
        dataset=ds,
        total_epochs=schedule_epochs,
        init_temp=train_cfg.get("initial_temp", 1.0),
        final_temp=train_cfg.get("final_temp", 0.1),
    )
    args = TrainingArguments(
        output_dir=f"{output_dir}/stage{stage_id}",
        num_train_epochs=schedule_epochs,
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
        seed=train_cfg["seed"],
        data_seed=int(train_cfg.get("data_order_seed", 42)) + int(stage_id),
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
        keep_keys=train_cfg.get("mmrl_diagnostics_keep_keys"),
    )
    grad_monitor_cb = SharedRepGradMonitorCallback(
        save_dir=f"{output_dir}/stage{stage_id}/metrics",
        stage_name=f"stage{stage_id}",
        ema_alpha=train_cfg.get("grad_spike_ema_alpha", 0.10),
        ratio_threshold=train_cfg.get("grad_spike_ratio_threshold", 3.0),
        abs_threshold=train_cfg.get("grad_spike_abs_threshold", 0.5),
        cooldown_steps=train_cfg.get("grad_spike_cooldown_steps", 20),
    )

    callbacks = [
        cb,
        StopAfterEpochCallback(actual_epochs),
        grad_monitor_cb,
        metrics_cb,
        diag_cb,
    ]
    if train_cfg.get("eval_each_epoch", False):
        callbacks.append(EpochEvaluationCallback(
            processor,
            output_dir,
            stage_id,
            actual_epochs,
        ))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    # trainer.save_model(f"{output_dir}/stage{stage_id}")


def run_stage(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    if stage_id == 1:
        _write_experiment_manifest(output_dir, train_cfg)
        print_joint_trainable_params_before_stage1(model)
        run_stage1_light(model, processor, data_cfg, train_cfg, output_dir)
    elif stage_id == 3:
        run_stage3_full(model, processor, data_cfg, train_cfg, output_dir)
    else:
        raise ValueError("stage_id must be 1 or 3")
