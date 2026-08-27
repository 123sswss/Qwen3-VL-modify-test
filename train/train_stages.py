# train_stages.py
import os
import csv
import math
from contextlib import contextmanager, nullcontext
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
from mmrl_checkpoint import save_mmrl_checkpoint

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
            "stage3_per_device_train_batch_size": train_cfg.get(
                "stage3_per_device_train_batch_size"
            ),
            "stage3_gradient_accumulation_steps": train_cfg.get(
                "stage3_gradient_accumulation_steps"
            ),
            "stage3_dataloader_num_workers": train_cfg.get(
                "stage3_dataloader_num_workers"
            ),
            "stage3_dataloader_pin_memory": train_cfg.get(
                "stage3_dataloader_pin_memory"
            ),
            "learning_rate": train_cfg.get("learning_rate", {}).get(stage_id),
            "epochs": train_cfg.get("epochs", {}).get(stage_id),
            "console_log_every": train_cfg.get("console_log_every"),
            "metric_smooth_window": train_cfg.get("metric_smooth_window"),
            "metric_ema_alpha": train_cfg.get("metric_ema_alpha"),
            "metric_scatter_stride": train_cfg.get("metric_scatter_stride"),
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
        resample_on_first_epoch=True,
        total_steps=None,
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.resample_on_first_epoch = bool(resample_on_first_epoch)
        self.total_steps = int(total_steps) if total_steps is not None else None

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self.resample_on_first_epoch and float(state.epoch or 0.0) == 0.0:
            return
        self.dataset.resample_data()

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]

        if self.total_steps is not None:
            prog = min(float(state.global_step) / max(float(self.total_steps), 1.0), 1.0)
        else:
            ep = state.epoch if state.epoch is not None else 0.0
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


class EpochCompactCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        output_dir,
        stage_id,
        base_model_path,
        metadata=None,
    ):
        self.processor = processor
        self.output_dir = output_dir
        self.stage_id = int(stage_id)
        self.base_model_path = base_model_path
        self.metadata = dict(metadata or {})
        self.completed_epochs = set()

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        epoch_id = max(1, int(round(float(state.epoch or 0.0))))
        if epoch_id in self.completed_epochs:
            return control
        self.completed_epochs.add(epoch_id)
        checkpoint_dir = os.path.join(
            self.output_dir,
            "checkpoints",
            f"stage{self.stage_id}_epoch_{epoch_id}",
        )
        save_mmrl_checkpoint(
            kwargs["model"],
            self.processor,
            checkpoint_dir,
            self.base_model_path,
            metadata={
                **self.metadata,
                "stage": self.stage_id,
                "epoch": epoch_id,
                "global_step": int(state.global_step),
            },
        )
        print(
            f"[EPOCH-CHECKPOINT] stage={self.stage_id} epoch={epoch_id} "
            f"global_step={int(state.global_step)} saved={checkpoint_dir}"
        )
        return control


def _parse_step_evaluation_steps(value):
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        value = value.replace(";", ",").split(",")
    steps = []
    for item in value:
        step = int(str(item).strip())
        if step <= 0:
            raise ValueError(
                f"intermediate evaluation steps must be positive, got {step}"
            )
        steps.append(step)
    return tuple(sorted(set(steps)))


class StepEvaluationCallback(TrainerCallback):
    def __init__(self, processor, output_dir, stage_id, steps):
        self.processor = processor
        self.output_dir = output_dir
        self.stage_id = int(stage_id)
        self.steps = tuple(steps)
        self.completed_steps = set()

    def on_train_begin(self, args, state, control, **kwargs):
        print(
            f"[STEP-EVAL] stage={self.stage_id} scheduled_steps={self.steps}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        step = int(state.global_step)
        if step not in self.steps or step in self.completed_steps:
            return control

        self.completed_steps.add(step)
        eval_dir = os.path.join(
            self.output_dir,
            "eval_steps",
            f"stage{self.stage_id}_step{step}",
        )
        log_path = os.path.join(eval_dir, "test.log")
        print(
            f"[STEP-EVAL] stage={self.stage_id} step={step} begin "
            f"log_path={log_path}"
        )
        summary = run_live_epoch_evaluation(
            kwargs["model"],
            self.processor,
            log_path,
        )
        summary_path = os.path.join(eval_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        print(
            f"[STEP-EVAL] stage={self.stage_id} step={step} "
            f"score={summary.get('score')} completed summary={summary_path}"
        )
        return control


class MMRLDiagnosticsCallback(TrainerCallback):
    DEFAULT_KEEP_KEYS = {
        "ce_loss",
        "delta_pool_common_mode_ratio",
        "delta_pool_specificity_ratio",
    }

    def __init__(self, save_dir, every_steps=1000, stage_name="stage", keep_keys=None):
        self.save_dir = save_dir
        self.every_steps = int(max(1, every_steps))
        self.stage_name = stage_name
        self.keep_keys = set(keep_keys or self.DEFAULT_KEEP_KEYS)
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
        self.model.MMRL.synchronize_layer_projector_initialization()
        self.soft_prompt_length = int(getattr(config, "MMRL_SOFT_PROMPT_LENGTH", 0))
        self.soft_prompt = None
        self._soft_prompt_inputs_preexpanded = False
        if self.soft_prompt_length > 0:
            initial_prompt = self._sample_soft_prompt_from_embeddings()
            self.soft_prompt = nn.Parameter(initial_prompt.float())
            self.soft_prompt.register_hook(self._soft_prompt_grad_hook)
        # 补充 generation_config，避免 save_pretrained 时报错
        from transformers import GenerationConfig
        self.generation_config = GenerationConfig.from_model_config(config)

        self.ce_loss_weight = 1.0
        self.alpha_loss_weight = 0.0
        self.current_stage_id = 0
        self.current_stage_step = 0
        self.current_stage_progress = 0.0
        self.enable_alpha_guide_loss = False
        self.mmrl_relation_loss_weight = 0.0

        self.temperature_override = None

        self.debug_mode = True
        self.debug_loss_threshold = 20.0
        self._last_metrics = {}
        self._last_shared_rep_grad = {}
        self._shared_rep_grad_hook_handle = None
        self._shared_rep_grad_hook_param_id = None
        self._validated_mmrl_supervision_contract = False
        self._mmrl_grad_squared = 0.0
        self._mmrl_grad_hook_handles = []
        self._register_mmrl_grad_hooks()
        self._ensure_shared_rep_grad_hook()

    def _get_mmrl_module(self):
        return getattr(self.model, "MMRL", None)

    def _sample_soft_prompt_from_embeddings(self):
        embeddings = self.get_input_embeddings().weight.detach()
        generator = torch.Generator(device="cpu").manual_seed(
            int(getattr(self.config, "MMRL_SOFT_PROMPT_INIT_SEED", 44))
        )
        sampled_rows = torch.randint(
            embeddings.shape[0],
            (self.soft_prompt_length,),
            generator=generator,
        )
        return embeddings[sampled_rows.to(embeddings.device)].clone()

    def reset_soft_prompt_from_base_embeddings(self):
        if self.soft_prompt is None:
            return
        initial_prompt = self._sample_soft_prompt_from_embeddings()
        self.soft_prompt.data.copy_(
            initial_prompt.to(
                device=self.soft_prompt.device,
                dtype=self.soft_prompt.dtype,
            )
        )

    def _soft_prompt_grad_hook(self, grad):
        grad_norm = grad.detach().float().norm()
        if isinstance(getattr(self, "_last_metrics", None), dict):
            self._last_metrics["soft_prompt_grad_norm"] = grad_norm
        return grad

    def _register_mmrl_grad_hooks(self):
        if self._mmrl_grad_hook_handles:
            return

        def accumulate(grad):
            squared = float(grad.detach().float().pow(2).sum().item())
            self._mmrl_grad_squared += squared
            if isinstance(getattr(self, "_last_metrics", None), dict):
                self._last_metrics["mmrl_grad_norm"] = torch.tensor(
                    math.sqrt(self._mmrl_grad_squared),
                    device=grad.device,
                )
            return grad

        for parameter in self.model.MMRL.parameters():
            self._mmrl_grad_hook_handles.append(parameter.register_hook(accumulate))

    def _expand_soft_prompt_inputs(self, kwargs):
        if self.soft_prompt is None:
            return kwargs
        expanded = dict(kwargs)
        input_ids = expanded.pop("input_ids")
        attention_mask = expanded.pop("attention_mask")
        batch_size = input_ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        prompt_ids = torch.full(
            (batch_size, self.soft_prompt_length),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        prompt_attention = torch.ones(
            (batch_size, self.soft_prompt_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        expanded["input_ids"] = torch.cat((prompt_ids, input_ids), dim=1)
        expanded["attention_mask"] = torch.cat(
            (prompt_attention, attention_mask), dim=1
        )
        labels = expanded.get("labels")
        if labels is not None:
            ignored = torch.full(
                (batch_size, self.soft_prompt_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            expanded["labels"] = torch.cat((ignored, labels), dim=1)
        gating_mask = expanded.get("mmrl_gating_mask")
        if gating_mask is None:
            gating_mask = attention_mask
        excluded_prompt = torch.zeros(
            (batch_size, self.soft_prompt_length),
            dtype=gating_mask.dtype,
            device=gating_mask.device,
        )
        expanded["mmrl_gating_mask"] = torch.cat(
            (excluded_prompt, gating_mask), dim=1
        )
        return expanded

    @contextmanager
    def _inject_soft_prompt_embeddings(self):
        if self.soft_prompt is None:
            yield
            return
        embeddings = self.get_input_embeddings()

        def replace_prompt(_module, _inputs, output):
            if output.ndim != 3 or output.shape[1] < self.soft_prompt_length:
                return output
            prompt = self.soft_prompt.to(
                device=output.device, dtype=output.dtype
            ).unsqueeze(0).expand(output.shape[0], -1, -1)
            return torch.cat((prompt, output[:, self.soft_prompt_length:]), dim=1)

        handle = embeddings.register_forward_hook(replace_prompt)
        try:
            yield
        finally:
            handle.remove()

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

        layer_embeddings = getattr(mmrl, "layer_embeddings", None)
        if layer_embeddings is not None:
            result["layer_embeddings_norm"] = (
                layer_embeddings.detach().float().norm()
            )
            if layer_embeddings.grad is not None:
                result["layer_embeddings_grad_norm"] = (
                    layer_embeddings.grad.detach().float().norm()
                )

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
        _grad_stats(
            getattr(mmrl, "visual_memory_pooling", None),
            "visual_memory_pooling",
        )
        _grad_stats(
            getattr(mmrl, "text_memory_pooling", None),
            "text_memory_pooling",
        )
        _grad_stats(
            getattr(mmrl, "cross_attention", None),
            "cross_attention",
        )
        cross_attention = getattr(mmrl, "cross_attention", None)
        output_projection = getattr(cross_attention, "output_projection", None)
        if output_projection is not None:
            result["cross_attention_output_weight_norm"] = (
                output_projection.weight.detach().float().norm()
            )
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
        if self.soft_prompt is not None and not self._soft_prompt_inputs_preexpanded:
            expanded = self._expand_soft_prompt_inputs({
                **kwargs,
                "input_ids": input_ids,
            })
            input_ids = expanded.pop("input_ids")
            kwargs = expanded
        self._last_shared_rep_grad = {}
        self._mmrl_grad_squared = 0.0
        self._ensure_shared_rep_grad_hook()
        self.model.visual.current_stage_id = int(self.current_stage_id)
        self.model.visual.current_stage_step = int(self.current_stage_step)
        if hasattr(self, "temperature_override") and self.temperature_override is not None:
            self.model.temperature_override = self.temperature_override
            kwargs["gating_temperature_override"] = self.temperature_override
        labels = kwargs.get("labels", None)
        if labels is not None:
            att_mask = kwargs.get("attention_mask")
            mmrl_gating_mask = kwargs.get("mmrl_gating_mask")
            if att_mask is None or mmrl_gating_mask is None:
                raise RuntimeError(
                    "Stage3 CE batches must provide attention_mask and an explicit "
                    "mmrl_gating_mask"
                )
            if (
                labels.dim() != 2
                or att_mask.shape != labels.shape
                or mmrl_gating_mask.shape != labels.shape
            ):
                raise RuntimeError(
                    "labels, attention_mask, and mmrl_gating_mask must share [B, S], "
                    f"got labels={tuple(labels.shape)} "
                    f"attention={tuple(att_mask.shape)} "
                    f"pooling={tuple(mmrl_gating_mask.shape)}"
                )

            pooling_bool = mmrl_gating_mask.bool()
            attention_bool = att_mask.bool()
            if not self._validated_mmrl_supervision_contract:
                if bool((pooling_bool & ~attention_bool).any()):
                    raise RuntimeError("mmrl_gating_mask must not include padding tokens")
                if bool((pooling_bool & (labels != -100)).any()):
                    raise RuntimeError(
                        "MMRL pooling context must not overlap supervised assistant tokens"
                    )
                if bool(((labels != -100) & ~attention_bool).any()):
                    raise RuntimeError("labels must not supervise padding tokens")
                self._validated_mmrl_supervision_contract = True

            kwargs["mmrl_gating_mask"] = pooling_bool.to(dtype=self.model.dtype)
        
        prompt_context = (
            nullcontext()
            if self._soft_prompt_inputs_preexpanded
            else self._inject_soft_prompt_embeddings()
        )
        with prompt_context:
            outputs = super().forward(
                input_ids=input_ids,
                images_per_sample=images_per_sample,
                **kwargs,
            )
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

        mmrl_relation_loss = getattr(self.model.visual, "mmrl_relation_loss", torch.tensor(0.0, device=input_ids.device))
        if not torch.is_tensor(mmrl_relation_loss):
            mmrl_relation_loss = torch.tensor(mmrl_relation_loss, device=input_ids.device, dtype=ce_loss.dtype)
        else:
            mmrl_relation_loss = mmrl_relation_loss.to(device=input_ids.device, dtype=ce_loss.dtype)
        scaled_mmrl_relation_loss = mmrl_relation_loss * float(self.mmrl_relation_loss_weight)
        outputs.loss = (
            self.ce_loss_weight * ce_loss
            + alpha_guide_loss
            + scaled_mmrl_relation_loss
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
                "mmrl_relation_loss": mmrl_relation_loss.detach(),
                "mmrl_relation_loss_scaled": scaled_mmrl_relation_loss.detach(),
                "alpha_mae": alpha_mae.detach(),
                "temperature": torch.tensor(float(self.temperature_override) if self.temperature_override is not None else float("nan"), device=input_ids.device),
                "stage_progress": torch.tensor(float(getattr(self, "current_stage_progress", 0.0)), device=input_ids.device),
                "mmrl_grad_norm": torch.tensor(0.0, device=input_ids.device),
            }
            visual_dbg = getattr(self.model.visual, "debug_context", {}) or {}
            for key, value in visual_dbg.items():
                self._last_metrics[key] = value
            model_dbg = getattr(self.model, "debug_context", {}) or {}
            for key, value in model_dbg.items():
                self._last_metrics[key] = value
            mmrl_dbg = getattr(self.model.MMRL, "debug_context", {}) or {}
            for key, value in mmrl_dbg.items():
                self._last_metrics[key] = value
            for key, value in self._collect_rep_grad_metrics(input_ids.device).items():
                self._last_metrics[key] = value
            if self.soft_prompt is not None:
                self._last_metrics["soft_prompt_norm"] = (
                    self.soft_prompt.detach().float().norm()
                )
                prompt_grad = self.soft_prompt.grad
                self._last_metrics["soft_prompt_grad_norm"] = (
                    prompt_grad.detach().float().norm()
                    if prompt_grad is not None
                    else torch.tensor(0.0, device=input_ids.device)
                )

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

    def generate(self, *args, **kwargs):
        if self.soft_prompt is None:
            return super().generate(*args, **kwargs)
        if args:
            if len(args) != 1 or "input_ids" in kwargs:
                raise TypeError("Soft Prompt generation accepts one positional input_ids")
            kwargs["input_ids"] = args[0]
        expanded = self._expand_soft_prompt_inputs(kwargs)
        self._soft_prompt_inputs_preexpanded = True
        try:
            with self._inject_soft_prompt_embeddings():
                return super().generate(**expanded)
        finally:
            self._soft_prompt_inputs_preexpanded = False


def build_model_and_processor(model_path, experiment_cfg=None):
    experiment_cfg = experiment_cfg or {}
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.RP_SPACE_LENGTH = int(os.getenv(
        "MMRL_RP_SPACE_LENGTH",
        str(experiment_cfg.get("rp_space_length", 40)),
    ))
    config.MMRL_MEMORY_QUERY_COUNT = int(os.getenv(
        "MMRL_MEMORY_QUERY_COUNT",
        str(experiment_cfg.get("memory_query_count", 128)),
    ))
    config.MMRL_MEMORY_ATTENTION_DIM = int(os.getenv(
        "MMRL_MEMORY_ATTENTION_DIM",
        str(experiment_cfg.get("memory_attention_dim", 128)),
    ))
    config.MMRL_PROJECTOR_HIDDEN_DIM = int(os.getenv(
        "MMRL_PROJECTOR_HIDDEN_DIM",
        str(experiment_cfg.get(
            "projector_hidden_dim",
            config.vision_config.hidden_size,
        )),
    ))
    config.MMRL_CROSS_ATTENTION_HEADS = int(os.getenv(
        "MMRL_CROSS_ATTENTION_HEADS",
        str(experiment_cfg.get("cross_attention_heads", 8)),
    ))
    config.MMRL_QUERY_ARCHITECTURE = os.getenv(
        "MMRL_QUERY_ARCHITECTURE",
        str(experiment_cfg.get("query_architecture", "layer_mlp_post_cross")),
    )
    config.MMRL_SAME_INIT_LAYER_PROJECTORS = os.getenv(
        "MMRL_SAME_INIT_LAYER_PROJECTORS",
        (
            "1"
            if experiment_cfg.get("same_init_layer_projectors", True)
            else "0"
        ),
    ) == "1"
    config.MMRL_USE_DYNAMIC_CROSS_ATTENTION = os.getenv(
        "MMRL_USE_DYNAMIC_CROSS_ATTENTION",
        (
            "1"
            if experiment_cfg.get("use_dynamic_cross_attention", True)
            else "0"
        ),
    ) == "1"
    config.MMRL_MEMORY_POOLING_MODE = os.getenv(
        "MMRL_MEMORY_POOLING_MODE",
        str(experiment_cfg.get("memory_pooling_mode", "multi_query")),
    )
    config.MMRL_FUSION_MODE = os.getenv(
        "MMRL_FUSION_MODE",
        str(experiment_cfg.get("fusion_mode", "cross_attention")),
    )
    config.MMRL_RELATION_LOSS_WEIGHT = float(experiment_cfg.get(
        "mmrl_relation_loss_weight",
        os.getenv("MMRL_RELATION_LOSS_WEIGHT", "0.0"),
    ))
    config.MMRL_SOFT_PROMPT_LENGTH = int(experiment_cfg.get(
        "soft_prompt_length", 0
    ))
    config.MMRL_SOFT_PROMPT_INIT_SEED = int(experiment_cfg.get(
        "soft_prompt_init_seed", 44
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print("Base model loaded. ")
    print("Tokenizer loaded. Now building MMRL model...")
    model = Qwen3VLMMRLForStages(config, tokenizer).to("cuda").to(torch.bfloat16)
    if model.soft_prompt is not None:
        model.soft_prompt.data = model.soft_prompt.data.float()
    model._ensure_shared_rep_grad_hook()
    visual = model.model.visual
    mmrl = model.model.MMRL
    propagation_checks = {
        "MMRL_RELATION_MAX_TOKENS": (
            config.MMRL_RELATION_MAX_TOKENS,
            visual.mmrl_relation_max_tokens,
        ),
        "MMRL_SAME_INIT_LAYER_PROJECTORS": (
            config.MMRL_SAME_INIT_LAYER_PROJECTORS,
            mmrl.same_init_layer_projectors,
        ),
        "MMRL_USE_DYNAMIC_CROSS_ATTENTION": (
            config.MMRL_USE_DYNAMIC_CROSS_ATTENTION,
            mmrl.use_dynamic_cross_attention,
        ),
        "MMRL_VARIANCE_FLOOR_RATIO": (
            config.MMRL_VARIANCE_FLOOR_RATIO,
            visual.mmrl_variance_floor_ratio,
        ),
        "MMRL_VARIANCE_FLOOR_WEIGHT": (
            config.MMRL_VARIANCE_FLOOR_WEIGHT,
            visual.mmrl_variance_floor_weight,
        ),
        "MMRL_MEMORY_QUERY_COUNT": (
            config.MMRL_MEMORY_QUERY_COUNT,
            mmrl.memory_query_count,
        ),
        "MMRL_MEMORY_ATTENTION_DIM": (
            config.MMRL_MEMORY_ATTENTION_DIM,
            mmrl.memory_attention_dim,
        ),
        "MMRL_PROJECTOR_HIDDEN_DIM": (
            config.MMRL_PROJECTOR_HIDDEN_DIM,
            mmrl.projector_hidden_dim,
        ),
    }
    if config.MMRL_FUSION_MODE == "cross_attention":
        propagation_checks["MMRL_CROSS_ATTENTION_HEADS"] = (
            config.MMRL_CROSS_ATTENTION_HEADS,
            mmrl.cross_attention.num_heads,
        )
    for name, (requested, actual) in propagation_checks.items():
        if abs(float(requested) - float(actual)) > 1e-9:
            raise RuntimeError(
                f"{name} config propagation failed: requested={requested} actual={actual}"
            )
    if config.MMRL_MEMORY_POOLING_MODE != mmrl.memory_pooling_mode:
        raise RuntimeError(
            "MMRL_MEMORY_POOLING_MODE config propagation failed: "
            f"requested={config.MMRL_MEMORY_POOLING_MODE} "
            f"actual={mmrl.memory_pooling_mode}"
        )
    if config.MMRL_FUSION_MODE != mmrl.fusion_mode:
        raise RuntimeError(
            "MMRL_FUSION_MODE config propagation failed: "
            f"requested={config.MMRL_FUSION_MODE} actual={mmrl.fusion_mode}"
        )
    if config.MMRL_QUERY_ARCHITECTURE != mmrl.query_architecture:
        raise RuntimeError(
            "MMRL_QUERY_ARCHITECTURE config propagation failed: "
            f"requested={config.MMRL_QUERY_ARCHITECTURE} "
            f"actual={mmrl.query_architecture}"
        )
    memory_tokens_per_image = 0
    if config.MMRL_USE_DYNAMIC_CROSS_ATTENTION:
        if config.MMRL_MEMORY_POOLING_MODE == "multi_query":
            memory_tokens_per_image = 2 * config.MMRL_MEMORY_QUERY_COUNT
        elif config.MMRL_MEMORY_POOLING_MODE == "text_guided":
            memory_tokens_per_image = config.MMRL_MEMORY_QUERY_COUNT
        else:
            memory_tokens_per_image = 2
    print(
        "[MMRL_STRUCTURE_AUDIT] "
        f"query_architecture={config.MMRL_QUERY_ARCHITECTURE} "
        f"rp_space_length={config.RP_SPACE_LENGTH} "
        f"memory_query_count={config.MMRL_MEMORY_QUERY_COUNT} "
        f"memory_attention_dim={config.MMRL_MEMORY_ATTENTION_DIM} "
        f"projector_hidden_dim={config.MMRL_PROJECTOR_HIDDEN_DIM} "
        f"cross_attention_heads={config.MMRL_CROSS_ATTENTION_HEADS} "
        f"same_init_layer_projectors={config.MMRL_SAME_INIT_LAYER_PROJECTORS} "
        "dynamic_cross_attention="
        f"{config.MMRL_USE_DYNAMIC_CROSS_ATTENTION} "
        f"memory_pooling_mode={config.MMRL_MEMORY_POOLING_MODE} "
        f"fusion_mode={config.MMRL_FUSION_MODE} "
        "train_gate_mode=open_full_ce relation_mode=uniform "
        "memory_tokens_per_image="
        f"{memory_tokens_per_image}"
    )
    component_parameters = {
        "shared_rep": model.model.MMRL.shared_represent_space.numel(),
        "layer_embeddings": model.model.MMRL.layer_embeddings.numel(),
        "layer_projectors": sum(
            parameter.numel()
            for parameter in model.model.MMRL.v_r_token_projector.parameters()
        ),
        "visual_memory_pooling": sum(
            parameter.numel()
            for parameter in model.model.MMRL.visual_memory_pooling.parameters()
        ),
        "text_memory_pooling": sum(
            parameter.numel()
            for parameter in (
                model.model.MMRL.text_memory_pooling.parameters()
                if model.model.MMRL.text_memory_pooling is not None
                else []
            )
        ),
        "cross_attention": sum(
            parameter.numel()
            for parameter in model.model.MMRL.cross_attention.parameters()
        ),
    }
    print(
        "[MMRL_PARAMETER_AUDIT] "
        f"components={component_parameters} "
        f"total={sum(parameter.numel() for parameter in model.model.MMRL.parameters())}"
    )
    expected_mmrl_parameters = experiment_cfg.get("expected_mmrl_parameters")
    if expected_mmrl_parameters is not None:
        actual_mmrl_parameters = sum(
            parameter.numel() for parameter in model.model.MMRL.parameters()
        )
        if actual_mmrl_parameters != int(expected_mmrl_parameters):
            raise RuntimeError(
                "MMRL parameter count audit failed: "
                f"expected={int(expected_mmrl_parameters)} "
                f"actual={actual_mmrl_parameters} components={component_parameters}"
            )
        print(
            "[MMRL_EXPECTED_PARAMETER_AUDIT] "
            f"expected={int(expected_mmrl_parameters)} "
            f"actual={actual_mmrl_parameters} pass=True"
        )
    model.model.load_state_dict(base.model.state_dict(), strict=False)
    model.lm_head.load_state_dict(base.lm_head.state_dict(), strict=False)
    model.reset_soft_prompt_from_base_embeddings()

    cross_output_norm = float(
        model.model.MMRL.cross_attention.output_projection.weight
        .detach()
        .float()
        .norm()
        .item()
    )
    cross_output_bias_norm = float(
        model.model.MMRL.cross_attention.output_projection.bias
        .detach()
        .float()
        .norm()
        .item()
    )
    if cross_output_norm != 0.0 or cross_output_bias_norm != 0.0:
        raise RuntimeError(
            "Dynamic Rep Cross-Attention output projection must start at zero, "
            f"got weight_norm={cross_output_norm} "
            f"bias_norm={cross_output_bias_norm}"
        )
    print(
        "[MMRL_CROSS_ATTENTION_INIT_AUDIT] "
        f"output_weight_norm={cross_output_norm} "
        f"output_bias_norm={cross_output_bias_norm} residual_identity=True"
    )

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
        f"rp_space_length={config.RP_SPACE_LENGTH} "
        f"memory_query_count={config.MMRL_MEMORY_QUERY_COUNT} "
        f"memory_attention_dim={config.MMRL_MEMORY_ATTENTION_DIM} "
        f"projector_hidden_dim={config.MMRL_PROJECTOR_HIDDEN_DIM} "
        f"cross_attention_heads={config.MMRL_CROSS_ATTENTION_HEADS} "
        f"query_architecture={config.MMRL_QUERY_ARCHITECTURE} "
        "dynamic_cross_attention="
        f"{config.MMRL_USE_DYNAMIC_CROSS_ATTENTION} "
        f"memory_pooling_mode={config.MMRL_MEMORY_POOLING_MODE} "
        f"fusion_mode={config.MMRL_FUSION_MODE} "
        f"mmrl_relation_loss_weight={config.MMRL_RELATION_LOSS_WEIGHT} "
        f"mmrl_variance_floor_ratio={config.MMRL_VARIANCE_FLOOR_RATIO} "
        f"mmrl_variance_floor_weight={config.MMRL_VARIANCE_FLOOR_WEIGHT}"
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
        mods = [model.model.MMRL]
        if getattr(model, "soft_prompt", None) is not None:
            mods.append(model.soft_prompt)
        # v.blocks_with_rep stay frozen while the Rep-Token branch remains active.
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


def _build_stage3_grouped_optimizer(model, train_cfg):
    experiment_cfg = train_cfg.get("experiment_cfg", {}) or {}
    if "stage3_mmrl_learning_rate" not in experiment_cfg:
        return None

    mmrl_lr = float(experiment_cfg.get(
        "stage3_mmrl_learning_rate",
        train_cfg["learning_rate"][3],
    ))
    adam_beta1 = float(experiment_cfg.get("stage3_adam_beta1", 0.9))
    adam_beta2 = float(experiment_cfg.get("stage3_adam_beta2", 0.999))
    adam_epsilon = float(experiment_cfg.get("stage3_adam_epsilon", 1e-8))
    weight_decay = float(experiment_cfg.get("stage3_weight_decay", 0.0))
    if not 0.0 <= adam_beta1 < 1.0 or not 0.0 <= adam_beta2 < 1.0:
        raise ValueError(
            "Stage 3 Adam betas must be in [0, 1), got "
            f"beta1={adam_beta1} beta2={adam_beta2}"
        )
    if adam_epsilon <= 0.0:
        raise ValueError(f"Stage 3 Adam epsilon must be positive, got {adam_epsilon}")
    if weight_decay < 0.0:
        raise ValueError(
            f"Stage 3 weight decay must be non-negative, got {weight_decay}"
        )
    mmrl_parameters = [
        parameter
        for parameter in model.model.MMRL.parameters()
        if parameter.requires_grad
    ]
    prompt_parameter = getattr(model, "soft_prompt", None)
    prompt_parameters = (
        [prompt_parameter]
        if prompt_parameter is not None and prompt_parameter.requires_grad
        else []
    )
    active_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not mmrl_parameters:
        raise RuntimeError("Stage 3 MMRL optimizer group is empty")
    grouped_parameters = mmrl_parameters + prompt_parameters
    if {id(parameter) for parameter in active_parameters} != {
        id(parameter) for parameter in grouped_parameters
    }:
        raise RuntimeError(
            "Stage 3 trainable parameters must belong to MMRL or Soft Prompt"
        )
    expected_total = experiment_cfg.get("expected_total_trainable_parameters")
    actual_total = sum(parameter.numel() for parameter in grouped_parameters)
    if expected_total is not None and actual_total != int(expected_total):
        raise RuntimeError(
            "Stage 3 trainable parameter audit failed: "
            f"expected={int(expected_total)} actual={actual_total}"
        )
    print(
        "[STAGE3_TRAINABLE_PARAMETER_AUDIT] "
        f"mmrl={sum(parameter.numel() for parameter in mmrl_parameters)} "
        f"soft_prompt={sum(parameter.numel() for parameter in prompt_parameters)} "
        f"total={actual_total} expected={expected_total}"
    )

    parameter_groups = [{
        "params": mmrl_parameters,
        "lr": mmrl_lr,
        "weight_decay": weight_decay,
        "group_name": "mmrl",
    }]
    if prompt_parameters:
        prompt_lr = float(experiment_cfg.get("stage3_prompt_learning_rate", 0.3))
        parameter_groups.append({
            "params": prompt_parameters,
            "lr": prompt_lr,
            "weight_decay": 0.0,
            "group_name": "soft_prompt",
        })
    print(
        "[STAGE3_OPTIMIZER_GROUPS] "
        + " ".join(
            f"{group['group_name']}:lr={group['lr']}:tensors={len(group['params'])}"
            for group in parameter_groups
        )
    )
    print(
        "[STAGE3_OPTIMIZER] "
        f"AdamW beta1={adam_beta1} beta2={adam_beta2} "
        f"eps={adam_epsilon} weight_decay={weight_decay}"
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )
    optimizer.soft_prompt_warmup_ratio = float(
        experiment_cfg.get("stage3_prompt_warmup_ratio", 0.03)
    )
    return optimizer


def _warmup_hold_cosine_factor(step, warmup_steps, hold_until_step, total_steps):
    step = int(step)
    warmup_steps = int(warmup_steps)
    hold_until_step = int(hold_until_step)
    total_steps = int(total_steps)
    if not 0 < warmup_steps <= hold_until_step < total_steps:
        raise ValueError(
            "Stage3 step schedule requires "
            f"0 < warmup_steps <= hold_until_step < total_steps, got "
            f"{warmup_steps}, {hold_until_step}, {total_steps}"
        )
    if step < warmup_steps:
        return float(step) / float(warmup_steps)
    if step <= hold_until_step:
        return 1.0
    if step >= total_steps:
        return 0.0
    progress = float(step - hold_until_step) / float(total_steps - hold_until_step)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_stage3_step_scheduler(optimizer, experiment_cfg):
    total_steps = int(experiment_cfg["stage3_max_steps"])
    warmup_steps = int(experiment_cfg["stage3_warmup_steps"])
    hold_until_step = int(experiment_cfg["stage3_hold_until_step"])

    def mmrl_lr_lambda(current_step):
        return _warmup_hold_cosine_factor(
            current_step,
            warmup_steps=warmup_steps,
            hold_until_step=hold_until_step,
            total_steps=total_steps,
        )

    audit_steps = (0, warmup_steps, hold_until_step, 625, total_steps)
    audit = {
        step: round(mmrl_lr_lambda(step), 8)
        for step in dict.fromkeys(audit_steps)
    }
    print(
        "[STAGE3_STEP_SCHEDULE] "
        f"max_steps={total_steps} warmup_steps={warmup_steps} "
        f"hold_until_step={hold_until_step} decay=cosine final_factor=0 "
        f"factor_audit={audit}"
    )
    lr_lambdas = [mmrl_lr_lambda]
    prompt_groups = [
        group for group in optimizer.param_groups
        if group.get("group_name") == "soft_prompt"
    ]
    if prompt_groups:
        prompt_warmup_ratio = float(
            getattr(optimizer, "soft_prompt_warmup_ratio", 0.03)
        )
        prompt_warmup_steps = max(1, int(total_steps * prompt_warmup_ratio))

        def prompt_lr_lambda(current_step):
            step = int(current_step)
            if step < prompt_warmup_steps:
                return float(step) / float(prompt_warmup_steps)
            return max(
                0.0,
                float(total_steps - step)
                / float(max(1, total_steps - prompt_warmup_steps)),
            )

        lr_lambdas.append(prompt_lr_lambda)
        prompt_audit = {
            step: round(prompt_lr_lambda(step), 8)
            for step in (0, prompt_warmup_steps, total_steps)
        }
        print(
            "[STAGE3_PROMPT_LINEAR_SCHEDULE] "
            f"total_steps={total_steps} warmup_steps={prompt_warmup_steps} "
            f"factor_audit={prompt_audit}"
        )
    if len(lr_lambdas) != len(optimizer.param_groups):
        raise RuntimeError("Stage3 scheduler/group count mismatch")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambdas
    )


def _build_stage3_epoch_decay_scheduler(
    optimizer,
    updates_per_epoch,
    total_epochs,
    warmup_ratio,
    epoch_decay,
):
    updates_per_epoch = int(updates_per_epoch)
    total_epochs = int(total_epochs)
    warmup_steps = max(1, int(updates_per_epoch * float(warmup_ratio)))
    epoch_decay = float(epoch_decay)
    if updates_per_epoch < 1 or total_epochs < 2:
        raise ValueError(
            "Stage3 epoch-decay schedule requires positive steps and at least two epochs"
        )
    if not 0.0 < epoch_decay <= 1.0:
        raise ValueError(
            f"Stage3 epoch LR decay must be in (0, 1], got {epoch_decay}"
        )

    def mmrl_lr_lambda(current_step):
        step = int(current_step)
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        epoch_index = min(step // updates_per_epoch, total_epochs - 1)
        return epoch_decay ** epoch_index

    audit_steps = [0, warmup_steps]
    audit_steps.extend(epoch * updates_per_epoch for epoch in range(1, total_epochs))
    audit = {
        step: round(mmrl_lr_lambda(step), 8)
        for step in dict.fromkeys(audit_steps)
    }
    print(
        "[STAGE3_EPOCH_DECAY_SCHEDULE] "
        f"updates_per_epoch={updates_per_epoch} total_epochs={total_epochs} "
        f"warmup_steps={warmup_steps} epoch_decay={epoch_decay} "
        f"factor_audit={audit}"
    )
    lr_lambdas = [mmrl_lr_lambda]
    if any(
        group.get("group_name") == "soft_prompt"
        for group in optimizer.param_groups
    ):
        total_steps = updates_per_epoch * total_epochs
        prompt_warmup_ratio = float(
            getattr(optimizer, "soft_prompt_warmup_ratio", 0.03)
        )
        prompt_warmup_steps = max(1, int(total_steps * prompt_warmup_ratio))

        def prompt_lr_lambda(current_step):
            step = int(current_step)
            if step < prompt_warmup_steps:
                return float(step) / float(prompt_warmup_steps)
            return max(
                0.0,
                float(total_steps - step)
                / float(max(1, total_steps - prompt_warmup_steps)),
            )

        lr_lambdas.append(prompt_lr_lambda)
        prompt_audit = {
            step: round(prompt_lr_lambda(step), 8)
            for step in (0, prompt_warmup_steps, total_steps)
        }
        print(
            "[STAGE3_PROMPT_LINEAR_SCHEDULE] "
            f"total_steps={total_steps} warmup_steps={prompt_warmup_steps} "
            f"factor_audit={prompt_audit}"
        )
    if len(lr_lambdas) != len(optimizer.param_groups):
        raise RuntimeError("Stage3 scheduler/group count mismatch")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambdas
    )


# =========================
#  Stage1 lightweight classifier pretraining
# =========================
def _extract_mm_pooled_vision(v, pixel_values, image_grid_thw):
    return v.extract_gate_pooled_vision(pixel_values, image_grid_thw)


def run_stage1_light(model, processor, data_cfg, train_cfg, output_dir):
    stage_id = 1
    set_trainable_stage(model, stage_id)
    model.train()
    first_insert_layer = min(model.model.visual.insert_layers)
    print(
        "[STAGE1_CLASSIFIER_INPUT] "
        "prompt_only=True add_generation_prompt=True "
        f"gate_input_after_blocks_0based=0..{first_insert_layer - 1} "
        f"gate_input_before_block_0based={first_insert_layer}"
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

    ds = data_cfg.get("stage1_dataset")
    if ds is not None:
        collator = data_cfg.get("stage1_data_collator")
        if collator is None:
            raise ValueError("stage1_dataset requires stage1_data_collator")
        print(
            "[STAGE1_EXTERNAL_DATASET] "
            f"type={type(ds).__name__} samples={len(ds)}"
        )
    else:
        stage1_expert_json = [
            *data_cfg["expert_json"],
            *data_cfg.get("stage1_only_expert_json", []),
        ]
        stage1_expert_img_dir = [
            *data_cfg["expert_img_dir"],
            *data_cfg.get("stage1_only_expert_img_dir", []),
        ]
        print(
            "[STAGE1_ONLY_EXPERT_DATA] "
            f"json={data_cfg.get('stage1_only_expert_json', [])} "
            f"image_dirs={data_cfg.get('stage1_only_expert_img_dir', [])}"
        )
        ds = FourViewMMRLDataset(
            processor=processor,
            expert_json=stage1_expert_json,
            expert_img_dir=stage1_expert_img_dir,
            general_json=data_cfg["general_json"],
            general_img_dir=data_cfg["general_img_dir"],
            total_limit=data_cfg["total_limit"],
            enable_views=("expert-mm", "expert-text", "general-mm", "general-text"),
            mode=f"stage{stage_id}",
            ce_enabled=False,
            seed=train_cfg.get("data_sampling_seed", 42),
            deterministic_sampling=train_cfg.get("deterministic_sampling", False),
            classifier_prompt_only=True,
        )
        collator = MMRLDataCollator(processor)
    data_generator = torch.Generator()
    data_generator.manual_seed(int(train_cfg.get("data_order_seed", 42)) + int(stage_id))

    print(
        "[STAGE1_BATCH] "
        f"micro_batch={train_cfg['per_device_train_batch_size']} "
        f"gradient_accumulation={train_cfg['gradient_accumulation_steps']} "
        "effective_batch="
        f"{train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']} "
        f"workers={train_cfg.get('dataloader_num_workers', 4)} "
        f"pin_memory={bool(train_cfg.get('dataloader_pin_memory', False))}"
    )

    dl = DataLoader(
        ds,
        batch_size=train_cfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("dataloader_num_workers", 4),
        pin_memory=bool(train_cfg.get("dataloader_pin_memory", False)),
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
            text_pool_mask = model.model.build_mmrl_text_pooling_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audit_context="stage1_train",
            )
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
    visual = model.model.visual
    set_trainable_stage(model, stage_id)
    pooling_modules = {
        "hidden_state_pooling": visual.hidden_state_pooling,
        "embedding_pooling": visual.embedding_pooling,
    }
    unexpectedly_trainable = [
        name
        for name, module in pooling_modules.items()
        if any(parameter.requires_grad for parameter in module.parameters())
    ]
    if unexpectedly_trainable:
        raise RuntimeError(
            "Stage 3 must preserve the Stage 1 gate pooling coordinate systems: "
            f"{unexpectedly_trainable}"
        )
    classifier_trainable = any(
        parameter.requires_grad
        for parameter in visual.Task_classifier.parameters()
    )
    if classifier_trainable:
        raise RuntimeError("Stage 3 must keep the Stage 1 task classifier frozen")
    frozen_mmrl_parameters = [
        name
        for name, parameter in model.model.MMRL.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen_mmrl_parameters:
        raise RuntimeError(
            "Stage 3 must train the complete dynamic Rep generator: "
            f"{frozen_mmrl_parameters}"
        )
    print(
        "[STAGE3_TRAINABILITY] "
        "dynamic_rep_generator=True gate_pooling=False task_classifier=False"
    )
    experiment_cfg = train_cfg.get("experiment_cfg", {}) or {}
    actual_epochs = int(train_cfg["epochs"][stage_id])
    schedule_epochs = int(
        train_cfg.get("schedule_epochs", {}).get(stage_id, actual_epochs)
    )
    stage3_max_steps = int(experiment_cfg.get("stage3_max_steps", -1))
    use_step_schedule = stage3_max_steps > 0
    epoch_lr_decay = experiment_cfg.get("stage3_epoch_lr_decay")
    use_epoch_decay_schedule = (
        epoch_lr_decay is not None and actual_epochs > 1 and not use_step_schedule
    )
    if actual_epochs < 1 or schedule_epochs < actual_epochs:
        raise ValueError(
            "Stage3 requires 1 <= actual epochs <= schedule epochs, got "
            f"actual={actual_epochs}, schedule={schedule_epochs}"
        )
    print(
        f"[Stage3] actual_epochs={actual_epochs} "
        f"schedule_epochs={schedule_epochs} "
        f"max_steps={stage3_max_steps} "
        f"peak_lr={train_cfg['learning_rate'][stage_id]} "
        f"warmup={experiment_cfg.get('stage3_warmup_steps') if use_step_schedule else train_cfg.get('stage3_warmup_ratio', 0.05)}"
        f"{'_steps' if use_step_schedule else '_ratio'} "
        f"scheduler={('warmup_hold_cosine' if use_step_schedule else 'fixed_warmup_epoch_decay' if use_epoch_decay_schedule else train_cfg.get('stage3_lr_scheduler_type', 'cosine'))} "
        f"temperature={train_cfg.get('initial_temp', 1.0)}"
        f"->{train_cfg.get('final_temp', 0.1)}"
    )

    model.ce_loss_weight = 1.0
    model.alpha_loss_weight = 0.0
    model.enable_alpha_guide_loss = False

    model.current_stage_id = int(stage_id)
    model.current_stage_step = 0
    model.current_stage_progress = 0.0
    model.mmrl_relation_loss_weight = float(train_cfg.get(
        "mmrl_relation_loss_weight",
        experiment_cfg.get("mmrl_relation_loss_weight", 0.0),
    ))
    model.model.visual.mmrl_relation_loss_weight = model.mmrl_relation_loss_weight
    stage3_ce_only = (
        model.ce_loss_weight == 1.0
        and not model.enable_alpha_guide_loss
        and model.alpha_loss_weight == 0.0
        and model.mmrl_relation_loss_weight == 0.0
    )
    print(
        "[STAGE3_LOSS_AUDIT] "
        f"ce_weight={model.ce_loss_weight} "
        f"alpha_enabled={model.enable_alpha_guide_loss} "
        f"alpha_weight={model.alpha_loss_weight} "
        f"relation_weight={model.mmrl_relation_loss_weight} "
        f"ce_only={stage3_ce_only}"
    )
    print(
        "[MMRL_RELATION] "
        f"weight={model.mmrl_relation_loss_weight} "
        f"max_tokens={model.model.visual.mmrl_relation_max_tokens} "
        f"variance_floor_ratio={model.model.visual.mmrl_variance_floor_ratio} "
        f"variance_floor_weight={model.model.visual.mmrl_variance_floor_weight}"
    )
    stage3_views = ("expert-mm",)
    print(f"[Stage{stage_id}] enable_views={stage3_views}")
    assistant_turn_policy = str(
        experiment_cfg.get("stage3_assistant_turn_policy", "all")
    )
    print(
        f"[Stage{stage_id}] assistant_turn_policy={assistant_turn_policy}"
    )

    stage3_data_seed = train_cfg.get("data_sampling_seed", 42)
    if train_cfg.get("deterministic_sampling", False):
        data_seed_offset = int(experiment_cfg.get("stage3_data_seed_offset", 1))
        stage3_data_seed = int(stage3_data_seed) + data_seed_offset
        print(
            f"[Stage{stage_id}] data_sampling_seed="
            f"{train_cfg.get('data_sampling_seed', 42)} "
            f"offset={data_seed_offset} effective={stage3_data_seed}"
        )

    ds = data_cfg.get("stage3_dataset")
    if ds is not None:
        collator = data_cfg.get("data_collator")
        if collator is None:
            raise ValueError("stage3_dataset requires data_collator")
        print(
            "[STAGE3_EXTERNAL_DATASET] "
            f"type={type(ds).__name__} samples={len(ds)}"
        )
    else:
        stage3_expert_json = experiment_cfg.get(
            "stage3_expert_json",
            data_cfg["expert_json"],
        )
        stage3_expert_img_dir = experiment_cfg.get(
            "stage3_expert_img_dir",
            data_cfg["expert_img_dir"],
        )
        stage3_total_limit = int(
            experiment_cfg.get("stage3_total_limit", data_cfg["total_limit"])
        )
        if "stage3_expert_json" in experiment_cfg:
            print(
                "[STAGE3_DATA_OVERRIDE] "
                f"json={stage3_expert_json} "
                f"image_dirs={stage3_expert_img_dir} "
                f"total_limit={stage3_total_limit}"
            )

        ds = FourViewMMRLDataset(
            processor=processor,
            expert_json=stage3_expert_json,
            expert_img_dir=stage3_expert_img_dir,
            general_json=data_cfg["general_json"],
            general_img_dir=data_cfg["general_img_dir"],
            total_limit=stage3_total_limit,
            enable_views=stage3_views,
            mode=f"stage{stage_id}",
            ce_enabled=True,
            seed=stage3_data_seed,
            deterministic_sampling=train_cfg.get("deterministic_sampling", False),
            assistant_turn_policy=assistant_turn_policy,
        )
        collator = MMRLDataCollator(processor)
    stage3_batch_size = int(train_cfg.get(
        "stage3_per_device_train_batch_size",
        train_cfg["per_device_train_batch_size"],
    ))
    stage3_gradient_accumulation = int(train_cfg.get(
        "stage3_gradient_accumulation_steps",
        train_cfg["gradient_accumulation_steps"],
    ))
    stage3_workers = int(train_cfg.get(
        "stage3_dataloader_num_workers",
        train_cfg.get("dataloader_num_workers", 4),
    ))
    stage3_pin_memory = bool(train_cfg.get(
        "stage3_dataloader_pin_memory",
        False,
    ))
    if stage3_batch_size < 1 or stage3_gradient_accumulation < 1:
        raise ValueError(
            "Stage3 batch size and gradient accumulation must be positive, got "
            f"batch={stage3_batch_size} accumulation={stage3_gradient_accumulation}"
        )
    print(
        "[STAGE3_BATCH] "
        f"micro_batch={stage3_batch_size} "
        f"gradient_accumulation={stage3_gradient_accumulation} "
        f"effective_batch={stage3_batch_size * stage3_gradient_accumulation} "
        f"workers={stage3_workers} pin_memory={stage3_pin_memory} "
        "persistent_workers=False"
    )
    micro_batches = math.ceil(len(ds) / stage3_batch_size)
    updates_per_epoch = math.ceil(
        micro_batches / stage3_gradient_accumulation
    )
    if use_step_schedule:
        print(
            "[STAGE3_DATA_BUDGET] "
            f"view_samples={len(ds)} micro_batches={micro_batches} "
            f"available_optimizer_steps={updates_per_epoch} "
            f"requested_max_steps={stage3_max_steps}"
        )
        if stage3_max_steps > updates_per_epoch:
            raise RuntimeError(
                "Stage3 max_steps would cross the first dataset pass: "
                f"requested={stage3_max_steps} available={updates_per_epoch}"
            )
    temperature_schedule_epochs = experiment_cfg.get(
        "stage3_temperature_schedule_epochs"
    )
    if temperature_schedule_epochs is not None:
        temperature_schedule_epochs = int(temperature_schedule_epochs)
        if not 1 <= temperature_schedule_epochs <= schedule_epochs:
            raise ValueError(
                "Stage3 temperature schedule epochs must be between 1 and the "
                f"schedule length, got {temperature_schedule_epochs}"
            )
        temperature_steps = updates_per_epoch * temperature_schedule_epochs
        print(
            "[STAGE3_TEMPERATURE_SCHEDULE] "
            f"decay_epochs={temperature_schedule_epochs} "
            f"decay_steps={temperature_steps} hold_final_afterward=True"
        )
    else:
        temperature_steps = stage3_max_steps if use_step_schedule else None
    cb = StageScheduleCallback(
        dataset=ds,
        total_epochs=schedule_epochs,
        init_temp=train_cfg.get("initial_temp", 1.0),
        final_temp=train_cfg.get("final_temp", 0.1),
        resample_on_first_epoch=False,
        total_steps=temperature_steps,
    )
    args = TrainingArguments(
        output_dir=f"{output_dir}/stage{stage_id}",
        num_train_epochs=schedule_epochs,
        max_steps=stage3_max_steps,
        per_device_train_batch_size=stage3_batch_size,
        gradient_accumulation_steps=stage3_gradient_accumulation,
        learning_rate=train_cfg["learning_rate"][stage_id],
        lr_scheduler_type=(
            "constant"
            if use_step_schedule
            else train_cfg.get("stage3_lr_scheduler_type", "cosine")
        ),
        warmup_ratio=(
            0.0
            if use_step_schedule or use_epoch_decay_schedule
            else train_cfg.get("stage3_warmup_ratio", 0.05)
        ),
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=True,
        dataloader_pin_memory=stage3_pin_memory,
        dataloader_num_workers=stage3_workers,
        dataloader_persistent_workers=False,
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
    intermediate_eval_steps = _parse_step_evaluation_steps(
        experiment_cfg.get(
            "intermediate_eval_steps",
            os.getenv("MMRL_INTERMEDIATE_EVAL_STEPS", ""),
        )
    )
    if intermediate_eval_steps:
        callbacks.append(
            StepEvaluationCallback(
                processor,
                output_dir,
                stage_id,
                intermediate_eval_steps,
            )
        )
    if train_cfg.get("eval_each_epoch", False):
        callbacks.append(EpochEvaluationCallback(
            processor,
            output_dir,
            stage_id,
            actual_epochs,
        ))
    if train_cfg.get("save_each_epoch", False):
        base_model_path = train_cfg.get("checkpoint_base_model_path")
        if not base_model_path:
            raise ValueError(
                "save_each_epoch requires checkpoint_base_model_path"
            )
        callbacks.append(EpochCompactCheckpointCallback(
            processor,
            output_dir,
            stage_id,
            base_model_path,
            metadata={
                "experiment": train_cfg.get("experiment_name"),
                "seed": train_cfg.get("seed"),
                "data_seed": train_cfg.get("data_sampling_seed"),
            },
        ))

    grouped_optimizer = _build_stage3_grouped_optimizer(model, train_cfg)
    trainer_kwargs = {}
    if grouped_optimizer is not None:
        grouped_scheduler = None
        if use_step_schedule:
            grouped_scheduler = _build_stage3_step_scheduler(
                grouped_optimizer,
                experiment_cfg,
            )
        elif use_epoch_decay_schedule:
            grouped_scheduler = _build_stage3_epoch_decay_scheduler(
                grouped_optimizer,
                updates_per_epoch=updates_per_epoch,
                total_epochs=schedule_epochs,
                warmup_ratio=train_cfg.get("stage3_warmup_ratio", 0.05),
                epoch_decay=epoch_lr_decay,
            )
        trainer_kwargs["optimizers"] = (grouped_optimizer, grouped_scheduler)
    elif use_step_schedule or use_epoch_decay_schedule:
        raise RuntimeError(
            "Stage3 explicit schedule requires the grouped optimizer configuration"
        )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=callbacks,
        **trainer_kwargs,
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
