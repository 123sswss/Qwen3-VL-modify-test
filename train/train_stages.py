# train_stages.py
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
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

import numbers


def _build_experiment_context(train_cfg, output_dir, stage_id):
    experiment_name = train_cfg.get("experiment_name", "experiment")
    experiment_cfg = dict(train_cfg.get("experiment_cfg", {}))
    recorded_keys = (
        "seed",
        "deterministic_sampling",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "visual_residual_adapter_count",
        "adapter_usage_balance_loss_weight",
        "adapter_sample_entropy_loss_weight",
        "adapter_sample_entropy_target",
        "mmrl_residual_guard_loss_weight",
        "mmrl_semantic_anchor_loss_weight",
        "mmrl_residual_ratio_upper",
        "mmrl_warmup_fraction",
        "stage4_mmrl_lr_scale",
        "stage4_router_lr_scale",
        "enable_router_kmeans_prior",
        "router_calibration_samples",
        "router_prior_temperature",
        "router_prior_entropy_target",
        "router_residual_start_fraction",
        "router_residual_max_scale",
        "router_residual_logit_bound",
    )
    return {
        "experiment_name": experiment_name,
        "stage_name": f"stage{stage_id}",
        "stage_id": stage_id,
        "output_dir": output_dir,
        "stage_output_dir": f"{output_dir}/stage{stage_id}",
        "metrics_dir": f"{output_dir}/stage{stage_id}/metrics",
        "train_cfg": {
            **{key: train_cfg.get(key) for key in recorded_keys},
            "learning_rate": train_cfg.get("learning_rate", {}).get(stage_id),
            "epochs": train_cfg.get("epochs", {}).get(stage_id),
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
    if "alpha_mae" in row:
        print(f"[Alpha Statistics]")
        print(f"  └─ Alpha MAE:             {_fmt(row.get('alpha_mae')):>10}")

    print(f"[Schedule]")
    if "temperature" in row:
        print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), 4):>10}")
    if "learning_rate" in row:
        print(f"  └─ Learning Rate:         {_fmt(row.get('learning_rate'), 8):>10}")
    print("=" * 72 + "\n")

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


class MMRLDiagnosticsCallback(TrainerCallback):
    DEFAULT_KEEP_KEYS = {
        "ce_loss",
        "expert_delta_to_org_ratio",
        "expert_delta_common_mode_ratio",
        "adapter_route_entropy_norm",
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


class GradientPressureCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        metrics = getattr(model, "_last_metrics", None)
        collect_pressure = getattr(model, "_collect_grad_pressure_metrics", None)
        if not isinstance(metrics, dict) or not callable(collect_pressure):
            return
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        metrics.update(collect_pressure(device))
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
        self.current_stage_id = 4
        self.current_stage_progress = 1.0
        self.mmrl_warmup_fraction = 1.0 / 3.0
        self.adapter_usage_balance_loss_weight = 0.0
        self.adapter_sample_entropy_loss_weight = 0.0
        self.mmrl_residual_guard_loss_weight = 0.0
        self.mmrl_semantic_anchor_loss_weight = 0.0
        self.temperature_override = None

        self.debug_mode = True
        self.debug_loss_threshold = 20.0
        self._last_metrics = {}
        self._grad_pressure_hook_handles = []
        self._grad_pressure_accumulators = {}
        self._ensure_grad_pressure_hooks()

    def _get_mmrl_module(self):
        return getattr(self.model, "MMRL", None)

    def _ensure_grad_pressure_hooks(self):
        if self._grad_pressure_hook_handles:
            return

        groups = {
            "mmrl": self._get_mmrl_module(),
            "adapter_router": getattr(self.model.visual, "adapter_router", None),
            "visual_adapter": getattr(self.model.visual, "residual_adapters", None),
        }
        for group_name, module in groups.items():
            if module is None:
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                def _capture(grad, name=group_name, tracked_param=param):
                    acc = self._grad_pressure_accumulators.setdefault(
                        name, {"grad_sq": [], "param_sq": [], "numel": 0}
                    )
                    grad_f = grad.detach().float()
                    param_f = tracked_param.detach().float()
                    acc["grad_sq"].append(grad_f.square().sum())
                    acc["param_sq"].append(param_f.square().sum())
                    acc["numel"] += grad_f.numel()
                    return grad
                self._grad_pressure_hook_handles.append(param.register_hook(_capture))

    def _reset_grad_pressure_accumulators(self):
        self._grad_pressure_accumulators = {}

    def _collect_grad_pressure_metrics(self, device):
        result = {}
        for name, acc in self._grad_pressure_accumulators.items():
            count = max(int(acc["numel"]), 1)
            grad_rms = (torch.stack(acc["grad_sq"]).sum() / count).clamp_min(0.0).sqrt()
            param_rms = (torch.stack(acc["param_sq"]).sum() / count).clamp_min(0.0).sqrt()
            result[f"{name}_grad_rms"] = grad_rms.to(device=device)
            result[f"{name}_grad_pressure"] = (
                grad_rms / param_rms.clamp_min(1e-12)
            ).to(device=device)
        return result

    def forward(self, input_ids=None, alpha_labels=None, images_per_sample=None, task_type_ids=None, **kwargs):
        self._reset_grad_pressure_accumulators()
        visual = self.model.visual
        visual.current_stage_id = int(self.current_stage_id)
        visual.current_stage_progress = float(self.current_stage_progress)
        visual.mmrl_warmup_fraction = float(self.mmrl_warmup_fraction)
        visual.experts_enabled = int(self.current_stage_id) >= 4

        if self.temperature_override is not None:
            self.model.temperature_override = self.temperature_override
            kwargs["gating_temperature_override"] = self.temperature_override

        labels = kwargs.get("labels")
        if labels is not None and "attention_mask" in kwargs:
            prompt_mask = labels == -100
            attention_mask = kwargs["attention_mask"]
            if attention_mask.dim() == 2 and prompt_mask.dim() == 2:
                prompt_mask = prompt_mask & (attention_mask == 1)
            kwargs["mmrl_gating_mask"] = prompt_mask.to(dtype=self.model.dtype)

        outputs = super().forward(
            input_ids=input_ids,
            images_per_sample=images_per_sample,
            **kwargs,
        )
        if labels is None:
            return outputs

        ce_loss = outputs.loss
        if ce_loss is None:
            ce_loss = torch.tensor(0.0, device=outputs.logits.device)

        def _loss_tensor(name):
            value = getattr(visual, name, 0.0)
            if torch.is_tensor(value):
                return value.to(device=ce_loss.device, dtype=ce_loss.dtype)
            return torch.tensor(float(value), device=ce_loss.device, dtype=ce_loss.dtype)

        usage_loss = _loss_tensor("adapter_usage_balance_loss")
        entropy_loss = _loss_tensor("adapter_sample_entropy_loss")
        mmrl_guard_loss = _loss_tensor("mmrl_residual_guard_loss")
        mmrl_semantic_anchor_loss = _loss_tensor("mmrl_semantic_anchor_loss")
        scaled_usage_loss = usage_loss * float(self.adapter_usage_balance_loss_weight)
        scaled_entropy_loss = entropy_loss * float(self.adapter_sample_entropy_loss_weight)
        scaled_mmrl_guard_loss = (
            mmrl_guard_loss * float(self.mmrl_residual_guard_loss_weight)
        )
        stage_id = int(self.current_stage_id)
        if stage_id == 3:
            semantic_anchor_weight = float(self.mmrl_semantic_anchor_loss_weight)
        elif stage_id == 4:
            semantic_anchor_weight = 0.25 * float(
                self.mmrl_semantic_anchor_loss_weight
            )
        else:
            semantic_anchor_weight = 0.0
        scaled_mmrl_semantic_anchor_loss = (
            mmrl_semantic_anchor_loss * semantic_anchor_weight
        )
        outputs.loss = (
            self.ce_loss_weight * ce_loss
            + scaled_usage_loss
            + scaled_entropy_loss
            + scaled_mmrl_guard_loss
            + scaled_mmrl_semantic_anchor_loss
        )

        with torch.no_grad():
            self._last_metrics = {
                "total_loss": outputs.loss.detach(),
                "ce_loss": ce_loss.detach(),
                "adapter_usage_balance_loss_scaled": scaled_usage_loss.detach(),
                "adapter_sample_entropy_loss_scaled": scaled_entropy_loss.detach(),
                "mmrl_residual_guard_loss_scaled": scaled_mmrl_guard_loss.detach(),
                "mmrl_semantic_anchor_loss_scaled": (
                    scaled_mmrl_semantic_anchor_loss.detach()
                ),
                "stage_progress": torch.tensor(
                    float(self.current_stage_progress), device=ce_loss.device
                ),
            }
            self._last_metrics.update(visual.debug_context or {})

        if self.debug_mode and (
            not torch.isfinite(outputs.loss)
            or outputs.loss.detach().float().item() > self.debug_loss_threshold
        ):
            print(
                "[HIGH-LOSS-DBG] "
                f"stage={self.current_stage_id} "
                f"total={outputs.loss.detach().float().item():.6f} "
                f"ce={ce_loss.detach().float().item():.6f}"
            )
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
    config.ADAPTER_SAMPLE_ENTROPY_TARGET = float(experiment_cfg.get(
        "adapter_sample_entropy_target", 0.55
    ))
    config.MMRL_RESIDUAL_RATIO_UPPER = float(experiment_cfg.get(
        "mmrl_residual_ratio_upper", 0.20
    ))
    config.ENABLE_ROUTER_KMEANS_PRIOR = bool(experiment_cfg.get(
        "enable_router_kmeans_prior", False
    ))
    config.ROUTER_PRIOR_TEMPERATURE = float(experiment_cfg.get(
        "router_prior_temperature", 0.25
    ))
    config.ROUTER_RESIDUAL_START_FRACTION = float(experiment_cfg.get(
        "router_residual_start_fraction", 0.30
    ))
    config.ROUTER_RESIDUAL_MAX_SCALE = float(experiment_cfg.get(
        "router_residual_max_scale", 0.50
    ))
    config.ROUTER_RESIDUAL_LOGIT_BOUND = float(experiment_cfg.get(
        "router_residual_logit_bound", 1.0
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print("Base model loaded. ")
    print("Tokenizer loaded. Now building MMRL model...")
    model = Qwen3VLMMRLForStages(config, tokenizer).to("cuda").to(torch.bfloat16)
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
        f"adapter_sample_entropy_target={config.ADAPTER_SAMPLE_ENTROPY_TARGET} "
        f"mmrl_residual_ratio_upper={config.MMRL_RESIDUAL_RATIO_UPPER} "
        f"enable_router_kmeans_prior={config.ENABLE_ROUTER_KMEANS_PRIOR}"
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
        modules = {
            "vision_pool": v.hidden_state_pooling,
            "text_pool": v.embedding_pooling,
            "task_classifier": v.Task_classifier,
        }
    elif stage == 2:
        modules = {
            "vision_pool": v.hidden_state_pooling,
            "text_pool": v.embedding_pooling,
            "task_classifier": v.Task_classifier,
            "vision_gate": v.visionGating,
        }
    elif stage == 3:
        modules = {"mmrl": model.model.MMRL}
    elif stage == 4:
        modules = {
            "mmrl": model.model.MMRL,
            "adapter_router": v.adapter_router,
            "residual_adapters": v.residual_adapters,
        }
    else:
        raise ValueError("stage must be 1/2/3/4")

    audit = []
    for name, module in modules.items():
        if isinstance(module, nn.Parameter):
            module.requires_grad = True
            count = module.numel()
        else:
            for p in module.parameters():
                p.requires_grad = True
            count = sum(p.numel() for p in module.parameters())
        audit.append(f"{name}:{count}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[STAGE_TRAINABLE_AUDIT] stage={stage} total={total} groups={audit}")


def print_stage4_trainable_params_before_stage1(model):
    set_trainable_stage(model, 4)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in trainable)
    print(f"[STAGE4-TRAINABLE][TOTAL] tensors={len(trainable)} total_numel={total}")


# =========================
#  轻前向工具（Stage1/2）
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


@torch.no_grad()
def _fit_balanced_spherical_kmeans(
    features,
    cluster_count,
    seed,
    iterations=20,
    sinkhorn_temperature=0.10,
    sinkhorn_iterations=5,
):
    features = F.normalize(features.float(), dim=-1)
    sample_count = features.shape[0]
    if sample_count < cluster_count:
        raise ValueError(
            f"router calibration needs at least {cluster_count} samples, got {sample_count}"
        )

    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    first = int(torch.randint(sample_count, (1,), generator=generator).item())
    centroid_list = [features[first]]
    for _ in range(1, cluster_count):
        existing = torch.stack(centroid_list, dim=0)
        nearest_similarity = (features @ existing.t()).max(dim=1).values
        centroid_list.append(features[nearest_similarity.argmin()])
    centroids = F.normalize(torch.stack(centroid_list, dim=0), dim=-1)

    temperature = max(float(sinkhorn_temperature), 1e-6)
    for _ in range(max(int(iterations), 1)):
        logits = (features @ centroids.t()) / temperature
        logits = logits - logits.max()
        assignments = logits.exp().t().clamp_min(1e-12)
        assignments = assignments / assignments.sum().clamp_min(1e-12)
        for _ in range(max(int(sinkhorn_iterations), 1)):
            assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
            assignments = assignments / cluster_count
            assignments = assignments / assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
            assignments = assignments / sample_count
        assignments = (assignments * sample_count).t()
        centroids = F.normalize(assignments.t() @ features, dim=-1)
    return centroids


@torch.no_grad()
def _calibrate_prior_temperature_and_bias(
    similarities,
    cluster_count,
    entropy_target=0.75,
):
    target_usage = torch.full((cluster_count,), 1.0 / cluster_count)
    entropy_target = min(max(float(entropy_target), 0.05), 0.99)

    def _evaluate(temperature):
        logits = similarities / max(float(temperature), 1e-6)
        bias = torch.zeros(cluster_count, dtype=logits.dtype)
        for _ in range(100):
            usage = torch.softmax(logits + bias, dim=-1).mean(dim=0)
            correction = (
                target_usage.clamp_min(1e-8).log()
                - usage.clamp_min(1e-8).log()
            )
            bias = bias + 0.5 * correction
            bias = bias - bias.mean()
        probs = torch.softmax(logits + bias, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / torch.log(probs.new_tensor(float(cluster_count)))
        return entropy.mean().item(), bias, probs

    low, high = 0.005, 2.0
    for _ in range(24):
        middle = (low * high) ** 0.5
        entropy, _, _ = _evaluate(middle)
        if entropy < entropy_target:
            low = middle
        else:
            high = middle
    temperature = (low * high) ** 0.5
    entropy, bias, probs = _evaluate(temperature)
    return temperature, bias, probs, entropy


@torch.no_grad()
def _calibrate_router_prior(model, processor, data_cfg, train_cfg, output_dir):
    visual = model.model.visual
    cluster_count = int(visual.visual_residual_adapter_count)
    sample_limit = min(
        int(train_cfg.get("router_calibration_samples", 4096)),
        int(data_cfg["total_limit"]),
    )
    if sample_limit < cluster_count:
        raise ValueError(
            f"router_calibration_samples must be >= {cluster_count}, got {sample_limit}"
        )

    dataset = FourViewMMRLDataset(
        processor=processor,
        expert_json=data_cfg["expert_json"],
        expert_img_dir=data_cfg["expert_img_dir"],
        general_json=data_cfg["general_json"],
        general_img_dir=data_cfg["general_img_dir"],
        total_limit=sample_limit,
        enable_views=("expert-mm",),
        mode="stage2_router_prior_calibration",
        ce_enabled=False,
        seed=train_cfg["seed"],
        deterministic_sampling=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_train_batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("dataloader_num_workers", 4),
        pin_memory=False,
        collate_fn=MMRLDataCollator(processor),
        drop_last=False,
    )

    was_training = visual.training
    visual.eval()
    feature_batches = []
    for batch in loader:
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        if pixel_values is None or image_grid_thw is None:
            continue
        pixel_values = pixel_values.cuda(non_blocking=True)
        image_grid_thw = image_grid_thw.cuda(non_blocking=True)
        if image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.squeeze(1)
        batch_features = visual.extract_router_calibration_features(
            pixel_values,
            image_grid_thw,
        )
        feature_batches.append(batch_features.float().cpu())
    visual.train(was_training)

    if not feature_batches:
        raise RuntimeError("router calibration collected no vision features")
    features = F.normalize(torch.cat(feature_batches, dim=0), dim=-1)
    centroids = _fit_balanced_spherical_kmeans(
        features,
        cluster_count=cluster_count,
        seed=train_cfg["seed"],
        iterations=train_cfg.get("router_kmeans_iterations", 20),
        sinkhorn_temperature=train_cfg.get("router_kmeans_temperature", 0.10),
        sinkhorn_iterations=train_cfg.get("router_sinkhorn_iterations", 5),
    )

    similarities = features @ centroids.t()
    temperature, bias, prior_probs, prior_entropy = (
        _calibrate_prior_temperature_and_bias(
            similarities,
            cluster_count=cluster_count,
            entropy_target=train_cfg.get("router_prior_entropy_target", 0.75),
        )
    )
    usage = prior_probs.mean(dim=0)
    visual.set_router_prior(centroids, bias, temperature)

    stage2_dir = os.path.join(output_dir, "stage2")
    os.makedirs(stage2_dir, exist_ok=True)
    prior_path = os.path.join(stage2_dir, "router_prior.pt")
    torch.save(
        {
            "centroids": centroids,
            "bias": bias,
            "usage": usage,
            "temperature": float(temperature),
            "normalized_entropy": float(prior_entropy),
            "sample_count": int(features.shape[0]),
        },
        prior_path,
    )
    print(
        "[ROUTER_PRIOR_CALIBRATION] "
        f"samples={features.shape[0]} clusters={cluster_count} "
        f"temperature={temperature:.6f} entropy={prior_entropy:.6f} "
        f"usage={[round(x, 6) for x in usage.tolist()]} "
        f"saved={prior_path}"
    )
    torch.cuda.empty_cache()


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

    if stage_id == 2 and train_cfg.get("enable_router_kmeans_prior", False):
        _calibrate_router_prior(
            model=model,
            processor=processor,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            output_dir=output_dir,
        )

    # save_path = f"{output_dir}/stage{stage_id}"
    # model.save_pretrained(save_path)
    # print(f"[Stage{stage_id}] saved to {save_path}")


# =========================
#  Stage3/4 (完整Trainer)
# =========================
def _build_stage34_optimizer(model, stage_id, train_cfg):
    base_lr = float(train_cfg["learning_rate"][stage_id])
    mmrl_params = [p for p in model.model.MMRL.parameters() if p.requires_grad]
    if stage_id == 3:
        group_specs = [("mmrl", mmrl_params, base_lr)]
    else:
        mmrl_ids = {id(p) for p in mmrl_params}
        router_params = [
            p for p in model.model.visual.adapter_router.parameters()
            if p.requires_grad
        ]
        router_ids = {id(p) for p in router_params}
        adapter_params = [
            p for p in model.parameters()
            if p.requires_grad
            and id(p) not in mmrl_ids
            and id(p) not in router_ids
        ]
        mmrl_lr = base_lr * float(train_cfg.get("stage4_mmrl_lr_scale", 1.0))
        router_lr = base_lr * float(train_cfg.get("stage4_router_lr_scale", 1.0))
        group_specs = [
            ("mmrl", mmrl_params, mmrl_lr),
            ("adapter", adapter_params, base_lr),
            ("router", router_params, router_lr),
        ]

    optimizer_groups = []
    grouped_ids = set()
    for name, params, lr in group_specs:
        if not params:
            continue
        grouped_ids.update(id(p) for p in params)
        count = sum(p.numel() for p in params)
        print(
            f"[MMRL_OPTIMIZER_GROUP] stage={stage_id} name={name} "
            f"lr={lr:.8g} tensors={len(params)} params={count}"
        )
        optimizer_groups.append({"params": params, "lr": lr})
    if not optimizer_groups:
        raise RuntimeError(f"Stage {stage_id} has no trainable parameters")
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if grouped_ids != trainable_ids:
        raise RuntimeError(
            f"Stage {stage_id} optimizer coverage mismatch: "
            f"grouped={len(grouped_ids)} trainable={len(trainable_ids)}"
        )
    return torch.optim.AdamW(optimizer_groups, weight_decay=0.01)


def run_stage34_full(stage_id, model, processor, data_cfg, train_cfg, output_dir):
    assert stage_id in [3, 4]
    set_trainable_stage(model, stage_id, train_cfg=train_cfg)

    model.ce_loss_weight = 1.0
    model.current_stage_id = int(stage_id)
    model.current_stage_progress = 0.0
    model.mmrl_warmup_fraction = float(train_cfg.get("mmrl_warmup_fraction", 1.0 / 3.0))
    model.adapter_usage_balance_loss_weight = float(
        train_cfg.get("adapter_usage_balance_loss_weight", 0.0)
    )
    model.adapter_sample_entropy_loss_weight = float(
        train_cfg.get("adapter_sample_entropy_loss_weight", 0.0)
    )
    model.mmrl_residual_guard_loss_weight = float(
        train_cfg.get("mmrl_residual_guard_loss_weight", 0.0)
    )
    model.mmrl_semantic_anchor_loss_weight = float(
        train_cfg.get("mmrl_semantic_anchor_loss_weight", 0.0)
    )

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
    schedule_cb = StageScheduleCallback(
        dataset=ds,
        total_epochs=train_cfg["epochs"][stage_id],
        init_temp=train_cfg.get("final_temp", 0.1),
        final_temp=train_cfg.get("final_temp", 0.1),
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
        keep_keys=train_cfg.get("mmrl_diagnostics_keep_keys"),
    )
    optimizer = _build_stage34_optimizer(model, stage_id, train_cfg)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[
            schedule_cb,
            GradientPressureCallback(),
            metrics_cb,
            diag_cb,
        ],
        optimizers=(optimizer, None),
    )
    trainer.train()
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
