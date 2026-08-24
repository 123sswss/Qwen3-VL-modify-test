import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_epoch_eval import (
    preflight_live_epoch_evaluation,
    run_live_epoch_evaluation,
    run_professional_gate_preflight,
)
from mmrl_checkpoint import save_mmrl_checkpoint
from train_stages import build_model_and_processor, run_stage


SCORE_PATTERN = re.compile(r"百分制分数\s*[:：]\s*(-?\d+(?:\.\d+)?)")
SELECTED_EXPERIMENT = os.getenv(
    "MMRL_EXPERIMENT",
    "dynamic_rep_cross_attention_v1",
)

EXPERIMENTS = {
    "dynamic_rep_cross_attention_v1": {
        "rp_space_length": 40,
        "memory_query_count": 128,
        "memory_attention_dim": 128,
        "projector_hidden_dim": 1024,
        "cross_attention_heads": 8,
        "same_init_layer_projectors": True,
        "use_dynamic_cross_attention": True,
        "memory_pooling_mode": "multi_query",
        "mmrl_relation_loss_weight": 0.05,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage1_learning_rate": 1e-4,
        "stage3_learning_rate": 6e-5,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "constant_with_warmup",
        "diag_every_steps": 250,
    },
}

if SELECTED_EXPERIMENT not in EXPERIMENTS:
    raise ValueError(
        f"Unknown MMRL_EXPERIMENT={SELECTED_EXPERIMENT!r}, "
        f"choices={sorted(EXPERIMENTS)}"
    )
EXP_CFG = EXPERIMENTS[SELECTED_EXPERIMENT]


def seed_before_model_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[Reproducibility] seeded model init and training RNGs with seed={seed}")


def _read_evaluation_score(log_path):
    try:
        lines = Path(log_path).read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except OSError:
        return None
    score = None
    for line in lines:
        match = SCORE_PATTERN.search(line)
        if match:
            score = float(match.group(1))
    return score


def _should_keep_extrema_checkpoint(output_dir, score):
    output_dir = Path(output_dir).resolve()
    checkpoint_root = output_dir.parent
    retained_scores = []
    if checkpoint_root.is_dir():
        for experiment_dir in checkpoint_root.iterdir():
            if (
                not experiment_dir.is_dir()
                or experiment_dir.name == "trash"
                or experiment_dir.resolve() == output_dir
                or not (experiment_dir / "final").is_dir()
            ):
                continue
            prior_score = _read_evaluation_score(experiment_dir / "eval" / "test.log")
            if prior_score is not None and math.isfinite(prior_score):
                retained_scores.append(prior_score)

    if not retained_scores:
        return True, None, None
    current_min = min(retained_scores)
    current_max = max(retained_scores)
    return score <= current_min or score >= current_max, current_min, current_max


CFG = {
    "model_path": os.getenv("MMRL_MODEL_PATH", "/root/autodl-tmp/model"),
    "output_dir": os.getenv(
        "MMRL_OUTPUT_DIR",
        f"./output/{SELECTED_EXPERIMENT}",
    ),
    "experiment_name": SELECTED_EXPERIMENT,
    "experiment": EXP_CFG,
    "data": {
        "expert_json": [
            "/root/autodl-tmp/dataset/1json.json",
            "/root/autodl-tmp/dataset/2conv_c.json",
            "/root/autodl-tmp/dataset/1conv_c.json",
            "/root/autodl-tmp/dataset/4conv_c.json",
            "/root/autodl-tmp/dataset/14json.json",
            "/root/autodl-tmp/dataset/prof_test.json",
            "/root/autodl-tmp/dataset/test2_train.json",
            "/root/autodl-tmp/dataset/test7_train.json",
        ],
        "expert_img_dir": [
            "/root/autodl-tmp/dataset/1/train",
            "/root/autodl-tmp/dataset/2/train",
            "/root/autodl-tmp/dataset/4/train",
            "/root/autodl-tmp/dataset/14",
        ],
        "stage1_only_expert_json": [],
        "stage1_only_expert_img_dir": [],
        "general_json": [
            "/root/autodl-tmp/dataset/llava_instruct_150k.json",
            "/root/autodl-tmp/dataset/gen_test.json",
            "/root/autodl-tmp/dataset/conversation_58k.json",
        ],
        "general_img_dir": [
            "/root/autodl-tmp/dataset/gen/train2017",
            "/root/autodl-tmp/dataset/gen/val2017",
        ],
        "total_limit": 20000,
    },
    "train": {
        "experiment_name": SELECTED_EXPERIMENT,
        "experiment_cfg": EXP_CFG,
        "seed": int(os.getenv("MMRL_SEED", "44")),
        "data_sampling_seed": int(os.getenv("MMRL_DATA_SAMPLING_SEED", "42")),
        "data_order_seed": int(os.getenv("MMRL_DATA_ORDER_SEED", "42")),
        "deterministic_sampling": os.getenv(
            "MMRL_DETERMINISTIC_SAMPLING",
            "1",
        ) == "1",
        "eval_each_epoch": os.getenv("MMRL_EVAL_EACH_EPOCH", "0") == "1",
        "live_final_eval": os.getenv("MMRL_LIVE_FINAL_EVAL", "0") == "1",
        "save_extrema_checkpoints": os.getenv(
            "MMRL_SAVE_EXTREMA_CHECKPOINTS",
            "1",
        ) == "1",
        "stage1_gate_preflight": os.getenv(
            "MMRL_STAGE1_GATE_PREFLIGHT",
            "1",
        ) == "1",
        "mmrl_relation_loss_weight": float(os.getenv(
            "MMRL_RELATION_LOSS_WEIGHT",
            str(EXP_CFG["mmrl_relation_loss_weight"]),
        )),
        "mmrl_relation_max_tokens": int(EXP_CFG["mmrl_relation_max_tokens"]),
        "mmrl_variance_floor_ratio": float(EXP_CFG["mmrl_variance_floor_ratio"]),
        "mmrl_variance_floor_weight": float(EXP_CFG["mmrl_variance_floor_weight"]),
        "per_device_train_batch_size": int(os.getenv("MMRL_BATCH_SIZE", "2")),
        "gradient_accumulation_steps": int(os.getenv("MMRL_GRAD_ACCUM", "16")),
        "grad_spike_ema_alpha": 0.10,
        "grad_spike_ratio_threshold": 3.0,
        "grad_spike_abs_threshold": 0.5,
        "grad_spike_cooldown_steps": 20,
        "console_log_every": 50,
        "metric_smooth_window": 30,
        "metric_ema_alpha": 0.12,
        "metric_scatter_stride": 5,
        "save_debug_figure": False,
        "diag_every_steps": int(os.getenv(
            "MMRL_DIAG_EVERY_STEPS",
            str(EXP_CFG["diag_every_steps"]),
        )),
        "mmrl_diagnostics_keep_keys": [
            "ce_loss",
            "delta_to_org_ratio",
            "mmrl_delta_to_org_ratio",
            "mmrl_relation_loss_scaled",
            "mmrl_relation_gram_loss",
            "mmrl_variance_floor_loss",
            "delta_pool_common_mode_ratio",
            "delta_pool_specificity_ratio",
            "shared_rep_norm_mean",
            "shared_rep_grad_norm",
            "v_projector_grad_norm_mean",
            "dynamic_rep_base_norm_mean",
            "dynamic_rep_cross_delta_norm_mean",
            "dynamic_rep_cross_delta_ratio",
            "visual_memory_norm_mean",
            "text_memory_norm_mean",
            "visual_memory_pooling_grad_norm_mean",
            "text_memory_pooling_grad_norm_mean",
            "cross_attention_grad_norm_mean",
            "cross_attention_output_weight_norm",
            "text_guided_visual_attention_entropy_norm",
            "text_guided_visual_attention_peak_mean",
            "text_guided_visual_slot_specificity_ratio",
            "text_guided_visual_slot_pairwise_cos_mean",
            "temperature",
        ],
        "learning_rate": {
            1: float(EXP_CFG["stage1_learning_rate"]),
            3: float(EXP_CFG["stage3_learning_rate"]),
        },
        "epochs": {
            1: int(os.getenv("MMRL_STAGE1_EPOCHS", "1")),
            3: int(os.getenv("MMRL_STAGE3_EPOCHS", "1")),
        },
        "schedule_epochs": {
            3: int(os.getenv("MMRL_STAGE3_EPOCHS", "1")),
        },
        "stage3_warmup_ratio": float(EXP_CFG["stage3_warmup_ratio"]),
        "stage3_lr_scheduler_type": str(EXP_CFG["stage3_lr_scheduler_type"]),
        "initial_temp": 1.0,
        "final_temp": 0.775,
    },
    "ablation_order": [1, 3],
}


def main():
    seed_before_model_init(CFG["train"]["seed"])
    live_final_eval = CFG["train"].get("live_final_eval", False)
    if CFG["train"].get("eval_each_epoch", False) or live_final_eval:
        preflight_live_epoch_evaluation()
    model, processor = build_model_and_processor(
        CFG["model_path"],
        experiment_cfg=CFG["experiment"],
    )

    for stage_id in CFG["ablation_order"]:
        print(f"\n========== Running Stage {stage_id} ==========")
        run_stage(
            stage_id=stage_id,
            model=model,
            processor=processor,
            data_cfg=CFG["data"],
            train_cfg=CFG["train"],
            output_dir=CFG["output_dir"],
        )
        if stage_id == 1 and CFG["train"].get("stage1_gate_preflight", True):
            run_professional_gate_preflight(model, processor)

    final_dir = Path(CFG["output_dir"]) / "final"
    if live_final_eval:
        log_path = Path(CFG["output_dir"]) / "eval" / "test.log"
        print("[FINAL-EVAL] online evaluation begin")
        try:
            summary = run_live_epoch_evaluation(model, processor, log_path)
        except Exception:
            recovery_dir = Path(CFG["output_dir"]) / "eval_failed_recovery"
            save_mmrl_checkpoint(
                model,
                processor,
                recovery_dir,
                CFG["model_path"],
                metadata={
                    "experiment": SELECTED_EXPERIMENT,
                    "reason": "final_evaluation_failed_recovery",
                },
            )
            print(
                "[FINAL-EVAL-RECOVERY] evaluation failed; trained weights saved to "
                f"{recovery_dir}"
            )
            raise
        score = float(summary["score"])
        print(f"[FINAL-EVAL] score={score} completed")

        if CFG["train"].get("save_extrema_checkpoints", True):
            keep, prior_min, prior_max = _should_keep_extrema_checkpoint(
                CFG["output_dir"],
                score,
            )
            if keep:
                save_mmrl_checkpoint(
                    model,
                    processor,
                    final_dir,
                    CFG["model_path"],
                    metadata={
                        "experiment": SELECTED_EXPERIMENT,
                        "score": score,
                        "seed": CFG["train"]["seed"],
                    },
                )
                print(
                    f"[CHECKPOINT] saved extrema candidate to {final_dir}; "
                    f"score={score} prior_min={prior_min} prior_max={prior_max}"
                )
            else:
                print(
                    f"[CHECKPOINT] skipped non-extrema score={score}; "
                    f"retained_range=[{prior_min}, {prior_max}]"
                )
    else:
        save_mmrl_checkpoint(
            model,
            processor,
            final_dir,
            CFG["model_path"],
            metadata={
                "experiment": SELECTED_EXPERIMENT,
                "seed": CFG["train"]["seed"],
            },
        )
        print(f"final model saved to {final_dir}")

    print("All stages done.")


if __name__ == "__main__":
    main()
