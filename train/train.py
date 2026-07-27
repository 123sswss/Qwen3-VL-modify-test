# new_train.py
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import random
import re
from pathlib import Path

import numpy as np
import torch

from train_stages import build_model_and_processor, run_stage
from live_epoch_eval import (
    preflight_live_epoch_evaluation,
    run_live_epoch_evaluation,
    run_professional_gate_preflight,
)


SCORE_PATTERN = re.compile(r"百分制分数\s*[:：]\s*(-?\d+(?:\.\d+)?)")


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
        lines = Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()
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


BASE_VISUAL_ROUTER_MODEL = {
    "rp_space_length": 40,
    "visual_residual_adapter_count": 4,
    # Historical runs were effectively random-initialized because outer post_init
    # overwrote the adapter constructor's zero initialization.
    "random_init_adapter_output_count": 4,
    "zero_init_adapter_router_output": False,
    "adapter_usage_balance_loss_weight": 0.005,
    "adapter_sample_entropy_loss_weight": 0.02,
    "adapter_common_mode_loss_weight": 0.03,
    "adapter_effective_delta_loss_weight": 0.0,
    "mmrl_relation_loss_weight": 0.0,
    "mmrl_relation_max_tokens": 64,
    "mmrl_variance_floor_ratio": 0.50,
    "mmrl_variance_floor_weight": 0.10,
    "adapter_diversity_loss_weight": 0.0,
    "adapter_diversity_target_low": 0.30,
    "adapter_diversity_target_high": 0.58,
    "adapter_diversity_upper_weight": 2.0,
    "adapter_diversity_worst_pair_weight": 1.0,
    "enable_adapter_router_identity_residual": False,
    "direct_mmrl_output": False,
    "raw_visual_adapter": False,
    "enable_early_mmrl_guard": False,
    "early_mmrl_guard_ratio": 1.25,
    "early_mmrl_guard_hold_steps": 250,
    "early_mmrl_guard_release_steps": 125,
    "enable_deepstack_mmrl_residual": False,
    "deepstack_mmrl_residual_scale": 0.0,
    "adapter_sample_entropy_target": 0.55,
    "adapter_common_mode_target": 0.85,
    "adapter_effective_delta_target_low": 0.78,
    "adapter_effective_delta_target_high": 1.10,
    "ablate_visual_gate": False,
    "ablate_direct_learnable_rep": False,
    "diag_every_steps": 250,
}


V4_RECOVER_BASE = {
    **BASE_VISUAL_ROUTER_MODEL,
    "adapter_usage_balance_loss_weight": 0.0026,
    "adapter_sample_entropy_loss_weight": 0.020,
    "adapter_common_mode_loss_weight": 0.0,
    "adapter_effective_delta_loss_weight": 0.0003,
    "adapter_diversity_loss_weight": 0.0,
    "adapter_diversity_target_low": 0.30,
    "adapter_diversity_target_high": 0.58,
    "adapter_sample_entropy_target": 0.72,
    "adapter_common_mode_target": 0.94,
    "adapter_effective_delta_target_low": 0.52,
    "adapter_effective_delta_target_high": 0.98,
    "enable_deepstack_mmrl_residual": False,
    "deepstack_mmrl_residual_scale": 0.0,
}


EXPERIMENTS = {
    "visual_router_layer_fixed_v4_diversity_recover": {
        **V4_RECOVER_BASE,
    },
    "visual_router_relation_alignment_diag": {
        **V4_RECOVER_BASE,
        "mmrl_relation_loss_weight": 0.010,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
    },
    "visual_router_relation_heterogeneous_adapter_lr_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "mmrl_relation_loss_weight": 0.010,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "constant_with_warmup",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
    },
    "visual_router_relation_multiturn_fixed_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "mmrl_relation_loss_weight": 0.010,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "constant_with_warmup",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
    },
    "visual_router_relation_joint_multiturn_cosine_v2": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "mmrl_relation_loss_weight": 0.010,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_joint_multiturn_cosine_ce_only_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.0,
        "adapter_sample_entropy_loss_weight": 0.0,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.0,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_joint_multiturn_cosine_usage_only_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.0,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.0,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_joint_multiturn_cosine_usage_1e2_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.01,
        "adapter_sample_entropy_loss_weight": 0.0,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.0,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_joint_multiturn_cosine_entropy_only_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.0,
        "adapter_sample_entropy_loss_weight": 0.02,
        "adapter_sample_entropy_target": 0.72,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.0,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_joint_multiturn_cosine_route_pair_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.02,
        "adapter_sample_entropy_target": 0.72,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.0,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_joint_multiturn_cosine_route_pair_relation_2p5e3_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.02,
        "adapter_sample_entropy_target": 0.72,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.0025,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_joint_multiturn_cosine_route_pair_relation_5e3_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.02,
        "adapter_sample_entropy_target": 0.72,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.005,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_relation_joint_multiturn_cosine_mmrl_lr4e5_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "mmrl_relation_loss_weight": 0.010,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 4e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_relation_joint_multiturn_cosine_mmrl_lr3e5_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "mmrl_relation_loss_weight": 0.010,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 3e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_relation_joint_multiturn_cosine_no_usage_v3": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "adapter_usage_balance_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.010,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "cosine",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_assistant_turn_policy": "joint",
        "stage3_data_seed_offset": 0,
    },
    "visual_router_no_relation_heterogeneous_adapter_lr_v1": {
        **V4_RECOVER_BASE,
        "random_init_adapter_output_count": 0,
        "mmrl_relation_loss_weight": 0.0,
        "mmrl_relation_max_tokens": 64,
        "mmrl_variance_floor_ratio": 0.50,
        "mmrl_variance_floor_weight": 0.10,
        "stage3_learning_rate": 6e-5,
        "stage3_schedule_epochs": 1,
        "stage3_warmup_ratio": 0.10,
        "stage3_lr_scheduler_type": "constant_with_warmup",
        "stage3_initial_temp": 1.0,
        "stage3_final_temp": 0.775,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
    },
    "visual_router_raw_adapter_v1": {
        **V4_RECOVER_BASE,
        "raw_visual_adapter": True,
        "mmrl_relation_loss_weight": 0.0,
        "direct_mmrl_output": False,
        "enable_adapter_router_identity_residual": False,
    },
    "visual_router_direct_mmrl_seed44": {
        **BASE_VISUAL_ROUTER_MODEL,
        "adapter_usage_balance_loss_weight": 0.0,
        "adapter_sample_entropy_loss_weight": 0.0,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "direct_mmrl_output": True,
        "enable_adapter_router_identity_residual": False,
        "enable_deepstack_mmrl_residual": False,
        "deepstack_mmrl_residual_scale": 0.0,
    },
    "visual_router_two_adapter_seed44": {
        **BASE_VISUAL_ROUTER_MODEL,
        "visual_residual_adapter_count": 2,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.020,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0003,
        "adapter_diversity_loss_weight": 0.0,
        "adapter_diversity_target_low": 0.30,
        "adapter_diversity_target_high": 0.58,
        "adapter_sample_entropy_target": 0.72,
        "adapter_common_mode_target": 0.94,
        "adapter_effective_delta_target_low": 0.52,
        "adapter_effective_delta_target_high": 0.98,
        "enable_deepstack_mmrl_residual": False,
        "deepstack_mmrl_residual_scale": 0.0,
    },
    "visual_router_single_adapter_seed44": {
        **BASE_VISUAL_ROUTER_MODEL,
        "visual_residual_adapter_count": 1,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.020,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0003,
        "adapter_diversity_loss_weight": 0.0,
        "adapter_diversity_target_low": 0.30,
        "adapter_diversity_target_high": 0.58,
        "adapter_sample_entropy_target": 0.72,
        "adapter_common_mode_target": 0.94,
        "adapter_effective_delta_target_low": 0.52,
        "adapter_effective_delta_target_high": 0.98,
        "enable_deepstack_mmrl_residual": False,
        "deepstack_mmrl_residual_scale": 0.0,
    },
}
EXPERIMENTS["visual_router_spatial_grounding_v1"] = {
    **EXPERIMENTS["visual_router_relation_joint_multiturn_cosine_v2"],
    "stage3_expert_json": [
        "/root/autodl-tmp/dataset/spatial_grounding_1.json",
        "/root/autodl-tmp/dataset/spatial_grounding_14.json",
    ],
    "stage3_expert_img_dir": [
        "/root/autodl-tmp/dataset/1/train",
        "/root/autodl-tmp/dataset/14",
    ],
    "stage3_total_limit": 20000,
}
EXPERIMENTS["visual_router_legacy_3cdf58d_joint_cosine_v2"] = (
    EXPERIMENTS["visual_router_relation_joint_multiturn_cosine_v2"].copy()
)
EXPERIMENTS["visual_router_legacy_3cdf58d_joint_cosine_v2"].update({
    "expert_json": [
        "/root/autodl-tmp/dataset/1json.json",
        # "/root/autodl-tmp/dataset/2conv_c.json",
        "/root/autodl-tmp/dataset/1conv_c.json",
        # "/root/autodl-tmp/dataset/4conv_c.json",
        "/root/autodl-tmp/dataset/14json.json",
        "/root/autodl-tmp/dataset/prof_test.json",
        "/root/autodl-tmp/dataset/test2_train.json",
        # "/root/autodl-tmp/dataset/test7_train.json",
    ],
    "expert_img_dir": [
        "/root/autodl-tmp/dataset/1/train",
        "/root/autodl-tmp/dataset/2/train",
        # "/root/autodl-tmp/dataset/4/train",
        "/root/autodl-tmp/dataset/14",
    ],
    "stage1_only_expert_json": [],
    "stage1_only_expert_img_dir": [],
})
SELECTED_EXPERIMENT = os.getenv(
    "MMRL_EXPERIMENT",
    "visual_router_layer_fixed_v4_diversity_recover",
)
if SELECTED_EXPERIMENT not in EXPERIMENTS:
    raise ValueError(f"Unknown MMRL_EXPERIMENT={SELECTED_EXPERIMENT!r}, choices={sorted(EXPERIMENTS)}")
EXP_CFG = EXPERIMENTS[SELECTED_EXPERIMENT]

# ===== 可调参数统一字典（不使用 argparse）=====
CFG = {
    "model_path": os.getenv("MMRL_MODEL_PATH", "/root/autodl-tmp/model"),
    "output_dir": os.getenv("MMRL_OUTPUT_DIR", f"./output/{SELECTED_EXPERIMENT}"),
    "experiment_name": SELECTED_EXPERIMENT,
    "experiment": EXP_CFG,
    "data": {
        "expert_json": EXP_CFG.get("expert_json", [
            "/root/autodl-tmp/dataset/1json.json",
            "/root/autodl-tmp/dataset/2conv_c.json",
            "/root/autodl-tmp/dataset/1conv_c.json",
            "/root/autodl-tmp/dataset/4conv_c.json",
            "/root/autodl-tmp/dataset/14json.json",
            "/root/autodl-tmp/dataset/prof_test.json",
            "/root/autodl-tmp/dataset/test2_train.json",
            "/root/autodl-tmp/dataset/test7_train.json",
        ]),
        "expert_img_dir": EXP_CFG.get("expert_img_dir", [
            "/root/autodl-tmp/dataset/1/train",
            "/root/autodl-tmp/dataset/2/train",
            "/root/autodl-tmp/dataset/4/train",
            "/root/autodl-tmp/dataset/14",
        ]),
        "stage1_only_expert_json": EXP_CFG.get("stage1_only_expert_json", []),
        "stage1_only_expert_img_dir": EXP_CFG.get("stage1_only_expert_img_dir", []),
        "general_json": [
            "/root/autodl-tmp/dataset/llava_instruct_150k.json",
            "/root/autodl-tmp/dataset/gen_test.json",
            "/root/autodl-tmp/dataset/conversation_58k.json",
        ],
        "general_img_dir": [
            "/root/autodl-tmp/dataset/gen/train2017",
            "/root/autodl-tmp/dataset/gen/val2017",
        ],
        "total_limit": 20000,  # Stage1: expert/general 各 20000；Stage3: expert-only 20000
    },
    "train": {
        "experiment_name": SELECTED_EXPERIMENT,
        "experiment_cfg": EXP_CFG,
        "seed": int(os.getenv("MMRL_SEED", "42")),
        "data_sampling_seed": int(os.getenv("MMRL_DATA_SAMPLING_SEED", "42")),
        "data_order_seed": int(os.getenv("MMRL_DATA_ORDER_SEED", "42")),
        "deterministic_sampling": os.getenv("MMRL_DETERMINISTIC_SAMPLING", "1") == "1",
        "eval_each_epoch": os.getenv("MMRL_EVAL_EACH_EPOCH", "0") == "1",
        "live_final_eval": os.getenv("MMRL_LIVE_FINAL_EVAL", "0") == "1",
        "save_extrema_checkpoints": os.getenv("MMRL_SAVE_EXTREMA_CHECKPOINTS", "1") == "1",
        "stage1_gate_preflight": os.getenv(
            "MMRL_STAGE1_GATE_PREFLIGHT",
            "1",
        ) == "1",
        "visual_residual_adapter_count": int(os.getenv(
            "MMRL_VISUAL_RESIDUAL_ADAPTER_COUNT",
            str(EXP_CFG.get("visual_residual_adapter_count", 4)),
        )),
        "adapter_usage_balance_loss_weight": float(os.getenv(
            "MMRL_ADAPTER_USAGE_BALANCE_LOSS_WEIGHT",
            str(EXP_CFG.get("adapter_usage_balance_loss_weight", 0.0)),
        )),
        "adapter_sample_entropy_loss_weight": float(os.getenv(
            "MMRL_ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT",
            str(EXP_CFG.get("adapter_sample_entropy_loss_weight", 0.0)),
        )),
        "adapter_common_mode_loss_weight": float(os.getenv(
            "MMRL_ADAPTER_COMMON_MODE_LOSS_WEIGHT",
            str(EXP_CFG.get("adapter_common_mode_loss_weight", 0.0)),
        )),
        "adapter_effective_delta_loss_weight": float(os.getenv(
            "MMRL_ADAPTER_EFFECTIVE_DELTA_LOSS_WEIGHT",
            str(EXP_CFG.get("adapter_effective_delta_loss_weight", 0.0)),
        )),
        "mmrl_relation_loss_weight": float(os.getenv(
            "MMRL_RELATION_LOSS_WEIGHT",
            str(EXP_CFG.get("mmrl_relation_loss_weight", 0.0)),
        )),
        "mmrl_relation_max_tokens": int(os.getenv(
            "MMRL_RELATION_MAX_TOKENS",
            str(EXP_CFG.get("mmrl_relation_max_tokens", 64)),
        )),
        "mmrl_variance_floor_ratio": float(os.getenv(
            "MMRL_VARIANCE_FLOOR_RATIO",
            str(EXP_CFG.get("mmrl_variance_floor_ratio", 0.50)),
        )),
        "mmrl_variance_floor_weight": float(os.getenv(
            "MMRL_VARIANCE_FLOOR_WEIGHT",
            str(EXP_CFG.get("mmrl_variance_floor_weight", 0.10)),
        )),
        "adapter_diversity_loss_weight": float(os.getenv(
            "MMRL_ADAPTER_DIVERSITY_LOSS_WEIGHT",
            str(EXP_CFG.get("adapter_diversity_loss_weight", 0.0)),
        )),
        "adapter_diversity_target_low": float(os.getenv(
            "MMRL_ADAPTER_DIVERSITY_TARGET_LOW",
            str(EXP_CFG.get("adapter_diversity_target_low", 0.30)),
        )),
        "adapter_diversity_target_high": float(os.getenv(
            "MMRL_ADAPTER_DIVERSITY_TARGET_HIGH",
            str(EXP_CFG.get("adapter_diversity_target_high", 0.58)),
        )),
        "adapter_diversity_upper_weight": float(os.getenv(
            "MMRL_ADAPTER_DIVERSITY_UPPER_WEIGHT",
            str(EXP_CFG.get("adapter_diversity_upper_weight", 2.0)),
        )),
        "adapter_diversity_worst_pair_weight": float(os.getenv(
            "MMRL_ADAPTER_DIVERSITY_WORST_PAIR_WEIGHT",
            str(EXP_CFG.get("adapter_diversity_worst_pair_weight", 1.0)),
        )),
        "enable_adapter_router_identity_residual": os.getenv(
            "MMRL_ENABLE_ADAPTER_ROUTER_IDENTITY_RESIDUAL",
            "1" if EXP_CFG.get("enable_adapter_router_identity_residual", False) else "0",
        ) == "1",
        "direct_mmrl_output": os.getenv(
            "MMRL_DIRECT_MMRL_OUTPUT",
            "1" if EXP_CFG.get("direct_mmrl_output", False) else "0",
        ) == "1",
        "raw_visual_adapter": os.getenv(
            "MMRL_RAW_VISUAL_ADAPTER",
            "1" if EXP_CFG.get("raw_visual_adapter", False) else "0",
        ) == "1",
        "enable_deepstack_mmrl_residual": os.getenv(
            "MMRL_ENABLE_DEEPSTACK_MMRL_RESIDUAL",
            "1" if EXP_CFG.get("enable_deepstack_mmrl_residual", False) else "0",
        ) == "1",
        "deepstack_mmrl_residual_scale": float(os.getenv(
            "MMRL_DEEPSTACK_MMRL_RESIDUAL_SCALE",
            str(EXP_CFG.get("deepstack_mmrl_residual_scale", 0.0)),
        )),
        "adapter_sample_entropy_target": float(os.getenv(
            "MMRL_ADAPTER_SAMPLE_ENTROPY_TARGET",
            str(EXP_CFG.get("adapter_sample_entropy_target", 0.40)),
        )),
        "adapter_common_mode_target": float(os.getenv(
            "MMRL_ADAPTER_COMMON_MODE_TARGET",
            str(EXP_CFG.get("adapter_common_mode_target", 0.85)),
        )),
        "adapter_effective_delta_target_low": float(os.getenv(
            "MMRL_ADAPTER_EFFECTIVE_DELTA_TARGET_LOW",
            str(EXP_CFG.get("adapter_effective_delta_target_low", 0.78)),
        )),
        "adapter_effective_delta_target_high": float(os.getenv(
            "MMRL_ADAPTER_EFFECTIVE_DELTA_TARGET_HIGH",
            str(EXP_CFG.get("adapter_effective_delta_target_high", 1.10)),
        )),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "grad_spike_ema_alpha": float(os.getenv("MMRL_GRAD_SPIKE_EMA_ALPHA", "0.10")),
        "grad_spike_ratio_threshold": float(os.getenv("MMRL_GRAD_SPIKE_RATIO_THRESHOLD", "3.0")),
        "grad_spike_abs_threshold": float(os.getenv("MMRL_GRAD_SPIKE_ABS_THRESHOLD", "0.5")),
        "grad_spike_cooldown_steps": int(os.getenv("MMRL_GRAD_SPIKE_COOLDOWN_STEPS", "20")),

        "console_log_every": 50,
        "metric_smooth_window": 30,
        "metric_ema_alpha": 0.12,
        "metric_scatter_stride": 5,
        "save_debug_figure": False,
        "diag_every_steps": int(os.getenv(
            "MMRL_DIAG_EVERY_STEPS",
            str(EXP_CFG.get("diag_every_steps", 1000)),
        )),
        "mmrl_diagnostics_keep_keys": [
            "ce_loss",
            "delta_to_org_ratio",
            "mmrl_delta_to_org_ratio",
            "mmrl_relation_loss_scaled",
            "mmrl_relation_gram_loss",
            "mmrl_variance_floor_loss",
            "adapter_effective_delta_ratio_mean",
            "final_to_gated_ratio",
            "delta_transform_cos_mean",
            "adapter_route_entropy_norm",
            "adapter_route_confidence",
            "adapter_usage_0",
            "adapter_usage_1",
            "adapter_usage_2",
            "adapter_usage_3",
            "adapter_usage_balance_loss_scaled",
            "adapter_sample_entropy_loss_scaled",
            "adapter_effective_delta_loss_scaled",
            "adapter_pairwise_cos_mean",
            "adapter_pairwise_cos_max",
            "adapter_pairwise_cos_0_1",
            "adapter_pairwise_cos_0_2",
            "adapter_pairwise_cos_0_3",
            "adapter_pairwise_cos_1_2",
            "adapter_pairwise_cos_1_3",
            "adapter_pairwise_cos_2_3",
            "adapter_output_norm_0",
            "adapter_output_norm_1",
            "adapter_output_norm_2",
            "adapter_output_norm_3",
            "adapter_contribution_norm_0",
            "adapter_contribution_norm_1",
            "adapter_contribution_norm_2",
            "adapter_contribution_norm_3",
            "shared_rep_norm_mean",
            "shared_rep_grad_norm",
            "v_projector_grad_norm_mean",
            "adapter_router_grad_norm_mean",
            "visual_adapter_grad_norm_mean",
            "temperature",
        ],

        "learning_rate": {
            1: 1e-4,  # 仅分类器
            3: float(EXP_CFG.get("stage3_learning_rate", 8e-5)),
        },
        "epochs": {
            1: 1,
            3: 1,
        },
        "schedule_epochs": {
            3: int(EXP_CFG.get("stage3_schedule_epochs", 4)),
        },
        "stage3_warmup_ratio": float(EXP_CFG.get("stage3_warmup_ratio", 0.05)),
        "stage3_lr_scheduler_type": str(EXP_CFG.get("stage3_lr_scheduler_type", "cosine")),
        "initial_temp": float(EXP_CFG.get("stage3_initial_temp", 1.0)),
        "final_temp": float(EXP_CFG.get("stage3_final_temp", 0.1)),
    },
    "ablation_order": [1, 3]
}


def main():
    seed_before_model_init(CFG["train"]["seed"])
    live_final_eval = CFG["train"].get("live_final_eval", False)
    if CFG["train"].get("eval_each_epoch", False) or live_final_eval:
        preflight_live_epoch_evaluation()
    model, processor = build_model_and_processor(CFG["model_path"], experiment_cfg=CFG["experiment"])

    for sid in CFG["ablation_order"]:
        print(f"\n========== Running Stage {sid} ==========")
        run_stage(
            stage_id=sid,
            model=model,
            processor=processor,
            data_cfg=CFG["data"],
            train_cfg=CFG["train"],
            output_dir=CFG["output_dir"]
        )
        if sid == 1 and CFG["train"].get("stage1_gate_preflight", True):
            run_professional_gate_preflight(model, processor)
    final_dir = Path(CFG["output_dir"]) / "final"
    if live_final_eval:
        log_path = Path(CFG["output_dir"]) / "eval" / "test.log"
        print("[FINAL-EVAL] online evaluation begin")
        summary = run_live_epoch_evaluation(model, processor, log_path)
        score = float(summary["score"])
        print(f"[FINAL-EVAL] score={score} completed")

        if CFG["train"].get("save_extrema_checkpoints", True):
            keep, prior_min, prior_max = _should_keep_extrema_checkpoint(
                CFG["output_dir"], score
            )
            if keep:
                model.save_pretrained(final_dir)
                processor.save_pretrained(final_dir)
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
            print("[CHECKPOINT] final checkpoint saving disabled")
    else:
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f"final model saved to {final_dir}")

    print("All stages done.")


if __name__ == "__main__":
    main()
