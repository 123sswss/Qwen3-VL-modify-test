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
    # When enabled, Stage 3 trains an exact copy of the Stage 1 gate poolers.
    "decouple_stage_pooling": os.getenv(
        "MMRL_DECOUPLE_STAGE_POOLING", "0"
    ) == "1",
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
})
EXPERIMENTS["visual_router_fixed_stage1_constant_v1"] = {
    **EXPERIMENTS["visual_router_legacy_3cdf58d_joint_cosine_v2"],
    "stage3_lr_scheduler_type": "constant_with_warmup",
}
EXPERIMENTS["visual_router_fixed_stage1_late_decay_v1"] = {
    **EXPERIMENTS["visual_router_fixed_stage1_constant_v1"],
    "stage3_max_steps": 625,
    "stage3_warmup_steps": 63,
    "stage3_hold_until_step": 500,
}
EXPERIMENTS["visual_router_fixed_stage1_late_decay_balanced_loss_v1"] = {
    **EXPERIMENTS["visual_router_fixed_stage1_late_decay_v1"],
    "mmrl_relation_loss_weight": 0.0125,
    "adapter_effective_delta_loss_weight": 0.0004,
    "adapter_effective_delta_target_low": 0.58,
}
for relation_tag, relation_weight in (
    ("r0125", 0.0125),
    ("r0250", 0.0250),
    ("r0500", 0.0500),
):
    for delta_tag, delta_floor in (
        ("d058", 0.58),
        ("d070", 0.70),
    ):
        EXPERIMENTS[f"visual_router_loss_matrix_{relation_tag}_{delta_tag}_v1"] = {
            **EXPERIMENTS["visual_router_fixed_stage1_late_decay_v1"],
            "mmrl_relation_loss_weight": relation_weight,
            "adapter_effective_delta_loss_weight": 0.0004,
            "adapter_effective_delta_target_low": delta_floor,
            "adapter_effective_delta_target_high": 0.98,
        }

STABILITY_R0125_D058_BASE = EXPERIMENTS[
    "visual_router_loss_matrix_r0125_d058_v1"
]
EXPERIMENTS["visual_router_stability_r0125_d058_heterogeneous_v1"] = {
    **STABILITY_R0125_D058_BASE,
}
EXPERIMENTS["visual_router_stability_r0125_d058_homogeneous_v1"] = {
    **STABILITY_R0125_D058_BASE,
    "stage3_adapter_learning_rates": [7e-5, 7e-5, 7e-5, 7e-5],
}

STABILITY_R0125_D058_COMMON_EXPECTED = {
    "random_init_adapter_output_count": 0,
    "adapter_usage_balance_loss_weight": 0.0026,
    "adapter_sample_entropy_loss_weight": 0.020,
    "adapter_sample_entropy_target": 0.72,
    "adapter_effective_delta_loss_weight": 0.0004,
    "adapter_effective_delta_target_low": 0.58,
    "adapter_effective_delta_target_high": 0.98,
    "mmrl_relation_loss_weight": 0.0125,
    "direct_mmrl_output": False,
    "raw_visual_adapter": False,
    "visual_residual_adapter_count": 4,
    "stage3_mmrl_learning_rate": 6e-5,
    "stage3_router_learning_rate": 8e-5,
    "stage3_lr_scheduler_type": "constant_with_warmup",
    "stage3_max_steps": 625,
    "stage3_warmup_steps": 63,
    "stage3_hold_until_step": 500,
}
STABILITY_R0125_D058_EXPECTED = {
    "visual_router_stability_r0125_d058_heterogeneous_v1": {
        **STABILITY_R0125_D058_COMMON_EXPECTED,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
    },
    "visual_router_stability_r0125_d058_homogeneous_v1": {
        **STABILITY_R0125_D058_COMMON_EXPECTED,
        "stage3_adapter_learning_rates": [7e-5, 7e-5, 7e-5, 7e-5],
    },
}

FINAL_CONSTRAINT_MATRIX_3X3_COMMON_EXPECTED = {
    "random_init_adapter_output_count": 0,
    "adapter_usage_balance_loss_weight": 0.0026,
    "adapter_sample_entropy_loss_weight": 0.020,
    "adapter_sample_entropy_target": 0.72,
    "adapter_effective_delta_target_low": 0.70,
    "adapter_effective_delta_target_high": 0.98,
    "direct_mmrl_output": False,
    "raw_visual_adapter": False,
    "visual_residual_adapter_count": 4,
    "stage3_mmrl_learning_rate": 6e-5,
    "stage3_router_learning_rate": 8e-5,
    "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
    "stage3_lr_scheduler_type": "constant_with_warmup",
    "stage3_max_steps": 625,
    "stage3_warmup_steps": 63,
    "stage3_hold_until_step": 500,
}
FINAL_CONSTRAINT_MATRIX_3X3_EXPECTED = {}
for relation_tag, relation_weight in (
    ("r0125", 0.0125),
    ("r0250", 0.0250),
    ("r0375", 0.0375),
):
    for effective_tag, effective_weight in (
        ("w0004", 0.0004),
        ("w0008", 0.0008),
        ("w0016", 0.0016),
    ):
        experiment_name = (
            "visual_router_final_constraint_matrix_"
            f"{relation_tag}_{effective_tag}_d070_v1"
        )
        EXPERIMENTS[experiment_name] = {
            **EXPERIMENTS["visual_router_fixed_stage1_late_decay_v1"],
            "mmrl_relation_loss_weight": relation_weight,
            "adapter_effective_delta_loss_weight": effective_weight,
            "adapter_effective_delta_target_low": 0.70,
            "adapter_effective_delta_target_high": 0.98,
        }
        FINAL_CONSTRAINT_MATRIX_3X3_EXPECTED[experiment_name] = {
            **FINAL_CONSTRAINT_MATRIX_3X3_COMMON_EXPECTED,
            "mmrl_relation_loss_weight": relation_weight,
            "adapter_effective_delta_loss_weight": effective_weight,
        }

FINAL_TUNING_BASE = EXPERIMENTS[
    "visual_router_loss_matrix_r0250_d058_v1"
]
EXPERIMENTS["visual_router_final_mmrl_lr8e5_v1"] = {
    **FINAL_TUNING_BASE,
    "stage3_pooling_learning_rate": 6e-5,
    "stage3_mmrl_learning_rate": 8e-5,
}
EXPERIMENTS["visual_router_final_entropy_target080_v1"] = {
    **FINAL_TUNING_BASE,
    "stage3_pooling_learning_rate": 6e-5,
    "adapter_sample_entropy_target": 0.80,
}
EXPERIMENTS["visual_router_custom_optimizer_beta2_098_v1"] = {
    **FINAL_TUNING_BASE,
    "stage3_pooling_learning_rate": 6e-5,
    "stage3_adam_beta1": 0.9,
    "stage3_adam_beta2": 0.98,
    "stage3_adam_epsilon": 1e-8,
    "stage3_weight_decay": 0.0,
}
EXPERIMENTS["visual_router_custom_adapter_lr_compressed_v1"] = {
    **FINAL_TUNING_BASE,
    "stage3_pooling_learning_rate": 6e-5,
    "stage3_adapter_learning_rates": [5.5e-5, 6.5e-5, 7.5e-5, 8.5e-5],
    "stage3_adam_beta1": 0.9,
    "stage3_adam_beta2": 0.999,
    "stage3_adam_epsilon": 1e-8,
    "stage3_weight_decay": 0.0,
}
EXPERIMENTS["visual_router_custom_adapter_lr_expanded_v1"] = {
    **FINAL_TUNING_BASE,
    "stage3_pooling_learning_rate": 6e-5,
    "stage3_adapter_learning_rates": [2.5e-5, 5.5e-5, 8.5e-5, 11.5e-5],
    "stage3_adam_beta1": 0.9,
    "stage3_adam_beta2": 0.999,
    "stage3_adam_epsilon": 1e-8,
    "stage3_weight_decay": 0.0,
}

FINAL_TUNING_EXPECTED = {
    "visual_router_final_mmrl_lr8e5_v1": {
        "adapter_sample_entropy_target": 0.72,
        "stage3_mmrl_learning_rate": 8e-5,
    },
    "visual_router_final_entropy_target080_v1": {
        "adapter_sample_entropy_target": 0.80,
        "stage3_mmrl_learning_rate": 6e-5,
    },
    "visual_router_custom_optimizer_beta2_098_v1": {
        "adapter_sample_entropy_target": 0.72,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_adam_beta1": 0.9,
        "stage3_adam_beta2": 0.98,
        "stage3_adam_epsilon": 1e-8,
        "stage3_weight_decay": 0.0,
    },
    "visual_router_custom_adapter_lr_compressed_v1": {
        "adapter_sample_entropy_target": 0.72,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_adam_beta1": 0.9,
        "stage3_adam_beta2": 0.999,
        "stage3_adam_epsilon": 1e-8,
        "stage3_weight_decay": 0.0,
    },
    "visual_router_custom_adapter_lr_expanded_v1": {
        "adapter_sample_entropy_target": 0.72,
        "stage3_mmrl_learning_rate": 6e-5,
        "stage3_adam_beta1": 0.9,
        "stage3_adam_beta2": 0.999,
        "stage3_adam_epsilon": 1e-8,
        "stage3_weight_decay": 0.0,
    },
}
for expected_values in FINAL_TUNING_EXPECTED.values():
    expected_values.update({
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.020,
        "adapter_effective_delta_loss_weight": 0.0004,
        "adapter_effective_delta_target_low": 0.58,
        "adapter_effective_delta_target_high": 0.98,
        "mmrl_relation_loss_weight": 0.0250,
        "stage3_pooling_learning_rate": 6e-5,
        "stage3_router_learning_rate": 8e-5,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "stage3_lr_scheduler_type": "constant_with_warmup",
        "stage3_max_steps": 625,
        "stage3_warmup_steps": 63,
        "stage3_hold_until_step": 500,
    })
FINAL_TUNING_EXPECTED[
    "visual_router_custom_adapter_lr_compressed_v1"
]["stage3_adapter_learning_rates"] = [5.5e-5, 6.5e-5, 7.5e-5, 8.5e-5]
FINAL_TUNING_EXPECTED[
    "visual_router_custom_adapter_lr_expanded_v1"
]["stage3_adapter_learning_rates"] = [2.5e-5, 5.5e-5, 8.5e-5, 11.5e-5]

ABLATION_FULL_BASE = EXPERIMENTS[
    "visual_router_loss_matrix_r0125_d070_v1"
]
EXPERIMENTS["visual_router_ablation_full_r0125_d070_v1"] = {
    **ABLATION_FULL_BASE,
}
EXPERIMENTS["visual_router_ablation_wo_rep_token_v1"] = {
    **ABLATION_FULL_BASE,
    "raw_visual_adapter": True,
    "mmrl_relation_loss_weight": 0.0,
}
EXPERIMENTS["visual_router_ablation_single_adapter_v1"] = {
    **ABLATION_FULL_BASE,
    "visual_residual_adapter_count": 1,
    "stage3_adapter_learning_rates": [7e-5],
}
EXPERIMENTS["visual_router_ablation_homogeneous_adapter_lr_v1"] = {
    **ABLATION_FULL_BASE,
    "stage3_adapter_learning_rates": [7e-5, 7e-5, 7e-5, 7e-5],
}
EXPERIMENTS["visual_router_ablation_no_relation_v1"] = {
    **ABLATION_FULL_BASE,
    "mmrl_relation_loss_weight": 0.0,
}

ABLATION_COMMON_EXPECTED = {
    "random_init_adapter_output_count": 0,
    "adapter_usage_balance_loss_weight": 0.0026,
    "adapter_sample_entropy_loss_weight": 0.020,
    "adapter_sample_entropy_target": 0.72,
    "adapter_effective_delta_loss_weight": 0.0004,
    "adapter_effective_delta_target_low": 0.70,
    "adapter_effective_delta_target_high": 0.98,
    "direct_mmrl_output": False,
    "stage3_mmrl_learning_rate": 6e-5,
    "stage3_router_learning_rate": 8e-5,
    "stage3_lr_scheduler_type": "constant_with_warmup",
    "stage3_max_steps": 625,
    "stage3_warmup_steps": 63,
    "stage3_hold_until_step": 500,
}
ABLATION_EXPECTED = {
    "visual_router_ablation_full_r0125_d070_v1": {
        **ABLATION_COMMON_EXPECTED,
        "raw_visual_adapter": False,
        "visual_residual_adapter_count": 4,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "mmrl_relation_loss_weight": 0.0125,
    },
    "visual_router_ablation_wo_rep_token_v1": {
        **ABLATION_COMMON_EXPECTED,
        "raw_visual_adapter": True,
        "visual_residual_adapter_count": 4,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "mmrl_relation_loss_weight": 0.0,
    },
    "visual_router_ablation_single_adapter_v1": {
        **ABLATION_COMMON_EXPECTED,
        "raw_visual_adapter": False,
        "visual_residual_adapter_count": 1,
        "stage3_adapter_learning_rates": [7e-5],
        "mmrl_relation_loss_weight": 0.0125,
    },
    "visual_router_ablation_homogeneous_adapter_lr_v1": {
        **ABLATION_COMMON_EXPECTED,
        "raw_visual_adapter": False,
        "visual_residual_adapter_count": 4,
        "stage3_adapter_learning_rates": [7e-5, 7e-5, 7e-5, 7e-5],
        "mmrl_relation_loss_weight": 0.0125,
    },
    "visual_router_ablation_no_relation_v1": {
        **ABLATION_COMMON_EXPECTED,
        "raw_visual_adapter": False,
        "visual_residual_adapter_count": 4,
        "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
        "mmrl_relation_loss_weight": 0.0,
    },
}

LOSS_TUNING_3X1_BASE = EXPERIMENTS[
    "visual_router_loss_matrix_r0125_d058_v1"
]
EXPERIMENTS["visual_router_loss_tuning_no_effective_delta_v1"] = {
    **LOSS_TUNING_3X1_BASE,
    "adapter_effective_delta_loss_weight": 0.0,
}
EXPERIMENTS["visual_router_loss_tuning_half_usage_v1"] = {
    **LOSS_TUNING_3X1_BASE,
    "adapter_usage_balance_loss_weight": 0.0013,
}
EXPERIMENTS["visual_router_loss_tuning_relation_r0100_v1"] = {
    **LOSS_TUNING_3X1_BASE,
    "mmrl_relation_loss_weight": 0.0100,
}

LOSS_TUNING_3X1_EXPECTED = {
    "visual_router_loss_tuning_no_effective_delta_v1": {
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_effective_delta_loss_weight": 0.0,
        "mmrl_relation_loss_weight": 0.0125,
    },
    "visual_router_loss_tuning_half_usage_v1": {
        "adapter_usage_balance_loss_weight": 0.0013,
        "adapter_effective_delta_loss_weight": 0.0004,
        "mmrl_relation_loss_weight": 0.0125,
    },
    "visual_router_loss_tuning_relation_r0100_v1": {
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_effective_delta_loss_weight": 0.0004,
        "mmrl_relation_loss_weight": 0.0100,
    },
}
for expected_values in LOSS_TUNING_3X1_EXPECTED.values():
    expected_values.update({
        "adapter_sample_entropy_loss_weight": 0.020,
        "adapter_sample_entropy_target": 0.72,
        "adapter_effective_delta_target_low": 0.58,
        "adapter_effective_delta_target_high": 0.98,
        "stage3_lr_scheduler_type": "constant_with_warmup",
        "stage3_max_steps": 625,
        "stage3_warmup_steps": 63,
        "stage3_hold_until_step": 500,
    })

EXPERIMENTS["visual_router_fixed_stage1_lr5e5_v1"] = {
    **EXPERIMENTS["visual_router_legacy_3cdf58d_joint_cosine_v2"],
    "stage1_learning_rate": 5e-5,
}
EXPERIMENTS["visual_router_fixed_stage1_pooling_lr1e5_v1"] = {
    **EXPERIMENTS["visual_router_legacy_3cdf58d_joint_cosine_v2"],
    "stage3_pooling_learning_rate": 1e-5,
}
EXPERIMENTS["visual_router_fixed_stage1_global_lr_half_v1"] = {
    **EXPERIMENTS["visual_router_legacy_3cdf58d_joint_cosine_v2"],
    "stage3_learning_rate": 3e-5,
    "stage3_mmrl_learning_rate": 3e-5,
    "stage3_router_learning_rate": 4e-5,
    "stage3_adapter_learning_rates": [2e-5, 3e-5, 4e-5, 5e-5],
}
EXPERIMENTS["visual_router_fixed_stage1_mmrl_only_lr3e5_v1"] = {
    **EXPERIMENTS["visual_router_legacy_3cdf58d_joint_cosine_v2"],
    "stage3_pooling_learning_rate": 6e-5,
    "stage3_mmrl_learning_rate": 3e-5,
    "stage3_router_learning_rate": 8e-5,
    "stage3_adapter_learning_rates": [4e-5, 6e-5, 8e-5, 1e-4],
}
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
            1: float(EXP_CFG.get("stage1_learning_rate", 1e-4)),
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


def audit_loss_tuning_config():
    expected = LOSS_TUNING_3X1_EXPECTED.get(SELECTED_EXPERIMENT)
    audit_label = "LOSS_TUNING_3X1_AUDIT"
    if expected is None:
        expected = FINAL_TUNING_EXPECTED.get(SELECTED_EXPERIMENT)
        audit_label = "FINAL_TUNING_AUDIT"
    if expected is None:
        expected = ABLATION_EXPECTED.get(SELECTED_EXPERIMENT)
        audit_label = "ABLATION_CONFIG_AUDIT"
    if expected is None:
        expected = STABILITY_R0125_D058_EXPECTED.get(SELECTED_EXPERIMENT)
        audit_label = "STABILITY_CONFIG_AUDIT"
    if expected is None:
        expected = FINAL_CONSTRAINT_MATRIX_3X3_EXPECTED.get(SELECTED_EXPERIMENT)
        audit_label = "FINAL_CONSTRAINT_MATRIX_3X3_AUDIT"
    if expected is None:
        return

    effective = dict(CFG["experiment"])
    for key in (
        "adapter_usage_balance_loss_weight",
        "adapter_sample_entropy_loss_weight",
        "adapter_sample_entropy_target",
        "adapter_effective_delta_loss_weight",
        "adapter_effective_delta_target_low",
        "adapter_effective_delta_target_high",
        "mmrl_relation_loss_weight",
    ):
        effective[key] = CFG["train"][key]

    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = effective.get(key)
        if isinstance(expected_value, float):
            matches = (
                isinstance(actual_value, (int, float))
                and math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches[key] = {
                "expected": expected_value,
                "actual": actual_value,
            }

    if mismatches:
        raise RuntimeError(
            f"Loss-tuning config audit failed for {SELECTED_EXPERIMENT}: "
            f"{mismatches}"
        )

    audited = " ".join(
        f"{key}={effective[key]}" for key in expected
    )
    print(f"[{audit_label}] {SELECTED_EXPERIMENT} {audited}")


def main():
    audit_loss_tuning_config()
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
        try:
            summary = run_live_epoch_evaluation(model, processor, log_path)
        except Exception:
            recovery_dir = Path(CFG["output_dir"]) / "eval_failed_recovery"
            model.save_pretrained(recovery_dir)
            processor.save_pretrained(recovery_dir)
            print(
                "[FINAL-EVAL-RECOVERY] evaluation failed; trained weights saved to "
                f"{recovery_dir}"
            )
            raise
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
