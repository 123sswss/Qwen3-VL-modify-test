# new_train.py
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch

from train_stages import build_model_and_processor, run_stage
from live_epoch_eval import preflight_live_epoch_evaluation


def seed_before_model_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[Reproducibility] seeded model init and training RNGs with seed={seed}")


BASE_VISUAL_ROUTER_MODEL = {
    "rp_space_length": 40,
    "visual_residual_adapter_count": 4,
    "adapter_usage_balance_loss_weight": 0.005,
    "adapter_sample_entropy_loss_weight": 0.02,
    "adapter_common_mode_loss_weight": 0.03,
    "adapter_effective_delta_loss_weight": 0.0,
    "adapter_diversity_loss_weight": 0.0,
    "adapter_diversity_target_low": 0.30,
    "adapter_diversity_target_high": 0.58,
    "adapter_diversity_upper_weight": 2.0,
    "adapter_diversity_worst_pair_weight": 1.0,
    "prototype_anchor_loss_weight": 0.0,
    "prototype_anchor_temperature": 0.20,
    "prototype_anchor_momentum": 0.95,
    "prototype_anchor_min_confidence": 0.40,
    "prototype_anchor_assignment_power": 2.0,
    "prototype_anchor_init_noise": 0.10,
    "enable_adapter_router_identity_residual": False,
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


EXPERIMENTS = {
    "visual_router_layer_fixed_v4_diversity_recover": {
        **BASE_VISUAL_ROUTER_MODEL,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.020,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0003,
        "prototype_anchor_loss_weight": 0.0,
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
    "visual_router_layer_fixed_v4_diversity_recover_identity_residual": {
        **BASE_VISUAL_ROUTER_MODEL,
        "adapter_usage_balance_loss_weight": 0.0026,
        "adapter_sample_entropy_loss_weight": 0.020,
        "adapter_common_mode_loss_weight": 0.0,
        "adapter_effective_delta_loss_weight": 0.0003,
        "prototype_anchor_loss_weight": 0.0,
        "adapter_diversity_loss_weight": 0.0,
        "adapter_diversity_target_low": 0.30,
        "adapter_diversity_target_high": 0.58,
        "adapter_sample_entropy_target": 0.72,
        "adapter_common_mode_target": 0.94,
        "adapter_effective_delta_target_low": 0.52,
        "adapter_effective_delta_target_high": 0.98,
        "enable_adapter_router_identity_residual": True,
        "enable_deepstack_mmrl_residual": False,
        "deepstack_mmrl_residual_scale": 0.0,
    },
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
        "deterministic_sampling": os.getenv("MMRL_DETERMINISTIC_SAMPLING", "1") == "1",
        "eval_each_epoch": os.getenv("MMRL_EVAL_EACH_EPOCH", "0") == "1",
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
        "prototype_anchor_loss_weight": float(os.getenv(
            "MMRL_PROTOTYPE_ANCHOR_LOSS_WEIGHT",
            str(EXP_CFG.get("prototype_anchor_loss_weight", 0.0)),
        )),
        "prototype_anchor_temperature": float(os.getenv(
            "MMRL_PROTOTYPE_ANCHOR_TEMPERATURE",
            str(EXP_CFG.get("prototype_anchor_temperature", 0.20)),
        )),
        "prototype_anchor_momentum": float(os.getenv(
            "MMRL_PROTOTYPE_ANCHOR_MOMENTUM",
            str(EXP_CFG.get("prototype_anchor_momentum", 0.95)),
        )),
        "prototype_anchor_min_confidence": float(os.getenv(
            "MMRL_PROTOTYPE_ANCHOR_MIN_CONFIDENCE",
            str(EXP_CFG.get("prototype_anchor_min_confidence", 0.40)),
        )),
        "prototype_anchor_assignment_power": float(os.getenv(
            "MMRL_PROTOTYPE_ANCHOR_ASSIGNMENT_POWER",
            str(EXP_CFG.get("prototype_anchor_assignment_power", 2.0)),
        )),
        "prototype_anchor_init_noise": float(os.getenv(
            "MMRL_PROTOTYPE_ANCHOR_INIT_NOISE",
            str(EXP_CFG.get("prototype_anchor_init_noise", 0.10)),
        )),
        "enable_adapter_router_identity_residual": os.getenv(
            "MMRL_ENABLE_ADAPTER_ROUTER_IDENTITY_RESIDUAL",
            "1" if EXP_CFG.get("enable_adapter_router_identity_residual", False) else "0",
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
            "adapter_effective_delta_ratio_mean",
            "adapter_effective_delta_ratio_min",
            "adapter_effective_delta_ratio_max",
            "delta_pool_common_mode_ratio",
            "delta_pool_specificity_ratio",
            "final_to_gated_ratio",
            "delta_transform_cos_mean",
            "delta_transform_cos_std",
            "adapter_route_entropy_norm",
            "adapter_route_confidence",
            "adapter_usage_max",
            "adapter_usage_min",
            "adapter_usage_0",
            "adapter_usage_1",
            "adapter_usage_2",
            "adapter_usage_3",
            "route_proto_kl",
            "route_proto_agreement",
            "prototype_usage_0",
            "prototype_usage_1",
            "prototype_usage_2",
            "prototype_usage_3",
            "adapter_usage_balance_loss_scaled",
            "adapter_sample_entropy_loss_scaled",
            "adapter_effective_delta_loss_scaled",
            "adapter_pairwise_cos_mean",
            "adapter_pairwise_cos_max",
            "adapter_pairwise_cos_below_band",
            "adapter_pairwise_cos_above_band",
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
            "adapter_utility_ema_0",
            "adapter_utility_ema_1",
            "adapter_utility_ema_2",
            "adapter_utility_ema_3",
            "route_utility_agreement",
            "route_utility_regret",
            "utility_margin",
            "utility_grad_norm",
            "route_utility_valid_fraction",
            "shared_rep_norm_mean",
            "shared_rep_grad_norm",
            "v_projector_grad_norm_mean",
            "adapter_router_grad_norm_mean",
            "visual_adapter_grad_norm_mean",
            "temperature",
        ],

        "learning_rate": {
            1: 1e-4,  # 仅分类器
            3: 8e-5,  # MMRL + router-adapter 联合适配
        },
        "epochs": {
            1: 1,
            3: 1,
        },
        # 实际只训练一轮，但保持本次 64.36 分实验的四轮调度轨迹。
        "schedule_epochs": {
            3: 4,
        },
    },
    "ablation_order": [1, 3]
}


def main():
    seed_before_model_init(CFG["train"]["seed"])
    if CFG["train"].get("eval_each_epoch", False):
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
    final_dir = f"{CFG['output_dir']}/final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"final model saved to {final_dir}")

    print("All stages done.")


if __name__ == "__main__":
    main()
