# new_train.py
import os

from train_stages import build_model_and_processor, run_stage


BASE_VISUAL_ROUTER_MODEL = {
    "active_rep_token_count": 5,
    "rp_space_length": 40,
    "text_gate_selection_mode": "text_adapter_router",
    "text_rep_init_scale": 1e-3,
    "visual_residual_adapter_count": 4,
    "text_adapter_token_count": 5,
    "text_residual_adapter_count": 4,
    "text_common_mode_loss_weight": 0.05,
    "text_common_mode_target": 0.75,
    "text_adapter_balance_loss_weight": 0.01,
    "text_adapter_sample_entropy_loss_weight": 0.05,
    "text_adapter_sample_entropy_target": 0.55,
    # v2.1 visual-safe baseline: loosen usage balance, strengthen common-mode suppression.
    "adapter_usage_balance_loss_weight": 0.005,
    "adapter_sample_entropy_loss_weight": 0.02,
    "adapter_common_mode_loss_weight": 0.03,
    "adapter_sample_entropy_target": 0.55,
    "adapter_common_mode_target": 0.85,
    "ablate_visual_gate": False,
    "ablate_direct_learnable_rep": False,
    "disable_text_gate": False,
    "disable_text_prompt_insert": False,
    "mask_rep_placeholders_when_text_disabled": True,
    "disable_general_mm_stage34": False,
    "diag_every_steps": 1000,
}


EXPERIMENTS = {
    "text5_adapter_router_v1": {
        **BASE_VISUAL_ROUTER_MODEL,
        "diag_every_steps": 500,
    },
    "text5_adapter_router_v1_cm_0p07": {
        **BASE_VISUAL_ROUTER_MODEL,
        "diag_every_steps": 500,
        "text_common_mode_loss_weight": 0.07,
    },
    "text5_adapter_router_v1_cm_0p075": {
        **BASE_VISUAL_ROUTER_MODEL,
        "diag_every_steps": 500,
        "text_common_mode_loss_weight": 0.075,
    },
    "text5_adapter_router_v1_cm_0p08": {
        **BASE_VISUAL_ROUTER_MODEL,
        "diag_every_steps": 500,
        "text_common_mode_loss_weight": 0.08,
    },
    "text5_adapter_router_v1_cm_0p06": {
        **BASE_VISUAL_ROUTER_MODEL,
        "diag_every_steps": 500,
        "text_common_mode_loss_weight": 0.06,
    },
    "text5_adapter_router_v1_cm_0p09": {
        **BASE_VISUAL_ROUTER_MODEL,
        "diag_every_steps": 500,
        "text_common_mode_loss_weight": 0.09,
    },
    "text5_adapter_router_text_off": {
        **BASE_VISUAL_ROUTER_MODEL,
        "disable_text_gate": True,
        "disable_text_prompt_insert": True,
        "mask_rep_placeholders_when_text_disabled": True,
        "disable_general_mm_stage34": True,
        "diag_every_steps": 500,
    },
    "text5_adapter_router_v3": {
        **BASE_VISUAL_ROUTER_MODEL,
        "text_rep_init_scale": 1e-3,
    },
    "text5_adapter_router_v4": {
        **BASE_VISUAL_ROUTER_MODEL,
        "diag_every_steps": 500,
        "text_rep_init_scale": 5e-4,
        "text_common_mode_loss_weight": 0.01,
        "text_common_mode_target": 0.9,
        "text_adapter_balance_loss_weight": 0.05,
        "text_adapter_sample_entropy_loss_weight": 0.10,
        "text_adapter_sample_entropy_target": 0.90,
    },




}

SELECTED_EXPERIMENT = os.getenv("MMRL_EXPERIMENT", "text5_adapter_router_v1")
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
        "total_limit": 20000,  # Stage1/2: expert 20000 + general 20000；Stage3/4: expert-only 20000
    },
    "train": {
        "experiment_name": SELECTED_EXPERIMENT,
        "experiment_cfg": EXP_CFG,
        "seed": 42,
        "deterministic_sampling": os.getenv("MMRL_DETERMINISTIC_SAMPLING", "0") == "1",
        "visual_residual_adapter_count": int(os.getenv(
            "MMRL_VISUAL_RESIDUAL_ADAPTER_COUNT",
            str(EXP_CFG.get("visual_residual_adapter_count", 4)),
        )),
        "text_adapter_token_count": int(os.getenv(
            "MMRL_TEXT_ADAPTER_TOKEN_COUNT",
            str(EXP_CFG.get("text_adapter_token_count", 5)),
        )),
        "text_residual_adapter_count": int(os.getenv(
            "MMRL_TEXT_RESIDUAL_ADAPTER_COUNT",
            str(EXP_CFG.get("text_residual_adapter_count", 4)),
        )),
        "text_common_mode_loss_weight": float(os.getenv(
            "MMRL_TEXT_COMMON_MODE_LOSS_WEIGHT",
            str(EXP_CFG.get("text_common_mode_loss_weight", 0.0)),
        )),
        "text_common_mode_target": float(os.getenv(
            "MMRL_TEXT_COMMON_MODE_TARGET",
            str(EXP_CFG.get("text_common_mode_target", 0.85)),
        )),
        "text_adapter_balance_loss_weight": float(os.getenv(
            "MMRL_TEXT_ADAPTER_BALANCE_LOSS_WEIGHT",
            str(EXP_CFG.get("text_adapter_balance_loss_weight", 0.0)),
        )),
        "text_adapter_sample_entropy_loss_weight": float(os.getenv(
            "MMRL_TEXT_ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT",
            str(EXP_CFG.get("text_adapter_sample_entropy_loss_weight", 0.0)),
        )),
        "text_adapter_sample_entropy_target": float(os.getenv(
            "MMRL_TEXT_ADAPTER_SAMPLE_ENTROPY_TARGET",
            str(EXP_CFG.get("text_adapter_sample_entropy_target", 1.0)),
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
        "adapter_sample_entropy_target": float(os.getenv(
            "MMRL_ADAPTER_SAMPLE_ENTROPY_TARGET",
            str(EXP_CFG.get("adapter_sample_entropy_target", 0.40)),
        )),
        "adapter_common_mode_target": float(os.getenv(
            "MMRL_ADAPTER_COMMON_MODE_TARGET",
            str(EXP_CFG.get("adapter_common_mode_target", 0.85)),
        )),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "enable_gate_loss_stage2": os.getenv("MMRL_ENABLE_GATE_LOSS_STAGE2", "1") == "1",
        "enable_alpha_guide_loss_s4": False,
        "alpha_loss_weight_s4": 0.0,
        "disable_general_mm_stage34": os.getenv(
            "MMRL_DISABLE_GENERAL_MM_STAGE34",
            "1" if EXP_CFG.get("disable_general_mm_stage34", False) else "0",
        ) == "1",
        "enable_expert_floor_loss_s4": os.getenv("MMRL_ENABLE_EXPERT_FLOOR_LOSS_S4", "0") == "1",
        "expert_floor_loss_weight": float(os.getenv("MMRL_EXPERT_FLOOR_LOSS_WEIGHT", "0.0")),
        "expert_min_active_tokens": float(os.getenv("MMRL_EXPERT_MIN_ACTIVE_TOKENS", "25.0")),
        "enable_collapse_loss_s4": os.getenv("MMRL_ENABLE_COLLAPSE_LOSS_S4", "0") == "1",
        "anti_collapse_weight": float(os.getenv("MMRL_ANTI_COLLAPSE_WEIGHT", "0.10")),
        "collapse_max_ratio": float(os.getenv("MMRL_COLLAPSE_MAX_RATIO", "0.45")),
        "collapse_min_ratio": float(os.getenv("MMRL_COLLAPSE_MIN_RATIO", "0.05")),
        "collapse_entropy_target": float(os.getenv("MMRL_COLLAPSE_ENTROPY_TARGET", "0.75")),
        "group_usage_dead_threshold": float(os.getenv("MMRL_GROUP_USAGE_DEAD_THRESHOLD", "0.03")),
        "group_usage_ema_alpha": float(os.getenv("MMRL_GROUP_USAGE_EMA_ALPHA", "0.10")),
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

        "learning_rate": {
            1: 1e-4,  # 仅分类器
            2: 1e-4,  # 分类器+门控预热
            3: 8e-5,  # 联合训练无tax
            4: 6e-5,  # 联合训练+relative tax
        },
        "epochs": {
            1: 1,
            2: 1,
            3: 2,
            4: 2,
        },
    },
    "ablation_order": [1, 2, 3, 4]
}


def main():
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
