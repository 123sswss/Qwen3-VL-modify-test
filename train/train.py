import os

from train_stages import build_model_and_processor, run_stage


BASE_VISUAL_ROUTER_MODEL = {
    "rp_space_length": 40,
    "visual_residual_adapter_count": 4,
    "adapter_usage_balance_loss_weight": 0.0025,
    "adapter_sample_entropy_loss_weight": 0.012,
    "adapter_sample_entropy_target": 0.65,
    "expert_residual_guard_loss_weight": 0.010,
    "expert_residual_ratio_upper": 0.35,
    "mmrl_residual_guard_loss_weight": 0.020,
    "mmrl_residual_ratio_upper": 0.20,
    "mmrl_warmup_fraction": 1.0 / 3.0,
    "stage4_mmrl_lr_scale": 0.02,
    "ablate_visual_gate": False,
    "ablate_direct_learnable_rep": False,
    "diag_every_steps": 250,
}


EXPERIMENTS = {
    "visual_router_shared_trust_region_v2": {
        **BASE_VISUAL_ROUTER_MODEL,
    },
}

SELECTED_EXPERIMENT = os.getenv(
    "MMRL_EXPERIMENT",
    "visual_router_shared_trust_region_v2",
)
if SELECTED_EXPERIMENT not in EXPERIMENTS:
    raise ValueError(
        f"Unknown MMRL_EXPERIMENT={SELECTED_EXPERIMENT!r}, "
        f"choices={sorted(EXPERIMENTS)}"
    )
EXP_CFG = EXPERIMENTS[SELECTED_EXPERIMENT]


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
        "seed": 42,
        "deterministic_sampling": os.getenv(
            "MMRL_DETERMINISTIC_SAMPLING", "0"
        ) == "1",
        "visual_residual_adapter_count": EXP_CFG["visual_residual_adapter_count"],
        "adapter_usage_balance_loss_weight": EXP_CFG[
            "adapter_usage_balance_loss_weight"
        ],
        "adapter_sample_entropy_loss_weight": EXP_CFG[
            "adapter_sample_entropy_loss_weight"
        ],
        "adapter_sample_entropy_target": EXP_CFG["adapter_sample_entropy_target"],
        "expert_residual_guard_loss_weight": EXP_CFG[
            "expert_residual_guard_loss_weight"
        ],
        "expert_residual_ratio_upper": EXP_CFG["expert_residual_ratio_upper"],
        "mmrl_residual_guard_loss_weight": EXP_CFG[
            "mmrl_residual_guard_loss_weight"
        ],
        "mmrl_residual_ratio_upper": EXP_CFG["mmrl_residual_ratio_upper"],
        "mmrl_warmup_fraction": EXP_CFG["mmrl_warmup_fraction"],
        "stage4_mmrl_lr_scale": EXP_CFG["stage4_mmrl_lr_scale"],
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "final_temp": 0.1,
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
            "stage_progress",
            "mmrl_shared_strength",
            "G_mean",
            "mmrl_raw_delta_to_org_ratio",
            "mmrl_pre_cap_delta_to_org_ratio",
            "mmrl_cap_scale_mean",
            "mmrl_cap_active_fraction",
            "mmrl_shared_delta_to_org_ratio",
            "mmrl_shared_common_mode_ratio",
            "mmrl_shared_specificity_ratio",
            "mmrl_shared_token_specificity_ratio",
            "expert_residual_to_shared_ratio",
            "expert_residual_ratio_p95",
            "expert_residual_shared_cos",
            "final_delta_to_org_ratio",
            "adapter_pairwise_cos_mean",
            "adapter_route_entropy_norm",
            "adapter_usage_max",
            "adapter_usage_min",
            "adapter_usage_balance_loss_scaled",
            "adapter_sample_entropy_loss_scaled",
            "expert_residual_guard_loss_scaled",
            "mmrl_residual_guard_loss_scaled",
            "mmrl_grad_pressure",
            "adapter_router_grad_pressure",
            "visual_adapter_grad_pressure",
        ],
        "learning_rate": {
            1: 1e-4,
            2: 1e-4,
            3: 8e-5,
            4: 6e-5,
        },
        "epochs": {
            1: 1,
            2: 1,
            3: 1,
            4: 2,
        },
    },
    "ablation_order": [1, 2, 3, 4],
}


def main():
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

    final_dir = f"{CFG['output_dir']}/final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"final model saved to {final_dir}")
    print("All stages done.")


if __name__ == "__main__":
    main()
