# new_train.py
from train_stages import build_model_and_processor, run_stage

# ===== 可调参数统一字典（不使用 argparse）=====
CFG = {
    "model_path": "/root/autodl-tmp/model",
    "output_dir": "./output",
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

            "/root/autodl-tmp/dataset/1json.translated.json",
            "/root/autodl-tmp/dataset/2conv_c.translated.json",
            "/root/autodl-tmp/dataset/1conv_c.translated.json",
            "/root/autodl-tmp/dataset/4conv_c.translated.json",
            "/root/autodl-tmp/dataset/14json.translated.json",
            "/root/autodl-tmp/dataset/prof_test.translated.json",
            "/root/autodl-tmp/dataset/test2_train.translated.json",
            "/root/autodl-tmp/dataset/test7_train.translated.json",
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
            "/root/autodl-tmp/dataset/gen_test.translated.json",
            "/root/autodl-tmp/dataset/conversation_58k.json",
        ],
        "general_img_dir": [
            "/root/autodl-tmp/dataset/gen/train2017",
            "/root/autodl-tmp/dataset/gen/val2017",
        ],
        "total_limit": 50000,
    },
    "train": {
        "seed": 42,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        # Stage4 的预算正则请显式配置，不再依赖 tax_loss_weight 回退。
        # 之前 stage4 只有 1 个 epoch，同时 warmup=1，导致整个 stage4 的 budget_weight 始终为 0。
        "budget_loss_weight": 3.0,
        "budget_warmup_epochs": 0,
        "tax_loss_weight": 4.0,

        # 让 selector 更早进入相对离散的决策区，避免长期高温下“全开更安全”的解。
        "initial_temp": 0.7,
        "final_temp": 0.2,

        # 加强 stage4 中 alpha 监督，避免预算头和 alpha 脱钩。
        "alpha_loss_weight_s3": 0.5,
        "alpha_loss_weight_s4": 1.5,

        "console_log_every": 50,
        "metric_smooth_window": 30,
        "metric_ema_alpha": 0.12,
        "metric_scatter_stride": 5,
        "save_debug_figure": False,

        "learning_rate": {
            1: 1e-4,  # 仅分类器
            2: 1e-4,  # 分类器+门控预热
            3: 1e-4,  # 联合训练无budget，主要做对齐与过渡
            4: 6e-5,  # 联合训练+budget，略微回升 LR，让主知识学习更多落在 stage4
        },
        "epochs": {
            1: 1,
            2: 1,
            3: 1,
            4: 2,
        },
        "enable_k_loss_s4": True,
        "k_general_target_s4": 0.0,
        "k_expert_target_s4": 8.0,
        "k_general_lambda_init_s4": 0.0,
        "k_expert_lambda_init_s4": 0.0,
        "k_lambda_lr_general_s4": 0.02,
        "k_lambda_lr_expert_s4": 0.01,
        "k_lambda_max_general_s4": 5.0,
        "k_lambda_max_expert_s4": 5.0,
        "enable_dataset_k_loss_s4": True,
        "k_group_constraints_s4": {
            "general": {"k_min": 0.0, "k_max": 1.5, "range_weight": 1.0, "mean_weight": 0.02, "mean_anchor": 0.3},
            "vqa":     {"k_min": 0.0, "k_max": 6.0, "range_weight": 1.0, "mean_weight": 0.01, "mean_anchor": 2.0},
            "report":  {"k_min": 6.0, "k_max": 18.0, "range_weight": 1.0, "mean_weight": 0.02, "mean_anchor": 10.0},
            "test":    {"k_min": 2.0, "k_max": 28.0, "range_weight": 0.6, "mean_weight": 0.005, "mean_anchor": 12.0},
        },
        "k_range_global_weight_s4": 1.0,
        "k_reg_start_scale_s4": 0.25,
        "k_reg_target_scale_s4": 1.0,
        "enable_slot_collapse_s4": True,
        "slot_collapse_weight_s4": 0.02,
        "slot_neff_min_s4": 6.0,
        "slot_top1_max_s4": 0.35,
        "slot_top1_weight_s4": 0.2,
        "slot_collapse_expert_only_s4": True,
        "collapse_reg_start_scale_s4": 0.1,
        "collapse_reg_target_scale_s4": 1.0,

        "enable_dataset_slot_loss_s4": True,
        "slot_group_constraints_s4": {
            "report": {"entropy_min": 2.20, "entropy_max": 3.60, "weight": 0.020},
            "vqa":    {"entropy_min": 0.80, "entropy_max": 2.00, "weight": 0.015},
            "test":   {"entropy_min": 1.20, "entropy_max": 2.80, "weight": 0.010},
        },
    },
    "ablation_order": [1, 2, 3, 4]
}


def main():
    model, processor = build_model_and_processor(CFG["model_path"])

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