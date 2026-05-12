# new_train.py
import os

from train_stages import build_model_and_processor, run_stage


EXPERIMENTS = {
    "dynamic_prefix": {
        "active_rep_token_count": 40,
        "text_gate_selection_mode": "dynamic_prefix",
        "text_gate_group_size": 5,
        "text_gate_num_groups": 8,
        "text_gate_selected_group_count": 4,
        "text_gate_selected_groups": "0,1,2,3",
        "enable_text_gating": True,
        "enable_text_gate_tax_loss": True,
        "enable_text_gate_collapse_loss": False,
        "enable_k_loss_s4": False,
        "enable_tax_loss_s4": True,
        "tax_loss_weight": 0.8,
        "k_general_target_s4": 0.0,
        "k_expert_target_s4": 20.0,
    },
    "fixed_group20": {
        "active_rep_token_count": 20,
        "text_gate_selection_mode": "fixed_group20",
        "text_gate_group_size": 5,
        "text_gate_num_groups": 8,
        "text_gate_selected_group_count": 4,
        "text_gate_selected_groups": "0,1,2,3",
        "enable_text_gating": True,
        "enable_text_gate_tax_loss": False,
        "enable_text_gate_collapse_loss": False,
        "enable_k_loss_s4": False,
        "enable_tax_loss_s4": False,
        "tax_loss_weight": 0.0,
        "k_general_target_s4": 0.0,
        "k_expert_target_s4": 20.0,
    },
    "group_top4": {
        "active_rep_token_count": 20,
        "text_gate_selection_mode": "group_top4",
        "text_gate_group_size": 5,
        "text_gate_num_groups": 8,
        "text_gate_selected_group_count": 4,
        "text_gate_selected_groups": "0,1,2,3",
        "enable_text_gating": True,
        "enable_text_gate_tax_loss": True,
        "enable_text_gate_collapse_loss": False,
        "enable_k_loss_s4": False,
        "enable_tax_loss_s4": True,
        "tax_loss_weight": 0.6,
        "enable_text_gate_group_anti_collapse": True,
        "text_gate_group_anti_collapse_weight": 0.02,
        "group_anti_collapse_weight_s4": 0.02,
        "group_anti_collapse_warmup_epochs_s4": 0.0,
        "text_gate_group_usage_ema_momentum": 0.98,
        "k_general_target_s4": 0.0,
        "k_expert_target_s4": 20.0,
    },
    "token_top20": {
        "active_rep_token_count": 20,
        "text_gate_selection_mode": "token_top20",
        "text_gate_group_size": 5,
        "text_gate_num_groups": 8,
        "text_gate_selected_group_count": 4,
        "text_gate_selected_groups": "0,1,2,3",
        "enable_text_gating": True,
        "enable_text_gate_tax_loss": True,
        "enable_text_gate_collapse_loss": False,
        "enable_k_loss_s4": False,
        "enable_tax_loss_s4": True,
        "tax_loss_weight": 1.0,
        "k_general_target_s4": 0.0,
        "k_expert_target_s4": 20.0,
    },
}

SELECTED_EXPERIMENT = os.getenv("MMRL_EXPERIMENT", "dynamic_prefix")
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

            # "/root/autodl-tmp/dataset/1json.translated.json",
            # "/root/autodl-tmp/dataset/2conv_c.translated.json",
            # "/root/autodl-tmp/dataset/1conv_c.translated.json",
            # "/root/autodl-tmp/dataset/4conv_c.translated.json",
            # "/root/autodl-tmp/dataset/14json.translated.json",
            # "/root/autodl-tmp/dataset/prof_test.translated.json",
            # "/root/autodl-tmp/dataset/test2_train.translated.json",
            # "/root/autodl-tmp/dataset/test7_train.translated.json",
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
            # "/root/autodl-tmp/dataset/gen_test.translated.json",
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
        "deterministic_sampling": os.getenv("MMRL_DETERMINISTIC_SAMPLING", "0") == "1",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "tax_loss_weight": EXP_CFG["tax_loss_weight"],
        "enable_gate_loss_stage2": os.getenv("MMRL_ENABLE_GATE_LOSS_STAGE2", "1") == "1",
        "enable_alpha_guide_loss_s3": os.getenv("MMRL_ENABLE_ALPHA_GUIDE_LOSS_S3", "1") == "1",
        "enable_alpha_guide_loss_s4": os.getenv("MMRL_ENABLE_ALPHA_GUIDE_LOSS_S4", "1") == "1",
        "enable_tax_loss_s4": EXP_CFG["enable_tax_loss_s4"],
        "enable_group_anti_collapse_s4": EXP_CFG.get("enable_text_gate_group_anti_collapse", False),

        "console_log_every": 50,
        "metric_smooth_window": 30,
        "metric_ema_alpha": 0.12,
        "metric_scatter_stride": 5,
        "save_debug_figure": False,

        "learning_rate": {
            1: 1e-4,  # 仅分类器
            2: 1e-4,  # 分类器+门控预热
            3: 8e-5,  # 联合训练无tax
            4: 6e-5,  # 联合训练+tax/K
        },
        "epochs": {
            1: 1,
            2: 1,
            3: 2,
            4: 2,
        },
        "enable_k_loss_s4": EXP_CFG["enable_k_loss_s4"],
        "group_anti_collapse_weight_s4": EXP_CFG.get("group_anti_collapse_weight_s4", EXP_CFG.get("text_gate_group_anti_collapse_weight", 0.0)),
        "group_anti_collapse_warmup_epochs_s4": EXP_CFG.get("group_anti_collapse_warmup_epochs_s4", 0.0),
        "group_usage_ema_momentum": EXP_CFG.get("text_gate_group_usage_ema_momentum", 0.98),
        "k_general_target_s4": EXP_CFG["k_general_target_s4"],
        "k_expert_target_s4": EXP_CFG["k_expert_target_s4"],
        "k_general_lambda_init_s4": 0.0,
        "k_expert_lambda_init_s4": 0.0,
        "k_lambda_lr_general_s4": 0.02,
        "k_lambda_lr_expert_s4": 0.01,
        "k_lambda_max_general_s4": 5.0,
        "k_lambda_max_expert_s4": 5.0,
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