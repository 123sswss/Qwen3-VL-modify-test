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
        "total_limit": 500,
    },
    "train": {
        "seed": 42,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "tax_loss_weight": 4.0,
        "learning_rate": {
            1: 1e-4,  # 仅分类器
            2: 1e-4,  # 分类器+门控预热
            3: 1e-4,  # 联合训练无tax
            4: 8e-5,  # 联合训练+tax/K
        },
        "epochs": {
            1: 1,
            2: 1,
            3: 2,
            4: 1,
        }
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