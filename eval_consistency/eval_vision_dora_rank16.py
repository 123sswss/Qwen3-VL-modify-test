from common import parse_args, run_peft_entry

CHECKPOINT = "/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/dora_vision_attn/rank16/final"

if __name__ == "__main__":
    args = parse_args("Vision-encoder DoRA rank-16 disruption", "vision_dora_rank16")
    run_peft_entry(args, "Vision Encoder DoRA rank 16", "vision_dora", CHECKPOINT)
