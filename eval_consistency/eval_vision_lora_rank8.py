from common import parse_args, run_peft_entry

CHECKPOINT = "/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/lora_vision_attn/rank8/final"

if __name__ == "__main__":
    args = parse_args("Vision-encoder LoRA rank-8 disruption", "vision_lora_rank8")
    run_peft_entry(args, "Vision Encoder LoRA rank 8", "vision_lora", CHECKPOINT)
