from common import parse_args, run_peft_entry

CHECKPOINT = (
    "/root/autodl-tmp/Qwen3-VL-modify-test/loraTest/runs/"
    "lora_vision_last8/last8_attention_rank32/final"
)

if __name__ == "__main__":
    args = parse_args(
        "Last-8 vision-encoder attention LoRA rank-32 disruption",
        "last8_lora_attention_rank32",
    )
    run_peft_entry(
        args,
        "Last-8 Vision Encoder Attention LoRA rank 32",
        "last8_vision_lora_attention",
        CHECKPOINT,
    )
