"""Reserved entry point for MMRL; enable it after the checkpoint is ready."""

# from common import parse_args, run_experiment
# from your_mmrl_loader import load_ours
#
# CHECKPOINT = "/root/autodl-tmp/Qwen3-VL-modify-test/mmrl_output"
#
# if __name__ == "__main__":
#     args = parse_args("Our zero-pollution method", "ours")
#     model, processor = load_ours(CHECKPOINT)
#     run_experiment("Ours", "ours", model, processor, args, CHECKPOINT)

if __name__ == "__main__":
    raise SystemExit("Ours is intentionally disabled until its checkpoint is ready.")
