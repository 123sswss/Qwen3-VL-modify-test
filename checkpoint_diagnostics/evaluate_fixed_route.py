from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from evaluate_hybrid import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASETS,
    DEFAULT_IMAGE_DIRS,
    SCRIPT_DIR,
    TEST_DIR,
    Tee,
    summarize_sources,
)


PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEST_DIR))


class FixedExpertRouter(nn.Module):
    def __init__(self, adapter_count: int, expert_index: int):
        super().__init__()
        self.adapter_count = adapter_count
        self.expert_index = expert_index

    def forward(self, pooled_vision_states, pooled_text_states, alpha_logits=None):
        del pooled_text_states, alpha_logits
        route = pooled_vision_states.new_zeros(
            (pooled_vision_states.shape[0], self.adapter_count)
        )
        route[:, self.expert_index] = 1
        return route


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint with a fixed one-hot expert route."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expert-index", type=int, default=3)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--image-dir", action="append", dest="image_dirs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    from inferEngine import ModelInterface
    from test import run_evaluation

    checkpoint = Path(args.checkpoint).resolve()
    datasets = args.datasets or DEFAULT_DATASETS
    image_dirs = args.image_dirs or DEFAULT_IMAGE_DIRS

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SCRIPT_DIR / "outputs" / f"fixed_expert_{args.expert_index}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "test.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[CHECKPOINT] {checkpoint}")
    print(f"[FIXED_EXPERT] {args.expert_index}")
    print(f"[OUTPUT] {output_dir}")

    interface = ModelInterface(str(checkpoint), args.base_model)
    visual = interface.model.model.visual
    adapter_count = len(visual.residual_adapters)
    if not 0 <= args.expert_index < adapter_count:
        raise ValueError(
            f"Expert index must be in [0, {adapter_count - 1}], got {args.expert_index}"
        )
    visual.adapter_router = FixedExpertRouter(adapter_count, args.expert_index).to(
        device=next(visual.parameters()).device
    )
    interface.model.eval()
    print(
        f"[ROUTE_OVERRIDE] every sample uses one-hot expert {args.expert_index}; "
        "Gate and all model weights are unchanged"
    )

    with torch.no_grad():
        evaluation = run_evaluation(
            datasets,
            interface,
            image_dirs,
            max_new_tokens=args.max_new_tokens,
        )
    log_file.flush()
    source_scores = summarize_sources(log_path, datasets)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "fixed_expert_index": args.expert_index,
        "evaluation": evaluation,
        "by_source": source_scores,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[FIXED_ROUTE_RESULT] overall={evaluation['score']} by_source={source_scores}")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
