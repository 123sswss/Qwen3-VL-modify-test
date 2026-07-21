from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one checkpoint after scaling the MMRL delta presented to "
            "every visual adapter. Model weights and checkpoints remain unchanged."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--image-dir", action="append", dest="image_dirs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def install_adapter_input_scale(visual, scale: float):
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    adapters = getattr(visual, "residual_adapters", None)
    if adapters is None or len(adapters) == 0:
        raise RuntimeError("The loaded checkpoint has no visual residual adapters")

    def scale_input(_module, args):
        if len(args) != 1:
            raise RuntimeError(f"Expected one adapter input tensor, got {len(args)}")
        return (args[0] * scale,)

    handles = [adapter.register_forward_pre_hook(scale_input) for adapter in adapters]
    return handles


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
        scale_tag = str(args.scale).replace(".", "p")
        output_dir = SCRIPT_DIR / "outputs" / f"mmrl_scale_{scale_tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "test.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[CHECKPOINT] {checkpoint}")
    print(f"[MMRL_ADAPTER_INPUT_SCALE] {args.scale}")
    print(f"[OUTPUT] {output_dir}")

    interface = ModelInterface(str(checkpoint), args.base_model)
    visual = interface.model.model.visual
    handles = install_adapter_input_scale(visual, args.scale)
    interface.model.eval()
    print(
        f"[PATCH_COMPLETE] installed {len(handles)} adapter pre-hooks; "
        "Gate, router, final residual, weights, and checkpoint files are unchanged"
    )

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
        "mmrl_adapter_input_scale": args.scale,
        "patch_location": "forward pre-hook on each visual residual adapter",
        "evaluation": evaluation,
        "by_source": source_scores,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[MMRL_SCALE_RESULT] scale={args.scale} "
        f"overall={evaluation['score']} by_source={source_scores}"
    )
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
