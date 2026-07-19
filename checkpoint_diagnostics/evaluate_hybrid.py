from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DIR = PROJECT_ROOT / "test"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEST_DIR))

DEFAULT_BASE_MODEL = "/root/autodl-tmp/model"
DEFAULT_DATASETS = [
    "/root/autodl-tmp/dataset/test2_val.json",
    "/root/autodl-tmp/dataset/seen_simple/llava_test.json",
]
DEFAULT_IMAGE_DIRS = [
    "/root/autodl-tmp/dataset/2/train",
    "/root/autodl-tmp/dataset/seen_simple/image",
]
MODES = ("high_adapters_low_router", "low_adapters_high_router")
EVAL_ROW_RE = re.compile(
    r"^\[(?P<done>\d+)/(?P<total>\d+)\]\s+\S+\s+"
    r"GT=(?P<gt>\S+)\s+Pred=(?P<pred>\S+)"
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a high/low checkpoint hybrid without writing a full checkpoint."
    )
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--low-checkpoint", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--permutation",
        default="1,3,0,2",
        help="Low expert index corresponding to each high expert index.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--image-dir", action="append", dest="image_dirs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def parse_permutation(value: str, adapter_count: int):
    permutation = [int(part.strip()) for part in value.split(",") if part.strip()]
    expected = list(range(adapter_count))
    if len(permutation) != adapter_count or sorted(permutation) != expected:
        raise ValueError(
            f"Permutation must contain each index in {expected} exactly once; got {permutation}"
        )
    return permutation


def checkpoint_weights(checkpoint: Path):
    path = checkpoint / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Expected a single safetensors checkpoint at {path}")
    return path


def read_module_state(weights, prefix: str, local_keys):
    state = {}
    missing = []
    available = set(weights.keys())
    for local_key in local_keys:
        checkpoint_key = prefix + local_key
        if checkpoint_key not in available:
            missing.append(checkpoint_key)
            continue
        state[local_key] = weights.get_tensor(checkpoint_key)
    if missing:
        raise KeyError(f"Missing checkpoint tensors: {missing}")
    return state


def install_low_router(visual, weights, permutation):
    router = visual.adapter_router
    state = read_module_state(
        weights,
        "model.visual.adapter_router.",
        router.state_dict().keys(),
    )
    for key in ("output_head.weight", "output_head.bias"):
        state[key] = state[key][permutation].contiguous()
    router.load_state_dict(state, strict=True)


def install_low_adapters(visual, weights, permutation):
    for high_index, low_index in enumerate(permutation):
        target = visual.residual_adapters[high_index]
        state = read_module_state(
            weights,
            f"model.visual.residual_adapters.{low_index}.",
            target.state_dict().keys(),
        )
        target.load_state_dict(state, strict=True)


def patch_hybrid(model_interface, low_checkpoint: Path, mode: str, permutation):
    visual = model_interface.model.model.visual
    if len(visual.residual_adapters) != len(permutation):
        raise ValueError(
            f"Checkpoint has {len(visual.residual_adapters)} adapters, "
            f"but permutation has {len(permutation)} entries"
        )

    with safe_open(
        str(checkpoint_weights(low_checkpoint)),
        framework="pt",
        device="cpu",
    ) as weights:
        if mode == "high_adapters_low_router":
            install_low_router(visual, weights, permutation)
        elif mode == "low_adapters_high_router":
            install_low_adapters(visual, weights, permutation)
        else:
            raise ValueError(f"Unsupported hybrid mode: {mode}")
    model_interface.model.eval()


def dataset_ranges(dataset_paths):
    ranges = []
    offset = 0
    for dataset_path in dataset_paths:
        path = Path(dataset_path)
        with path.open("r", encoding="utf-8") as handle:
            count = len(json.load(handle))
        name = f"{path.parent.name}__{path.stem}" if path.parent.name else path.stem
        ranges.append((offset, offset + count, name))
        offset += count
    return ranges


def summarize_sources(log_path: Path, dataset_paths):
    ranges = dataset_ranges(dataset_paths)
    counts = {
        name: {"evaluated": 0, "correct": 0, "wrong": 0}
        for _, _, name in ranges
    }
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = EVAL_ROW_RE.match(line.strip())
            if match is None:
                continue
            index = int(match.group("done")) - 1
            source = next(
                (name for start, end, name in ranges if start <= index < end),
                None,
            )
            if source is None:
                continue
            is_correct = match.group("gt") == match.group("pred")
            counts[source]["evaluated"] += 1
            counts[source]["correct" if is_correct else "wrong"] += 1

    for values in counts.values():
        evaluated = values["evaluated"]
        values["score"] = round(100.0 * values["correct"] / evaluated, 2) if evaluated else 0.0
    return counts


def main():
    args = parse_args()
    from inferEngine import ModelInterface
    from test import run_evaluation

    high_checkpoint = Path(args.high_checkpoint).resolve()
    low_checkpoint = Path(args.low_checkpoint).resolve()
    datasets = args.datasets or DEFAULT_DATASETS
    image_dirs = args.image_dirs or DEFAULT_IMAGE_DIRS

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SCRIPT_DIR / "outputs" / f"hybrid_{args.mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "test.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"[HYBRID_MODE] {args.mode}")
    print(f"[HIGH_CHECKPOINT] {high_checkpoint}")
    print(f"[LOW_CHECKPOINT] {low_checkpoint}")
    print(f"[OUTPUT] {output_dir}")

    interface = ModelInterface(str(high_checkpoint), args.base_model)
    adapter_count = len(interface.model.model.visual.residual_adapters)
    permutation = parse_permutation(args.permutation, adapter_count)
    print(f"[EXPERT_ALIGNMENT] high[i] <- low[permutation[i]]: {permutation}")
    patch_hybrid(interface, low_checkpoint, args.mode, permutation)
    print("[PATCH_COMPLETE] all unmentioned modules remain from the high checkpoint")

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
        "mode": args.mode,
        "high_checkpoint": str(high_checkpoint),
        "low_checkpoint": str(low_checkpoint),
        "expert_permutation_high_to_low": permutation,
        "evaluation": evaluation,
        "by_source": source_scores,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[HYBRID_RESULT] overall={evaluation['score']} by_source={source_scores}")
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
