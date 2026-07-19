from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
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
    parser.add_argument(
        "--stop-on-regex-fail",
        action="store_true",
        help="Stop this expert at the first output from which no A-D answer can be extracted.",
    )
    return parser.parse_args()


def run_evaluation_with_early_stop(
    dataset_paths,
    model,
    image_dirs,
    max_new_tokens,
    eval_module,
):
    dataset = eval_module.load_combined_dataset(dataset_paths)
    total = len(dataset)
    correct = 0
    wrong = 0
    regex_fail = 0
    inference_errors = 0
    skip_reasons = {"missing_gt": 0, "missing_image": 0, "image_read_error": 0}
    stopped_early = False
    stop_record = None
    processed = 0
    started = time.time()

    print(f"{'=' * 60}")
    print(f"开始早停评测 | 共 {total} 题 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("早停条件: 第一次无法从模型输出提取 A-D 答案")
    print(f"{'=' * 60}")

    for index, item in enumerate(dataset):
        processed = index + 1
        item_id = item.get("id", f"unknown_{index}")
        source_json = item.get("_source_json")
        human_text = ""
        gt_answer = None
        for conversation in item.get("conversations", []):
            if conversation.get("from") == "human":
                human_text = conversation.get("value", "").replace("<image>\n", "").replace("<image>", "")
            elif conversation.get("from") == "gpt":
                gt_answer = eval_module.extract_gt_answer(conversation.get("value", ""))

        if gt_answer is None:
            skip_reasons["missing_gt"] += 1
            continue

        image_path = eval_module.resolve_image_path(
            item.get("image", ""),
            image_dirs,
            source_json_path=source_json,
        )
        if image_path is None:
            skip_reasons["missing_image"] += 1
            continue

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except Exception:
            skip_reasons["image_read_error"] += 1
            continue

        try:
            output = model.infer(image, human_text, max_new_tokens=max_new_tokens, temperature=0.2)
        except Exception as exc:
            inference_errors += 1
            print(f"[EVAL-ERROR] index={index} id={item_id} error={exc}")
            continue

        pred_answer, extracted = eval_module.extract_answer(output)
        if not extracted:
            regex_fail += 1
            wrong += 1
            stopped_early = True
            stop_record = {
                "global_index": index,
                "item_id": item_id,
                "source_json": source_json,
                "gt": gt_answer,
                "output_preview": output[:500],
            }
            print(f"[{index + 1}/{total}] REGEX_FAIL GT={gt_answer} Pred=None Acc={100.0 * correct / max(correct + wrong, 1):.1f}% | {item_id}")
            print(
                f"[EARLY-STOP REGEX_FAIL] index={index} id={item_id} "
                f"output_preview={output[:500]!r}"
            )
            break

        if pred_answer == gt_answer:
            correct += 1
            status = "CORRECT"
        else:
            wrong += 1
            status = "WRONG"
        evaluated = correct + wrong
        print(
            f"[{index + 1}/{total}] {status} GT={gt_answer} Pred={pred_answer} "
            f"Acc={100.0 * correct / max(evaluated, 1):.1f}% | {item_id}"
        )

    elapsed = time.time() - started
    evaluated = correct + wrong
    skipped = sum(skip_reasons.values())
    result = {
        "json_paths": list(dataset_paths),
        "image_dirs": list(image_dirs),
        "total_in_dataset": total,
        "processed": processed,
        "evaluated": evaluated,
        "skipped": skipped,
        "remaining_unprocessed": total - processed,
        "correct": correct,
        "wrong": wrong,
        "regex_fail": regex_fail,
        "inference_errors": inference_errors,
        "skip_reasons": skip_reasons,
        "stopped_early": stopped_early,
        "stop_record": stop_record,
        "score": round(100.0 * correct / evaluated, 2) if evaluated else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_question": round(elapsed / max(evaluated, 1), 2),
    }
    print(
        f"[EARLY-EVAL-RESULT] processed={processed}/{total} evaluated={evaluated} "
        f"score={result['score']} stopped_early={stopped_early}"
    )
    return result


def main():
    args = parse_args()
    from inferEngine import ModelInterface
    import test as eval_module

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
        if args.stop_on_regex_fail:
            evaluation = run_evaluation_with_early_stop(
                datasets,
                interface,
                image_dirs,
                args.max_new_tokens,
                eval_module,
            )
        else:
            evaluation = eval_module.run_evaluation(
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
        "stop_on_regex_fail": args.stop_on_regex_fail,
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
