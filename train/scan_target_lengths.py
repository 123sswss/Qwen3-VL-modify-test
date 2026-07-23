import argparse
import os
import sys
from collections import Counter

from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config as cfg
import processingWithMMRL
from data_pipeline import (
    FourViewMMRLDataset,
    build_compact_target_window,
    build_target_supervision_masks,
    normalize_conversation_role,
)


DEFAULT_EXPERT_JSON = [
    "/root/autodl-tmp/dataset/1json.json",
    "/root/autodl-tmp/dataset/2conv_c.json",
    "/root/autodl-tmp/dataset/1conv_c.json",
    "/root/autodl-tmp/dataset/4conv_c.json",
    "/root/autodl-tmp/dataset/14json.json",
    "/root/autodl-tmp/dataset/prof_test.json",
    "/root/autodl-tmp/dataset/test2_train.json",
    "/root/autodl-tmp/dataset/test7_train.json",
]
DEFAULT_EXPERT_IMG_DIR = [
    "/root/autodl-tmp/dataset/1/train",
    "/root/autodl-tmp/dataset/2/train",
    "/root/autodl-tmp/dataset/4/train",
    "/root/autodl-tmp/dataset/14",
]


def build_processor(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    return processingWithMMRL.Qwen3ProcessorWithMMRL(
        image_processor=image_processor,
        tokenizer=tokenizer,
        cfg=cfg,
    )


def valid_length(attention_mask):
    active = (attention_mask != 0).nonzero(as_tuple=True)[0]
    return int(active[-1].item()) + 1 if active.numel() else 0


def encode_dataset_item(ds, sample, max_length=None):
    image = None
    if sample["image_path"] is not None:
        image = Image.open(sample["image_path"]).convert("RGB")

    qwen_conv = ds._build_qwen_conv(sample["conversations"], image)
    text_inputs = ds.processor.apply_chat_template(
        qwen_conv,
        tokenize=False,
        add_generation_prompt=False,
    )
    kwargs = {
        "text": text_inputs,
        "padding": "max_length" if max_length else False,
        "truncation": bool(max_length),
        "return_tensors": "pt",
    }
    if max_length:
        kwargs["max_length"] = max_length
    if sample["view_type"] == "mm":
        kwargs["images"] = image
    encoded = ds.processor(**kwargs)
    return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)


def target_span(ds, input_ids, attention_mask, target_assistant_ordinal):
    valid = valid_length(attention_mask)
    headers = []
    header_ids = ds.assistant_header_ids
    first = int(header_ids[0])
    for start in (input_ids[:valid] == first).nonzero(as_tuple=True)[0].tolist():
        end = start + len(header_ids)
        if end <= valid and input_ids[start:end].tolist() == list(header_ids):
            headers.append(start)
    if len(headers) < int(target_assistant_ordinal):
        return {
            "valid_tokens": valid,
            "target_start": None,
            "target_end": None,
            "target_tokens": None,
            "complete": False,
            "reason": f"missing header found={len(headers)} expected={target_assistant_ordinal}",
        }

    target_start = headers[int(target_assistant_ordinal) - 1] + len(ds.assistant_label_prefix_ids)
    target_tail = input_ids[target_start:valid]
    end_offsets = (target_tail == int(ds.im_end_token_id)).nonzero(as_tuple=True)[0]
    if end_offsets.numel() == 0:
        return {
            "valid_tokens": valid,
            "target_start": target_start,
            "target_end": None,
            "target_tokens": valid - target_start,
            "complete": False,
            "reason": "missing im_end",
        }
    target_end = target_start + int(end_offsets[0].item()) + 1
    return {
        "valid_tokens": valid,
        "target_start": target_start,
        "target_end": target_end,
        "target_tokens": target_end - target_start,
        "complete": True,
        "reason": "",
    }


def percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((len(ordered) - 1) * pct / 100.0))
    return ordered[idx]


def describe_lengths(name, values):
    if not values:
        print(f"[{name}] no values")
        return
    print(
        f"[{name}] count={len(values)} min={min(values)} "
        f"p50={percentile(values, 50)} p90={percentile(values, 90)} "
        f"p95={percentile(values, 95)} p99={percentile(values, 99)} max={max(values)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.getenv("MMRL_MODEL_PATH", "/root/autodl-tmp/model"))
    parser.add_argument("--total-limit", type=int, default=int(os.getenv("MMRL_TOTAL_LIMIT", "20000")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("MMRL_DATA_SAMPLING_SEED", "42")))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--limit-samples", type=int, default=0)
    args = parser.parse_args()

    processor = build_processor(args.model_path)
    ds = FourViewMMRLDataset(
        processor=processor,
        expert_json=DEFAULT_EXPERT_JSON,
        expert_img_dir=DEFAULT_EXPERT_IMG_DIR,
        general_json=[],
        general_img_dir=[],
        total_limit=args.total_limit,
        enable_views=("expert-mm",),
        mode="stage3_joint",
        ce_enabled=True,
        seed=args.seed,
        deterministic_sampling=True,
    )

    rows = []
    full_target_lengths = []
    compact_target_lengths = []
    compact_total_lengths = []
    source_counter = Counter()
    hard_overflow = 0
    checked = 0

    for idx, sample in enumerate(ds.data):
        if args.limit_samples and checked >= args.limit_samples:
            break
        checked += 1

        full_ids, full_mask = encode_dataset_item(ds, sample, max_length=None)
        full_span = target_span(ds, full_ids, full_mask, sample["target_assistant_ordinal"])

        compact_sample = dict(sample)
        compact_sample["conversations"] = build_compact_target_window(
            sample["conversations"],
            require_image=(sample["view_type"] == "mm" and sample["image_path"] is not None),
        )
        compact_ordinal = sum(
            normalize_conversation_role(turn.get("from")) == "assistant"
            for turn in compact_sample["conversations"]
        )
        compact_ids, compact_mask = encode_dataset_item(ds, compact_sample, max_length=None)
        compact_span = target_span(ds, compact_ids, compact_mask, compact_ordinal)

        full_target_len = full_span["target_tokens"] or 0
        compact_target_len = compact_span["target_tokens"] or 0
        compact_total_len = compact_span["valid_tokens"]
        full_target_lengths.append(full_target_len)
        compact_target_lengths.append(compact_target_len)
        compact_total_lengths.append(compact_total_len)

        would_fail_full = False
        would_fail_compact = False
        try:
            full_trunc_ids, full_trunc_mask = encode_dataset_item(ds, sample, max_length=args.max_length)
            build_target_supervision_masks(
                input_ids=full_trunc_ids,
                attention_mask=full_trunc_mask,
                assistant_header_ids=ds.assistant_header_ids,
                assistant_label_prefix_ids=ds.assistant_label_prefix_ids,
                im_end_token_id=ds.im_end_token_id,
                target_assistant_ordinal=sample["target_assistant_ordinal"],
            )
        except Exception:
            would_fail_full = True

        try:
            compact_trunc_ids, compact_trunc_mask = encode_dataset_item(
                ds,
                compact_sample,
                max_length=args.max_length,
            )
            build_target_supervision_masks(
                input_ids=compact_trunc_ids,
                attention_mask=compact_trunc_mask,
                assistant_header_ids=ds.assistant_header_ids,
                assistant_label_prefix_ids=ds.assistant_label_prefix_ids,
                im_end_token_id=ds.im_end_token_id,
                target_assistant_ordinal=compact_ordinal,
            )
        except Exception:
            would_fail_compact = True

        if would_fail_compact:
            hard_overflow += 1
            source_counter[sample["source_name"]] += 1

        rows.append({
            "idx": idx,
            "source": sample["source_name"],
            "image": sample["image_path"],
            "target_turn_index": sample["target_turn_index"],
            "target_assistant_ordinal": sample["target_assistant_ordinal"],
            "raw_turn_count": sample["raw_turn_count"],
            "full_total_tokens": full_span["valid_tokens"],
            "full_target_tokens": full_target_len,
            "compact_total_tokens": compact_total_len,
            "compact_target_tokens": compact_target_len,
            "would_fail_full_1024": would_fail_full,
            "would_fail_compact_1024": would_fail_compact,
            "compact_reason": compact_span["reason"],
        })

        if checked % 500 == 0:
            print(f"[SCAN_PROGRESS] checked={checked}/{len(ds.data)} hard_overflow={hard_overflow}", flush=True)

    print("\n========== Target Length Scan ==========")
    print(f"checked={checked} max_length={args.max_length} seed={args.seed} total_limit={args.total_limit}")
    describe_lengths("full_target_tokens", full_target_lengths)
    describe_lengths("compact_target_tokens", compact_target_lengths)
    describe_lengths("compact_total_tokens", compact_total_lengths)
    print(f"hard_overflow_after_compact={hard_overflow}")
    print(f"hard_overflow_sources={dict(source_counter)}")

    print(f"\n========== Top {args.top} By Compact Total Tokens ==========")
    for row in sorted(rows, key=lambda x: x["compact_total_tokens"], reverse=True)[:args.top]:
        print(row)

    print(f"\n========== Top {args.top} By Compact Target Tokens ==========")
    for row in sorted(rows, key=lambda x: x["compact_target_tokens"], reverse=True)[:args.top]:
        print(row)

    print(f"\n========== Compact Still Fails @ {args.max_length} ==========")
    for row in [r for r in rows if r["would_fail_compact_1024"]][:args.top]:
        print(row)


if __name__ == "__main__":
    main()
