#!/usr/bin/env python3
"""Read-only V0 seed44 epoch3 forward probe and paired Validation interventions."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnostics.compare_pathvqa_conditioning_mismatches import (
    clustered_paired_bootstrap, compare_variant, load_json, percentile,
)
from pathvqa.data_pipeline import PathVQAParquetStore, build_target_supervision_masks
from pathvqa.pathvqa_official_eval import build_prompt, image_fingerprint
from slake.visual_selection_offset import locate_question_mask
from slake.visual_selection_offset_interface import VisualSelectionOffsetInterface


def rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a, b = a.float().flatten(), b.float().flatten()
    if not a.square().sum() or not b.square().sum():
        return None
    return float(F.cosine_similarity(a[None], b[None]))


def pairs_mean(values: torch.Tensor, kind: str) -> float | None:
    n = values.shape[0]
    if n < 2:
        return None
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if kind == "mask":
                out.append(float((values[i] == values[j]).float().mean()))
            else:
                score = cosine(values[i], values[j])
                if score is not None:
                    out.append(score)
    return sum(out) / len(out) if out else None


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p.float(), q.float()
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("Same-image maps must have exactly matching token geometry")
    midpoint = (p + q) / 2
    return float((F.kl_div(midpoint.clamp_min(1e-12).log(), p, reduction="sum")
                  + F.kl_div(midpoint.clamp_min(1e-12).log(), q, reduction="sum")) / 2)


def numeric_summary(rows: list[dict]) -> dict:
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return {
        key: {"count": len(values), "mean": sum(values) / len(values),
              "median": percentile(values, .5), "p10": percentile(values, .1),
              "p90": percentile(values, .9), "p05": percentile(values, .05),
              "p95": percentile(values, .95)}
        for key in keys
        if (values := [float(row[key]) for row in rows if row.get(key) is not None])
    }


def select_samples(baseline: list[dict], count: int = 128) -> tuple[list[str], dict[str, str]]:
    if len(baseline) != 6259:
        raise ValueError(f"V0 baseline must be complete Validation, got {len(baseline)}")
    rng = random.Random(42)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in baseline:
        groups[str(row["image_id"])].append(row)
    paired_groups = [group for group in groups.values()
                     if len({str(row["question"]) for row in group}) >= 2]
    rng.shuffle(paired_groups)
    selected: list[dict] = []
    partner: dict[str, str] = {}
    for group in paired_groups[:16]:
        first = group[0]
        second = next(row for row in group[1:] if row["question"] != first["question"])
        selected.extend((first, second))
        partner[str(first["question_id"])] = str(second["question_id"])
        partner[str(second["question_id"])] = str(first["question_id"])
    if len(partner) < 32:
        raise RuntimeError("Insufficient same-image/different-question pairs")
    picked = {str(row["question_id"]) for row in selected}
    pool = [row for row in baseline if str(row["question_id"]) not in picked]
    rng.shuffle(pool)
    for predicate, target in (
        (lambda row: row["question_type"] == "where", 32),
        (lambda row: row["answer_type"] == "yes/no", 32),
        (lambda row: row["answer_type"] == "free-form", 64),
    ):
        for row in pool:
            if sum(bool(predicate(item)) for item in selected) >= target:
                break
            qid = str(row["question_id"])
            if qid not in picked and predicate(row):
                selected.append(row)
                picked.add(qid)
    for row in pool:
        if len(selected) >= count:
            break
        qid = str(row["question_id"])
        if qid not in picked:
            selected.append(row)
            picked.add(qid)
    if len(selected) != count:
        raise RuntimeError(f"Could not select {count} fixed Validation questions")
    if (sum(row["question_type"] == "where" for row in selected) < 32
        or sum(row["answer_type"] == "yes/no" for row in selected) < 32
        or sum(row["answer_type"] == "free-form" for row in selected) < 64):
        raise RuntimeError("Fixed probe selection did not meet answer/question-type quotas")
    return [str(row["question_id"]) for row in selected], partner


def train_question_ids(processor, image, question: str, answer: str) -> list[int]:
    conversation = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    ids = processor(images=image, text=rendered, padding=False,
                    truncation=False, return_tensors="pt")["input_ids"][0]
    tokenizer = processor.tokenizer
    labels, context = build_target_supervision_masks(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        assistant_header_ids=tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
        assistant_label_prefix_ids=tokenizer.encode("<|im_start|>assistant", add_special_tokens=False),
        im_end_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        target_assistant_ordinal=1,
    )
    mask = locate_question_mask(ids, question, tokenizer, context_mask=context.bool())
    if bool((mask & labels.ne(-100)).any()):
        raise RuntimeError("Training question mask overlaps teacher-forced answer")
    return ids[mask].tolist()


def forward_row(probe: dict, qid: str) -> dict:
    valid = probe["valid"][0]
    text = probe["text"][0, valid]
    condition = probe["condition"][0]
    bias = probe["bias"]
    pre = probe["preactivation"][0, valid]
    delta = probe["offset"][0, valid]
    delta_without_c = probe["offset_condition_off"][0, valid]
    question = probe["question"][0, valid]
    centered = delta - delta.mean(dim=0, keepdim=True)
    effect = delta - delta_without_c
    effect_centered = effect - effect.mean(dim=0, keepdim=True)
    active = pre > 0
    text_token_rms = text.square().mean(dim=-1).sqrt().tolist()
    pre_token_rms = pre.square().mean(dim=-1).sqrt().tolist()
    offset_abs_rms = rms(delta)
    effect_abs_rms = rms(effect)
    result = {
        "question_id": qid, "question_tokens": int(valid.sum()),
        "text_rms": rms(text), "condition_rms": rms(condition),
        "bias_rms": rms(bias), "preactivation_rms": rms(pre),
        "text_token_rms_p10": percentile(text_token_rms, .1),
        "text_token_rms_median": percentile(text_token_rms, .5),
        "text_token_rms_p90": percentile(text_token_rms, .9),
        "preactivation_token_rms_p10": percentile(pre_token_rms, .1),
        "preactivation_token_rms_median": percentile(pre_token_rms, .5),
        "preactivation_token_rms_p90": percentile(pre_token_rms, .9),
        "condition_to_text_rms": rms(condition) / max(rms(text), 1e-12),
        "relu_mask_agreement": pairs_mean(active, "mask"),
        "relu_positive_fraction": float(active.float().mean()),
        "relu_dead_channel_fraction": float((~active.any(dim=0)).float().mean()),
        "offset_pair_cosine": pairs_mean(delta, "cosine"),
        "offset_zero_token_count": int((delta.square().sum(dim=-1) <= 1e-24).sum()),
        "offset_rms": offset_abs_rms, "offset_centered_rms": rms(centered),
        "offset_centered_ratio": (rms(centered) / offset_abs_rms
                                  if offset_abs_rms > 1e-12 else None),
        "offset_is_zero": offset_abs_rms <= 1e-12,
        "question_rms": rms(question),
        "offset_to_question_rms": rms(delta) / max(rms(question), 1e-12),
        "condition_off_offset_rms": rms(delta_without_c),
        "condition_effect_rms": effect_abs_rms,
        "condition_effect_common_rms": rms(effect.mean(dim=0)),
        "condition_effect_centered_rms": rms(effect_centered),
        "condition_effect_common_fraction": (rms(effect.mean(dim=0)) / effect_abs_rms
                                            if effect_abs_rms > 1e-12 else None),
        "condition_effect_is_zero": effect_abs_rms <= 1e-12,
        "condition_off_cosine": cosine(delta.mean(dim=0), delta_without_c.mean(dim=0)),
    }
    for i, layer in enumerate((5, 11, 17)):
        result[f"layer{layer}_weight"] = float(probe["layer_weights"][0, i])
    for left, right in ((5, 11), (5, 17), (11, 17)):
        result[f"summary{left}_{right}_cosine"] = cosine(
            probe[f"summary{left}"][0], probe[f"summary{right}"][0]
        )
    if "alternative" in probe:
        alt = probe["alternative"]
        result["alternative_layer_weight_l1"] = float(
            (probe["layer_weights"] - alt["layer_weights"]).abs().sum()
        )
        for layer in (5, 11, 17):
            result[f"same_image_swap_map{layer}_js"] = js_divergence(
                probe[f"map{layer}"][0], alt[f"map{layer}"][0]
            )
            result[f"same_image_swap_summary{layer}_cosine"] = cosine(
                probe[f"summary{layer}"][0], alt[f"summary{layer}"][0]
            )
    return result


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def existing_training_audit(baseline_root: Path) -> dict:
    first_path = baseline_root / "v0_first_batch_audit.json"
    trajectory_path = baseline_root / "v0_diagnostics.jsonl"
    if not first_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("Original V0 first-batch and training-trajectory audit files are required")
    first = load_json(first_path)
    with trajectory_path.open("r", encoding="utf-8") as handle:
        trajectory = [json.loads(line) for line in handle if line.strip()]
    if not trajectory:
        raise ValueError("Original V0 training diagnostic trajectory is empty")
    return {
        "source_first_batch": str(first_path),
        "source_trajectory": str(trajectory_path),
        "first_batch": first,
        "trajectory_count": len(trajectory),
        "trajectory_first": trajectory[0],
        "trajectory_last": trajectory[-1],
        "trajectory_statistics": numeric_summary(trajectory),
        "gradient_timing": {
            "first_backward_parameter_hook": "first microbatch backward; before gradient accumulation and clipping",
            "on_pre_optimizer_step_callback": "after gradient accumulation and clipping; before optimizer step",
            "max_grad_norm": 1.0,
        },
        "rms_timing": {
            "condition_rms": "after summary LayerNorm, value projection and beta weighting; before offset MLP addition",
            "down_rms": "offset_down(question), including linear bias b1; before condition addition",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(
        path.name != "diagnostic.log"
        for path in args.output_dir.iterdir()
    ):
        raise FileExistsError(f"Diagnostic output must be a fresh directory: {args.output_dir}")
    if args.output_dir.resolve() == args.baseline_dir.resolve():
        raise ValueError("Diagnostic output may not replace V0 baseline")
    baseline_summary = load_json(args.baseline_dir / "pathvqa_summary.json")
    baseline = load_json(args.baseline_dir / "pathvqa_comparisons.json")
    baseline_details = load_json(args.baseline_dir / "pathvqa_details.json")
    if len(baseline_details) != 6259 or any(row.get("status") != "ok" for row in baseline_details):
        raise ValueError("Original V0 Validation predictions are incomplete or contain errors")
    expected = ("validation", 32, 0, "raw", "short-answer")
    actual = tuple(baseline_summary.get(k) for k in
                   ("split", "max_new_tokens", "temperature", "answer_mode", "instruction"))
    if actual != expected or baseline_summary.get("partial_evaluation"):
        raise ValueError(f"Original V0 generation protocol differs; expected={expected}, actual={actual}")
    training_audit = existing_training_audit(args.baseline_dir.parent.parent)
    sample_ids, partners = select_samples(baseline)
    baseline_by_id = {str(row["question_id"]): row for row in baseline}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "existing_training_audit.json", training_audit)
    write_json(args.output_dir / "protocol_audit.json", {
        "baseline_dir": str(args.baseline_dir), "checkpoint": str(args.checkpoint),
        "validation_count": len(baseline), "baseline_summary": baseline_summary,
        "intervention_scope": "offset MLP condition addition or final question-token delta only",
        "vision_mapping": "V0 requires no window reorder, merge size 2, contiguous groups of four patches, one image per sample; runtime map probability and post-merger length guards remain active",
        "deepstack_and_cache": "original V0 generate path; no new vision/LLM pass; decode uses unchanged native KV cache",
    })
    write_json(args.output_dir / "sample_ids.json", {
        "selection_seed": 42, "question_ids": sample_ids, "same_image_pairs": partners,
    })
    store = PathVQAParquetStore(args.data_root, "validation", cache_dir=args.cache_dir)
    records = {str(row["question_id"]): row for row in store.samples}
    model = VisualSelectionOffsetInterface(str(args.checkpoint), args.base_model)
    model.model.diagnostic_capture = True
    detail_rows = []
    deterministic = None
    for index, qid in enumerate(sample_ids):
        record = records[qid]
        image = store.load_image(record)
        try:
            if image_fingerprint(image) != str(baseline_by_id[qid]["image_id"]):
                raise RuntimeError(f"Validation image differs from original V0 prediction at {qid}")
            prompt = build_prompt(record["question"], None)
            prefill = model.prepare_inputs(image, prompt, question=record["question"])
            eval_ids = prefill["question_source_ids"][0, prefill["question_source_mask"][0]].tolist()
            training_ids = train_question_ids(model.processor, image, record["question"], record["answer"])
            if eval_ids != training_ids:
                raise RuntimeError(f"Train/prefill question source differs at {qid}: "
                                   f"train={training_ids} prefill={eval_ids}")
            alt_qid = partners.get(qid)
            model.model.diagnostic_alternative_question_ids = (
                torch.tensor(model.processor.tokenizer.encode(records[alt_qid]["question"],
                    add_special_tokens=False), dtype=torch.long)
                if alt_qid else None
            )
            model.infer(image, prompt, max_new_tokens=1, temperature=0, question=record["question"])
            probe = model.model.last_forward_probe
            if probe is None:
                raise RuntimeError("V0 diagnostic probe was not captured")
            row = forward_row(probe, qid)
            row.update(image_id=baseline_by_id[qid]["image_id"],
                       answer_type=baseline_by_id[qid]["answer_type"],
                       question_type=baseline_by_id[qid]["question_type"],
                       alternate_question_id=alt_qid)
            detail_rows.append(row)
            if index == 0:
                original = {key: [item.clone() for item in value] if isinstance(value, list)
                            else value.clone() for key, value in probe.items()
                            if key.startswith("map") or key == "condition"}
                model.infer(image, prompt, max_new_tokens=1, temperature=0, question=record["question"])
                repeat = model.model.last_forward_probe
                deterministic = {
                    "condition_max_abs_difference": float((original["condition"] - repeat["condition"]).abs().max()),
                    **{f"map{layer}_max_abs_difference": float((original[f"map{layer}"][0]
                        - repeat[f"map{layer}"][0]).abs().max()) for layer in (5, 11, 17)},
                }
                if any(value > 1e-6 for value in deterministic.values()):
                    raise RuntimeError(f"Same-image/same-question V0 forward is not deterministic: {deterministic}")
        finally:
            image.close()
        if (index + 1) % 16 == 0:
            print(f"[V0_PROBE] {index + 1}/{len(sample_ids)}", flush=True)
    write_json(args.output_dir / "forward_rows.json", detail_rows)
    write_json(args.output_dir / "forward_summary.json", {
        "count": len(detail_rows), "same_image_question_pairs": len(partners) // 2,
        "same_question_determinism": deterministic,
        "statistics": numeric_summary(detail_rows),
        "measurement_note": "text excludes b1; condition includes value LayerNorm/projection and beta; first-batch hook pre-clipping, optimizer-step callback after clipping",
    })
    model.model.diagnostic_capture = False
    model.model.diagnostic_alternative_question_ids = None
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for mode in ("offset_off", "condition_off"):
        output = args.output_dir / mode
        command = [sys.executable, "-m", "pathvqa.pathvqa_official_eval",
                   "--backend", "visual-selection-offset", "--base-model", args.base_model,
                   "--checkpoint", str(args.checkpoint), "--data-root", str(args.data_root),
                   "--cache-dir", str(args.cache_dir), "--split", "validation",
                   "--v0-intervention", mode, "--output-dir", str(output)]
        print("[V0_INTERVENTION] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
        comparisons = load_json(output / "pathvqa_comparisons.json")
        details = load_json(output / "pathvqa_details.json")
        if len(comparisons) != len(baseline) or len(details) != len(baseline) or any(
            row.get("status") != "ok" for row in details
        ):
            raise RuntimeError(f"Incomplete V0 intervention output: {mode}")
    comparison = []
    for mode in ("offset_off", "condition_off"):
        variant = load_json(args.output_dir / mode / "pathvqa_comparisons.json")
        item = compare_variant(mode, baseline, variant,
            load_json(args.output_dir / mode / "pathvqa_summary.json"), 10000, 42)
        variant_by_id = {str(row["question_id"]): row for row in variant}
        for field, label, destination in (
            ("answer_type", "yes/no", item["per_answer_type"]["yes/no"]),
            ("answer_type", "free-form", item["per_answer_type"]["free-form"]),
            ("question_type", "where", item["per_question_type"]["where"]),
        ):
            subset_base = [row for row in baseline if row[field] == label]
            subset_variant = [variant_by_id[str(row["question_id"])] for row in subset_base]
            destination["clustered_paired_delta_ci"] = clustered_paired_bootstrap(
                subset_base, subset_variant, iterations=10000, seed=42,
            )
        comparison.append(item)
    write_json(args.output_dir / "paired_comparison.json", comparison)
    for item in comparison:
        overall = item["overall"]
        print(f"[V0_PAIRED] {item['variant']} overall={overall['variant_accuracy']:.4f} "
              f"delta={overall['delta']:+.4f} variant_only={overall['variant_only_correct']} "
              f"baseline_only={overall['baseline_only_correct']} "
              f"image_clustered_ci={overall['clustered_paired_delta_ci']}", flush=True)
        for label, metrics in (("yes/no", item["per_answer_type"]["yes/no"]),
                               ("free-form", item["per_answer_type"]["free-form"]),
                               ("where", item["per_question_type"]["where"])):
            print(f"[V0_PAIRED_BREAKDOWN] {item['variant']} {label} "
                  f"accuracy={metrics['variant_accuracy']:.4f} delta={metrics['delta']:+.4f} "
                  f"variant_only={metrics['variant_only_correct']} "
                  f"baseline_only={metrics['baseline_only_correct']} "
                  f"image_clustered_ci={metrics['clustered_paired_delta_ci']}", flush=True)
    print(f"[V0_DIAGNOSTIC_DONE] output={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
