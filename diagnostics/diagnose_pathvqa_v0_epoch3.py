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

MASK_POLICY = "standalone_training_source_with_prefill_overlap_targets_v2"


def audit_question_alignment(processor, prefill, question, prompt):
    """Audit source identity separately from contextual write positions."""
    tokenizer = processor.tokenizer
    question = question.strip()
    source = tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
    contextual = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = prefill["input_ids"][0].tolist()
    prompt_ids = contextual["input_ids"]
    starts = [i for i in range(len(ids) - len(prompt_ids) + 1)
              if ids[i:i + len(prompt_ids)] == prompt_ids]
    if len(starts) != 1:
        raise RuntimeError("Cannot audit unique full-prompt span")
    positions = prefill["question_mask"][0].nonzero().flatten().tolist()
    expected = [starts[0] + i for i, (a, b) in enumerate(contextual["offset_mapping"])
                if a < len(question) and b > a and b > 0]
    source_ids = prefill["question_source_ids"][0, prefill["question_source_mask"][0]].tolist()
    if source_ids != source["input_ids"] or positions != expected or len(positions) != len(source_ids):
        raise RuntimeError("Question source/target coverage or count differs from v2 policy")
    correspondence = []
    for ordinal, position in enumerate(positions):
        a, b = contextual["offset_mapping"][position - starts[0]]
        source_span = tuple(source["offset_mapping"][ordinal])
        if source_span != (max(a, 0), min(b, len(question))):
            raise RuntimeError("Source and target tokens have different question character spans")
        if b > len(question) and prompt[len(question):b].strip():
            raise RuntimeError("Question write target includes non-whitespace instruction text")
        if not bool(prefill["attention_mask"][0, position]):
            raise RuntimeError("Question write target overlaps padding")
        correspondence.append({"source_id": source_ids[ordinal], "source_span": source_span,
                               "prefill_position": position, "prefill_id": ids[position],
                               "prefill_span": [a, b]})
    return {"policy": MASK_POLICY, "source_ids": source_ids,
            "target_positions": positions, "correspondence": correspondence}


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
        "condition_off_offset_centered_rms": rms(delta_without_c - delta_without_c.mean(dim=0)),
        "condition_off_pair_cosine": pairs_mean(delta_without_c, "cosine"),
        "condition_effect_rms": effect_abs_rms,
        "condition_effect_common_rms": rms(effect.mean(dim=0)),
        "condition_effect_centered_rms": rms(effect_centered),
        "condition_effect_centered_ratio": (rms(effect_centered) / effect_abs_rms
                                            if effect_abs_rms > 1e-12 else None),
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
            li = (5, 11, 17).index(layer)
            result[f"same_image_swap_layer{layer}_weight_delta"] = float(
                alt["layer_weights"][0, li] - probe["layer_weights"][0, li])
            result[f"same_image_swap_map{layer}_js"] = js_divergence(
                probe[f"map{layer}"][0], alt[f"map{layer}"][0]
            )
            result[f"same_image_swap_summary{layer}_cosine"] = cosine(
                probe[f"summary{layer}"][0], alt[f"summary{layer}"][0]
            )
    return result


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


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
    if (baseline_summary.get("v0_question_mask_policy") != MASK_POLICY
        or baseline_summary.get("v0_intervention") != "normal"
        or baseline_summary.get("backend") != "visual-selection-offset"):
        raise ValueError("Baseline must be the completed normal V0 v2 evaluation")
    for key, expected_score in (("overall_accuracy", 56.7503),
                                ("yes_no_accuracy", 89.5040),
                                ("free_form_accuracy", 24.0906)):
        if abs(float(baseline_summary[key]) - expected_score) > 0.00015:
            raise ValueError(f"Wrong repaired baseline score for {key}")
    if abs(float(baseline_summary["per_question_type_accuracy"]["where"]) - 59.4132) > .00015:
        raise ValueError("Wrong repaired baseline where score")
    if (Path(baseline_summary["checkpoint"]).resolve() != args.checkpoint.resolve()
        or Path(baseline_summary["base_model"]).resolve() != Path(args.base_model).resolve()
        or Path(baseline_summary["data_root"]).resolve() != args.data_root.resolve()):
        raise ValueError("Checkpoint, backbone or data root differs from repaired baseline")
    training_audit = existing_training_audit(args.checkpoint.parent.parent)
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
    if model.question_mask_policy != MASK_POLICY:
        raise RuntimeError("Active interface does not implement the required v2 policy")
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
            alignment = audit_question_alignment(model.processor, prefill, record["question"], prompt)
            eval_ids = prefill["question_source_ids"][0, prefill["question_source_mask"][0]].tolist()
            training_ids = train_question_ids(model.processor, image, record["question"], record["answer"])
            if eval_ids != training_ids:
                raise RuntimeError(f"Train/prefill question source differs at {qid}: "
                                   f"train={training_ids} prefill={eval_ids}")
            with (args.output_dir / "question_alignment.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"question_id": qid, **alignment}) + "\n")
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
            with (args.output_dir / "forward_rows.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if index == 0:
                original = {key: [item.clone() for item in value] if isinstance(value, list)
                            else value.clone() for key, value in probe.items()
                            if key.startswith(("map", "summary")) or key in {"condition", "layer_weights", "offset"}}
                model.infer(image, prompt, max_new_tokens=1, temperature=0, question=record["question"])
                repeat = model.model.last_forward_probe
                deterministic = {
                    "condition_max_abs_difference": float((original["condition"] - repeat["condition"]).abs().max()),
                    **{f"map{layer}_max_abs_difference": float((original[f"map{layer}"][0]
                        - repeat[f"map{layer}"][0]).abs().max()) for layer in (5, 11, 17)},
                    **{f"summary{layer}_max_abs_difference": float((original[f"summary{layer}"]
                        - repeat[f"summary{layer}"]).abs().max()) for layer in (5, 11, 17)},
                    **{f"{key}_max_abs_difference": float((original[key] - repeat[key]).abs().max())
                       for key in ("layer_weights", "offset")},
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
        "by_answer_type": {label: numeric_summary([r for r in detail_rows if r["answer_type"] == label])
                           for label in ("yes/no", "free-form")},
        "where_statistics": numeric_summary([r for r in detail_rows if r["question_type"] == "where"]),
        "mask_policy": MASK_POLICY,
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
        summary = load_json(output / "pathvqa_summary.json")
        if (summary.get("v0_question_mask_policy") != MASK_POLICY
            or summary.get("v0_intervention") != mode
            or any(summary.get(k) != baseline_summary.get(k) for k in
                   ("checkpoint", "base_model", "data_root", "split", "max_new_tokens",
                    "temperature", "answer_mode", "instruction"))):
            raise RuntimeError(f"Intervention protocol differs from repaired baseline: {mode}")
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
