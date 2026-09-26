#!/usr/bin/env python3
"""Read-only small-sample fitting audit for normalized V1 seed44."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import importlib.metadata
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import torch.nn.functional as F

from pathvqa.data_pipeline import PathVQAParquetStore, build_target_supervision_masks
from pathvqa.pathvqa_official_eval import build_prompt, image_fingerprint
from pathvqa.pathvqa_vqa_metric import (
    answer_type, normalize_pathvqa_answer, question_type,
)
from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface


QUOTAS = {"yes/no": 96, "what": 96, "where": 48, "other": 16}
MARKER = "__QDPT_DIAGNOSTIC_ANSWER_BODY_9F31A7__"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def stratum(record: dict[str, Any]) -> str:
    kind = question_type(record["question"], record["answer"])
    return kind if kind in ("yes/no", "what", "where") else "other"


def percentiles(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean")}
    def q(probability):
        position = (len(ordered) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    return {"min": ordered[0], "p10": q(.10), "p25": q(.25), "p50": q(.50),
            "p75": q(.75), "p90": q(.90), "max": ordered[-1], "mean": mean(ordered)}


def fingerprint_for(store, record):
    image = store.load_image(record)
    try:
        return image_fingerprint(image)
    finally:
        image.close()


def select_records(store, split: str, comparison_by_id=None):
    rng = random.Random(42)
    grouped = defaultdict(list)
    for record in store.samples:
        grouped[stratum(record)].append(record)
    selected, selected_images, shortages = [], set(), {}
    fingerprint_cache = {}
    for group, quota in QUOTAS.items():
        candidates = list(grouped[group])
        rng.shuffle(candidates)
        chosen, deferred = [], []
        for record in candidates:
            qid = str(record["question_id"])
            if comparison_by_id is not None:
                image_id = str(comparison_by_id[qid]["image_id"])
            else:
                image_id = fingerprint_cache.setdefault(qid, fingerprint_for(store, record))
            enriched = {**record, "image_id": image_id, "stratum": group}
            if image_id not in selected_images and len(chosen) < quota:
                chosen.append(enriched)
                selected_images.add(image_id)
            else:
                deferred.append(enriched)
            if len(chosen) == quota:
                break
        if len(chosen) < quota:
            already = {row["question_id"] for row in chosen}
            for record in deferred:
                if record["question_id"] not in already:
                    chosen.append(record)
                    already.add(record["question_id"])
                    if len(chosen) == quota:
                        break
        shortages[group] = max(0, quota - len(chosen))
        selected.extend(chosen)
    if len(selected) != sum(QUOTAS.values()):
        raise RuntimeError(f"Cannot meet fixed sample quotas for {split}: {shortages}")
    return selected, {"seed": 42, "quotas": QUOTAS, "shortages": shortages,
                      "policy": "shuffle_within_stratum_then_unique_image_first_then_repeat",
                      "selected_images": len({row["image_id"] for row in selected})}


def answer_frequency(train_records):
    return Counter(normalize_pathvqa_answer(row["answer"]) for row in train_records)


def add_background(rows, frequencies):
    for row in rows:
        answer = str(row["answer"])
        row["normalized_answer"] = normalize_pathvqa_answer(answer)
        row["answer_frequency_in_train"] = frequencies[row["normalized_answer"]]
        row["answer_characters"] = len(answer)
        row["answer_words"] = len(answer.split())
        row["answer_type"] = answer_type(answer)
        row["question_type"] = question_type(row["question"], answer)


def overlap_audit(train_rows, validation_rows):
    train_images = {row["image_id"] for row in train_rows}
    val_images = {row["image_id"] for row in validation_rows}
    train_qa = {(normalize_pathvqa_answer(row["question"]), row["normalized_answer"]) for row in train_rows}
    val_qa = {(normalize_pathvqa_answer(row["question"]), row["normalized_answer"]) for row in validation_rows}
    image_overlap = sorted(train_images & val_images)
    qa_overlap = sorted(train_qa & val_qa)
    return {"image_overlap_count": len(image_overlap), "image_ids": image_overlap,
            "exact_normalized_question_answer_overlap_count": len(qa_overlap),
            "question_answers": [{"question": q, "answer": a} for q, a in qa_overlap]}


def expand_image_tokens(processor, text: str, grid: torch.Tensor) -> str:
    count = int(grid.reshape(-1, 3)[0].prod().item()) // int(processor.image_processor.merge_size) ** 2
    if text.count(processor.image_token) != 1:
        raise ValueError("Fitting audit expects one image placeholder per PathVQA item")
    return text.replace(processor.image_token, "<|placeholder|>" * count, 1).replace(
        "<|placeholder|>", processor.image_token,
    )


def teacher_inputs(interface, image, record):
    processor, tokenizer = interface.processor, interface.processor.tokenizer
    def conversation(answer):
        return [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(record["question"])},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    full_text = processor.apply_chat_template(conversation(str(record["answer"])), tokenize=False,
                                              add_generation_prompt=False)
    marker_text = processor.apply_chat_template(conversation(MARKER), tokenize=False,
                                                add_generation_prompt=False)
    if marker_text.count(MARKER) != 1 or marker_text.replace(MARKER, str(record["answer"])) != full_text:
        raise RuntimeError("Cannot establish exact assistant answer character boundary")
    inputs = dict(processor(images=image, text=full_text, padding=False, truncation=False,
                            return_tensors="pt"))
    ids = inputs["input_ids"].squeeze(0)
    attention = inputs["attention_mask"].squeeze(0)
    expanded_marker = expand_image_tokens(processor, marker_text, inputs["image_grid_thw"])
    expanded_full = expand_image_tokens(processor, full_text, inputs["image_grid_thw"])
    if expanded_marker.replace(MARKER, str(record["answer"])) != expanded_full:
        raise RuntimeError("Expanded multimodal text answer boundary is ambiguous")
    answer_start = expanded_marker.index(MARKER)
    answer_end = answer_start + len(str(record["answer"]))
    direct = tokenizer(expanded_full, return_offsets_mapping=True, return_tensors="pt")
    if not torch.equal(direct["input_ids"], inputs["input_ids"]):
        raise RuntimeError("Tokenizer offset pass differs from processor input IDs")
    offsets = direct["offset_mapping"].squeeze(0)
    assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_prefix = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    labels, _ = build_target_supervision_masks(
        input_ids=ids, attention_mask=attention,
        assistant_header_ids=assistant_header,
        assistant_label_prefix_ids=assistant_prefix,
        im_end_token_id=im_end, target_assistant_ordinal=1,
    )
    body_mask = torch.zeros_like(labels, dtype=torch.bool)
    cross_boundary = []
    for index, (start, end) in enumerate(offsets.tolist()):
        if labels[index].item() == -100 or end <= start:
            continue
        overlap = max(0, min(end, answer_end) - max(start, answer_start))
        if overlap > 0:
            body_mask[index] = True
            if start < answer_start or end > answer_end:
                cross_boundary.append({"token_index": index, "offset": [start, end],
                                       "token": tokenizer.decode([int(ids[index])], skip_special_tokens=False)})
    supervised = labels.ne(-100)
    template_mask = supervised & ~body_mask
    if not bool(body_mask.any()) or not torch.equal(body_mask | template_mask, supervised):
        raise RuntimeError("Answer body/template supervision partition failed")
    question_ids = tokenizer.encode(str(record["question"]).strip(), add_special_tokens=False)
    if not question_ids:
        raise RuntimeError("Empty standalone condition question")
    inputs["labels"] = labels.unsqueeze(0)
    inputs["question_source_ids"] = torch.tensor(question_ids, dtype=ids.dtype).unsqueeze(0)
    inputs["question_source_mask"] = torch.ones_like(inputs["question_source_ids"], dtype=torch.bool)
    audit = {"answer_character_span": [answer_start, answer_end],
             "cross_boundary_policy": "any_positive_character_overlap_is_answer_body",
             "cross_boundary_tokens": cross_boundary,
             "supervised_decoded": tokenizer.decode(ids[supervised], skip_special_tokens=False),
             "body_decoded": tokenizer.decode(ids[body_mask], skip_special_tokens=False),
             "template_decoded": tokenizer.decode(ids[template_mask], skip_special_tokens=False)}
    return inputs, body_mask, template_mask, audit


def teacher_forced_metrics(interface, image, record):
    inputs, body_mask, template_mask, boundary = teacher_inputs(interface, image, record)
    device = interface.device
    moved = {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
             for key, value in inputs.items()}
    moved["use_cache"] = False
    with torch.inference_mode():
        outputs = interface.model(**moved)
    expanded_labels = torch.cat((moved["labels"].new_full((1, 20), -100), moved["labels"]), dim=1)
    expanded_body = torch.cat((body_mask.new_zeros((20,)), body_mask)).to(device)
    expanded_template = torch.cat((template_mask.new_zeros((20,)), template_mask)).to(device)
    targets = expanded_labels[:, 1:]
    per_token = F.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
                                targets.reshape(-1), ignore_index=-100, reduction="none").reshape_as(targets)
    body_values = per_token[0, expanded_body[1:]]
    template_values = per_token[0, expanded_template[1:]]
    if body_values.numel() == 0:
        raise RuntimeError("No answer-body causal targets after shifting")
    result = {
        "model_forward_loss": float(outputs.loss),
        "body_nll_sum": float(body_values.sum()), "body_token_count": int(body_values.numel()),
        "body_mean_ce": float(body_values.mean()), "first_body_token_nll": float(body_values[0]),
        "first_body_token_probability": math.exp(-float(body_values[0])),
        "body_geometric_mean_token_probability": math.exp(-float(body_values.mean())),
        "template_nll_sum": float(template_values.sum()),
        "template_token_count": int(template_values.numel()),
        "template_mean_ce": float(template_values.mean()) if template_values.numel() else None,
        "boundary": boundary,
    }
    del outputs
    return result


def infer_train(interface, image, record):
    return interface.infer(image, build_prompt(str(record["question"]), None),
                           max_new_tokens=32, temperature=0.0, question=str(record["question"]))


def supervision_preflight(interface, samples, output_dir: Path):
    rows = []
    for split, store, record in samples:
        image = store.load_image(record)
        try:
            _, _, _, audit = teacher_inputs(interface, image, record)
        finally:
            image.close()
        if normalize_pathvqa_answer(audit["body_decoded"]) != normalize_pathvqa_answer(record["answer"]):
            raise RuntimeError(f"Decoded answer-body supervision differs at {record['question_id']}")
        rows.append({"split": split, "question_id": record["question_id"],
                     "reference": record["answer"], **audit})
    write_json(output_dir / "supervision_boundary_preflight.json", rows)
    print("[V1_FIT_SUPERVISION_PREFLIGHT] " + json.dumps(rows, ensure_ascii=False), flush=True)
    return rows


def summarize_rows(rows):
    def one(items):
        body_tokens = sum(row["body_token_count"] for row in items)
        template_tokens = sum(row["template_token_count"] for row in items)
        return {
            "count": len(items), "image_count": len({row["image_id"] for row in items}),
            "accuracy": 100.0 * sum(bool(row["correct"]) for row in items) / len(items) if items else None,
            "body_ce_question_equal": mean(row["body_mean_ce"] for row in items) if items else None,
            "body_ce_token_equal": sum(row["body_nll_sum"] for row in items) / body_tokens if body_tokens else None,
            "body_mean_ce_distribution": percentiles([row["body_mean_ce"] for row in items]),
            "first_body_nll_distribution": percentiles([row["first_body_token_nll"] for row in items]),
            "body_token_count_distribution": percentiles([row["body_token_count"] for row in items]),
            "template_ce_question_equal": mean(row["template_mean_ce"] for row in items
                                                if row["template_mean_ce"] is not None) if template_tokens else None,
            "template_ce_token_equal": sum(row["template_nll_sum"] for row in items) / template_tokens
                                       if template_tokens else None,
        }
    groups = {"sample_total": rows}
    for name in QUOTAS:
        groups[name] = [row for row in rows if row["stratum"] == name]
    length_groups = {
        "answer_tokens_1": [row for row in rows if row["body_token_count"] == 1],
        "answer_tokens_2": [row for row in rows if row["body_token_count"] == 2],
        "answer_tokens_3_plus": [row for row in rows if row["body_token_count"] >= 3],
    }
    frequency_groups = {
        "answer_frequency_1": [row for row in rows if row["answer_frequency_in_train"] == 1],
        "answer_frequency_2_5": [row for row in rows if 2 <= row["answer_frequency_in_train"] <= 5],
        "answer_frequency_6_20": [row for row in rows if 6 <= row["answer_frequency_in_train"] <= 20],
        "answer_frequency_21_plus": [row for row in rows if row["answer_frequency_in_train"] >= 21],
    }
    return {"by_stratum": {name: one(items) for name, items in groups.items()},
            "by_answer_length": {name: one(items) for name, items in length_groups.items()},
            "by_answer_frequency": {name: one(items) for name, items in frequency_groups.items()}}


def parse_training_trajectory(run_root: Path):
    log_rows = []
    train_log = run_root / "train.log"
    if train_log.is_file():
        for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
            start, end = line.find("{'loss':"), line.rfind("}")
            if start >= 0 and end > start:
                try:
                    row = ast.literal_eval(line[start:end + 1])
                except (SyntaxError, ValueError):
                    continue
                if isinstance(row, dict) and "loss" in row:
                    log_rows.append(row)
    diagnostics = []
    diagnostics_path = run_root / "v1_diagnostics.jsonl"
    if diagnostics_path.is_file():
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                diagnostics.append(json.loads(line))
    def epoch_summary(rows):
        grouped = defaultdict(list)
        for row in rows:
            epoch = max(1, min(3, int(math.ceil(float(row.get("epoch", 0.0))))))
            grouped[str(epoch)].append(row)
        result = {}
        for epoch, items in grouped.items():
            result[epoch] = {"records": len(items)}
            for key in ("loss", "grad_norm", "offset_to_p20_rms"):
                values = [float(item[key]) for item in items if key in item]
                if values:
                    result[epoch][key] = {"first": values[0], "last": values[-1],
                                          "mean": mean(values), "distribution": percentiles(values)}
        return result
    tail_logs = log_rows[max(0, math.floor(len(log_rows) * .8)):]
    tail_diag = diagnostics[max(0, math.floor(len(diagnostics) * .8)):]
    gradient_keys = sorted({key for row in diagnostics for key in row if key.endswith("_grad_norm")})
    return {
        "source": {"train_log": str(train_log), "diagnostics": str(diagnostics_path)},
        "missing": {"train_log": not train_log.is_file(), "diagnostics": not diagnostics_path.is_file(),
                    "early_checkpoints": True},
        "gradient_semantics": {"train_log_grad_norm": "Trainer clip_grad_norm return, pre-clip total norm",
                               "v1_diagnostics_group_grad_norms": "on_pre_optimizer_step, after global clipping"},
        "log_records": len(log_rows), "diagnostic_records": len(diagnostics),
        "log_by_epoch": epoch_summary(log_rows), "diagnostics_by_epoch": epoch_summary(diagnostics),
        "last_20_percent": {
            "loss": percentiles([row["loss"] for row in tail_logs if "loss" in row]),
            "preclip_total_grad_norm": percentiles([row["grad_norm"] for row in tail_logs if "grad_norm" in row]),
            "offset_to_p20_rms": percentiles([row["offset_to_p20_rms"] for row in tail_diag
                                               if "offset_to_p20_rms" in row]),
            "clipped_group_gradients": {key: percentiles([row[key] for row in tail_diag if key in row])
                                         for key in gradient_keys},
        },
        "records": {"trainer": log_rows, "diagnostics": diagnostics},
    }


def fixed_review(rows, split, output_dir: Path, store):
    rng = random.Random(42 + (0 if split == "train" else 1))
    free = [row for row in rows if row["answer_type"] == "free-form"]
    wrong, correct = [row for row in free if not row["correct"]], [row for row in free if row["correct"]]
    rng.shuffle(wrong)
    rng.shuffle(correct)
    chosen = [("wrong", row) for row in wrong[:20]] + [("correct_reference", row) for row in correct[:5]]
    folder = output_dir / "error_review" / split
    folder.mkdir(parents=True, exist_ok=True)
    result = []
    by_id = {row["question_id"]: row for row in store.samples}
    for index, (role, row) in enumerate(chosen):
        image = store.load_image(by_id[row["question_id"]])
        image_path = folder / f"{index:02d}_{role}_{row['question_id'].replace(':', '_')}.png"
        try:
            image.save(image_path)
        finally:
            image.close()
        result.append({"selection_role": role, "question_id": row["question_id"],
                       "image_id": row["image_id"], "image_path": str(image_path),
                       "question": row["question"], "reference": row["reference"],
                       "prediction": row["prediction"], "correct": row["correct"],
                       "answer_frequency_in_train": row["answer_frequency_in_train"],
                       "body_token_count": row["body_token_count"],
                       "body_mean_ce": row["body_mean_ce"],
                       "first_body_token_probability": row["first_body_token_probability"],
                       "review_category": "pending_visual_review",
                       "review_note": "Requires image-aware human/AI review; medical uncertainty must be explicit."})
    return result


def write_review_html(output_dir: Path, review_rows):
    blocks = []
    for row in review_rows:
        relative = Path(row["image_path"]).relative_to(output_dir).as_posix()
        blocks.append(
            f"<article><h3>{html.escape(row['selection_role'])} · {html.escape(row['question_id'])}</h3>"
            f"<img src='{html.escape(relative)}' style='max-width:420px;max-height:320px'>"
            f"<p>Q: {html.escape(row['question'])}<br>Ref: {html.escape(row['reference'])}<br>"
            f"Pred: {html.escape(row['prediction'])}<br>CE: {row['body_mean_ce']:.4f}; "
            f"first-token p: {row['first_body_token_probability']:.6f}</p></article>"
        )
    (output_dir / "error_review" / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>V1 fit audit review</title>" + "\n".join(blocks),
        encoding="utf-8",
    )


def concise_report(payload):
    train = payload["summary"]["train"]["by_stratum"]
    val = payload["summary"]["validation"]["by_stratum"]
    lines = ["# Normalized V1 seed44 fitting audit", "",
             "This is a stratified 256+256 diagnostic sample, not official Overall.", "",
             "| Stratum | Train acc | Validation acc | Train body CE(q/token) | Validation body CE(q/token) |",
             "|---|---:|---:|---:|---:|"]
    for name in ("yes/no", "what", "where", "other", "sample_total"):
        a, b = train[name], val[name]
        lines.append(f"| {name} | {a['accuracy']:.4f} | {b['accuracy']:.4f} | "
                     f"{a['body_ce_question_equal']:.4f}/{a['body_ce_token_equal']:.4f} | "
                     f"{b['body_ce_question_equal']:.4f}/{b['body_ce_token_equal']:.4f} |")
    lines += ["", f"Overlap audit: {payload['overlap']}", "",
              "Error images and fixed selections are in error_review/. Categories remain pending visual review.",
              "No training, optimizer update, Test evaluation, or checkpoint write was performed."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--validation-eval", type=Path, required=True)
    parser.add_argument("--train-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                             text=True, check=False).stdout.strip(),
               "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
               "accelerate": importlib.metadata.version("accelerate"),
               "checkpoint": str(args.checkpoint)}
    train_store = PathVQAParquetStore(args.data_root, "train", cache_dir=args.cache_dir)
    validation_store = PathVQAParquetStore(args.data_root, "validation", cache_dir=args.cache_dir)
    validation_comparisons = load_json(args.validation_eval / "pathvqa_comparisons.json")
    if len(validation_comparisons) != 6259:
        raise ValueError("Complete saved Validation comparisons required")
    validation_by_id = {str(row["question_id"]): row for row in validation_comparisons}
    frequencies = answer_frequency(train_store.samples)
    train_rows, train_sampling = select_records(train_store, "train")
    validation_rows, val_sampling = select_records(validation_store, "validation", validation_by_id)
    add_background(train_rows, frequencies)
    add_background(validation_rows, frequencies)
    overlap = overlap_audit(train_rows, validation_rows)
    write_json(args.output_dir / "sample_manifest.json",
               {"train": train_rows, "validation": validation_rows,
                "sampling": {"train": train_sampling, "validation": val_sampling}, "overlap": overlap})
    interface = VisualSelectionPrefixInterface(str(args.checkpoint), str(args.model_path))
    interface.model.eval()
    supervision_preflight(interface, [
        ("train", train_store, train_rows[0]), ("train", train_store, train_rows[96]),
        ("validation", validation_store, validation_rows[0]),
        ("validation", validation_store, validation_rows[96]),
    ], args.output_dir)
    results = {"train": [], "validation": []}
    decoded_audits = []
    for split, selected, store in (("train", train_rows, train_store),
                                   ("validation", validation_rows, validation_store)):
        for index, record in enumerate(selected):
            image = store.load_image(record)
            try:
                if split == "train":
                    prediction = infer_train(interface, image, record)
                    image_id = image_fingerprint(image)
                else:
                    saved = validation_by_id[record["question_id"]]
                    for key in ("question", "reference", "question_type", "answer_type"):
                        expected = (normalize_pathvqa_answer(record["answer"]) if key == "reference"
                                    else record.get(key))
                        if key == "question":
                            expected = str(record["question"])
                        if str(saved[key]) != str(expected):
                            raise ValueError(f"Saved Validation metadata mismatch at {record['question_id']}: {key}")
                    prediction = str(saved["prediction"])
                    image_id = str(saved["image_id"])
                ce = teacher_forced_metrics(interface, image, record)
            finally:
                image.close()
            reference = normalize_pathvqa_answer(record["answer"])
            predicted = normalize_pathvqa_answer(prediction)
            row = {key: record[key] for key in (
                "question_id", "question", "answer", "normalized_answer", "answer_frequency_in_train",
                "answer_characters", "answer_words", "answer_type", "question_type", "stratum",
            )}
            row.update({"split": split, "image_id": image_id, "reference": reference,
                        "prediction": predicted, "raw_prediction": prediction,
                        "correct": predicted == reference, **ce})
            results[split].append(row)
            if len(decoded_audits) < 8:
                decoded_audits.append({"split": split, "question_id": record["question_id"],
                                       **ce["boundary"]})
            if (index + 1) % 16 == 0:
                write_json(args.output_dir / f"{split}_per_question.partial.json", results[split])
                print(f"[V1_FIT_PROGRESS] split={split} completed={index + 1}/256", flush=True)
    write_json(args.output_dir / "train_per_question.json", results["train"])
    write_json(args.output_dir / "validation_per_question.json", results["validation"])
    write_json(args.output_dir / "decoded_supervision_audit.json", decoded_audits)
    summary = {split: summarize_rows(rows) for split, rows in results.items()}
    trajectory = parse_training_trajectory(args.train_run_root)
    review = []
    review += fixed_review(results["train"], "train", args.output_dir, train_store)
    review += fixed_review(results["validation"], "validation", args.output_dir, validation_store)
    write_json(args.output_dir / "error_review" / "manifest.json", review)
    write_review_html(args.output_dir, review)
    payload = {"runtime": runtime, "scope": {"training": False, "parameter_update": False,
                                               "test_evaluation": False,
                                               "validation_generation_reused": True},
               "sampling": {"train": train_sampling, "validation": val_sampling},
               "overlap": overlap, "summary": summary,
               "training_trajectory": trajectory,
               "review": {"selected": len(review), "pending_visual_review": True}}
    write_json(args.output_dir / "fit_audit.json", payload)
    (args.output_dir / "fit_audit_report.md").write_text(concise_report(payload), encoding="utf-8")
    digest = hashlib.sha256((args.output_dir / "sample_manifest.json").read_bytes()).hexdigest()
    print(f"[V1_FIT_AUDIT_DONE] output={args.output_dir} sample_manifest_sha256={digest} "
          "training=false test=false", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
