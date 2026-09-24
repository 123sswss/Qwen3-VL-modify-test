#!/usr/bin/env python3
"""Read-only, fixed-protocol diagnostics for normalized V1 seed44."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import torch

from diagnostics.compare_pathvqa_conditioning_mismatches import load_json, percentile
from pathvqa.data_pipeline import PathVQAParquetStore
from pathvqa.pathvqa_official_eval import build_prompt, run_inference
from pathvqa.pathvqa_vqa_metric import evaluate_pathvqa_predictions, normalize_pathvqa_answer
from slake.slake_official_eval import extract_generated_answer, write_json
from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface
from slake.visual_selection_offset import LAYERS


SCOPE = ("overall", "yes/no", "free-form", "what", "where")


def _accuracy(rows):
    return 100.0 * sum(bool(x["correct"]) for x in rows) / len(rows)


def paired(a, b, *, iterations=10000, seed=42):
    ai = {str(x["question_id"]): x for x in a}
    bi = {str(x["question_id"]): x for x in b}
    if len(ai) != len(a) or len(bi) != len(b) or ai.keys() != bi.keys():
        raise ValueError("Paired IDs duplicate or differ")
    result = {}
    for scope in SCOPE:
        aa, bb = [], []
        for qid in sorted(ai):
            x, y = ai[qid], bi[qid]
            for key in ("image_id", "reference", "answer_type", "question_type"):
                if x[key] != y[key]:
                    raise ValueError(f"Paired metadata mismatch: {qid} {key}")
            if scope == "overall" or x["answer_type"] == scope or x["question_type"] == scope:
                aa.append(x)
                bb.append(y)
        if not aa:
            raise ValueError(f"Empty comparison subgroup: {scope}")
        clusters = defaultdict(list)
        for x, y in zip(aa, bb):
            clusters[str(x["image_id"])].append(int(bool(y["correct"])) - int(bool(x["correct"])))
        cluster_ids = sorted(clusters)
        counts = [len(clusters[k]) for k in cluster_ids]
        deltas = [sum(clusters[k]) for k in cluster_ids]
        rng = random.Random(seed)
        samples = []
        for _ in range(iterations):
            draws = [rng.randrange(len(cluster_ids)) for _ in cluster_ids]
            samples.append(100.0 * sum(deltas[i] for i in draws) / sum(counts[i] for i in draws))
        result[scope] = {
            "questions": len(aa), "image_clusters": len(cluster_ids),
            "baseline": _accuracy(aa), "variant": _accuracy(bb),
            "delta": _accuracy(bb) - _accuracy(aa),
            "variant_only_correct": sum(y["correct"] and not x["correct"] for x, y in zip(aa, bb)),
            "baseline_only_correct": sum(x["correct"] and not y["correct"] for x, y in zip(aa, bb)),
            "paired_cluster_ci95": [percentile(samples, .025), percentile(samples, .975)],
            "bootstrap_iterations": iterations, "seed": seed,
        }
    result["normalized_prediction_change_fraction"] = sum(
        ai[k]["prediction"] != bi[k]["prediction"] for k in ai
    ) / len(ai)
    return result


def _normalize_question(text):
    return " ".join(str(text).casefold().split())


def _token_ids(interface, question):
    ids = interface.processor.tokenizer.encode(str(question).strip(), add_special_tokens=False)
    if not ids:
        raise ValueError("Empty independent question tokens")
    return ids


def _condition_question_type(question):
    """Question-only matching label: no reference answer or prediction read."""
    match = re.search(r"[a-z]+", str(question).casefold())
    first = match.group(0) if match else ""
    if first in {"what", "where", "when", "why", "who", "which", "how"}:
        return first
    if first in {"is", "are", "was", "were", "do", "does", "did", "can", "could",
                 "has", "have", "had", "will", "would", "should", "may"}:
        return "yes/no-form"
    return "other-form"


def pairs(rows, interface, mode, seed=42):
    rng = random.Random(seed)
    indexed = {str(x["question_id"]): x for x in rows}
    lengths = {qid: len(_token_ids(interface, row["question"])) for qid, row in indexed.items()}
    types = {qid: _condition_question_type(row["question"]) for qid, row in indexed.items()}
    by_type = defaultdict(list)
    for qid, row in indexed.items():
        by_type[types[qid]].append(qid)
    result, changed, transitions, length_diffs = {}, 0, Counter(), []
    for qid in sorted(indexed):
        row = indexed[qid]
        pool = by_type[types[qid]] if mode == "same_type" else [
            k for t, group in by_type.items() if t != types[qid] for k in group
        ]
        valid = [k for k in pool if k != qid and _normalize_question(indexed[k]["question"]) != _normalize_question(row["question"])]
        if not valid:
            result[qid] = None
            continue
        gap = min(abs(lengths[k] - lengths[qid]) for k in valid)
        nearest = [k for k in valid if abs(lengths[k] - lengths[qid]) == gap]
        donor = rng.choice(nearest)
        result[qid] = donor
        changed += 1
        transitions[f"{types[qid]}->{types[donor]}"] += 1
        length_diffs.append(lengths[donor] - lengths[qid])
    return result, {
        "seed": seed, "rule": "question-text-only first-word type (not reference answer); same/different label; exclude self and identical normalized question; minimum tokenizer length gap, seeded tie-break",
        "eligible": changed, "total": len(indexed), "coverage": changed / len(indexed),
        "type_transitions": dict(transitions),
        "length_difference": distribution(length_diffs),
    }


def distribution(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": sum(values) / len(values),
            "p05": percentile(values, .05), "p25": percentile(values, .25),
            "p50": percentile(values, .5), "p75": percentile(values, .75),
            "p95": percentile(values, .95)}


def cosine(a, b, eps=1e-10):
    a, b = a.float().flatten(), b.float().flatten()
    norms = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(norms) <= eps:
        return None
    return float(torch.dot(a, b) / norms)


def relative(a, b, eps=1e-10):
    a, b = a.float().flatten(), b.float().flatten()
    denominator = .5 * (torch.linalg.vector_norm(a) + torch.linalg.vector_norm(b))
    if float(denominator) <= eps:
        return None
    return float(torch.linalg.vector_norm(a - b) / (denominator + eps))


def js(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.numel() != b.numel() or a.numel() == 0:
        raise ValueError("Mapped distributions have different grids")
    if not torch.allclose(a.sum(), a.new_tensor(1.), atol=1e-5) or not torch.allclose(b.sum(), b.new_tensor(1.), atol=1e-5):
        raise ValueError("Spatial maps are not normalized")
    m = .5 * (a + b)
    term = lambda p: torch.where(p > 0, p * (p.clamp_min(1e-20).log() - m.clamp_min(1e-20).log()), 0.).sum()
    return float(.5 * (term(a) + term(b)))


def _moved(interface, inputs):
    return {k: v.to(device=interface.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            if torch.is_tensor(v) else v for k, v in inputs.items()}


def capture(interface, store, record):
    image = store.load_image(record)
    try:
        prompt = build_prompt(record["question"], None)
        inputs = _moved(interface, interface.prepare_inputs(image, prompt, question=record["question"]))
        model = interface.model
        model.diagnostic_probe = {}
        model.diagnostic_capture_features = True
        before = model.diagnostic_prefill_calls
        with torch.inference_mode():
            model(**inputs)
        if model.diagnostic_prefill_calls != before + 1:
            raise RuntimeError("Probe did not perform exactly one prefill")
        return model.diagnostic_probe
    finally:
        interface.model.diagnostic_probe = None
        interface.model.diagnostic_capture_features = False
        image.close()


def recondition(interface, source, question, *, features=None):
    model = interface.model
    grid = source["grid"]
    ids = torch.tensor([_token_ids(interface, question)], device=interface.device)
    q, valid = model._question_embeddings(ids, torch.ones_like(ids, dtype=torch.bool))
    model.diagnostic_condition_features = features or source["features"]
    probe = {}
    try:
        with torch.inference_mode():
            condition, _ = model._condition(q, valid, grid.to(interface.device), probe=probe)
            shift = model.prefix_output(torch.relu(condition))
        probe["shift"] = shift.detach().float().cpu()
        return probe
    finally:
        model.diagnostic_condition_features = None


def _probe_metrics(probe):
    out = {"shift_norm": float(torch.linalg.vector_norm(probe["shift"])),
           "shift_rms": float(probe["shift"].square().mean().sqrt()),
           "shift_to_p20_rms": float(probe["shift"].square().mean().sqrt()) / probe["p20_rms"]}
    for i, a in enumerate(LAYERS):
        for b in LAYERS[i + 1:]:
            out[f"map_js_{a}_{b}"] = js(probe[f"map{a}"][0], probe[f"map{b}"][0])
            out[f"summary_cos_{a}_{b}"] = cosine(probe[f"summary{a}"][0], probe[f"summary{b}"][0])
            out[f"summary_relative_{a}_{b}"] = relative(probe[f"summary{a}"][0], probe[f"summary{b}"][0])
    return out


def _probe_change(a, b):
    out = {"shift_cos": cosine(a["shift"], b["shift"]),
           "shift_relative": relative(a["shift"], b["shift"])}
    for layer in LAYERS:
        out[f"map_js_{layer}"] = js(a[f"map{layer}"][0], b[f"map{layer}"][0])
        out[f"summary_cos_{layer}"] = cosine(a[f"summary{layer}"][0], b[f"summary{layer}"][0])
        out[f"summary_relative_{layer}"] = relative(a[f"summary{layer}"][0], b[f"summary{layer}"][0])
    out["layer_weight_relative"] = relative(a["layer_weights"], b["layer_weights"])
    return out


def _select_probe(rows, seed=42):
    groups = defaultdict(list)
    for row in rows:
        kind = row["question_type"]
        groups[kind if kind in {"yes/no", "what", "where"} else "other"].append(row)
    rng = random.Random(seed)
    selected = []
    for kind in ("yes/no", "what", "where", "other"):
        selected.extend(rng.sample(groups[kind], min(28, len(groups[kind]))))
    selected_ids = {x["question_id"] for x in selected}
    image_groups = defaultdict(list)
    for row in rows:
        image_groups[row["image_id"]].append(row)
    multi = [g for g in image_groups.values() if len(g) > 1]
    rng.shuffle(multi)
    for group in multi[:8]:
        for row in group[:2]:
            if row["question_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["question_id"])
    remaining = [r for r in rows if r["question_id"] not in selected_ids]
    selected.extend(rng.sample(remaining, 128 - len(selected)))
    return selected


def _state_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _summary(values):
    keys = sorted({key for row in values for key in row})
    return {key: {**distribution([row[key] for row in values if row.get(key) is not None]),
                  "undefined_or_near_zero_norm": sum(row.get(key) is None for row in values)}
            for key in keys}


def _image_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    return groups


def do_probe(interface, store, comparison_rows, output):
    selected = _select_probe(comparison_rows)
    by_id = {row["question_id"]: row for row in store.samples}
    by_image = _image_groups(comparison_rows)
    rng = random.Random(42)
    bank = {}
    rows, question_changes, image_changes, shifts = [], [], [], []
    centered_pairs = []
    for index, row in enumerate(selected, 1):
        source = capture(interface, store, by_id[row["question_id"]])
        same = recondition(interface, source, row["question"])
        if not torch.allclose(source["shift"], same["shift"], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"Same-image same-question determinism failed: {row['question_id']}")
        for layer in LAYERS:
            if not torch.allclose(source[f"map{layer}"][0], same[f"map{layer}"][0], atol=1e-6):
                raise RuntimeError(f"Same-question map determinism failed: {row['question_id']} layer={layer}")
        base = _probe_metrics(source)
        base.update(question_id=row["question_id"], image_id=row["image_id"],
                    question_type=row["question_type"])
        shifts.append(source["shift"].flatten())
        choices = [x for x in by_image[row["image_id"]]
                   if x["question_id"] != row["question_id"]
                   and _normalize_question(x["question"]) != _normalize_question(row["question"])]
        if choices:
            donor = rng.choice(choices)
            changed = recondition(interface, source, donor["question"])
            metrics = _probe_change(source, changed)
            question_changes.append(metrics)
            centered_pairs.append((metrics, source["shift"].flatten(), changed["shift"].flatten()))
            base["same_image_question_donor"] = donor["question_id"]
            base["same_image_question_change"] = metrics
        key = tuple(source["grid"].flatten().tolist())
        if key in bank and bank[key]["image_id"] != row["image_id"]:
            donor = bank[key]
            changed = recondition(interface, source, row["question"], features=donor["features"])
            metrics = _probe_change(source, changed)
            image_changes.append(metrics)
            centered_pairs.append((metrics, source["shift"].flatten(), changed["shift"].flatten()))
            base["conditional_image_donor"] = donor["question_id"]
            base["conditional_image_change"] = metrics
        elif key not in bank and len(bank) < 16:
            bank[key] = {"image_id": row["image_id"], "question_id": row["question_id"],
                         "features": source["features"]}
        rows.append(base)
        if index % 16 == 0:
            print(f"[V1_PROBE] {index}/128")
    center = torch.stack(shifts).mean(dim=0)
    for metrics, original, changed in centered_pairs:
        metrics["centered_shift_cos"] = cosine(original - center, changed - center)
        metrics["centered_shift_relative"] = relative(original - center, changed - center)
    for row, shift in zip(rows, shifts):
        row["centered_shift_norm"] = float(torch.linalg.vector_norm(shift - center))
    report = {
        "selection": "seed42 stratified 28 each yes/no, what, where, other; add up to 8 same-image question pairs; fill to 128 uniformly without replacement; no correctness selection",
        "sample_ids": [x["question_id"] for x in selected],
        "normal": _summary([{k: v for k, v in row.items() if isinstance(v, (float, int))} for row in rows]),
        "same_image_different_question": {"covered": len(question_changes), "statistics": _summary(question_changes)},
        "same_question_different_condition_image": {"covered": len(image_changes), "statistics": _summary(image_changes),
            "policy": "same post-merger grid only; native LLM image unchanged; no answer generation"},
        "center_norm": float(torch.linalg.vector_norm(center)),
        "rows": rows,
    }
    write_json(output / "probe_128.json", report)
    return report


def disagreement_samples(interface, store, normal, cocoop, output):
    n = {row["question_id"]: row for row in normal}
    c = {row["question_id"]: row for row in cocoop}
    records = {row["question_id"]: row for row in store.samples}
    rng = random.Random(42)
    results = {}
    image_dir = output / "disagreement_images"
    image_dir.mkdir(exist_ok=True)
    for kind in ("what", "where"):
        for winner in ("v1_only", "cocoop_only"):
            candidates = []
            for qid in sorted(n):
                if qid not in c or n[qid]["question_type"] != kind:
                    continue
                v1_only = bool(n[qid]["correct"]) and not bool(c[qid]["correct"])
                co_only = bool(c[qid]["correct"]) and not bool(n[qid]["correct"])
                if (winner == "v1_only" and v1_only) or (winner == "cocoop_only" and co_only):
                    candidates.append(qid)
            picked = rng.sample(candidates, min(15, len(candidates)))
            rows = []
            for qid in picked:
                image_path = image_dir / (qid.replace(":", "_") + ".jpg")
                image = store.load_image(records[qid])
                try:
                    image.thumbnail((768, 768))
                    image.save(image_path, format="JPEG", quality=88)
                finally:
                    image.close()
                rows.append({"question_id": qid, "image_id": n[qid]["image_id"],
                             "image_file": str(image_path), "question": records[qid]["question"],
                             "reference": n[qid]["reference"],
                             "v1_prediction": n[qid]["prediction"],
                             "cocoop_prediction": c[qid]["prediction"],
                             "error_type": "uncertain_pending_image_review"})
            results[f"{kind}_{winner}"] = {"population": len(candidates), "sample": rows,
                                             "seed": 42, "selection": "uniform without replacement"}
    write_json(output / "disagreements.json", results)
    return results


def _infer_one(interface, store, record):
    image = store.load_image(record)
    try:
        text = interface.infer(image, build_prompt(record["question"], None),
                               max_new_tokens=32, temperature=0., question=record["question"])
        return extract_generated_answer(text, "raw")
    finally:
        image.close()


def render_report(report, output):
    normal = report["normal"]
    lines = [
        "# Normalized V1 seed44 epoch3 read-only diagnosis",
        "",
        f"Normal Validation: {normal['overall_accuracy']:.4f} Overall, "
        f"{normal['yes_no_accuracy']:.4f} Yes/No, {normal['free_form_accuracy']:.4f} Free-form; "
        f"what {normal['per_question_type_accuracy']['what']:.4f}, "
        f"where {normal['per_question_type_accuracy']['where']:.4f} (6259 questions, 832 images).",
        "",
        "## Paired baselines (V1 minus comparator; image-cluster bootstrap 10000, seed42)",
        "",
        "| Comparator | Overall delta | 95% paired CI | V1-only | Comparator-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["paired_baselines"].items():
        if "overall" not in result:
            lines.append(f"| {name} | unavailable | missing predictions | — | — |")
            continue
        row = result["overall"]
        lo, hi = row["paired_cluster_ci95"]
        lines.append(f"| {name} | {row['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{row['variant_only_correct']} | {row['baseline_only_correct']} |")
    lines += ["", "## Current-checkpoint interventions", "",
              "| Mode | Overall | Paired delta | 95% paired CI | Normalized prediction changed |",
              "|---|---:|---:|---:|---:|"]
    for name, item in report["interventions"].items():
        row = item["paired"]["overall"]
        lo, hi = row["paired_cluster_ci95"]
        lines.append(f"| {name} | {item['summary']['overall_accuracy']:.4f} | {row['delta']:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {item['paired']['normalized_prediction_change_fraction']:.4f} |")
    probe = report["probe"]
    lines += ["", "## Forward probe", "",
              f"Selected IDs: `probe_128.json`; same-image Q-swap coverage "
              f"{probe['same_image_different_question']['covered']}/128, "
              f"same-Q condition-image-swap coverage {probe['same_question_different_condition_image']['covered']}/128.",
              "Map/summary/offset distributions and zero-norm counts are in `probe_128.json`.",
              "Question mismatch uses question-text-only labels, not reference-answer-derived evaluation types.",
              "These interventions measure dependency of this trained checkpoint only; they are not retrained ablations or proof of correct localization.",
              "Historical comparator normalization protocols remain unverified; what/where subgroup statistics are exploratory.",
              "Disagreement images are exported for visual review; cases without visual review remain uncertain.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(interface, store, normal, row_map, pair_maps, output):
    model = interface.model
    before_hash = _state_digest(model)
    sample = [store.samples[i] for i in (0, 23, 48, 127, 256, 512, 1024, 2048)]
    audit = {"normal_ids": [], "modes": {}, "parameter_sha256_before": before_hash}
    for record in sample:
        qid = record["question_id"]
        prior = normalize_pathvqa_answer(normal[qid]["answer"])
        current = normalize_pathvqa_answer(_infer_one(interface, store, record))
        if current != prior:
            raise RuntimeError(f"Normal greedy preflight differs from saved prediction: {qid} {current!r}!={prior!r}")
        audit["normal_ids"].append(qid)
    image = store.load_image(sample[0])
    try:
        prepared = interface.prepare_inputs(image, build_prompt(sample[0]["question"], None), question=sample[0]["question"])
        audit_donor = pair_maps["cross_type"].get(sample[0]["question_id"])
        if audit_donor is None:
            raise RuntimeError("No cross-type condition donor for native-input preflight")
        model.diagnostic_question_ids = torch.tensor([_token_ids(interface, row_map[audit_donor]["question"])])
        prepared_with_donor = interface.prepare_inputs(image, build_prompt(sample[0]["question"], None), question=sample[0]["question"])
        for key in ("input_ids", "pixel_values", "image_grid_thw", "question_source_ids"):
            if key in prepared and not torch.equal(prepared[key], prepared_with_donor[key]):
                raise RuntimeError(f"Condition donor altered native input: {key}")
        audit["native_input_equal_with_condition_donor"] = True
    finally:
        model.diagnostic_question_ids = None
        image.close()
    for mode in ("same_type", "cross_type", "uniform_map"):
        model.diagnostic_uniform_maps = mode == "uniform_map"
        count = 0
        for record in sample[:3]:
            qid = record["question_id"]
            donor = pair_maps.get(mode, {}).get(qid)
            if mode != "uniform_map" and donor is None:
                continue
            if donor is not None:
                model.diagnostic_question_ids = torch.tensor([_token_ids(interface, row_map[donor]["question"])])
            model.diagnostic_probe = {}
            before = model.diagnostic_prefill_calls
            _infer_one(interface, store, record)
            if model.diagnostic_prefill_calls != before + 1:
                raise RuntimeError("KV-cache generated another conditional prefill")
            if model.last_injection_audit is None or not model.last_injection_audit["prefix_only"] or not model.last_injection_audit["native_embeddings_unchanged"]:
                raise RuntimeError("Intervention changed native embeddings")
            if mode == "uniform_map":
                p = model.diagnostic_probe
                for layer in LAYERS:
                    mapped = p[f"map{layer}"][0]
                    if not torch.allclose(mapped, torch.full_like(mapped, 1 / mapped.numel()), atol=1e-7):
                        raise RuntimeError("Uniform map contains invalid/nonuniform positions")
                    expected = int(p["grid"].prod(dim=1).sum()) // 4
                    if mapped.numel() != expected:
                        raise RuntimeError("Uniform map includes padding or temporary prompts")
            count += 1
            model.diagnostic_question_ids = None
            model.diagnostic_probe = None
        audit["modes"][mode] = {"checked": count}
        model.diagnostic_uniform_maps = False
    for record in sample[:2]:
        qid = record["question_id"]
        if normalize_pathvqa_answer(_infer_one(interface, store, record)) != normalize_pathvqa_answer(normal[qid]["answer"]):
            raise RuntimeError(f"Normal mode changed after interventions: {qid}")
    audit["parameter_sha256_after"] = _state_digest(model)
    if audit["parameter_sha256_after"] != before_hash:
        raise RuntimeError("Diagnostic altered trainable parameters")
    write_json(output / "preflight.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normal-eval", type=Path, required=True)
    parser.add_argument("--original-v1-eval", type=Path, required=True)
    parser.add_argument("--v0-eval", type=Path, required=True)
    parser.add_argument("--cocoop-eval", type=Path, required=True)
    parser.add_argument("--static-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    store = PathVQAParquetStore(args.data_root, "validation", cache_dir=args.cache_dir)
    normal_predictions = load_json(args.normal_eval / "pathvqa_predictions.json")
    normal_details = load_json(args.normal_eval / "pathvqa_details.json")
    normal_summary = load_json(args.normal_eval / "pathvqa_summary.json")
    normal_comparisons = load_json(args.normal_eval / "pathvqa_comparisons.json")
    if len(store.samples) != 6259 or len(normal_predictions) != 6259:
        raise RuntimeError("Incomplete normal validation")
    regenerated_summary, regenerated = evaluate_pathvqa_predictions(
        store.samples, normal_details, bootstrap_iterations=1, bootstrap_seed=42)
    if len(regenerated) != 6259 or len(_image_groups(regenerated)) != 832:
        raise RuntimeError("Normal Validation count/cluster mismatch")
    if {x["question_id"]: x["answer"] for x in normal_predictions} != {x["question_id"]: x["answer"] for x in normal_details}:
        raise RuntimeError("Official predictions differ from saved detailed outputs")
    if any(x != y for x, y in zip(regenerated, normal_comparisons)):
        raise RuntimeError("Normal comparisons/reference/score differ from current official evaluator")
    for key in ("overall_accuracy", "yes_no_accuracy", "free_form_accuracy", "per_question_type_accuracy"):
        if regenerated_summary[key] != normal_summary[key]:
            raise RuntimeError(f"Normal summary mismatch: {key}")
    write_json(output / "normal_verified.json", {"summary": normal_summary, "count": 6259,
               "image_clusters": 832, "source": str(args.normal_eval), "scoring_recomputed": True})
    print(f"[V1_NORMAL_VERIFIED] overall={normal_summary['overall_accuracy']:.4f} "
          f"yes_no={normal_summary['yes_no_accuracy']:.4f} free_form={normal_summary['free_form_accuracy']:.4f}")
    write_json(output / "config.json", {
        "checkpoint": str(args.checkpoint), "normal_eval": str(args.normal_eval),
        "split": "validation", "model_seed": 44, "data_seed": 42,
        "interventions": ["same_type", "cross_type", "uniform_map"],
        "sampling_seed": 42, "bootstrap_seed": 42, "bootstrap_iterations": 10000,
        "max_new_tokens": 32, "temperature": 0.0, "answer_mode": "raw",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
    })
    paired_results = {}
    baselines = {"original_v1": args.original_v1_eval, "repaired_v0": args.v0_eval,
                 "cocoop": args.cocoop_eval, "static_p20": args.static_eval}
    for name, folder in baselines.items():
        comparison = folder / "pathvqa_comparisons.json"
        if comparison.is_file():
            paired_results[name] = paired(load_json(comparison), regenerated)
        else:
            paired_results[name] = {"status": "missing", "path": str(comparison)}
    write_json(output / "paired_baselines.json", paired_results)
    interface = VisualSelectionPrefixInterface(str(args.checkpoint), args.base_model)
    row_map = {row["question_id"]: row for row in regenerated}
    pair_maps, pair_metadata = {}, {}
    for mode in ("same_type", "cross_type"):
        pair_maps[mode], pair_metadata[mode] = pairs(regenerated, interface, mode)
    write_json(output / "condition_question_pairs.json", {"maps": pair_maps, "metadata": pair_metadata})
    before_hash = _state_digest(interface.model)
    probe = do_probe(interface, store, regenerated, output)
    preflight(interface, store, {x["question_id"]: x for x in normal_predictions}, row_map, pair_maps, output)
    intervention_results = {}
    for mode in ("same_type", "cross_type", "uniform_map"):
        folder = output / mode
        folder.mkdir()
        model = interface.model
        model.diagnostic_uniform_maps = mode == "uniform_map"
        prediction_rows = []
        # The ordinary evaluation runner owns image loading, prompt and decoding protocol.
        # For question mismatch only, wrap infer to set a temporary independent condition source.
        original_infer = interface.infer
        current_qid = {"value": None}
        def conditional_infer(image, prompt, max_new_tokens=32, temperature=0., *, question):
            qid = current_qid["value"]
            donor = pair_maps.get(mode, {}).get(qid)
            if donor is not None:
                model.diagnostic_question_ids = torch.tensor([_token_ids(interface, row_map[donor]["question"])])
            try:
                return original_infer(image, prompt, max_new_tokens, temperature, question=question)
            finally:
                model.diagnostic_question_ids = None
        interface.infer = conditional_infer
        class TrackedStore:
            samples = store.samples
            def load_image(self, record):
                current_qid["value"] = record["question_id"]
                return store.load_image(record)
        try:
            prediction_rows = run_inference(TrackedStore(), store.samples, interface, folder,
                max_new_tokens=32, temperature=0., instruction=None, answer_mode="raw",
                resume=False, overwrite=False, continue_on_error=False)
        finally:
            interface.infer = original_infer
            model.diagnostic_uniform_maps = False
            model.diagnostic_question_ids = None
        summary, comparisons = evaluate_pathvqa_predictions(store.samples, prediction_rows,
                                                               bootstrap_iterations=2000, bootstrap_seed=42)
        write_json(folder / "pathvqa_predictions.json", [{"question_id": r["question_id"], "answer": r["answer"]} for r in prediction_rows])
        write_json(folder / "pathvqa_details.json", prediction_rows)
        write_json(folder / "pathvqa_comparisons.json", comparisons)
        write_json(folder / "pathvqa_summary.json", summary)
        intervention_results[mode] = {"summary": summary, "paired": paired(regenerated, comparisons),
                                      "config": pair_metadata.get(mode, {"uniform_over_valid_post_merger_positions": True})}
        write_json(output / "interventions.json", intervention_results)
        print(f"[V1_INTERVENTION] mode={mode} overall={summary['overall_accuracy']:.4f} "
              f"delta={intervention_results[mode]['paired']['overall']['delta']:+.4f}")
        if _state_digest(model) != before_hash:
            raise RuntimeError(f"Parameters changed after {mode}")
    if (args.cocoop_eval / "pathvqa_comparisons.json").is_file():
        disagreement_samples(interface, store, regenerated,
                             load_json(args.cocoop_eval / "pathvqa_comparisons.json"), output)
    report = {"normal": normal_summary, "paired_baselines": paired_results,
              "probe": {k: v for k, v in probe.items() if k != "rows"},
              "interventions": intervention_results,
              "disagreement_note": "Images exported for visual review; no unsupported error-category claims.",
              "interpretation_boundary": "Checkpoint dependence, not retrained ablation or proof of correct spatial localization."}
    write_json(output / "report.json", report)
    render_report(report, output)
    print(f"[V1_DIAGNOSTIC_DONE] output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
