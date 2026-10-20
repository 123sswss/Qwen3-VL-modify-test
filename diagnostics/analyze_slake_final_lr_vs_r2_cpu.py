"""Archived final-LR V1 / r2 pairing and fixed stratified review; CPU only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from diagnostics.audit_lora_and_slake_cpu import stratified_sample
from diagnostics.run_slake_cocoop_seed44 import paired, scored, record_id, _metric


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    refs = read(root / "data/test.json")
    assert len(refs) == 2094
    scores, binding = {}, {}
    for name in ("v1", "r2"):
        source = root / "sources" / name
        report = read(source / "train_report.json")
        assert report["model_seed" if name == "v1" else "seed"] == 44
        assert report["data_seed"] == 42 and report["epochs"] == 5
        assert report["trainer_model_accepts_loss_kwargs"] is False
        if name == "v1":
            assert report["experiment"] == "slake_v1_read_write_lr3e4_norm_fixed_5ep_seed44"
            assert report["read_write_learning_rate"] == 3e-4
            rates = report["optimizer"]["group_learning_rates"]
            assert rates["value_projection"] == rates["prefix_output"] == 3e-4
        summary, comparisons = scored(source, refs)
        assert str(summary["checkpoint"]).endswith("/checkpoints/epoch_5")
        scores[name] = {str(r["question_id"]): r for r in comparisons}
        details = {str(r["question_id"]): r for r in read(source / "slake_details.json")}
        for ref in refs:
            assert details[str(record_id(ref))]["image"] == ref["img_name"]
        binding[name] = {"summary": summary, "train_report": report,
                         "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.json")}}
    # Direction throughout: V1 minus r2.
    result = paired(root / "sources/r2", root / "sources/v1", refs)
    result["budget_note"] = "Both seed44/data42, five epochs/fixed epoch5 bilingual Test; V1 batch2/accum16, r2 batch1/accum32; near parameter budgets."
    write(root / "paired_v1_minus_r2.json", result)
    freq = Counter(_metric.normalize_vqa_answer(r.get("answer", "")) for r in read(root / "data/train.json") if str(r.get("answer", "")).strip())
    rows = []
    for ref in refs:
        key = str(record_id(ref))
        a, b = scores["v1"][key], scores["r2"][key]
        for field in ("question", "ground_truth_answers", "answer_type", "question_type", "language"):
            assert a[field] == b[field]
        category = "both_correct" if a["correct"] and b["correct"] else "v1_only" if a["correct"] else "r2_only" if b["correct"] else "both_wrong"
        rows.append({"question_id": key, "image_id": ref["img_id"], "image": ref["img_name"],
                     "question": ref["question"], "reference": ref["answer"], "question_type": ref["base_type"],
                     "answer_type": ref["answer_type"], "language": ref["q_lang"],
                     "v1_prediction": a["predicted_answer"], "r2_prediction": b["predicted_answer"],
                     "v1_correct": a["correct"], "r2_correct": b["correct"], "category": category,
                     "train_normalized_answer_frequency": freq[_metric.normalize_vqa_answer(ref["answer"])]})
    assert len({r["question_id"] for r in rows}) == 2094
    pools = {c: [r for r in rows if r["category"] == c] for c in ("r2_only", "v1_only")}
    rng, sample, strata = random.Random(42), [], {}
    for category, n in (("r2_only", 40), ("v1_only", 20)):
        assert len(pools[category]) >= n
        part, strata[category] = stratified_sample(pools[category], n, rng)
        sample += part
    for i, row in enumerate(sample, 1):
        row["review_index"] = i
        row["exported_image"] = "images/" + str(row["image_id"]) + Path(row["image"]).suffix
    write(root / "slake_sample_60.json", {"seed": 42, "sampling_rule": "40 r2-only + 20 V1-only; same proportional question_type x answer_type x language strata and sorted-ID seeded-shuffle algorithm as 20261013 audit; no manual replacement", "strata": strata, "rows": sample})
    write(root / "all_paired_rows.json", rows)
    binding["data_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (root / "data").glob("*.json")}
    write(root / "binding.json", binding)
    write(root / "outcome_counts.json", dict(Counter(r["category"] for r in rows)))
    print(json.dumps({"counts": dict(Counter(r["category"] for r in rows)), "groups": result["groups"],
                      "unique_sample_images": len({r["image"] for r in sample})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
