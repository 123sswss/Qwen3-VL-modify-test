"""Compare every method with Base and write paper-auditable JSON reports."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from config import DEFAULT_OUTPUT_ROOT


def decode_tensor(payload: Dict[str, Any]) -> torch.Tensor:
    if payload["encoding"] != "base64" or payload["dtype"] != "float16":
        raise ValueError(f"Unsupported tensor encoding: {payload}")
    raw = base64.b64decode(payload["data"])
    array = np.frombuffer(raw, dtype=np.float16).copy().reshape(payload["shape"])
    return torch.from_numpy(array).float()


def load_run(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    if left["sample_id"] != right["sample_id"]:
        raise ValueError(f"Sample mismatch: {left['sample_id']} != {right['sample_id']}")
    if left["input_ids_sha256"] != right["input_ids_sha256"]:
        raise ValueError(f"Tokenized input mismatch for sample {left['sample_id']}")
    a = decode_tensor(left["first_token_logits"])
    b = decode_tensor(right["first_token_logits"])
    if a.shape != b.shape:
        raise ValueError(f"Vocabulary mismatch for sample {left['sample_id']}: {a.shape}, {b.shape}")

    log_a = F.log_softmax(a, dim=-1)
    log_b = F.log_softmax(b, dim=-1)
    p_a, p_b = log_a.exp(), log_b.exp()
    midpoint = 0.5 * (p_a + p_b)
    log_midpoint = torch.log(midpoint.clamp_min(1e-30))
    kl_ab = torch.sum(p_a * (log_a - log_b))
    kl_ba = torch.sum(p_b * (log_b - log_a))
    js = 0.5 * (
        torch.sum(p_a * (log_a - log_midpoint))
        + torch.sum(p_b * (log_b - log_midpoint))
    )
    top_a = left["top50_token_ids"]
    top_b = right["top50_token_ids"]
    right_timing = right.get("timing", {})
    return {
        "sample_id": left["sample_id"],
        "js_divergence": js.item(),
        "symmetric_kl": (0.5 * (kl_ab + kl_ba)).item(),
        "logits_rmse": torch.sqrt(torch.mean((a - b) ** 2)).item(),
        "logits_cosine_distance": (1.0 - F.cosine_similarity(a, b, dim=0)).item(),
        "top1_agreement": int(top_a[0] == top_b[0]),
        "top10_overlap": len(set(top_a[:10]) & set(top_b[:10])) / 10.0,
        "generated_exact_match": int(left["generated_token_ids"] == right["generated_token_ids"]),
        "first_token_latency_seconds": right_timing.get("first_token_latency_seconds"),
        "subsequent_token_mean_seconds": right_timing.get("subsequent_token_mean_seconds"),
        "generation_total_seconds": right_timing.get("generation_total_seconds"),
        "left_generated_text": left["generated_text"],
        "right_generated_text": right["generated_text"],
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"num_samples": len(rows)}
    numeric_keys = [
        "js_divergence",
        "symmetric_kl",
        "logits_rmse",
        "logits_cosine_distance",
        "top1_agreement",
        "top10_overlap",
        "generated_exact_match",
        "first_token_latency_seconds",
        "subsequent_token_mean_seconds",
        "generation_total_seconds",
    ]
    for key in numeric_keys:
        values = np.asarray(
            [row[key] for row in rows if row.get(key) is not None], dtype=np.float64
        )
        if values.size == 0:
            result[key] = None
            continue
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def paired_rows(left: Dict[str, Any], right: Dict[str, Any]) -> List[Dict[str, Any]]:
    left_map = {str(row["sample_id"]): row for row in left["samples"]}
    right_map = {str(row["sample_id"]): row for row in right["samples"]}
    if set(left_map) != set(right_map):
        raise ValueError("Runs do not contain exactly the same sample IDs.")
    return [metrics(left_map[key], right_map[key]) for key in left_map]


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-run-1", default="base_run_1")
    parser.add_argument("--base-run-2", default="base_run_2")
    args = parser.parse_args()

    root = Path(args.output_root)
    base1 = load_run(root / args.base_run_1 / "raw_results.json")
    base2 = load_run(root / args.base_run_2 / "raw_results.json")
    natural_rows = paired_rows(base1, base2)
    natural_summary = summarize(natural_rows)
    reports = root / "reports"
    write_report(
        reports / "base_natural_disruption.json",
        {
            "conclusion": {
                "status": "complete",
                "statement": "This report is the empirical natural-disruption floor from two independent Base runs.",
                "mean_natural_js": natural_summary["js_divergence"]["mean"],
            },
            "comparison": [args.base_run_1, args.base_run_2],
            "aggregate": natural_summary,
            "per_sample": natural_rows,
        },
    )

    run_paths = sorted(root.glob("*/raw_results.json"))
    overview = []
    natural_js = natural_summary["js_divergence"]["mean"]
    for run_path in run_paths:
        run = load_run(run_path)
        run_id = run["metadata"]["run_id"]
        if run_id in {args.base_run_1, args.base_run_2}:
            continue
        rows_to_base1 = paired_rows(base1, run)
        rows_to_base2 = paired_rows(base2, run)
        combined = []
        for first, second in zip(rows_to_base1, rows_to_base2):
            row = {"sample_id": first["sample_id"]}
            for key in first:
                if key in {"sample_id", "left_generated_text", "right_generated_text"}:
                    continue
                first_value = first[key]
                second_value = second[key]
                if first_value is None and second_value is None:
                    row[key] = None
                elif first_value is None:
                    row[key] = second_value
                elif second_value is None:
                    row[key] = first_value
                else:
                    row[key] = 0.5 * (first_value + second_value)
            row["method_generated_text"] = first["right_generated_text"]
            combined.append(row)
        aggregate = summarize(combined)
        method_js = aggregate["js_divergence"]["mean"]
        excess = method_js - natural_js
        ratio = method_js / natural_js if natural_js > 0 else None
        statement = (
            f"{run_id}: mean JS={method_js:.8g}; natural floor={natural_js:.8g}; "
            f"excess disruption={excess:.8g}."
        )
        report = {
            "conclusion": {
                "status": "complete",
                "statement": statement,
                "mean_js": method_js,
                "mean_natural_js": natural_js,
                "mean_excess_js": excess,
                "natural_disruption_ratio": ratio,
            },
            "comparison": {
                "method_run": run_id,
                "base_runs": [args.base_run_1, args.base_run_2],
                "distance_rule": "Per-sample arithmetic mean of distance to both Base runs.",
            },
            "aggregate": aggregate,
            "per_sample": combined,
        }
        write_report(reports / f"{run_id}.json", report)
        overview.append(report["conclusion"] | {"run_id": run_id})

    overview.sort(key=lambda row: row["mean_js"])
    write_report(
        reports / "all_experiments.json",
        {
            "conclusion": {
                "status": "complete",
                "statement": "Experiments are ordered by mean JS disruption; lower is better and the natural floor is the target.",
                "mean_natural_js": natural_js,
            },
            "ranking": overview,
            "natural_disruption": natural_summary,
        },
    )
    print(f"[done] reports written to {reports}")


if __name__ == "__main__":
    main()
