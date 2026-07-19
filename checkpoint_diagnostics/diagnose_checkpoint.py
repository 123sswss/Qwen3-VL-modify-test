from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DIR = PROJECT_ROOT / "test"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEST_DIR))


DEFAULT_CHECKPOINT = (
    "/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/"
    "visual_router_layer_fixed_v4_diversity_recover_20260719_5/final"
)
DEFAULT_BASE_MODEL = "/root/autodl-tmp/model"
DEFAULT_DATASETS = [
    "/root/autodl-tmp/dataset/test2_val.json",
    "/root/autodl-tmp/dataset/seen_simple/llava_test.json",
]
DEFAULT_IMAGE_DIRS = [
    "/root/autodl-tmp/dataset/2/train",
    "/root/autodl-tmp/dataset/seen_simple/image",
]

PROGRESS_RE = re.compile(
    r"^\[(?P<done>\d+)/(?P<total>\d+)\]\s+"
    r"(?P<icon>[^\s]+)\s+GT=(?P<gt>\S+)\s+Pred=(?P<pred>\S+)\s+"
    r"Acc=.*?\|\s+(?P<item_id>.+?)\s*$"
)
STATUS_BY_ICON = {"✓": "CORRECT", "✗": "WRONG", "⚠": "REGEX_FAIL"}


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
        description="Profile one trained MMRL checkpoint without regenerating answers."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--eval-log", default=None)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--image-dir", action="append", dest="image_dirs")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory. Defaults to checkpoint_diagnostics/outputs/<run>_<time>.",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=0,
        help="Profile at most N samples from each source; 0 means all samples.",
    )
    return parser.parse_args()


def source_name(json_path: str):
    path = Path(json_path)
    parent = path.parent.name
    return f"{parent}__{path.stem}" if parent else path.stem


def load_dataset(json_paths):
    rows = []
    for source_index, json_path in enumerate(json_paths):
        with Path(json_path).open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        if not isinstance(dataset, list):
            raise ValueError(f"Dataset is not a list: {json_path}")
        for source_item_index, item in enumerate(dataset):
            row = dict(item)
            row["_source_json"] = json_path
            row["_source_name"] = source_name(json_path)
            row["_source_index"] = source_index
            row["_source_item_index"] = source_item_index
            row["_global_index"] = len(rows)
            rows.append(row)
    return rows


def parse_original_eval_log(log_path: Path):
    records = {}
    if not log_path.is_file():
        print(f"[WARN] Existing eval log not found: {log_path}")
        return records

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = PROGRESS_RE.match(line.strip())
            if match is None:
                continue
            global_index = int(match.group("done")) - 1
            icon = match.group("icon")
            records[global_index] = {
                "status": STATUS_BY_ICON.get(icon, "UNKNOWN"),
                "gt": match.group("gt"),
                "pred": match.group("pred"),
                "logged_item_id": match.group("item_id"),
            }
    print(f"[INFO] Parsed {len(records)} scored items from {log_path}")
    return records


def extract_question_and_gt(item):
    question = ""
    gt = None
    for conversation in item.get("conversations", []):
        if conversation.get("from") == "human":
            question = conversation.get("value", "")
            question = question.replace("<image>\n", "").replace("<image>", "")
        elif conversation.get("from") == "gpt":
            match = re.search(r"\[\[([A-Da-d])\]\]", conversation.get("value", ""))
            if match:
                gt = match.group(1).upper()
    return question, gt


def resolve_image_path(image_file: str, image_dirs, source_json: str):
    candidates = [Path(image_dir) / image_file for image_dir in image_dirs]
    candidates.append(Path(source_json).resolve().parent / image_file)
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def scalar_tensor(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return finite_float(value.detach().float().item())
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return finite_float(value)
    return None


class AdapterVectorCapture:
    def __init__(self, visual):
        self.inputs = {}
        self.outputs = {}
        self.handles = []
        for index, adapter in enumerate(visual.residual_adapters):
            self.handles.append(adapter.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def capture(_module, inputs, output):
            if inputs and torch.is_tensor(inputs[0]):
                self.inputs[index] = inputs[0].detach()
            if torch.is_tensor(output):
                self.outputs[index] = output.detach()

        return capture

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()

    def snapshot(self, visual):
        adapter_count = len(visual.residual_adapters)
        if len(self.outputs) != adapter_count or 0 not in self.inputs:
            return None

        adapter_input = self.inputs[0].float().mean(dim=0)
        adapter_outputs = torch.stack(
            [self.outputs[index].float().mean(dim=0) for index in range(adapter_count)],
            dim=0,
        )
        routes = getattr(visual, "route_probs", None)
        if not torch.is_tensor(routes) or routes.numel() == 0:
            return None
        route_mean = routes.detach().float().mean(dim=0)
        route_mean = route_mean / route_mean.sum().clamp_min(1e-8)
        routed = (adapter_outputs * route_mean.unsqueeze(-1)).sum(dim=0)

        gate = getattr(visual, "G_list", None)
        gate_mean = gate.detach().float().mean() if torch.is_tensor(gate) else routed.new_tensor(1.0)
        final_vector = routed * gate_mean
        return {
            "adapter_input": adapter_input.cpu().half().numpy(),
            "adapter_outputs": adapter_outputs.cpu().half().numpy(),
            "route_probs": route_mean.cpu().numpy().astype(np.float32),
            "routed_delta": routed.cpu().half().numpy(),
            "final_delta": final_vector.cpu().half().numpy(),
            "gate": np.float32(gate_mean.cpu().item()),
        }


def prefill(interface, capture, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text_prompt = interface.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if hasattr(interface.model.model, "rope_deltas"):
        interface.model.model.rope_deltas = None
    inputs = interface.processor(
        text=[text_prompt],
        images=image,
        padding=False,
        max_length=False,
        truncation=False,
        return_tensors="pt",
    ).to(interface.model.device)

    capture.clear()
    with torch.inference_mode():
        output = interface.model(**inputs, use_cache=False, return_dict=True)
    del output, inputs


def collect_diagnostics(visual):
    diagnostics = {}
    route_probs = getattr(visual, "route_probs", None)
    if torch.is_tensor(route_probs) and route_probs.numel() > 0:
        routes = route_probs.detach().float()
        route_mean = routes.mean(dim=0)
        route_mean = route_mean / route_mean.sum().clamp_min(1e-8)
        diagnostics["route_probs"] = [finite_float(v) for v in route_mean.cpu().tolist()]
        diagnostics["route_winner"] = int(route_mean.argmax().item())
        diagnostics["route_confidence"] = finite_float(routes.max(dim=-1).values.mean().item())
        entropy = -(route_mean * route_mean.clamp_min(1e-8).log()).sum()
        if route_mean.numel() > 1:
            entropy = entropy / torch.log(route_mean.new_tensor(float(route_mean.numel())))
        diagnostics["route_entropy_norm"] = finite_float(entropy.item())

    gate = getattr(visual, "G_list", None)
    if torch.is_tensor(gate) and gate.numel() > 0:
        gate_f = gate.detach().float()
        diagnostics["gate_mean"] = finite_float(gate_f.mean().item())
        diagnostics["gate_min"] = finite_float(gate_f.min().item())
        diagnostics["gate_max"] = finite_float(gate_f.max().item())
        diagnostics["gate_active_fraction"] = finite_float((gate_f > 0.5).float().mean().item())

    alpha = getattr(visual, "alpha_list", None)
    if torch.is_tensor(alpha) and alpha.numel() > 0:
        alpha_prob = torch.sigmoid(alpha.detach().float())
        diagnostics["alpha_prob_mean"] = finite_float(alpha_prob.mean().item())
        diagnostics["alpha_prob_min"] = finite_float(alpha_prob.min().item())
        diagnostics["alpha_prob_max"] = finite_float(alpha_prob.max().item())

    for key, value in (getattr(visual, "debug_context", {}) or {}).items():
        scalar = scalar_tensor(value)
        if scalar is not None:
            diagnostics[key] = scalar
    return diagnostics


def numeric_metrics(rows):
    values = {}
    for row in rows:
        for key, value in row.get("diagnostics", {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(float(value)):
                values.setdefault(key, []).append(float(value))
    summary = {}
    for key, items in sorted(values.items()):
        array = np.asarray(items, dtype=np.float64)
        summary[key] = {
            "count": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return summary


def summarize_group(rows):
    statuses = Counter(row.get("status", "UNKNOWN") for row in rows)
    scored = statuses["CORRECT"] + statuses["WRONG"] + statuses["REGEX_FAIL"]
    score = 100.0 * statuses["CORRECT"] / scored if scored else None
    route_vectors = [
        row["diagnostics"].get("route_probs")
        for row in rows
        if row.get("diagnostics", {}).get("route_probs") is not None
    ]
    route_usage = None
    winner_counts = {}
    if route_vectors:
        route_array = np.asarray(route_vectors, dtype=np.float64)
        route_usage = route_array.mean(axis=0).tolist()
        winner_counts = {
            str(key): int(value)
            for key, value in sorted(Counter(route_array.argmax(axis=1).tolist()).items())
        }
    return {
        "profiled": len(rows),
        "status_counts": dict(statuses),
        "score_from_original_log": score,
        "route_usage": route_usage,
        "route_winner_counts": winner_counts,
        "metrics": numeric_metrics(rows),
    }


def build_summary(rows, checkpoint, eval_log, elapsed):
    sources = sorted({row["source_name"] for row in rows})
    by_source = {}
    for name in sources:
        source_rows = [row for row in rows if row["source_name"] == name]
        correct = [row for row in source_rows if row.get("status") == "CORRECT"]
        incorrect = [
            row
            for row in source_rows
            if row.get("status") in {"WRONG", "REGEX_FAIL"}
        ]
        by_source[name] = {
            "all": summarize_group(source_rows),
            "correct": summarize_group(correct),
            "incorrect": summarize_group(incorrect),
        }
    return {
        "checkpoint": str(checkpoint),
        "eval_log": str(eval_log),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "overall": summarize_group(rows),
        "by_source": by_source,
    }


def write_vectors(path, vector_rows):
    if not vector_rows:
        print("[WARN] No adapter vectors were captured")
        return
    np.savez_compressed(
        path,
        global_index=np.asarray([row["global_index"] for row in vector_rows], dtype=np.int64),
        item_id=np.asarray([row["item_id"] for row in vector_rows], dtype=str),
        source_name=np.asarray([row["source_name"] for row in vector_rows], dtype=str),
        status=np.asarray([row["status"] for row in vector_rows], dtype=str),
        adapter_input=np.stack([row["vectors"]["adapter_input"] for row in vector_rows]),
        adapter_outputs=np.stack([row["vectors"]["adapter_outputs"] for row in vector_rows]),
        route_probs=np.stack([row["vectors"]["route_probs"] for row in vector_rows]),
        routed_delta=np.stack([row["vectors"]["routed_delta"] for row in vector_rows]),
        final_delta=np.stack([row["vectors"]["final_delta"] for row in vector_rows]),
        gate=np.asarray([row["vectors"]["gate"] for row in vector_rows], dtype=np.float32),
    )
    print(f"[SAVE] vectors={path}")


def main():
    args = parse_args()
    from inferEngine import ModelInterface

    checkpoint = Path(args.checkpoint).resolve()
    base_model = Path(args.base_model).resolve()
    datasets = args.datasets or DEFAULT_DATASETS
    image_dirs = args.image_dirs or DEFAULT_IMAGE_DIRS
    eval_log = Path(args.eval_log).resolve() if args.eval_log else checkpoint.parent / "eval" / "test.log"

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        experiment_name = checkpoint.parent.name if checkpoint.name == "final" else checkpoint.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SCRIPT_DIR / "outputs" / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    console_log = (output_dir / "console.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, console_log)
    sys.stderr = Tee(sys.__stderr__, console_log)

    print(f"[OUTPUT] {output_dir}")
    print(f"[CHECKPOINT] {checkpoint}")
    print(f"[BASE] {base_model}")
    print(f"[EVAL_LOG] {eval_log}")
    print(f"[DATASETS] {datasets}")
    print("[MODE] visual prefill only; answers are read from the existing eval log")

    original_records = parse_original_eval_log(eval_log)
    dataset = load_dataset(datasets)
    interface = ModelInterface(str(checkpoint), str(base_model))
    visual = interface.model.model.visual
    capture = AdapterVectorCapture(visual)

    rows = []
    vector_rows = []
    source_seen = Counter()
    samples_path = output_dir / "samples.jsonl"
    started = time.time()

    try:
        with samples_path.open("w", encoding="utf-8", buffering=1) as samples_file:
            for item in dataset:
                name = item["_source_name"]
                if args.limit_per_source > 0 and source_seen[name] >= args.limit_per_source:
                    continue
                source_seen[name] += 1

                global_index = int(item["_global_index"])
                item_id = str(item.get("id", f"unknown_{global_index}"))
                question, dataset_gt = extract_question_and_gt(item)
                original = original_records.get(global_index, {})
                status = original.get("status", "UNKNOWN")
                image_path = resolve_image_path(
                    str(item.get("image", "")),
                    image_dirs,
                    item["_source_json"],
                )

                row = {
                    "global_index": global_index,
                    "source_name": name,
                    "source_json": item["_source_json"],
                    "source_item_index": int(item["_source_item_index"]),
                    "item_id": item_id,
                    "status": status,
                    "gt": original.get("gt", dataset_gt),
                    "pred": original.get("pred"),
                    "logged_item_id": original.get("logged_item_id"),
                    "image_path": str(image_path) if image_path else None,
                    "diagnostics": {},
                }

                if image_path is None or dataset_gt is None:
                    row["profile_status"] = "SKIP_MISSING_INPUT"
                else:
                    try:
                        with Image.open(image_path) as opened:
                            image = opened.convert("RGB")
                        prefill(interface, capture, image, question)
                        row["diagnostics"] = collect_diagnostics(visual)
                        row["profile_status"] = "OK"
                        vectors = capture.snapshot(visual)
                        if vectors is not None:
                            vector_rows.append(
                                {
                                    "global_index": global_index,
                                    "item_id": item_id,
                                    "source_name": name,
                                    "status": status,
                                    "vectors": vectors,
                                }
                            )
                    except Exception as exc:
                        row["profile_status"] = "ERROR"
                        row["profile_error"] = repr(exc)
                        print(f"[ERROR] index={global_index} id={item_id}: {exc}")

                rows.append(row)
                samples_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                done = len(rows)
                if done == 1 or done % 25 == 0:
                    route = row.get("diagnostics", {}).get("route_probs")
                    gate = row.get("diagnostics", {}).get("gate_mean")
                    print(
                        f"[PROGRESS] {done}/{len(dataset)} source={name} id={item_id} "
                        f"status={status} gate={gate} route={route}"
                    )
    finally:
        capture.close()

    elapsed = round(time.time() - started, 3)
    summary = build_summary(rows, checkpoint, eval_log, elapsed)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_vectors(output_dir / "vectors.npz", vector_rows)

    print(f"[SAVE] samples={samples_path}")
    print(f"[SAVE] summary={summary_path}")
    print(f"[DONE] profiled={len(rows)} elapsed={elapsed}s")
    for name, groups in summary["by_source"].items():
        group = groups["all"]
        metrics = group["metrics"]
        print(
            f"[SOURCE] {name} score={group['score_from_original_log']} "
            f"route={group['route_usage']} "
            f"gate={metrics.get('gate_mean', {}).get('mean')} "
            f"delta/org={metrics.get('delta_to_org_ratio', {}).get('mean')}"
        )


if __name__ == "__main__":
    main()
