from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two checkpoint diagnostic runs.")
    parser.add_argument("left", help="First diagnostic output directory")
    parser.add_argument("right", help="Second diagnostic output directory")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def cosine_rows(left, right):
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    numerator = (left * right).sum(axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-8)


def load_run(path):
    run_dir = Path(path).resolve()
    vectors = np.load(run_dir / "vectors.npz", allow_pickle=False)
    index = {int(value): offset for offset, value in enumerate(vectors["global_index"])}
    return run_dir, vectors, index


def expert_alignment(left_outputs, right_outputs):
    adapter_count = left_outputs.shape[1]
    similarity = np.zeros((adapter_count, adapter_count), dtype=np.float64)
    for left_index in range(adapter_count):
        for right_index in range(adapter_count):
            similarity[left_index, right_index] = cosine_rows(
                left_outputs[:, left_index],
                right_outputs[:, right_index],
            ).mean()

    best_permutation = None
    best_score = -float("inf")
    for permutation in itertools.permutations(range(adapter_count)):
        score = np.mean(
            [similarity[left_index, permutation[left_index]] for left_index in range(adapter_count)]
        )
        if score > best_score:
            best_score = float(score)
            best_permutation = permutation
    return similarity, list(best_permutation), best_score


def summarize_subset(left, right, source_name=None):
    mask = np.ones(left["global_index"].shape[0], dtype=bool)
    if source_name is not None:
        mask = left["source_name"] == source_name
    left_outputs = left["adapter_outputs"][mask]
    right_outputs = right["adapter_outputs"][mask]
    similarity, permutation, alignment_score = expert_alignment(left_outputs, right_outputs)

    right_routes_aligned = right["route_probs"][mask][:, permutation]
    left_routes = left["route_probs"][mask]
    return {
        "count": int(mask.sum()),
        "expert_similarity_matrix": similarity.tolist(),
        "best_right_expert_for_each_left": permutation,
        "expert_alignment_cosine": alignment_score,
        "route_argmax_agreement_raw": float(
            (left_routes.argmax(axis=1) == right["route_probs"][mask].argmax(axis=1)).mean()
        ),
        "route_argmax_agreement_aligned": float(
            (left_routes.argmax(axis=1) == right_routes_aligned.argmax(axis=1)).mean()
        ),
        "route_l1_aligned": float(np.abs(left_routes - right_routes_aligned).mean()),
        "adapter_input_cosine": float(
            cosine_rows(left["adapter_input"][mask], right["adapter_input"][mask]).mean()
        ),
        "routed_delta_cosine": float(
            cosine_rows(left["routed_delta"][mask], right["routed_delta"][mask]).mean()
        ),
        "final_delta_cosine": float(
            cosine_rows(left["final_delta"][mask], right["final_delta"][mask]).mean()
        ),
    }


def main():
    args = parse_args()
    left_dir, left_raw, left_index = load_run(args.left)
    right_dir, right_raw, right_index = load_run(args.right)
    common = sorted(set(left_index) & set(right_index))
    if not common:
        raise RuntimeError("The two runs have no common global sample indices")

    left_offsets = np.asarray([left_index[index] for index in common], dtype=np.int64)
    right_offsets = np.asarray([right_index[index] for index in common], dtype=np.int64)
    left = {key: left_raw[key][left_offsets] for key in left_raw.files}
    right = {key: right_raw[key][right_offsets] for key in right_raw.files}

    sources = sorted(set(left["source_name"].tolist()) & set(right["source_name"].tolist()))
    result = {
        "left": str(left_dir),
        "right": str(right_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "overall": summarize_subset(left, right),
        "by_source": {
            source: summarize_subset(left, right, source_name=source)
            for source in sources
        },
    }

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = SCRIPT_DIR / "outputs" / f"comparison_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()

