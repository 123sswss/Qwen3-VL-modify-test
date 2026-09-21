#!/usr/bin/env python3
"""Compare endpoint parameter geometry across trained QDPT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slake.dynamic_prompt_tuning import (
    DYNAMIC_PROMPT_CONFIG_NAME,
    DYNAMIC_PROMPT_WEIGHTS_NAME,
)


STATIC_VISUAL_KEYS = {
    "private_visual_prompt",
    "static_visual_prompt",
    "workspace_visual_anchor",
}


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected non-empty NAME=CHECKPOINT")
    return name.strip(), Path(path).expanduser().resolve()


def state_tensors(state: Mapping[str, Any] | None) -> dict[str, torch.Tensor]:
    if state is None:
        return {}
    return {
        str(key): value.detach().float().cpu()
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def checkpoint_groups(state: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    sparse = state_tensors(state.get("sparse_visual"))
    projection = state_tensors(state.get("workspace_text_projection"))
    groups: dict[str, dict[str, torch.Tensor]] = {}
    if torch.is_tensor(state.get("soft_prompt")):
        groups["soft_prompt_p20"] = {"soft_prompt": state["soft_prompt"].float()}
    if torch.is_tensor(state.get("workspace_text_anchor")):
        groups["workspace_text_anchor_z10"] = {
            "workspace_text_anchor": state["workspace_text_anchor"].float()
        }
    groups["workspace_text_projection"] = projection
    groups["visual_static_prompts"] = {
        key: value
        for key, value in sparse.items()
        if key in STATIC_VISUAL_KEYS
    }
    groups["question_pooling_generator"] = {
        key: value
        for key, value in sparse.items()
        if key.startswith("workspace_text_")
    }
    groups["cross_attention_generator"] = {
        key: value
        for key, value in sparse.items()
        if key.startswith("workspace_query_norm.")
        or key.startswith("workspace_visual_memory_")
        or key.startswith("workspace_cross_attention.")
    }
    groups["visual_write_generator"] = {
        key: value
        for key, value in sparse.items()
        if key.startswith("workspace_visual_delta.")
    }
    generator = {
        **{f"text_projection.{key}": value for key, value in projection.items()},
        **{
            f"sparse_visual.{key}": value
            for key, value in sparse.items()
            if key not in STATIC_VISUAL_KEYS
        },
    }
    groups["all_dynamic_generator_parameters"] = generator
    return {name: tensors for name, tensors in groups.items() if tensors}


def flatten_group(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensors[key].reshape(-1) for key in sorted(tensors)])


def effective_rank(tensor: torch.Tensor) -> float | None:
    if tensor.ndim < 2 or min(tensor.shape) < 2:
        return None
    matrix = tensor.reshape(tensor.shape[0], -1).float()
    singular = torch.linalg.svdvals(matrix)
    total = singular.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return 0.0
    probability = singular / total
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(entropy.exp().item())


def group_statistics(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    flat = flatten_group(tensors)
    ranks = {
        key: rank
        for key, value in tensors.items()
        if (rank := effective_rank(value)) is not None
    }
    return {
        "parameter_count": int(flat.numel()),
        "l2_norm": float(flat.norm().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "max_abs": float(flat.abs().max().item()),
        "tensor_effective_ranks": ranks,
        "mean_tensor_effective_rank": (
            sum(ranks.values()) / len(ranks) if ranks else None
        ),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float(torch.dot(left, right).div(denominator).item())


def analyze_checkpoints(
    checkpoints: Mapping[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required")
    configs = {}
    grouped = {}
    for name, checkpoint in checkpoints.items():
        with (checkpoint / DYNAMIC_PROMPT_CONFIG_NAME).open(
            "r", encoding="utf-8"
        ) as handle:
            configs[name] = json.load(handle)
        state = torch.load(
            checkpoint / DYNAMIC_PROMPT_WEIGHTS_NAME,
            map_location="cpu",
            weights_only=True,
        )
        grouped[name] = checkpoint_groups(state)

    reference_name = next(iter(checkpoints))
    reference = dict(configs[reference_name])
    reference.pop("init_seed", None)
    for name, config in configs.items():
        normalized = dict(config)
        normalized.pop("init_seed", None)
        if normalized != reference:
            raise ValueError(
                f"Checkpoint architecture differs for {name}; endpoint geometry "
                "must only compare identical QDPT variants"
            )

    common_groups = set.intersection(*(set(value) for value in grouped.values()))
    pairwise_rows = []
    for left, right in combinations(checkpoints, 2):
        for group in sorted(common_groups):
            left_tensors = grouped[left][group]
            right_tensors = grouped[right][group]
            if list(sorted(left_tensors)) != list(sorted(right_tensors)):
                raise ValueError(f"Tensor keys differ for group={group}")
            left_flat = flatten_group(left_tensors)
            right_flat = flatten_group(right_tensors)
            if left_flat.shape != right_flat.shape:
                raise ValueError(f"Tensor shape differs for group={group}")
            pairwise_rows.append(
                {
                    "left": left,
                    "right": right,
                    "group": group,
                    "parameter_count": int(left_flat.numel()),
                    "cosine_similarity": cosine(left_flat, right_flat),
                    "relative_l2_distance": float(
                        (left_flat - right_flat).norm().div(
                            left_flat.norm().clamp_min(1e-30)
                        ).item()
                    ),
                }
            )

    report = {
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "init_seeds": {
            name: config.get("init_seed") for name, config in configs.items()
        },
        "architecture_match_ignoring_init_seed": True,
        "per_checkpoint": {
            name: {
                group: group_statistics(tensors)
                for group, tensors in groups.items()
            }
            for name, groups in grouped.items()
        },
        "pairwise": pairwise_rows,
        "scope_note": (
            "Q10 and sample-conditioned dynamic Z/Prompt values are activations, "
            "not standalone checkpoint tensors; their training-time scalar traces "
            "remain in dynamic_prompt_diagnostics.jsonl."
        ),
    }
    return report, pairwise_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise ValueError("Duplicate checkpoint names are not allowed")
    report, pairwise_rows = analyze_checkpoints(checkpoints)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "checkpoint_representation_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    tsv_path = output_dir / "checkpoint_representation_pairwise.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(pairwise_rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print(
        "[QDPT_CHECKPOINT_REPRESENTATIONS] "
        f"checkpoints={list(checkpoints)} groups={sorted(report['per_checkpoint'][next(iter(checkpoints))])} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
