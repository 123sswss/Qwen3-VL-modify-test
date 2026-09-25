#!/usr/bin/env python3
"""Read-only V2 alpha audit from a saved checkpoint and training diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


EXPERIMENT = "pathvqa_v2_layer_mix_prefix_p20_norm_fixed_seed44"
LAYERS = (5, 11, 17)


def _json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _trajectory(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return sorted(rows, key=lambda row: int(row["step"]))


def _column_stats(alpha: torch.Tensor) -> dict[str, dict[str, float]]:
    return {
        str(layer): {
            "mean": float(alpha[:, index].mean()),
            "position_std_population": float(alpha[:, index].std(unbiased=False)),
            "min": float(alpha[:, index].min()),
            "max": float(alpha[:, index].max()),
            "mean_minus_one": float(alpha[:, index].mean() - 1.0),
        }
        for index, layer in enumerate(LAYERS)
    }


def _snapshot(row: dict) -> dict:
    alpha = torch.as_tensor(row["alpha_values"], dtype=torch.float64)
    if alpha.shape != (20, 3):
        raise ValueError(f"Invalid logged alpha at step={row['step']}: {tuple(alpha.shape)}")
    return {
        "step": int(row["step"]),
        "epoch": row.get("epoch"),
        "alpha_grad_norm_post_clip": row.get("alpha_grad_norm"),
        "alpha_change_rms": float((alpha - 1).square().mean().sqrt()),
        "alpha_column_mean": alpha.mean(dim=0).tolist(),
        "alpha_column_position_std": alpha.std(dim=0, unbiased=False).tolist(),
        "alpha_max_abs": float(alpha.abs().max()),
    }


def audit(run_root: Path, output_dir: Path) -> dict:
    checkpoint = run_root / "checkpoints" / "epoch_3" / "visual_selection_layer_mix.pt"
    config_path = checkpoint.parent / "visual_selection_layer_mix_config.json"
    trajectory_path = run_root / "v2_diagnostics.jsonl"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"V2 epoch3 checkpoint/config missing: {checkpoint.parent}")
    config = _json(config_path)
    if config.get("method") != "visual_selection_layer_mix_prefix_p20_v2" or int(config["init_seed"]) != 44:
        raise ValueError("Checkpoint is not the requested V2 seed44 architecture")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    alpha = state["alpha"].detach().to(dtype=torch.float64, device="cpu")
    weight = state["prefix_output.weight"].detach().to(dtype=torch.float64, device="cpu")
    if alpha.shape != (20, 3) or weight.shape != (2560, 192):
        raise ValueError(f"Unexpected alpha/weight shapes: {tuple(alpha.shape)}, {tuple(weight.shape)}")
    if not bool(torch.isfinite(alpha).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("Nonfinite checkpoint alpha/output weights")

    # The three column blocks are disjoint, so the Frobenius norms can be
    # computed exactly from three scalar block norms without constructing M_i.
    weight_norm_squared = torch.stack([
        weight[:, index * 64:(index + 1) * 64].square().sum()
        for index in range(3)
    ])
    alpha_mean = alpha.mean(dim=0)
    position_part = alpha - alpha_mean
    alpha_change = alpha - 1.0
    mapping_numerator = (position_part.square() * weight_norm_squared).sum()
    mapping_denominator = (alpha.square() * weight_norm_squared).sum()
    epsilon = torch.finfo(torch.float64).eps
    mapping_ratio = float((mapping_numerator / (mapping_denominator + epsilon)).sqrt())
    common_alpha_energy = 20.0 * (alpha_mean - 1.0).square().sum()
    position_alpha_energy = position_part.square().sum()
    common_mapping_change_energy = 20.0 * ((alpha_mean - 1.0).square() * weight_norm_squared).sum()
    position_mapping_change_energy = mapping_numerator

    rows = _trajectory(trajectory_path)
    selected = []
    if rows:
        indices = list(dict.fromkeys((0, len(rows) // 2, len(rows) - 1)))
        selected = [_snapshot(rows[index]) for index in indices]
    activation_paths = [
        run_root / "v2_real_batch_preflight.json",
        run_root / "v2_formula_preflight.json",
        run_root / "v2_greedy_equivalence.json",
    ]
    available_activations = []
    for path in activation_paths:
        if path.is_file():
            payload = _json(path)
            # Scalar RMS summaries and synthetic-formula checks cannot recover
            # actual sample-wise c, b_l, or per-position dynamic shifts.
            if isinstance(payload, dict) and any(
                isinstance(payload.get(key), list)
                for key in ("condition_vectors", "layer_components", "position_shifts")
            ):
                available_activations.append(str(path))
    activation_status = (
        "saved_arrays_require_separate_shape_audit" if available_activations
        else "not_saved_c_b_l_or_per_position_shifts_cannot_determine_sample_level_offset_difference"
    )
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=False).stdout.strip()
    result = {
        "experiment": EXPERIMENT,
        "scope": "existing_files_only_no_model_forward_no_training_no_evaluation",
        "audit_git_commit": git_commit,
        "source": {"run_root": str(run_root), "checkpoint": str(checkpoint),
                   "config": str(config_path),
                   "trajectory": str(trajectory_path) if trajectory_path.is_file() else None},
        "alpha": {
            "matrix": alpha.tolist(),
            "change_from_one_rms": float(alpha_change.square().mean().sqrt()),
            "change_from_one_max_abs": float(alpha_change.abs().max()),
            "column_stats": _column_stats(alpha),
            "common_shift_energy": float(common_alpha_energy),
            "position_difference_energy": float(position_alpha_energy),
            "position_share_of_change_energy": float(
                position_alpha_energy / (common_alpha_energy + position_alpha_energy + epsilon)
            ),
        },
        "effective_mapping": {
            "layer_weight_frobenius_norms": {
                str(layer): float(weight_norm_squared[index].sqrt())
                for index, layer in enumerate(LAYERS)
            },
            "R_position_relative": mapping_ratio,
            "numerator_frobenius_squared": float(mapping_numerator),
            "denominator_frobenius_squared": float(mapping_denominator),
            "common_change_frobenius_squared": float(common_mapping_change_energy),
            "position_change_frobenius_squared": float(position_mapping_change_energy),
            "position_share_of_mapping_change_energy": float(
                position_mapping_change_energy /
                (common_mapping_change_energy + position_mapping_change_energy + epsilon)
            ),
            "interpretation_boundary": "parameter_space_only_not_actual_sample_offsets",
        },
        "trajectory": {
            "records": len(rows),
            "first_step": int(rows[0]["step"]) if rows else None,
            "last_step": int(rows[-1]["step"]) if rows else None,
            "selected_early_middle_late": selected,
            "gradient_measurement": "post_global_clip_before_optimizer_step",
            "gradient_missing": not bool(rows),
        },
        "sample_level_offset": {
            "status": activation_status,
            "saved_activation_candidates": available_activations,
            "computed_relative_position_difference": None,
            "sample_count": None,
        },
        "limits": [
            "nonzero_gradient_is_not_evidence_of_useful_specialization",
            "R_is_not_sample_level_dynamic_shift_diversity",
            "single_seed_does_not_establish_general_utility",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "alpha_audit.json"
    report_path = output_dir / "alpha_audit.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    common_share = 1.0 - result["alpha"]["position_share_of_change_energy"]
    lines = [
        "# V2 seed44 alpha checkpoint audit (read-only)", "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Trajectory: `{trajectory_path}` ({len(rows)} logged records)",
        "- No model forward, training, validation, Test, or new seed.", "",
        "## Final alpha [20,3] (columns 5/11/17)", "",
        "```text",
        *["[" + ", ".join(f"{value:.8f}" for value in row) + "]" for row in alpha.tolist()],
        "```", "",
        f"Change from 1: RMS {result['alpha']['change_from_one_rms']:.8g}; "
        f"max abs {result['alpha']['change_from_one_max_abs']:.8g}. "
        f"Change-energy split: common {common_share:.3%}, across-position "
        f"{result['alpha']['position_share_of_change_energy']:.3%}.", "",
        "| Layer | Mean | Position std | Min | Max | W block Frobenius |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer in LAYERS:
        col = result["alpha"]["column_stats"][str(layer)]
        wnorm = result["effective_mapping"]["layer_weight_frobenius_norms"][str(layer)]
        lines.append(f"| {layer} | {col['mean']:.8f} | {col['position_std_population']:.8g} "
                     f"| {col['min']:.8f} | {col['max']:.8f} | {wnorm:.8g} |")
    lines.extend([
        "", f"Effective mapping position difference R = **{mapping_ratio:.8g}**. "
        "This is parameter-space difference, not observed sample-level shift difference.",
        "", "## Logged trajectory", "",
        "| Step | Epoch | alpha change RMS | alpha column means | alpha column position std | alpha grad norm |",
        "| ---: | ---: | ---: | --- | --- | ---: |",
    ])
    for item in selected:
        means = "/".join(f"{x:.6f}" for x in item["alpha_column_mean"])
        stds = "/".join(f"{x:.6g}" for x in item["alpha_column_position_std"])
        grad = item["alpha_grad_norm_post_clip"]
        lines.append(f"| {item['step']} | {item['epoch']} | {item['alpha_change_rms']:.8g} "
                     f"| {means} | {stds} | {grad if grad is not None else 'missing'} |")
    lines.extend([
        "", "Logged alpha gradients are **after global clipping, before optimizer.step** "
        "(the V2 callback is on_pre_optimizer_step).", "",
        "## Activation availability", "",
        f"- {activation_status}. No new forward is permitted or performed.",
        "- Inspect both alpha position variation and W-weighted R; do not infer usefulness "
        "from either alone. Retain normalized V1 as the reference; no new training is proposed.", "",
    ])
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"[V2_ALPHA_AUDIT] output={output_dir} R={mapping_ratio:.8g} "
          f"alpha_change_rms={result['alpha']['change_from_one_rms']:.8g} "
          f"position_share={result['alpha']['position_share_of_change_energy']:.6f} "
          f"activation_status={activation_status}")
    print(report_path.read_text(encoding="utf-8"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_root.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
