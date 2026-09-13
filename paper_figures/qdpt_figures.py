#!/usr/bin/env python3
"""Generate publication figures from QDPT logs and attention exports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SEED_COLORS = ("#167D77", "#D56A3A", "#3366A3")
METHOD_COLORS = ("#8B6F47", "#D56A3A", "#167D77", "#3366A3")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except ImportError as exc:
        raise RuntimeError(
            "Paper figures require matplotlib. Install pathvqa/requirements.txt."
        ) from exc
    return plt, colors


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No diagnostic rows found in {path}")
    return rows


def find_one(root: Path, names: Sequence[str]) -> Path:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    matches = sorted(
        path
        for name in names
        for path in root.rglob(Path(name).name)
        if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"None of {list(names)} found under {root}")
    return matches[0]


def parse_run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must use LABEL=/path/to/run")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Run label and path must be non-empty")
    return label.strip(), Path(raw_path).expanduser()


def finite_series(
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    selected_key = next((key for key in keys if any(key in row for row in rows)), None)
    if selected_key is None:
        raise KeyError(f"None of the requested metrics exist: {list(keys)}")
    points = []
    for row in rows:
        if selected_key not in row or "step" not in row:
            continue
        try:
            step = float(row["step"])
            metric = float(row[selected_key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(step) and math.isfinite(metric):
            points.append((step, metric))
    if not points:
        raise ValueError(f"No finite values for {selected_key}")
    x, y = zip(*points)
    return np.asarray(x), np.asarray(y), selected_key


def progress_percent(steps: np.ndarray, maximum: float | None = None) -> np.ndarray:
    denominator = float(maximum if maximum is not None else np.max(steps))
    if denominator <= 0:
        raise ValueError("Training steps must contain a positive maximum")
    return steps / denominator * 100.0


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values.copy()
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_losses(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    state_path = find_one(
        run_dir,
        ("trainer/trainer_state.json", "trainer_state.json"),
    )
    state = load_json(state_path)
    points = []
    for row in state.get("log_history", []):
        if "loss" not in row or "step" not in row:
            continue
        step, loss = float(row["step"]), float(row["loss"])
        if math.isfinite(step) and math.isfinite(loss):
            points.append((step, loss))
    if not points:
        raise ValueError(f"No finite step loss values in {state_path}")
    x, y = zip(*points)
    return np.asarray(x), np.asarray(y)


def load_diagnostics(run_dir: Path) -> list[dict[str, Any]]:
    return load_jsonl(find_one(run_dir, ("dynamic_prompt_diagnostics.jsonl",)))


def configure_style() -> None:
    plt, _ = _require_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#46534F",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#FFFEFA",
            "figure.facecolor": "#F7F4EC",
            "grid.color": "#DDD8CC",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def save_all(fig: Any, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
    ):
        fig.savefig(output_prefix.with_suffix(suffix), **kwargs)


def style_axis(axis: Any) -> None:
    axis.grid(axis="y", alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Training progress (%)")


def plot_dynamics(
    runs: Sequence[tuple[str, Path]],
    scores: Sequence[float] | None,
    output: Path,
    smooth_window: int,
) -> None:
    plt, _ = _require_matplotlib()
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    score_by_label = {
        label: score for (label, _), score in zip(runs, scores or ())
    }
    for index, (label, run_dir) in enumerate(runs):
        color = SEED_COLORS[index % len(SEED_COLORS)]
        diagnostics = load_diagnostics(run_dir)
        diag_steps = np.asarray([float(row["step"]) for row in diagnostics])
        loss_steps, losses = load_losses(run_dir)
        maximum = float(max(np.max(diag_steps), np.max(loss_steps)))
        line_label = label
        if label in score_by_label:
            line_label += f"  ({score_by_label[label]:.2f})"
        axes[0, 0].plot(
            progress_percent(loss_steps, maximum),
            moving_average(losses, smooth_window),
            color=color,
            linewidth=2.0,
            label=line_label,
        )
        for keys, axis, linestyle in (
            (("soft_prompt_norm",), axes[0, 1], "-"),
            (("workspace_text_anchor_norm",), axes[0, 1], "--"),
            (("workspace_norm_mean", "workspace_norm_mean_layer17"), axes[1, 0], "-"),
            (("workspace_visual_attention_entropy_norm", "workspace_visual_attention_entropy_norm_layer17"), axes[1, 1], "-"),
        ):
            steps, values, _ = finite_series(diagnostics, keys)
            axis.plot(
                progress_percent(steps, maximum),
                moving_average(values, smooth_window),
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
    titles = (
        (axes[0, 0], "Training loss", "Cross-entropy"),
        (axes[0, 1], "Prompt scale", "L2 norm"),
        (axes[1, 0], "Workspace Z10 scale", "Mean token norm"),
        (axes[1, 1], "Visual attention entropy", "Normalized entropy"),
    )
    for axis, title, ylabel in titles:
        style_axis(axis)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
    axes[0, 0].legend(title="Seed (final Val.)", ncol=1)
    prompt_handles = [
        axes[0, 1].plot([], [], color="#46534F", linestyle="-", label="P20")[0],
        axes[0, 1].plot(
            [], [], color="#46534F", linestyle="--", label="Text anchor"
        )[0],
    ]
    axes[0, 1].legend(handles=prompt_handles)
    fig.suptitle(
        "QDPT optimization follows seed-dependent trajectories",
        fontsize=16,
        fontweight="bold",
        color="#183C3A",
    )
    save_all(fig, output)
    plt.close(fig)


def plot_activity(run: tuple[str, Path], output: Path, smooth_window: int) -> None:
    plt, _ = _require_matplotlib()
    configure_style()
    label, run_dir = run
    diagnostics = load_diagnostics(run_dir)
    maximum = max(float(row["step"]) for row in diagnostics)
    metrics = (
        (
            "Cross-attention update / query",
            ("workspace_cross_delta_to_query_ratio", "workspace_cross_delta_to_query_ratio_layer17"),
            False,
        ),
        (
            "Dynamic text delta / anchor",
            ("workspace_text_delta_to_anchor_ratio",),
            False,
        ),
        (
            "Workspace slot cosine",
            ("workspace_slot_pairwise_cosine_mean", "workspace_slot_pairwise_cosine_mean_layer17"),
            False,
        ),
        ("Static visual Prompt gradient", ("sparse_visual_grad_norm",), True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), constrained_layout=True)
    for axis, (title, keys, log_scale) in zip(axes.flat, metrics):
        steps, values, _ = finite_series(diagnostics, keys)
        axis.plot(
            progress_percent(steps, maximum),
            moving_average(values, smooth_window),
            color="#167D77" if not log_scale else "#D56A3A",
            linewidth=2.2,
        )
        if log_scale:
            axis.set_yscale("log")
        style_axis(axis)
        axis.set_title(title, loc="left", fontweight="bold")
    fig.suptitle(
        f"QDPT modules remain active throughout training ({label})",
        fontsize=16,
        fontweight="bold",
        color="#183C3A",
    )
    save_all(fig, output)
    plt.close(fig)


def plot_stability(score_file: Path, output: Path) -> None:
    plt, _ = _require_matplotlib()
    configure_style()
    config = load_json(score_file)
    methods = config["methods"]
    seeds = config.get("seeds", list(range(len(methods[0]["scores"]))))
    fig, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for index, method in enumerate(methods):
        scores = np.asarray(method["scores"], dtype=float)
        x = np.full(scores.shape, index, dtype=float)
        offsets = np.linspace(-0.10, 0.10, scores.size)
        color = METHOD_COLORS[index % len(METHOD_COLORS)]
        axis.scatter(x + offsets, scores, s=48, color=color, zorder=3)
        mean = float(scores.mean())
        std = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
        axis.errorbar(
            index,
            mean,
            yerr=std,
            fmt="D",
            markersize=6,
            color="#1E2926",
            ecolor=color,
            elinewidth=2.2,
            capsize=5,
            zorder=4,
        )
        axis.text(index, scores.max() + 0.35, f"{mean:.2f} ± {std:.2f}", ha="center", fontsize=9)
        for offset, score, seed in zip(offsets, scores, seeds):
            axis.annotate(str(seed), (index + offset, score), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7, color="#606A65")
    axis.set_xticks(range(len(methods)), [method["name"] for method in methods])
    axis.set_ylabel("Overall accuracy (%)")
    axis.set_title(
        "Conditional Prompt methods show higher initialization sensitivity",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#183C3A",
    )
    axis.text(
        0,
        1.01,
        f"{config.get('dataset', '')} · dots are seeds · diamonds are mean ± sample standard deviation",
        transform=axis.transAxes,
        color="#606A65",
        fontsize=9,
    )
    axis.grid(axis="y", alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    save_all(fig, output)
    plt.close(fig)


def attention_maps(attention: np.ndarray, grid_thw: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if attention.ndim != 3:
        raise ValueError("Attention must have shape [heads, queries, visual_tokens]")
    temporal, height, width = (int(value) for value in grid_thw)
    expected = temporal * height * width
    if attention.shape[-1] != expected:
        raise ValueError(
            f"Attention token count {attention.shape[-1]} != grid product {expected}"
        )
    per_query = attention.mean(axis=0).reshape(attention.shape[1], temporal, height, width).mean(axis=1)
    aggregate = per_query.mean(axis=0)
    probabilities = attention.mean(axis=0)
    probabilities = probabilities / np.clip(probabilities.sum(axis=-1, keepdims=True), 1e-12, None)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=-1)
    entropy /= math.log(max(2, probabilities.shape[-1]))
    top_queries = np.argsort(entropy)[: min(3, attention.shape[1])]
    return aggregate, per_query, top_queries


def normalized_map(values: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(values, (2, 98))
    if upper <= lower:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def plot_attention(bundle_dir: Path, output: Path) -> None:
    plt, colors = _require_matplotlib()
    configure_style()
    manifest = load_json(bundle_dir / "manifest.json")
    image = plt.imread(bundle_dir / manifest["image"])
    entries = manifest["entries"]
    columns = 5
    fig, axes = plt.subplots(
        len(entries), columns, figsize=(15, 3.2 * len(entries)), constrained_layout=True
    )
    if len(entries) == 1:
        axes = np.expand_dims(axes, 0)
    cmap = plt.get_cmap("inferno")
    for row_index, entry in enumerate(entries):
        archive = np.load(bundle_dir / entry["attention"])
        aggregate, per_query, top_queries = attention_maps(
            archive["cross_attention"], archive["grid_thw"]
        )
        maps = [aggregate] + [per_query[index] for index in top_queries]
        titles = ["Mean over 10 queries"] + [f"Focused query Q{int(index) + 1}" for index in top_queries]
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title("Original image", fontweight="bold")
        axes[row_index, 0].set_ylabel(
            f"Question {row_index + 1}\n{entry['question']}\nPred: {entry['prediction']}",
            fontsize=8.5,
        )
        for column, (heatmap, title) in enumerate(zip(maps, titles), 1):
            axes[row_index, column].imshow(image)
            axes[row_index, column].imshow(
                normalized_map(heatmap),
                cmap=cmap,
                alpha=0.58,
                interpolation="bicubic",
                norm=colors.Normalize(0, 1),
                extent=(0, image.shape[1], image.shape[0], 0),
            )
            axes[row_index, column].set_title(title, fontweight="bold")
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "The same image yields question-dependent visual evidence",
        fontsize=17,
        fontweight="bold",
        color="#183C3A",
    )
    save_all(fig, output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dynamics = subparsers.add_parser("dynamics")
    dynamics.add_argument("--run", action="append", type=parse_run_argument, required=True)
    dynamics.add_argument("--score", action="append", type=float)
    dynamics.add_argument("--smooth-window", type=int, default=7)
    dynamics.add_argument("--output", type=Path, required=True)

    activity = subparsers.add_parser("activity")
    activity.add_argument("--run", type=parse_run_argument, required=True)
    activity.add_argument("--smooth-window", type=int, default=7)
    activity.add_argument("--output", type=Path, required=True)

    stability = subparsers.add_parser("stability")
    stability.add_argument(
        "--scores",
        type=Path,
        default=Path(__file__).with_name("pathvqa_seed_scores.json"),
    )
    stability.add_argument("--output", type=Path, required=True)

    attention = subparsers.add_parser("attention")
    attention.add_argument("--bundle-dir", type=Path, required=True)
    attention.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "dynamics":
        if args.score is not None and len(args.score) != len(args.run):
            raise ValueError("Provide one --score for every --run, or none")
        plot_dynamics(args.run, args.score, args.output, args.smooth_window)
    elif args.command == "activity":
        plot_activity(args.run, args.output, args.smooth_window)
    elif args.command == "stability":
        plot_stability(args.scores, args.output)
    elif args.command == "attention":
        plot_attention(args.bundle_dir, args.output)
    else:
        raise AssertionError(args.command)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
