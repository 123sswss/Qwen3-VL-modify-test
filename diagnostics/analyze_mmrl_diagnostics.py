from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "slake" / "outputs" / "mmrl"
DIAGNOSTICS_RELATIVE_PATH = Path("stage3/metrics/mmrl_diagnostics.jsonl")
SCORE_PATTERN = re.compile(r"^Overall Accuracy:\s*([-+]?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
STAT_SUFFIXES = ("_mean", "_std", "_latest")
IDENTITY_KEYS = {"stage_name", "step_start", "step_end", "window_size"}


@dataclass(frozen=True)
class MetricSummary:
    name: str
    samples: int
    global_mean: float
    global_std: float
    first_window_mean: float
    last_window_mean: float
    latest: float
    window_mean_min: float
    window_mean_max: float
    slope_per_100_steps: float


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metric_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    names = set()
    for row in rows:
        for key in row:
            if key in IDENTITY_KEYS:
                continue
            for suffix in STAT_SUFFIXES:
                if key.endswith(suffix):
                    names.add(key[: -len(suffix)])
                    break
    return sorted(names)


def weighted_slope(points: list[tuple[float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total_weight = sum(weight for _, _, weight in points)
    if total_weight <= 0:
        return 0.0
    mean_x = sum(x * weight for x, _, weight in points) / total_weight
    mean_y = sum(y * weight for _, y, weight in points) / total_weight
    denominator = sum(weight * (x - mean_x) ** 2 for x, _, weight in points)
    if denominator <= 0:
        return 0.0
    numerator = sum(
        weight * (x - mean_x) * (y - mean_y) for x, y, weight in points
    )
    return numerator / denominator


def summarize_metric(rows: list[dict[str, Any]], name: str) -> MetricSummary | None:
    windows = []
    for row in rows:
        mean = finite_number(row.get(f"{name}_mean"))
        if mean is None:
            continue
        std = finite_number(row.get(f"{name}_std")) or 0.0
        latest = finite_number(row.get(f"{name}_latest"))
        size = finite_number(row.get("window_size")) or 1.0
        size = max(size, 1.0)
        start = finite_number(row.get("step_start")) or 0.0
        end = finite_number(row.get("step_end")) or start
        midpoint = (start + end) / 2.0
        windows.append((mean, std, latest, size, midpoint))

    if not windows:
        return None

    sample_count = int(sum(window[3] for window in windows))
    total_weight = sum(window[3] for window in windows)
    global_mean = sum(mean * size for mean, _, _, size, _ in windows) / total_weight
    second_moment = sum(
        size * (std**2 + mean**2) for mean, std, _, size, _ in windows
    ) / total_weight
    global_std = math.sqrt(max(second_moment - global_mean**2, 0.0))
    latest_values = [latest for _, _, latest, _, _ in windows if latest is not None]
    means = [mean for mean, _, _, _, _ in windows]
    slope = weighted_slope(
        [(midpoint, mean, size) for mean, _, _, size, midpoint in windows]
    )
    return MetricSummary(
        name=name,
        samples=sample_count,
        global_mean=global_mean,
        global_std=global_std,
        first_window_mean=means[0],
        last_window_mean=means[-1],
        latest=latest_values[-1] if latest_values else means[-1],
        window_mean_min=min(means),
        window_mean_max=max(means),
        slope_per_100_steps=slope * 100.0,
    )


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc.msg}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"line {line_number}: expected object, got {type(payload).__name__}")
                continue
            rows.append(payload)
    rows.sort(key=lambda row: (row.get("step_start", 0), row.get("step_end", 0)))
    return rows, errors


def discover_paths(inputs: list[Path]) -> list[Path]:
    discovered = set()
    for raw_path in inputs or [DEFAULT_ROOT]:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            if path.name == DIAGNOSTICS_RELATIVE_PATH.name:
                discovered.add(path)
            continue
        direct = path / DIAGNOSTICS_RELATIVE_PATH
        if direct.is_file():
            discovered.add(direct)
            continue
        if path.is_dir():
            discovered.update(path.glob(f"*/{DIAGNOSTICS_RELATIVE_PATH.as_posix()}"))
    return sorted(
        discovered,
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )


def experiment_dir(path: Path) -> Path:
    try:
        return path.parents[2]
    except IndexError:
        return path.parent


def read_score(directory: Path) -> float | None:
    eval_log = directory / "eval.log"
    if not eval_log.is_file():
        return None
    content = eval_log.read_text(encoding="utf-8", errors="ignore")
    matches = SCORE_PATTERN.findall(content)
    return float(matches[-1]) if matches else None


def format_number(value: float) -> str:
    return f"{value:.6g}"


def print_summary(
    path: Path,
    rows: list[dict[str, Any]],
    errors: list[str],
    metric_filter: re.Pattern[str] | None,
) -> None:
    directory = experiment_dir(path)
    score = read_score(directory)
    names = metric_names(rows)
    if metric_filter is not None:
        names = [name for name in names if metric_filter.search(name)]
    summaries = [summary for name in names if (summary := summarize_metric(rows, name))]

    print(f"===== {directory.name} =====")
    print(f"diagnostics: {path}")
    print(f"modified: {datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(sep=' ', timespec='seconds')}")
    if score is not None:
        print(f"Overall Accuracy: {score:.2f}")
    if rows:
        print(
            f"windows={len(rows)} step={rows[0].get('step_start', '?')}.."
            f"{rows[-1].get('step_end', '?')} malformed_lines={len(errors)}"
        )
    else:
        print(f"windows=0 malformed_lines={len(errors)}")
    if errors:
        print("parse_errors: " + "; ".join(errors[:5]))

    if not summaries:
        print("没有可汇总的标量指标。\n")
        return

    print(
        f"{'metric':<38} {'mean±std':>22} {'firstW':>11} {'lastW':>11} "
        f"{'latest':>11} {'W[min,max]':>23} {'slope/100':>11}"
    )
    for summary in summaries:
        mean_std = f"{format_number(summary.global_mean)}±{format_number(summary.global_std)}"
        window_range = (
            f"[{format_number(summary.window_mean_min)},"
            f"{format_number(summary.window_mean_max)}]"
        )
        print(
            f"{summary.name:<38} {mean_std:>22} "
            f"{format_number(summary.first_window_mean):>11} "
            f"{format_number(summary.last_window_mean):>11} "
            f"{format_number(summary.latest):>11} "
            f"{window_range:>23} "
            f"{format_number(summary.slope_per_100_steps):>11}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解析并汇总 SLAKE 的 mmrl_diagnostics.jsonl 窗口指标。"
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="diagnostics 文件、单个实验目录或输出根目录；默认扫描 slake/outputs/mmrl。",
    )
    parser.add_argument(
        "--metric",
        help="只显示名称匹配该正则表达式的指标。",
    )
    args = parser.parse_args()

    try:
        metric_filter = re.compile(args.metric) if args.metric else None
    except re.error as exc:
        parser.error(f"invalid --metric regex: {exc}")

    paths = discover_paths(args.paths)
    if not paths:
        roots = ", ".join(str(path) for path in (args.paths or [DEFAULT_ROOT]))
        print(f"没有找到 {DIAGNOSTICS_RELATIVE_PATH.as_posix()}：{roots}")
        return

    print(f"找到 {len(paths)} 个 diagnostics（按修改时间从旧到新）\n")
    for path in paths:
        rows, errors = load_rows(path)
        print_summary(path, rows, errors, metric_filter)


if __name__ == "__main__":
    main()
