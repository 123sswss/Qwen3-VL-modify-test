#!/usr/bin/env python3
"""Generate the final QDPT marathon convergence figure without plot dependencies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ROWS = (
    (3, 56.4467, 89.9840, 23.0057, 9.644586),
    (4, 57.2456, 89.6000, 24.9840, 9.009147),
    (5, 58.6675, 90.5920, 26.8347, 8.660347),
    (6, 58.7794, 91.2320, 26.4199, 8.049925),
    (7, 57.4852, 90.3680, 24.6969, 8.733946),
    (8, 57.1817, 90.7520, 23.7077, 7.163892),
    (9, 57.3894, 90.4000, 24.4735, 5.893644),
    (10, 57.1657, 90.7200, 23.7077, 5.396529),
)
BASELINE_OVERALL = 59.5622


def load_rows(path: Path | None) -> Sequence[tuple[int, float, float, float, float]]:
    if path is None:
        return DEFAULT_ROWS
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    rows = [
        (
            int(row["epoch"]),
            float(row["overall"]),
            float(row["yes_no"]),
            float(row["free_form"]),
            float(row["latest_train_loss"]),
        )
        for row in records
        if row.get("status") == "complete"
    ]
    if [row[0] for row in rows] != list(range(3, 11)):
        raise ValueError("Expected complete marathon rows for epochs 3 through 10")
    return rows


def escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(
    title: str,
    subtitle: str,
    epochs: Sequence[int],
    values: Sequence[float],
    x: float,
    y: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    y_ticks: Iterable[float],
    color: str,
    baseline: float | None = None,
) -> str:
    left, right, top, bottom = 72, 28, 78, 52
    plot_x = x + left
    plot_y = y + top
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(epoch: int) -> float:
        return plot_x + (epoch - min(epochs)) * plot_w / (max(epochs) - min(epochs))

    def sy(value: float) -> float:
        return plot_y + (y_max - value) * plot_h / (y_max - y_min)

    parts = [
        f'<g class="panel"><rect x="{x}" y="{y}" width="{width}" height="{height}" rx="20"/>',
        f'<text class="panel-title" x="{x + 28}" y="{y + 36}">{escape(title)}</text>',
        f'<text class="panel-subtitle" x="{x + 28}" y="{y + 59}">{escape(subtitle)}</text>',
    ]
    for tick in y_ticks:
        tick_y = sy(float(tick))
        parts.extend(
            (
                f'<line class="grid" x1="{plot_x}" y1="{tick_y:.2f}" x2="{plot_x + plot_w}" y2="{tick_y:.2f}"/>',
                f'<text class="tick" x="{plot_x - 13}" y="{tick_y + 5:.2f}" text-anchor="end">{tick:g}</text>',
            )
        )
    for epoch in epochs:
        tick_x = sx(epoch)
        parts.append(
            f'<text class="tick" x="{tick_x:.2f}" y="{plot_y + plot_h + 29}" text-anchor="middle">{epoch}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{plot_x + plot_w / 2:.2f}" y="{plot_y + plot_h + 48}" text-anchor="middle">Epoch</text>'
    )
    if baseline is not None:
        baseline_y = sy(baseline)
        parts.extend(
            (
                f'<line class="baseline" x1="{plot_x}" y1="{baseline_y:.2f}" x2="{plot_x + plot_w}" y2="{baseline_y:.2f}"/>',
                f'<text class="baseline-label" x="{plot_x + plot_w - 4}" y="{baseline_y - 9:.2f}" text-anchor="end">Original 3-epoch: {baseline:.2f}</text>',
            )
        )
    points = " ".join(
        f"{sx(epoch):.2f},{sy(value):.2f}"
        for epoch, value in zip(epochs, values)
    )
    parts.append(f'<polyline class="series" stroke="{color}" points="{points}"/>')
    maximum = max(values)
    for epoch, value in zip(epochs, values):
        point_x, point_y = sx(epoch), sy(value)
        point_class = "peak-point" if value == maximum else "point"
        parts.append(
            f'<circle class="{point_class}" cx="{point_x:.2f}" cy="{point_y:.2f}" r="{6 if value == maximum else 4.5}" fill="{color}"/>'
        )
        parts.append(
            f'<text class="value" x="{point_x:.2f}" y="{point_y - 12:.2f}" text-anchor="middle">{value:.2f}</text>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def build_svg(rows: Sequence[tuple[int, float, float, float, float]]) -> str:
    epochs = [row[0] for row in rows]
    overall = [row[1] for row in rows]
    yes_no = [row[2] for row in rows]
    free_form = [row[3] for row in rows]
    losses = [row[4] for row in rows]
    panels = (
        panel("Overall accuracy", "Full PathVQA Validation", epochs, overall, 55, 145, 630, 310, 56, 60.2, range(56, 61), "#167D77", BASELINE_OVERALL),
        panel("Yes / No accuracy", "Binary-answer behavior", epochs, yes_no, 715, 145, 630, 310, 89, 92, (89, 90, 91, 92), "#D56A3A"),
        panel("Free-form accuracy", "Open-answer behavior", epochs, free_form, 55, 485, 630, 310, 22.5, 29, (23, 25, 27, 29), "#3366A3"),
        panel("Latest logged train loss", "Training fit continues after validation peaks", epochs, losses, 715, 485, 630, 310, 5, 10, (5, 6, 7, 8, 9, 10), "#8B5E3C"),
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<defs>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="5" stdDeviation="9" flood-color="#173B3A" flood-opacity="0.08"/></filter>
</defs>
<style>
  .background {{ fill: #F4F1E8; }}
  .panel rect {{ fill: #FFFEFA; stroke: #D8D3C6; stroke-width: 1.2; filter: url(#shadow); }}
  .title {{ font: 700 34px Georgia, serif; fill: #183C3A; letter-spacing: -0.4px; }}
  .kicker {{ font: 700 13px "Trebuchet MS", sans-serif; fill: #B6532D; letter-spacing: 2.2px; }}
  .note {{ font: 15px "Trebuchet MS", sans-serif; fill: #5E655F; }}
  .panel-title {{ font: 700 21px Georgia, serif; fill: #243D3B; }}
  .panel-subtitle {{ font: 13px "Trebuchet MS", sans-serif; fill: #7A7F78; }}
  .tick {{ font: 12px "Trebuchet MS", sans-serif; fill: #737A74; }}
  .axis-label {{ font: 12px "Trebuchet MS", sans-serif; fill: #515A55; }}
  .grid {{ stroke: #E5E1D8; stroke-width: 1; }}
  .series {{ fill: none; stroke-width: 3.4; stroke-linecap: round; stroke-linejoin: round; }}
  .point {{ stroke: #FFFEFA; stroke-width: 2; }}
  .peak-point {{ stroke: #F1B24A; stroke-width: 4; }}
  .value {{ font: 700 11px "Trebuchet MS", sans-serif; fill: #3F4B46; }}
  .baseline {{ stroke: #C64B3C; stroke-width: 2; stroke-dasharray: 8 7; }}
  .baseline-label {{ font: 700 12px "Trebuchet MS", sans-serif; fill: #B23E31; }}
  .footer {{ font: 13px "Trebuchet MS", sans-serif; fill: #68706A; }}
</style>
<rect class="background" width="1400" height="900"/>
<text class="kicker" x="55" y="48">QDPT CONVERGENCE DIAGNOSTIC</text>
<text class="title" x="55" y="91">Validation peaks early while training fit keeps improving</text>
<text class="note" x="55" y="120">QDPT-D768 · seed 44 · 10-epoch linear schedule · complete Validation at epochs 3–10</text>
{''.join(panels)}
<line x1="55" y1="835" x2="1345" y2="835" stroke="#D1CCC0"/>
<text class="footer" x="55" y="866">Peak marathon Overall: 58.78 at epoch 6. The original 3-epoch schedule remains higher at 59.56.</text>
<text class="footer" x="1345" y="866" text-anchor="end">Different scheduler horizons; baseline shown for reference, not as an earlier point on the same curve.</text>
</svg>'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/qdpt_d768_marathon_curve.svg"),
    )
    args = parser.parse_args()
    rows = load_rows(args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(rows), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
