# logger.py
import os
import math
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import TrainerCallback


def _to_float(x):
    try:
        if x is None:
            return np.nan
        if hasattr(x, "detach"):
            x = x.detach().float().item()
        x = float(x)
        if math.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


def _fmt(x, nd=4, default="nan"):
    x = _to_float(x)
    if np.isnan(x):
        return default
    return f"{x:.{nd}f}"


class StageMetricLogger:
    """
    更适合论文/汇报的轻量日志器：
    - 保存逐 step 指标 CSV
    - 输出单张 paper-style 主图
    - 可选输出更完整的 debug 图（默认不开）
    """
    def __init__(
        self,
        save_dir,
        stage_name,
        experiment_name=None,
        experiment_config=None,
        smooth_window=30,
        ema_alpha=0.15,
        scatter_stride=3,
        save_debug_figure=False,
    ):
        self.save_dir = save_dir
        self.stage_name = stage_name
        self.experiment_name = experiment_name
        self.experiment_config = experiment_config or {}
        self.smooth_window = int(max(3, smooth_window))
        self.ema_alpha = float(ema_alpha)
        self.scatter_stride = int(max(1, scatter_stride))
        self.save_debug_figure = bool(save_debug_figure)
        self.records = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _file_prefix(self):
        if self.experiment_name:
            return f"{self.experiment_name}_{self.stage_name}"
        return self.stage_name

    def _metadata(self):
        return {
            "experiment_name": self.experiment_name,
            "stage_name": self.stage_name,
            "save_dir": self.save_dir,
            "smooth_window": self.smooth_window,
            "ema_alpha": self.ema_alpha,
            "scatter_stride": self.scatter_stride,
            "save_debug_figure": self.save_debug_figure,
            "record_count": len(self.records),
            "generated_at": datetime.now().isoformat(),
            "experiment_config": self.experiment_config,
        }

    def log(self, step, **metrics):
        row = {"step": int(step)}
        for k, v in metrics.items():
            row[k] = _to_float(v)
        self.records.append(row)

    def _prepare_df(self):
        if len(self.records) == 0:
            return None
        df = pd.DataFrame(self.records).sort_values("step").reset_index(drop=True)
        for c in df.columns:
            if c != "step":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _smooth(self, s: pd.Series):
        return s.ewm(alpha=self.ema_alpha, adjust=False).mean()

    def _setup_style(self):
        plt.rcParams.update({
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def _plot_series(self, ax, x, y, label, color, raw_alpha=0.18, lw=2.2, draw_raw=True):
        y = pd.to_numeric(y, errors="coerce")
        if y.notna().sum() == 0:
            return
        ys = self._smooth(y)
        if draw_raw:
            ax.plot(x, y, color=color, alpha=raw_alpha, linewidth=1.0)
        ax.plot(x, ys, color=color, linewidth=lw, label=label)

    def _has_valid_data(self, df, column):
        return column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()

    def _plot_paper_figure(self, df):
        self._setup_style()
        x = df["step"].values

        C = {
            "total": "#222222",
            "ce": "#1f77b4",
            "alpha": "#d62728",
            "temp": "#17becf",
            "lr": "#8c564b",
            "std_alpha": "#e377c2",
            "cls": "#1f77b4",
            "gate": "#d62728",
        }
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.flatten()

        ax = axes[0]
        loss_candidates = [
            ("total_loss", "Total", C["total"]),
            ("ce_loss", "CE", C["ce"]),
            ("alpha_guide_loss", "Alpha", C["alpha"]),
            ("mmrl_relation_loss_scaled", "Relation", "#72b7b2"),
            ("cls_loss", "Cls", C["cls"]),
            ("gate_loss", "Gate", C["gate"]),
        ]
        plotted = 0
        for k, label, color in loss_candidates:
            if k in df.columns and df[k].notna().any():
                self._plot_series(ax, x, df[k], label, color)
                plotted += 1
        ax.set_title("Loss Dynamics")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25, linestyle="--")
        if plotted > 0:
            ax.legend(frameon=False, ncol=min(3, plotted))

        ax = axes[1]
        plotted = 0
        if "alpha_mae" in df.columns and df["alpha_mae"].notna().any():
            self._plot_series(ax, x, df["alpha_mae"], "Alpha MAE", C["std_alpha"])
            plotted += 1
        if self._has_valid_data(df, "G_mean"):
            self._plot_series(ax, x, df["G_mean"], "Gate Mean", C["alpha"])
            plotted += 1
        if self._has_valid_data(df, "alpha_prob_mean"):
            self._plot_series(ax, x, df["alpha_prob_mean"], "Alpha Mean", "#f58518")
            plotted += 1
        ax.set_title("Gate Health")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25, linestyle="--")
        if plotted > 0:
            ax.legend(frameon=False)

        ax = axes[2]
        plotted = 0
        if self._has_valid_data(df, "final_delta_norm_mean"):
            self._plot_series(ax, x, df["final_delta_norm_mean"], "Final Δ Norm", "#f58518")
            plotted += 1
        if self._has_valid_data(df, "delta_to_org_ratio"):
            self._plot_series(ax, x, df["delta_to_org_ratio"], "Δ / Org", "#54a24b")
            plotted += 1
        if self._has_valid_data(df, "delta_pool_common_mode_ratio"):
            self._plot_series(ax, x, df["delta_pool_common_mode_ratio"], "Common-Mode", "#72b7b2")
            plotted += 1
        if self._has_valid_data(df, "delta_pool_specificity_ratio"):
            self._plot_series(ax, x, df["delta_pool_specificity_ratio"], "Specificity", "#eeca3b")
            plotted += 1
        ax.set_title("Residual Health")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25, linestyle="--")
        if plotted > 0:
            ax.legend(frameon=False)

        ax = axes[3]
        plotted = 0
        if "temperature" in df.columns and df["temperature"].notna().any():
            self._plot_series(ax, x, df["temperature"], "Temperature", C["temp"])
            plotted += 1
        if "learning_rate" in df.columns and df["learning_rate"].notna().any():
            self._plot_series(ax, x, df["learning_rate"], "Learning Rate", C["lr"])
            plotted += 1
        ax.set_title("Schedule")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25, linestyle="--")
        if plotted > 0:
            ax.legend(frameon=False)

        fig_title = self.stage_name if not self.experiment_name else f"{self.experiment_name} / {self.stage_name}"
        fig.suptitle(f"Training Metrics - {fig_title}", fontsize=15, y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        prefix = self._file_prefix()
        png_path = os.path.join(self.save_dir, f"{prefix}_paper.png")
        pdf_path = os.path.join(self.save_dir, f"{prefix}_paper.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"[MetricLogger] Saved FIG: {png_path}")
        print(f"[MetricLogger] Saved FIG: {pdf_path}")

    def finalize(self):
        df = self._prepare_df()
        if df is None:
            print(f"[MetricLogger] No records for {self.stage_name}, skip.")
            return

        prefix = self._file_prefix()
        csv_path = os.path.join(self.save_dir, f"{prefix}_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[MetricLogger] Saved CSV: {csv_path}")

        metadata_path = os.path.join(self.save_dir, f"{prefix}_meta.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata(), f, ensure_ascii=False, indent=2)
        print(f"[MetricLogger] Saved META: {metadata_path}")

        self._plot_paper_figure(df)


class TrainerMetricsCallback(TrainerCallback):
    """
    每个 optimizer step：
    1) 从 model._last_metrics 拉取指标写入 logger
    2) 每隔 print_every 步打印一次合集日志
    """
    def __init__(self, metric_logger: StageMetricLogger, print_every=50, stage_name="stage"):
        self.metric_logger = metric_logger
        self.last_logged_step = -1
        self.print_every = int(max(1, print_every))
        self.stage_name = stage_name

    def _print_compact_summary(self, step, row):
        print("\n" + "=" * 72)
        print(f"[{self.stage_name} | Training Step {step}] Loss Breakdown")

        if "total_loss" in row:
            print(f"  ├─ Total Loss:            {_fmt(row.get('total_loss')):>10}")
        if "ce_loss" in row:
            print(f"  ├─ CE Loss:               {_fmt(row.get('ce_loss')):>10}")
        if "alpha_guide_loss" in row:
            print(f"  ├─ Alpha Guide Loss:      {_fmt(row.get('alpha_guide_loss')):>10}")
        if "cls_loss" in row:
            print(f"  ├─ Cls Loss:              {_fmt(row.get('cls_loss')):>10}")
        if "gate_loss" in row:
            print(f"  ├─ Gate Loss:             {_fmt(row.get('gate_loss')):>10}")
        if "mmrl_relation_loss_scaled" in row:
            print(f"  ├─ Relation Loss:         {_fmt(row.get('mmrl_relation_loss_scaled')):>10}")

        # Alpha 统计
        has_alpha = ("alpha_mae" in row) or ("alpha_std" in row) or ("label_alpha_std" in row)
        if has_alpha:
            print(f"[Alpha Statistics]")
            if "alpha_mae" in row:
                print(f"  ├─ Alpha MAE:             {_fmt(row.get('alpha_mae'), nd=4):>10}")
            if "alpha_std" in row:
                print(f"  ├─ Alpha Std:             {_fmt(row.get('alpha_std'), nd=4):>10}")
            if "label_alpha_std" in row:
                print(f"  └─ Label Alpha Std:       {_fmt(row.get('label_alpha_std'), nd=4):>10}")

        has_rep_grad = (
            ("shared_rep_grad_norm" in row) or
            ("shared_rep_grad_ema" in row) or
            ("shared_rep_grad_ratio" in row) or
            ("shared_rep_grad_spike_flag" in row)
        )
        if has_rep_grad:
            print(f"[Shared Rep Grad]")
            if "shared_rep_grad_norm" in row:
                print(f"  ├─ Grad Norm:            {_fmt(row.get('shared_rep_grad_norm'), nd=4):>10}")
            if "shared_rep_grad_ema" in row:
                print(f"  ├─ Grad EMA:             {_fmt(row.get('shared_rep_grad_ema'), nd=4):>10}")
            if "shared_rep_grad_ratio" in row:
                print(f"  ├─ Grad Ratio:           {_fmt(row.get('shared_rep_grad_ratio'), nd=4):>10}")
            if "shared_rep_grad_spike_flag" in row:
                print(f"  ├─ Spike Flag:           {_fmt(row.get('shared_rep_grad_spike_flag'), nd=0):>10}")
            if "shared_rep_grad_spike_count" in row:
                print(f"  └─ Spike Count:          {_fmt(row.get('shared_rep_grad_spike_count'), nd=0):>10}")

        has_visual_residual = (
            ("final_delta_norm_mean" in row) or
            ("delta_to_org_ratio" in row) or
            ("delta_pool_common_mode_ratio" in row) or
            ("delta_pool_specificity_ratio" in row)
        )
        if has_visual_residual:
            print(f"[Vision Residual]")
            if "final_delta_norm_mean" in row:
                print(f"  ├─ Final Δ Norm:        {_fmt(row.get('final_delta_norm_mean'), nd=4):>10}")
            if "delta_to_org_ratio" in row:
                print(f"  ├─ Δ / Org Ratio:       {_fmt(row.get('delta_to_org_ratio'), nd=4):>10}")
            if "delta_pool_common_mode_ratio" in row:
                print(f"  ├─ Common-Mode Ratio:   {_fmt(row.get('delta_pool_common_mode_ratio'), nd=4):>10}")
            if "delta_pool_specificity_ratio" in row:
                print(f"  └─ Specificity Ratio:   {_fmt(row.get('delta_pool_specificity_ratio'), nd=4):>10}")

        # 调度
        print(f"[Schedule]")
        if "temperature" in row:
            print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), nd=4):>10}")
        if "learning_rate" in row:
            print(f"  └─ Learning Rate:         {_fmt(row.get('learning_rate'), nd=8):>10}")
        print("=" * 72 + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step == self.last_logged_step:
            return

        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
        if model is None:
            return

        m = getattr(model, "_last_metrics", None)
        if isinstance(m, dict):
            row = dict(m)
            if optimizer is not None and len(optimizer.param_groups) > 0:
                row["learning_rate"] = optimizer.param_groups[0].get("lr", np.nan)

            self.metric_logger.log(step=step, **row)
            self.last_logged_step = step

            if step % self.print_every == 0:
                self._print_compact_summary(step, row)

    def on_train_end(self, args, state, control, **kwargs):
        self.metric_logger.finalize()
