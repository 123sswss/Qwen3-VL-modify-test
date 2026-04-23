# logger.py
import os
import math
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
        smooth_window=30,
        ema_alpha=0.15,
        scatter_stride=3,
        save_debug_figure=False,
    ):
        self.save_dir = save_dir
        self.stage_name = stage_name
        self.smooth_window = int(max(3, smooth_window))
        self.ema_alpha = float(ema_alpha)
        self.scatter_stride = int(max(1, scatter_stride))
        self.save_debug_figure = bool(save_debug_figure)
        self.records = []
        os.makedirs(self.save_dir, exist_ok=True)

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

    def _plot_paper_figure(self, df):
        self._setup_style()
        x = df["step"].values

        # 配色统一
        C = {
            "total": "#222222",
            "ce": "#1f77b4",
            "alpha": "#d62728",
            "tax": "#9467bd",
            "kg": "#2ca02c",
            "ke": "#ff7f0e",
            "temp": "#17becf",
            "lr": "#8c564b",
            "lambda_g": "#2ca02c",
            "lambda_e": "#ff7f0e",
            "std_alpha": "#e377c2",
            "std_label": "#7f7f7f",
            "cls": "#1f77b4",
            "gate": "#d62728",
        }

        is_stage4 = (
            ("k_general_mean" in df.columns and df["k_general_mean"].notna().any()) or
            ("tax_loss" in df.columns and df["tax_loss"].notna().any()) or
            ("dynamic_k_lambda_general" in df.columns and df["dynamic_k_lambda_general"].notna().any())
        )

        # Stage1/2/3: 1x3；Stage4: 2x2
        if is_stage4:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
            axes = axes.flatten()

        # ---------------- Panel 1: Loss ----------------
        ax = axes[0]
        loss_candidates = [
            ("total_loss", "Total", C["total"]),
            ("ce_loss", "CE", C["ce"]),
            ("alpha_guide_loss", "Alpha", C["alpha"]),
            ("cls_loss", "Cls", C["cls"]),
            ("gate_loss", "Gate", C["gate"]),
            ("tax_loss", "Tax", C["tax"]),
            ("k_general_loss", "K-General", C["kg"]),
            ("k_expert_loss", "K-Expert", C["ke"]),
        ]
        plotted = 0
        for k, label, color in loss_candidates:
            if k in df.columns and df[k].notna().any():
                # 避免 stage4 的 loss panel 太花：只保留最核心几个
                if is_stage4 and k in ("k_general_loss", "k_expert_loss") and "tax_loss" in df.columns:
                    continue
                self._plot_series(ax, x, df[k], label, color)
                plotted += 1
        ax.set_title("Loss Dynamics")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25, linestyle="--")
        if plotted > 0:
            ax.legend(frameon=False, ncol=min(3, plotted))

        # ---------------- Panel 2 ----------------
        ax = axes[1]
        if is_stage4:
            # K mean
            if "k_general_mean" in df.columns and df["k_general_mean"].notna().any():
                self._plot_series(ax, x, df["k_general_mean"], "General K", C["kg"])
            if "k_expert_mean" in df.columns and df["k_expert_mean"].notna().any():
                self._plot_series(ax, x, df["k_expert_mean"], "Expert K", C["ke"])
            ax.set_title("K Statistics")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("K")
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend(frameon=False)
        else:
            # alpha std
            plotted = 0
            if "alpha_std" in df.columns and df["alpha_std"].notna().any():
                self._plot_series(ax, x, df["alpha_std"], "Alpha Std", C["std_alpha"])
                plotted += 1
            if "label_alpha_std" in df.columns and df["label_alpha_std"].notna().any():
                self._plot_series(ax, x, df["label_alpha_std"], "Label Std", C["std_label"])
                plotted += 1
            ax.set_title("Alpha Dispersion")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Std")
            ax.grid(alpha=0.25, linestyle="--")
            if plotted > 0:
                ax.legend(frameon=False)

        # ---------------- Panel 3 ----------------
        ax = axes[2]
        if is_stage4:
            plotted = 0
            if "dynamic_k_lambda_general" in df.columns and df["dynamic_k_lambda_general"].notna().any():
                self._plot_series(ax, x, df["dynamic_k_lambda_general"], "Lambda General", C["lambda_g"])
                plotted += 1
            if "dynamic_k_lambda_expert" in df.columns and df["dynamic_k_lambda_expert"].notna().any():
                self._plot_series(ax, x, df["dynamic_k_lambda_expert"], "Lambda Expert", C["lambda_e"])
                plotted += 1
            ax.set_title("Dynamic K Lambda")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Lambda")
            ax.grid(alpha=0.25, linestyle="--")
            if plotted > 0:
                ax.legend(frameon=False)
        else:
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

        # ---------------- Panel 4 (only Stage4) ----------------
        if is_stage4:
            ax = axes[3]
            plotted = 0
            if "temperature" in df.columns and df["temperature"].notna().any():
                self._plot_series(ax, x, df["temperature"], "Temperature", C["temp"])
                plotted += 1
            if "tax_weight" in df.columns and df["tax_weight"].notna().any():
                self._plot_series(ax, x, df["tax_weight"], "Tax Weight", C["tax"])
                plotted += 1
            if "learning_rate" in df.columns and df["learning_rate"].notna().any():
                self._plot_series(ax, x, df["learning_rate"], "Learning Rate", C["lr"])
                plotted += 1
            ax.set_title("Schedule & Regularization")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.25, linestyle="--")
            if plotted > 0:
                ax.legend(frameon=False)

        fig.suptitle(f"Training Metrics - {self.stage_name}", fontsize=15, y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        png_path = os.path.join(self.save_dir, f"{self.stage_name}_paper.png")
        pdf_path = os.path.join(self.save_dir, f"{self.stage_name}_paper.pdf")
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

        csv_path = os.path.join(self.save_dir, f"{self.stage_name}_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[MetricLogger] Saved CSV: {csv_path}")

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
        if "k_general_loss" in row:
            print(f"  ├─ K Loss (General):      {_fmt(row.get('k_general_loss')):>10}")
        if "k_expert_loss" in row:
            print(f"  ├─ K Loss (Expert):       {_fmt(row.get('k_expert_loss')):>10}")
        if "tax_loss" in row:
            print(f"  ├─ Tax Loss:              {_fmt(row.get('tax_loss')):>10}")

        # K 统计
        has_k = ("k_general_mean" in row) or ("k_expert_mean" in row)
        if has_k:
            print(f"[K Statistics]")
            if "k_general_mean" in row:
                print(f"  ├─ General Mean K:        {_fmt(row.get('k_general_mean'), nd=3):>10}")
            if "k_expert_mean" in row:
                print(f"  ├─ Expert Mean K:         {_fmt(row.get('k_expert_mean'), nd=3):>10}")
            if "dynamic_k_lambda_general" in row:
                print(f"  ├─ Lambda General:        {_fmt(row.get('dynamic_k_lambda_general'), nd=4):>10}")
            if "dynamic_k_lambda_expert" in row:
                print(f"  └─ Lambda Expert:         {_fmt(row.get('dynamic_k_lambda_expert'), nd=4):>10}")

        # Alpha 统计
        has_alpha = ("alpha_std" in row) or ("label_alpha_std" in row)
        if has_alpha:
            print(f"[Alpha Statistics]")
            if "alpha_std" in row:
                print(f"  ├─ Alpha Std:             {_fmt(row.get('alpha_std'), nd=4):>10}")
            if "label_alpha_std" in row:
                print(f"  └─ Label Alpha Std:       {_fmt(row.get('label_alpha_std'), nd=4):>10}")

        # 调度
        print(f"[Schedule]")
        if "temperature" in row:
            print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), nd=4):>10}")
        if "tax_weight" in row:
            print(f"  ├─ Tax Weight:            {_fmt(row.get('tax_weight'), nd=4):>10}")
        if "learning_rate" in row:
            print(f"  └─ Learning Rate:         {_fmt(row.get('learning_rate'), nd=8):>10}")
        print("=" * 72 + "\n")

        if "k_dataset_loss" in row:
            print(f"  ├─ K Loss (Dataset):      {_fmt(row.get('k_dataset_loss')):>10}")
        if "slot_collapse_loss" in row:
            print(f"  ├─ Slot Collapse Loss:    {_fmt(row.get('slot_collapse_loss')):>10}")

            if "k_mean_report" in row:
                print(f"  ├─ Report Mean K:         {_fmt(row.get('k_mean_report'), nd=3):>10}")
            if "k_mean_vqa" in row:
                print(f"  ├─ VQA Mean K:            {_fmt(row.get('k_mean_vqa'), nd=3):>10}")
            if "k_mean_test" in row:
                print(f"  ├─ Test Mean K:           {_fmt(row.get('k_mean_test'), nd=3):>10}")

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