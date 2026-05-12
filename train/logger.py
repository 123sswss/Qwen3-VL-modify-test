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

    def _resolve_text_gate_mode(self, df=None):
        experiment_cfg = self.experiment_config.get("experiment_cfg", {}) or {}
        mode = experiment_cfg.get("text_gate_selection_mode", None)
        if isinstance(mode, str) and mode.strip():
            return mode.strip().lower()

        if df is not None and self._has_valid_data(df, "selection_mode_id"):
            val = pd.to_numeric(df["selection_mode_id"], errors="coerce").dropna()
            if len(val) > 0:
                mapping = {
                    0: "dynamic_prefix",
                    1: "fixed_prefix",
                    2: "fixed_group20",
                    3: "group_top4",
                    4: "token_top20",
                }
                return mapping.get(int(round(float(val.iloc[-1]))), None)
        return None

    def _group_usage_columns(self, df):
        cols = []
        for c in df.columns:
            if c.startswith("group_usage_"):
                suffix = c[len("group_usage_"):]
                if suffix.isdigit():
                    cols.append(c)
        cols.sort(key=lambda c: int(c.split("_")[-1]))
        return cols

    def _plot_group_usage_bar(self, ax, df):
        usage_cols = self._group_usage_columns(df)
        if not usage_cols:
            ax.set_visible(False)
            return

        usage = df[usage_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        usage = usage.fillna(0.0)
        total = float(usage.sum())
        if total > 0:
            usage = usage / total

        group_ids = [int(c.split("_")[-1]) for c in usage_cols]
        values = usage.values.astype(float)
        bars = ax.bar(group_ids, values, color="#4c78a8", alpha=0.9)
        ax.set_title("Global Group Usage")
        ax.set_xlabel("Group ID")
        ax.set_ylabel("Normalized Usage")
        ax.set_xticks(group_ids)
        ax.set_ylim(0.0, max(0.2, float(values.max()) * 1.15 if len(values) > 0 else 0.2))
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    def _plot_paper_figure(self, df):
        self._setup_style()
        x = df["step"].values
        stage_id = self.experiment_config.get("stage_id")
        gate_mode = self._resolve_text_gate_mode(df)

        # 配色统一
        C = {
            "total": "#222222",
            "ce": "#1f77b4",
            "alpha": "#d62728",
            "budget": "#8c6bb1",
            "tax": "#9467bd",
            "slot_dataset": "#bcbd22",
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

        if stage_id is not None:
            is_stage4 = int(stage_id) == 4
        else:
            is_stage4 = (
                ("k_general_mean" in df.columns and df["k_general_mean"].notna().any()) or
                ("k_expert_mean" in df.columns and df["k_expert_mean"].notna().any()) or
                ("k_general_loss" in df.columns and df["k_general_loss"].notna().any()) or
                ("k_expert_loss" in df.columns and df["k_expert_loss"].notna().any()) or
                ("tax_loss" in df.columns and df["tax_loss"].notna().any())
            )
        has_group_usage = len(self._group_usage_columns(df)) > 0

        # dynamic_prefix 等旧逻辑保持不变；仅对 group_top4 增量扩展一个 group usage 面板
        if is_stage4 and has_group_usage:
            fig, axes = plt.subplots(2, 3, figsize=(18, 9))
            axes = axes.flatten()
        elif is_stage4:
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
            ("anti_collapse_loss", "Anti-Collapse", "#bc5090"),
            ("cls_loss", "Cls", C["cls"]),
            ("gate_loss", "Gate", C["gate"]),
            ("tax_loss", "Tax", C["tax"]),
            ("slot_dataset_loss", "Slot-Group", C["slot_dataset"]),
            ("k_general_loss", "K-General", C["kg"]),
            ("k_expert_loss", "K-Expert", C["ke"]),
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

        # ---------------- Panel 2 ----------------
        ax = axes[1]
        if is_stage4:
            plotted = 0
            if gate_mode == "dynamic_prefix":
                if self._has_valid_data(df, "active_token_count_mean"):
                    self._plot_series(ax, x, df["active_token_count_mean"], "Active Tokens", C["kg"])
                    plotted += 1
                if self._has_valid_data(df, "k_budget_mean"):
                    self._plot_series(ax, x, df["k_budget_mean"], "K Budget", C["ke"])
                    plotted += 1
                if self._has_valid_data(df, "raw_budget_mean"):
                    self._plot_series(ax, x, df["raw_budget_mean"], "Raw Budget", C["tax"])
                    plotted += 1
                if plotted == 0:
                    if self._has_valid_data(df, "k_general_mean"):
                        self._plot_series(ax, x, df["k_general_mean"], "General K", C["kg"])
                        plotted += 1
                    if self._has_valid_data(df, "k_expert_mean"):
                        self._plot_series(ax, x, df["k_expert_mean"], "Expert K", C["ke"])
                        plotted += 1
                ax.set_title("K Statistics")
                ax.set_ylabel("K / Budget")
            elif gate_mode == "group_top4":
                if self._has_valid_data(df, "group_count_loss"):
                    self._plot_series(ax, x, df["group_count_loss"], "Group Count Loss", C["tax"])
                    plotted += 1
                if self._has_valid_data(df, "group_usage_ema_entropy"):
                    self._plot_series(ax, x, df["group_usage_ema_entropy"], "Usage Entropy", C["kg"])
                    plotted += 1
                if self._has_valid_data(df, "group_usage_dead_ratio"):
                    self._plot_series(ax, x, df["group_usage_dead_ratio"], "Dead Group Ratio", C["ke"])
                    plotted += 1
                ax.set_title("Group Statistics")
                ax.set_ylabel("Value")
            elif gate_mode == "token_top20":
                if self._has_valid_data(df, "active_token_count_mean"):
                    self._plot_series(ax, x, df["active_token_count_mean"], "Active Tokens", C["kg"])
                    plotted += 1
                if self._has_valid_data(df, "token_count_loss"):
                    self._plot_series(ax, x, df["token_count_loss"], "Token Count Loss", C["ke"])
                    plotted += 1
                if self._has_valid_data(df, "token_balance_loss"):
                    self._plot_series(ax, x, df["token_balance_loss"], "Token Balance Loss", C["tax"])
                    plotted += 1
                ax.set_title("Token Statistics")
                ax.set_ylabel("Value")
            else:
                if self._has_valid_data(df, "k_general_mean"):
                    self._plot_series(ax, x, df["k_general_mean"], "General K", C["kg"])
                    plotted += 1
                if self._has_valid_data(df, "k_expert_mean"):
                    self._plot_series(ax, x, df["k_expert_mean"], "Expert K", C["ke"])
                    plotted += 1
                ax.set_title("K Statistics")
                ax.set_ylabel("K")
            ax.set_xlabel("Global Step")
            ax.grid(alpha=0.25, linestyle="--")
            if plotted > 0:
                ax.legend(frameon=False)
        else:
            # alpha std
            plotted = 0
            if "alpha_mae" in df.columns and df["alpha_mae"].notna().any():
                self._plot_series(ax, x, df["alpha_mae"], "Alpha MAE", C["std_alpha"])
                plotted += 1
            elif "alpha_std" in df.columns and df["alpha_std"].notna().any():
                self._plot_series(ax, x, df["alpha_std"], "Alpha Std", C["std_alpha"])
                plotted += 1
            if "label_alpha_std" in df.columns and df["label_alpha_std"].notna().any():
                self._plot_series(ax, x, df["label_alpha_std"], "Label Std", C["std_label"])
                plotted += 1
            ax.set_title("Alpha Error")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.25, linestyle="--")
            if plotted > 0:
                ax.legend(frameon=False)

        # ---------------- Panel 3 ----------------
        ax = axes[2]
        if is_stage4:
            plotted = 0
            if gate_mode == "group_top4":
                if self._has_valid_data(df, "anti_collapse_weight"):
                    self._plot_series(ax, x, df["anti_collapse_weight"], "Anti-Collapse Weight", "#bc5090")
                    plotted += 1
                if self._has_valid_data(df, "group_usage_ema_max"):
                    self._plot_series(ax, x, df["group_usage_ema_max"], "EMA Usage Max", C["kg"])
                    plotted += 1
                if self._has_valid_data(df, "group_usage_ema_min"):
                    self._plot_series(ax, x, df["group_usage_ema_min"], "EMA Usage Min", C["ke"])
                    plotted += 1
                ax.set_title("Anti-Collapse")
                ax.set_ylabel("Value")
            elif gate_mode == "token_top20":
                if self._has_valid_data(df, "k_budget_mean"):
                    self._plot_series(ax, x, df["k_budget_mean"], "K Budget", C["kg"])
                    plotted += 1
                if self._has_valid_data(df, "batch_alpha_mean"):
                    self._plot_series(ax, x, df["batch_alpha_mean"], "Batch Alpha", C["alpha"])
                    plotted += 1
                if self._has_valid_data(df, "text_rel_mean"):
                    self._plot_series(ax, x, df["text_rel_mean"], "Text Relevance", C["temp"])
                    plotted += 1
                ax.set_title("Token Routing")
                ax.set_ylabel("Value")
            else:
                if self._has_valid_data(df, "dynamic_k_lambda_general"):
                    self._plot_series(ax, x, df["dynamic_k_lambda_general"], "Lambda General", C["lambda_g"])
                    plotted += 1
                if self._has_valid_data(df, "dynamic_k_lambda_expert"):
                    self._plot_series(ax, x, df["dynamic_k_lambda_expert"], "Lambda Expert", C["lambda_e"])
                    plotted += 1
                ax.set_title("Dynamic K Lambda")
                ax.set_ylabel("Lambda")
            ax.set_xlabel("Global Step")
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
            if "budget_weight" in df.columns and df["budget_weight"].notna().any():
                self._plot_series(ax, x, df["budget_weight"], "Budget Weight", C["budget"])
                plotted += 1
            if "tax_weight" in df.columns and df["tax_weight"].notna().any():
                self._plot_series(ax, x, df["tax_weight"], "Tax Weight", C["tax"])
                plotted += 1
            if "selector_explore" in df.columns and df["selector_explore"].notna().any():
                self._plot_series(ax, x, df["selector_explore"], "Explore Scale", C["temp"])
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

        if is_stage4 and has_group_usage:
            self._plot_group_usage_bar(axes[4], df)
            if len(axes) > 5:
                axes[5].axis("off")

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
        if "anti_collapse_loss" in row:
            print(f"  ├─ Anti-Collapse Loss:    {_fmt(row.get('anti_collapse_loss')):>10}")
        if "cls_loss" in row:
            print(f"  ├─ Cls Loss:              {_fmt(row.get('cls_loss')):>10}")
        if "gate_loss" in row:
            print(f"  ├─ Gate Loss:             {_fmt(row.get('gate_loss')):>10}")
        if "k_general_loss" in row:
            print(f"  ├─ K Loss (General):      {_fmt(row.get('k_general_loss')):>10}")
        if "k_expert_loss" in row:
            print(f"  ├─ K Loss (Expert):       {_fmt(row.get('k_expert_loss')):>10}")
        if "slot_dataset_loss" in row:
            print(f"  ├─ Slot-Group Loss:       {_fmt(row.get('slot_dataset_loss')):>10}")
        if "tax_loss" in row:
            print(f"  ├─ Tax Loss:              {_fmt(row.get('tax_loss')):>10}")
        if "group_usage_ema_max" in row or "group_usage_ema_min" in row or "group_usage_ema_entropy" in row:
            print(f"[Anti-Collapse]")
            if "group_usage_ema_max" in row:
                print(f"  ├─ EMA Usage Max:         {_fmt(row.get('group_usage_ema_max')):>10}")
            if "group_usage_ema_min" in row:
                print(f"  ├─ EMA Usage Min:         {_fmt(row.get('group_usage_ema_min')):>10}")
            if "group_usage_dead_ratio" in row:
                print(f"  ├─ Dead Group Ratio:      {_fmt(row.get('group_usage_dead_ratio')):>10}")
            if "group_usage_ema_entropy" in row:
                print(f"  └─ EMA Usage Entropy:     {_fmt(row.get('group_usage_ema_entropy')):>10}")

        # K 统计
        has_k = (
            ("k_selected_mean" in row) or
            ("span_gates_mean" in row) or
            ("k_general_mean" in row) or
            ("k_expert_mean" in row)
        )
        if has_k:
            print(f"[Span Selector Statistics]")
            if "k_selected_mean" in row:
                print(f"  ├─ Selected Mean K:       {_fmt(row.get('k_selected_mean'), nd=3):>10}")
            if "span_gates_mean" in row:
                print(f"  ├─ Spans Selected (avg):  {_fmt(row.get('span_gates_mean'), nd=3):>10}")
            if "span_budget_loss" in row:
                print(f"  ├─ Span Budget Loss:      {_fmt(row.get('span_budget_loss'), nd=4):>10}")
            if "k_general_mean" in row:
                print(f"  ├─ General Mean K:        {_fmt(row.get('k_general_mean'), nd=3):>10}")
            if "k_expert_mean" in row:
                print(f"  ├─ Expert Mean K:         {_fmt(row.get('k_expert_mean'), nd=3):>10}")
            if "k_mean_report" in row:
                print(f"  ├─ Report Mean K:         {_fmt(row.get('k_mean_report'), nd=3):>10}")
            if "k_mean_vqa" in row:
                print(f"  ├─ VQA Mean K:            {_fmt(row.get('k_mean_vqa'), nd=3):>10}")
            if "k_mean_test" in row:
                print(f"  ├─ Test Mean K:           {_fmt(row.get('k_mean_test'), nd=3):>10}")
            if "dynamic_k_lambda_general" in row:
                print(f"  ├─ Lambda General:        {_fmt(row.get('dynamic_k_lambda_general'), nd=4):>10}")
            if "dynamic_k_lambda_expert" in row:
                print(f"  └─ Lambda Expert:         {_fmt(row.get('dynamic_k_lambda_expert'), nd=4):>10}")
            if "alpha_open_rate" in row:
                print(f"  └─ Alpha Open Rate:       {_fmt(row.get('alpha_open_rate'), nd=4):>10}")

        has_slot_entropy = (
            ("slot_entropy_report" in row) or
            ("slot_entropy_vqa" in row) or
            ("slot_entropy_test" in row)
        )
        if has_slot_entropy:
            print(f"[Slot Entropy]")
            if "slot_entropy_report" in row:
                print(f"  ├─ Report Entropy:        {_fmt(row.get('slot_entropy_report'), nd=3):>10}")
            if "slot_entropy_vqa" in row:
                print(f"  ├─ VQA Entropy:           {_fmt(row.get('slot_entropy_vqa'), nd=3):>10}")
            if "slot_entropy_test" in row:
                print(f"  └─ Test Entropy:          {_fmt(row.get('slot_entropy_test'), nd=3):>10}")

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

        # 调度
        print(f"[Schedule]")
        if "temperature" in row:
            print(f"  ├─ Temperature:           {_fmt(row.get('temperature'), nd=4):>10}")
        if "selector_explore" in row:
            print(f"  ├─ Explore Scale:         {_fmt(row.get('selector_explore'), nd=4):>10}")
        if "budget_weight" in row:
            print(f"  ├─ Budget Weight:         {_fmt(row.get('budget_weight'), nd=4):>10}")
        if "tax_weight" in row:
            print(f"  ├─ Tax Weight:            {_fmt(row.get('tax_weight'), nd=4):>10}")
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