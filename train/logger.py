# metrics_logger.py
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


class StageMetricLogger:
    """
    Lightweight metric logger:
    - store step-wise metrics
    - save CSV
    - generate publication-style plots (raw scatter + smoothed curves)
    """
    def __init__(self, save_dir, stage_name, smooth_window=30, ema_alpha=0.15, scatter_stride=3):
        self.save_dir = save_dir
        self.stage_name = stage_name
        self.smooth_window = int(max(3, smooth_window))
        self.ema_alpha = float(ema_alpha)
        self.scatter_stride = int(max(1, scatter_stride))
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
        roll = s.rolling(window=self.smooth_window, min_periods=1).mean()
        ema = s.ewm(alpha=self.ema_alpha, adjust=False).mean()
        return roll, ema

    def finalize(self):
        df = self._prepare_df()
        if df is None:
            print(f"[MetricLogger] No records for {self.stage_name}, skip.")
            return

        csv_path = os.path.join(self.save_dir, f"{self.stage_name}_metrics.csv")
        df.to_csv(csv_path, index=False)

        # --- plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

        x = df["step"].values
        idx = np.arange(len(df))
        raw_idx = idx[::self.scatter_stride]

        # Panel 1: losses
        loss_keys = [
            "total_loss", "ce_loss", "alpha_guide_loss",
            "k_general_loss", "k_expert_loss", "tax_loss",
            "cls_loss", "gate_loss"
        ]
        for k in loss_keys:
            if k in df.columns and df[k].notna().any():
                y = df[k]
                r, e = self._smooth(y)
                ax1.scatter(x[raw_idx], y.iloc[raw_idx], s=9, alpha=0.25)
                ax1.plot(x, r, linewidth=1.8, label=f"{k} (rolling)")
                ax1.plot(x, e, linewidth=1.2, linestyle="--", alpha=0.9, label=f"{k} (ema)")
        ax1.set_title("Loss Dynamics")
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8, ncol=2)

        # Panel 2: alpha std metrics
        std_keys = ["alpha_std", "label_alpha_std"]
        for k in std_keys:
            if k in df.columns and df[k].notna().any():
                y = df[k]
                r, e = self._smooth(y)
                ax2.scatter(x[raw_idx], y.iloc[raw_idx], s=9, alpha=0.25)
                ax2.plot(x, r, linewidth=2.0, label=f"{k} (rolling)")
                ax2.plot(x, e, linewidth=1.2, linestyle="--", alpha=0.9, label=f"{k} (ema)")
        ax2.set_title("Alpha Dispersion (Std)")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Std")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8)

        # Panel 3: K values, dual-color (general vs expert)
        has_kg = "k_general_mean" in df.columns and df["k_general_mean"].notna().any()
        has_ke = "k_expert_mean" in df.columns and df["k_expert_mean"].notna().any()
        if has_kg:
            y = df["k_general_mean"]
            r, e = self._smooth(y)
            ax3.scatter(x[raw_idx], y.iloc[raw_idx], s=10, alpha=0.28, color="tab:blue", label="general raw")
            ax3.plot(x, r, color="tab:blue", linewidth=2.0, label="general rolling")
            ax3.plot(x, e, color="tab:blue", linewidth=1.2, linestyle="--", alpha=0.9, label="general ema")
        if has_ke:
            y = df["k_expert_mean"]
            r, e = self._smooth(y)
            ax3.scatter(x[raw_idx], y.iloc[raw_idx], s=10, alpha=0.28, color="tab:orange", label="expert raw")
            ax3.plot(x, r, color="tab:orange", linewidth=2.0, label="expert rolling")
            ax3.plot(x, e, color="tab:orange", linewidth=1.2, linestyle="--", alpha=0.9, label="expert ema")
        ax3.set_title("K Dynamics (General vs Expert)")
        ax3.set_xlabel("Global Step")
        ax3.set_ylabel("K")
        ax3.grid(alpha=0.25)
        ax3.legend(fontsize=8)

        # Panel 4: schedule / aux
        aux_keys = ["temperature", "tax_weight", "learning_rate"]
        for k in aux_keys:
            if k in df.columns and df[k].notna().any():
                y = df[k]
                r, e = self._smooth(y)
                ax4.scatter(x[raw_idx], y.iloc[raw_idx], s=9, alpha=0.25)
                ax4.plot(x, r, linewidth=1.8, label=f"{k} (rolling)")
                ax4.plot(x, e, linewidth=1.2, linestyle="--", alpha=0.9, label=f"{k} (ema)")
        ax4.set_title("Schedule and Auxiliary Signals")
        ax4.set_xlabel("Global Step")
        ax4.set_ylabel("Value")
        ax4.grid(alpha=0.25)
        ax4.legend(fontsize=8)

        fig.suptitle(f"Training Metrics - {self.stage_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])

        fig_path = os.path.join(self.save_dir, f"{self.stage_name}_metrics.png")
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)

        print(f"[MetricLogger] Saved CSV: {csv_path}")
        print(f"[MetricLogger] Saved FIG: {fig_path}")


class TrainerMetricsCallback(TrainerCallback):
    """
    Pull metrics from model._last_metrics at each optimizer step.
    """
    def __init__(self, metric_logger: StageMetricLogger):
        self.metric_logger = metric_logger
        self.last_logged_step = -1

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

    def on_train_end(self, args, state, control, **kwargs):
        self.metric_logger.finalize()