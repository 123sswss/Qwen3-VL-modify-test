import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def _ema_smooth(data, weight=0.9):
    """指数移动平均平滑，weight 越大越平滑"""
    smoothed, last = [], data[0]
    for val in data:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return smoothed


class MetricsLogger:
    """用于记录训练过程中的各项指标"""
    def __init__(self, batch_size: int, gradient_accumulation_steps: int):
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.samples_per_step = batch_size
        self.ce_loss_history = []
        self.alpha_loss_history = []
        self.k_general_loss_history = []
        self.k_expert_loss_history = []
        self.tax_loss_history = []
        self.alpha_std_history = []
        self.k_general_history = []
        self.k_expert_history = []
        self.sample_counts = []
        self.current_samples = 0

    def log_step(self, ce_loss, alpha_loss, k_general_loss, k_expert_loss, tax_loss,
                 alpha_probs, alpha_labels, mean_k_general, mean_k_expert):
        """记录一个 step 的指标"""
        self.current_samples += self.samples_per_step
        self.sample_counts.append(self.current_samples)
        self.ce_loss_history.append(ce_loss)
        self.alpha_loss_history.append(alpha_loss)
        self.k_general_loss_history.append(k_general_loss)
        self.k_expert_loss_history.append(k_expert_loss)
        self.tax_loss_history.append(tax_loss)
        if alpha_probs is not None and alpha_labels is not None:
            alpha_diff = alpha_probs.squeeze() - alpha_labels.squeeze()
            self.alpha_std_history.append(alpha_diff.std().item())
        else:
            self.alpha_std_history.append(0.0)
        self.k_general_history.append(mean_k_general)
        self.k_expert_history.append(mean_k_expert)

    def plot_curves(self, output_dir: str, total_target_samples: int = None, smooth_weight: float = 0.9):
        """
        绘制所有曲线，带 EMA 平滑，横坐标为 step
        :param output_dir: 保存路径
        :param total_target_samples: 伪造/映射的真实总样本数（不再影响横坐标，仅保留参数兼容）
        :param smooth_weight: EMA 平滑系数，0~1，越大越平滑
        """
        if len(self.sample_counts) == 0:
            print("[MetricsLogger] 没有记录到数据，跳过绘图")
            return

        num_steps = len(self.sample_counts)
        x = np.arange(1, num_steps + 1)  # 横坐标改为 step

        def beautify_plot(ax, y_data_lists):
            y_min = min([min(y) for y in y_data_lists])
            y_max = max([max(y) for y in y_data_lists])
            ax.margins(x=0)
            y_range = y_max - y_min if y_max != y_min else 1.0
            ax.set_ylim(bottom=y_min, top=y_max + y_range * 0.15)
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        def plot_with_smooth(ax, x, raw, color, label):
            """原始曲线淡显，平滑曲线加粗前景"""
            ax.plot(x, raw, color=color, linewidth=0.8, alpha=0.15)
            ax.plot(x, _ema_smooth(raw, smooth_weight), color=color, linewidth=2, label=label)

        save_kwargs = {'dpi': 200, 'bbox_inches': 'tight', 'pad_inches': 0.05}

        # 图1: CE Loss
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        plot_with_smooth(ax1, x, self.ce_loss_history, '#1f77b4', 'CE Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('CE Loss Curve', pad=10)
        beautify_plot(ax1, [self.ce_loss_history])
        ax1.legend(loc='upper right', framealpha=0.9)
        fig1.savefig(os.path.join(output_dir, 'ce_loss_curve.png'), **save_kwargs)
        plt.close(fig1)

        # 图2: 其他 Loss
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        colors2 = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for raw, color, label in zip(
            [self.alpha_loss_history, self.k_general_loss_history,
             self.k_expert_loss_history, self.tax_loss_history],
            colors2,
            ['Alpha Loss', 'K General Loss', 'K Expert Loss', 'Tax Loss']
        ):
            plot_with_smooth(ax2, x, raw, color, label)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Auxiliary Loss Curves', pad=10)
        beautify_plot(ax2, [self.alpha_loss_history, self.k_general_loss_history,
                            self.k_expert_loss_history, self.tax_loss_history])
        ax2.legend(loc='upper right', framealpha=0.9)
        fig2.savefig(os.path.join(output_dir, 'auxiliary_loss_curve.png'), **save_kwargs)
        plt.close(fig2)

        # 图3: Alpha 标准差
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        plot_with_smooth(ax3, x, self.alpha_std_history, '#9467bd', 'Alpha - Target Std Dev')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_title('Alpha Prediction vs Target Std Dev Curve', pad=10)
        beautify_plot(ax3, [self.alpha_std_history])
        ax3.legend(loc='upper right', framealpha=0.9)
        fig3.savefig(os.path.join(output_dir, 'alpha_std_curve.png'), **save_kwargs)
        plt.close(fig3)

        # 图4: K 值
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        plot_with_smooth(ax4, x, self.k_general_history, '#2ca02c', 'K (General Task)')
        plot_with_smooth(ax4, x, self.k_expert_history, '#d62728', 'K (Expert Task)')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('K')
        ax4.set_title('General Task K vs Expert Task K Curves', pad=10)
        beautify_plot(ax4, [self.k_general_history, self.k_expert_history])
        ax4.legend(loc='upper right', framealpha=0.9)
        fig4.savefig(os.path.join(output_dir, 'k_value_curve.png'), **save_kwargs)
        plt.close(fig4)

        print(f"[MetricsLogger] 曲线已保存至 {output_dir}")