import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

class MetricsLogger:
    """用于记录训练过程中的各项指标"""
    def __init__(self, batch_size: int, gradient_accumulation_steps: int):
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.samples_per_step = batch_size * gradient_accumulation_steps
        
        # Loss 记录
        self.ce_loss_history = []
        self.alpha_loss_history = []
        self.k_general_loss_history = []
        self.k_expert_loss_history = []
        self.tax_loss_history = []
        
        # Alpha 标准差记录
        self.alpha_std_history = []
        
        # K 值记录
        self.k_general_history = []
        self.k_expert_history = []
        
        # 样本数记录
        self.sample_counts = []
        self.current_samples = 0
    
    def log_step(self, ce_loss, alpha_loss, k_general_loss, k_expert_loss, tax_loss,
                 alpha_probs, alpha_labels, mean_k_general, mean_k_expert):
        """记录一个 step 的指标"""
        self.current_samples += self.samples_per_step
        self.sample_counts.append(self.current_samples)
        
        # 记录 Loss
        self.ce_loss_history.append(ce_loss)
        self.alpha_loss_history.append(alpha_loss)
        self.k_general_loss_history.append(k_general_loss)
        self.k_expert_loss_history.append(k_expert_loss)
        self.tax_loss_history.append(tax_loss)
        
        # 计算并记录 Alpha 标准差
        if alpha_probs is not None and alpha_labels is not None:
            alpha_diff = alpha_probs.squeeze() - alpha_labels.squeeze()
            alpha_std = alpha_diff.std().item()
            self.alpha_std_history.append(alpha_std)
        else:
            self.alpha_std_history.append(0.0)
        
        # 记录 K 值
        self.k_general_history.append(mean_k_general)
        self.k_expert_history.append(mean_k_expert)
    
    def plot_curves(self, output_dir: str):
        """绘制所有曲线"""
        if len(self.sample_counts) == 0:
            print("[MetricsLogger] 没有记录到数据，跳过绘图")
            return
        
        x = np.array(self.sample_counts)
        
        # 图1: CE Loss 单独绘制
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x, self.ce_loss_history, label='CE Loss', color='blue', linewidth=1.5)
        ax1.set_xlabel('图文样本对数量')
        ax1.set_ylabel('Loss')
        ax1.set_title('CE Loss 曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(os.path.join(output_dir, 'ce_loss_curve.png'), dpi=150)
        plt.close(fig1)
        
        # 图2: 其他 Loss 合并绘制
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(x, self.alpha_loss_history, label='Alpha Loss', linewidth=1.5)
        ax2.plot(x, self.k_general_loss_history, label='K General Loss', linewidth=1.5)
        ax2.plot(x, self.k_expert_loss_history, label='K Expert Loss', linewidth=1.5)
        ax2.plot(x, self.tax_loss_history, label='Tax Loss', linewidth=1.5)
        ax2.set_xlabel('图文样本对数量')
        ax2.set_ylabel('Loss')
        ax2.set_title('辅助 Loss 曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'auxiliary_loss_curve.png'), dpi=150)
        plt.close(fig2)
        
        # 图3: Alpha 标准差曲线
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(x, self.alpha_std_history, label='Alpha - Target 标准差', color='purple', linewidth=1.5)
        ax3.set_xlabel('图文样本对数量')
        ax3.set_ylabel('标准差')
        ax3.set_title('Alpha 预测与目标的标准差曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'alpha_std_curve.png'), dpi=150)
        plt.close(fig3)
        
        # 图4: K 值曲线（通用和专业在同一张图）
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(x, self.k_general_history, label='K (通用任务)', color='green', linewidth=1.5)
        ax4.plot(x, self.k_expert_history, label='K (专业任务)', color='red', linewidth=1.5)
        ax4.set_xlabel('图文样本对数量')
        ax4.set_ylabel('K 值')
        ax4.set_title('通用任务 K 与专业任务 K 曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(os.path.join(output_dir, 'k_value_curve.png'), dpi=150)
        plt.close(fig4)
        
        print(f"[MetricsLogger] 曲线已保存至 {output_dir}")