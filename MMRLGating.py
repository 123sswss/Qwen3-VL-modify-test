# MMRLGating.PY
from typing import Optional

import torch
from torch import nn
from utils import attention_pooling

class Task_classifier(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.vision_proj = nn.Linear(config.vision_token_dim, config.GATING_MID_DIM)
        self.text_proj = nn.Linear(config.text_token_dim, config.GATING_MID_DIM)
        self.fc_fusion = nn.Linear(config.GATING_MID_DIM * 2, config.GATING_MID_DIM)
        self.output_head = nn.Linear(config.GATING_MID_DIM, 1)
        self.relu = nn.ReLU()

        nn.init.constant_(self.output_head.bias, -0.5)

    def forward(self,
                vision_token_after_pooling: torch.Tensor,
                text_embedding_after_pooling: torch.Tensor):
        v_feat = self.relu(self.vision_proj(vision_token_after_pooling))
        t_feat = self.relu(self.text_proj(text_embedding_after_pooling))
        combined = torch.cat((v_feat, t_feat), dim=-1)
        combined = self.relu(self.fc_fusion(combined))
        alpha = self.output_head(combined)
        return alpha

class HardConcreteGate(nn.Module):
    def __init__(self,
                 temperature: float,
                 stretching_length: float = 0.1,
                 eps: float = 1e-7):
        super().__init__()
        self.temperature = temperature
        self.upper_bound = 1 + stretching_length
        self.lower_bound = 0 - stretching_length
        self.eps = eps
    ################## 传入HardConcreteGate的都必须是未经归一化的值 ##################
    def forward(self,
                logits: torch.Tensor,
                temperature_override: Optional[float] = None) -> torch.Tensor:
        temp = temperature_override if temperature_override is not None else self.temperature
        if self.training:
            u = torch.rand_like(logits)
            u = torch.clamp(u, min=self.eps, max=1 - self.eps)
            noise = torch.log(u) - torch.log(1 - u)
        else:
            noise = 0.0
        y = torch.sigmoid((noise + logits) / temp)
        y_stretched = y * (self.upper_bound - self.lower_bound) + self.lower_bound
        gate = torch.clamp(y_stretched, min=0, max=1)
        return gate


class textGating(nn.Module):
    """
    Multi-Span Selector: 从 num_spans(8) 个 span 中选 num_select(4) 个。
    每个 span 包含 span_size(5) 个连续 rep token。
    选中的 4 个 span 按顺序拼成 20 个 placeholder token。
    """
    def __init__(self,
                 config,
                 epsilon: float = 0.1,
                 temperature: float = 1.0,
                 lambda_=0.2):
        super().__init__()
        insert_layers = getattr(config, "INSERT_LAYER", None)
        num_layers = len(insert_layers) if insert_layers is not None else 8
        self.span_size = int(getattr(config, "RP_SPACE_LENGTH", 5))
        self.num_spans = num_layers  # 8
        self.num_select = int(getattr(config, "MULTI_SPAN_NUM_SELECT", 4))
        self.total_rep_num = self.span_size * self.num_spans  # 40
        self.window_size = self.span_size * self.num_select  # 20 (placeholder count)

        # 兼容旧接
        self.num_start_positions = 1  # dummy for backward compat
        self.start_exploration_scale = float(getattr(config, "START_EXPLORATION_SCALE", 0.0))

        self.attention_pooling = attention_pooling(config.vision_token_dim, config.GATING_MID_DIM)
        self.visual_semantic_proj = nn.Sequential(
            nn.Linear(config.vision_token_dim * 2, config.GATING_MID_DIM),
            nn.ReLU(),
        )
        self.confidence_proj = nn.Sequential(
            nn.Linear(8, config.GATING_MID_DIM // 2),
            nn.ReLU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
            nn.ReLU(),
        )
        self.text_relevance_head = nn.Sequential(
            nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, 1)
        )
        nn.init.constant_(self.text_relevance_head[-1].bias, 0.0)

        # 输出 num_spans 个 logits，决定每个 span 是否被选中
        feat_dim = config.GATING_MID_DIM * 2 + config.GATING_MID_DIM // 2 + 2
        self.span_logit_head = nn.Sequential(
            nn.Linear(feat_dim, config.GATING_MID_DIM),
            nn.ReLU(),
            nn.Linear(config.GATING_MID_DIM, self.num_spans)
        )
        # 初始化：轻微正偏置让初始时所有 span 趋向被选中（避免冷启动全关）
        nn.init.zeros_(self.span_logit_head[-1].weight)
        nn.init.constant_(self.span_logit_head[-1].bias, 0.5)

        self.hard_concrete = HardConcreteGate(temperature)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.debug_context = {}

        # 用于外部调度
        self.lambda_general = 1.0
        self.lambda_expert = 0.25
        self.k_cap_expert = 14.0
        self.k_min_expert = 2.0
        self.infer_alpha_threshold = 0.5

    def forward(self,
                delta_vision_token: torch.Tensor,
                alpha: torch.Tensor,
                batch_indices_token: torch.Tensor,
                batch_indices_img: torch.Tensor,
                batch_size: int,
                text_embedding: torch.Tensor,
                pooled_semantic_mean: Optional[torch.Tensor] = None,
                pooled_semantic_max: Optional[torch.Tensor] = None,
                confidence_stats: Optional[torch.Tensor] = None,
                has_image: Optional[torch.Tensor] = None,
                temperature_override: Optional[float] = None):
        pooled_vision = self.attention_pooling.forward_vectorized(
            delta_vision_token, batch_indices_token, batch_size
        )  # [B, Dv]

        alpha_prob = torch.sigmoid(alpha)
        batch_alpha = torch.zeros(batch_size, 1, device=alpha_prob.device, dtype=alpha_prob.dtype)
        batch_alpha.index_reduce_(0, batch_indices_img, alpha_prob, reduce="amax", include_self=False)

        if has_image is None:
            has_image = torch.ones(batch_size, 1, device=text_embedding.device, dtype=text_embedding.dtype)
        if pooled_semantic_mean is None:
            pooled_semantic_mean = pooled_vision
        if pooled_semantic_max is None:
            pooled_semantic_max = pooled_vision
        if confidence_stats is None:
            confidence_stats = torch.zeros(batch_size, 8, device=text_embedding.device, dtype=text_embedding.dtype)

        visual_semantic = torch.cat([pooled_semantic_mean, pooled_semantic_max], dim=-1)
        visual_semantic = self.visual_semantic_proj(visual_semantic)
        confidence_feat = self.confidence_proj(confidence_stats)
        text_feat = self.text_proj(text_embedding)

        feat = torch.cat([visual_semantic, text_feat, confidence_feat, batch_alpha, has_image], dim=-1)

        # 产出 8 个 span logits
        span_logits = self.span_logit_head(feat)  # [B, 8]

        # 训练时加探索噪声
        if self.training and self.start_exploration_scale > 0:
            explore_noise = (torch.rand_like(span_logits) * 2.0 - 1.0) * self.start_exploration_scale
            span_logits = span_logits + explore_noise

        # Hard Concrete gate 得到 soft binary masks
        span_gates = self.hard_concrete(span_logits, temperature_override)  # [B, 8], 值在 [0,1]

        # ---- 推理时：hard top-k 选 num_select 个 ----
        if not self.training:
            _, topk_indices = span_logits.topk(self.num_select, dim=-1)
            hard_gates = torch.zeros_like(span_gates)
            hard_gates.scatter_(1, topk_indices, 1.0)
            span_gates = hard_gates

        # ---- 训练时：straight-through top-k ----
        # 让 hard 选择的梯度通过 soft gates 回传
        if self.training:
            _, topk_indices = span_logits.topk(self.num_select, dim=-1)
            hard_gates = torch.zeros_like(span_gates)
            hard_gates.scatter_(1, topk_indices, 1.0)
            # straight-through: forward 用 hard，backward 用 soft
            span_gates_st = hard_gates + span_gates - span_gates.detach()
        else:
            span_gates_st = span_gates

        # ---- 构建 position_weights [B, 20, 40] ----
        # 每个选中的 span i 的 5 个 token 依次填入 placeholder
        # 需要按 span 顺序排列（保持层序）
        device = feat.device
        dtype = feat.dtype

        # 构建 span->token 映射 basis: [num_spans, span_size, total_rep_num]
        # basis[i, j, i*span_size+j] = 1
        basis = torch.zeros(self.num_spans, self.span_size, self.total_rep_num, device=device, dtype=dtype)
        for i in range(self.num_spans):
            for j in range(self.span_size):
                basis[i, j, i * self.span_size + j] = 1.0

        # 对每个 batch，按顺序取 topk span 的 token
        # 排序 topk indices 保持层序
        if self.training:
            sorted_topk, _ = topk_indices.sort(dim=-1)
        else:
            sorted_topk = topk_indices
            sorted_topk, _ = sorted_topk.sort(dim=-1)

        # position_weights: [B, window_size, total_rep_num]
        # 用 gather + einsum 的可微方式构建
        # 方法：对每个 batch sample，选中的 num_select 个 span 按序排列
        # span_gates_st 已经是 [B, 8] 的 0/1 (with gradient)
        # 我们需要一个 ordered selection

        # 用累积和来确定每个span在输出中的位置偏移
        # 由于 straight-through，hard_gates 决定了选哪些，但梯度走 span_gates
        # 简单做法：直接用 sorted topk 索引构建 position_weights

        batch_size_actual = feat.shape[0]
        position_weights = torch.zeros(batch_size_actual, self.window_size, self.total_rep_num, device=device, dtype=dtype)

        for b in range(batch_size_actual):
            for sel_idx in range(self.num_select):
                span_id = sorted_topk[b, sel_idx]
                gate_val = span_gates_st[b, span_id]
                offset = sel_idx * self.span_size
                for j in range(self.span_size):
                    position_weights[b, offset + j, span_id * self.span_size + j] = gate_val

        # selected_mask: [B, 40] 指示哪些 token 被选中
        selected_mask = position_weights.sum(dim=1).clamp(min=0.0, max=1.0)

        # k_selected: 实际选中的 token 数（训练时为 soft 值，推理时恒为 20）
        k_selected = span_gates_st.sum(dim=-1) * float(self.span_size)  # [B]

        # span budget loss: 鼓励选中恰好 num_select 个
        span_budget_loss = (span_gates.sum(dim=-1) - float(self.num_select)).pow(2).mean()

        self.debug_context = {
            "batch_alpha": batch_alpha.detach(),
            "span_logits": span_logits.detach(),
            "span_gates": span_gates.detach(),          # soft gates，用于观测真实门值
            "span_gates_hard": span_gates_st.detach(),  # ST/hard gates，用于观测实际前向选择
            "sorted_topk": sorted_topk.detach(),
            "exploration_scale": torch.tensor(float(self.start_exploration_scale), device=device),
            "selected_mask": selected_mask.detach(),
            "span_budget_loss": span_budget_loss.detach(),
        }

        return {
            "selected_mask": selected_mask,
            "position_weights": position_weights,
            "span_gates": span_gates,
            "span_gates_hard": span_gates_st,
            "span_logits": span_logits,
            "sorted_topk": sorted_topk,
            "k_selected": k_selected,
            "span_budget_loss": span_budget_loss,
            "tax_loss": None,
        }
