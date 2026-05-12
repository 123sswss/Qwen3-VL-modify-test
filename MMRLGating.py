from typing import Optional

import torch
from torch import nn
from utils import attention_pooling
import random


def _cfg_get(config, key, default):
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _parse_group_selection(value, default_groups):
    if value is None:
        return list(default_groups)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return list(default_groups)
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    raise TypeError(f"Unsupported TEXT_GATE_SELECTED_GROUPS type: {type(value).__name__}")


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
        # if self.training:
        #     ss = random.random()
        #     if ss < 0.1:
        #         t_feat = torch.zeros_like(t_feat)
        #     elif ss < 0.2 and ss >= 0.1:
        #         v_feat = torch.zeros_like(v_feat)
        #     else:
        #         pass

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
    def __init__(self,
                 config,
                 epsilon: float = 0.1,
                 temperature: float = 1.0,
                 lambda_=0.2):
        super().__init__()
        self.total_rep_num = config.RP_SPACE_LENGTH * 8
        self.attention_pooling = attention_pooling(config.vision_token_dim, config.GATING_MID_DIM)

        self.enable_text_gating = bool(_cfg_get(config, "ENABLE_TEXT_GATING", True))
        self.fixed_k_when_text_gating_disabled = float(
            _cfg_get(config, "FIXED_K_WHEN_TEXT_GATING_DISABLED", self.total_rep_num)
        )
        self.enable_tax_loss = bool(_cfg_get(config, "ENABLE_TEXT_GATE_TAX_LOSS", True))
        self.enable_collapse_loss = bool(_cfg_get(config, "ENABLE_TEXT_GATE_COLLAPSE_LOSS", True))

        self.group_size = int(_cfg_get(config, "TEXT_GATE_GROUP_SIZE", config.RP_SPACE_LENGTH))
        self.num_groups = int(_cfg_get(config, "TEXT_GATE_NUM_GROUPS", 8))
        self.selected_group_count = int(_cfg_get(config, "TEXT_GATE_SELECTED_GROUP_COUNT", 4))
        self.active_rep_token_count = int(_cfg_get(config, "ACTIVE_REP_TOKEN_COUNT", self.total_rep_num))
        self.selected_groups = _parse_group_selection(
            _cfg_get(config, "TEXT_GATE_SELECTED_GROUPS", [0, 1, 2, 3]),
            default_groups=[0, 1, 2, 3],
        )
        self.shared_groups = _parse_group_selection(
            _cfg_get(config, "TEXT_GATE_SHARED_GROUPS", [0, 1]),
            default_groups=[0, 1],
        )
        self.enable_group_anti_collapse = bool(_cfg_get(config, "ENABLE_TEXT_GATE_GROUP_ANTI_COLLAPSE", False))
        self.group_anti_collapse_weight = float(_cfg_get(config, "TEXT_GATE_GROUP_ANTI_COLLAPSE_WEIGHT", 0.0))
        self.group_usage_ema_momentum = float(_cfg_get(config, "TEXT_GATE_GROUP_USAGE_EMA_MOMENTUM", 0.98))
        self.group_usage_ema_eps = float(_cfg_get(config, "TEXT_GATE_GROUP_USAGE_EMA_EPS", 1e-6))
        requested_mode = str(_cfg_get(config, "TEXT_GATE_SELECTION_MODE", "dynamic_prefix")).strip().lower()
        if requested_mode in {"dynamic", "dynamic_k", "prefix"}:
            requested_mode = "dynamic_prefix"
        if requested_mode in {"group20", "fixed20", "fixed_group_20"}:
            requested_mode = "fixed_group20"
        if requested_mode in {"group_top4", "top4_group", "learnable_group_top4", "8choose4"}:
            requested_mode = "group_top4"
        if requested_mode in {"token_top20", "top20_token", "learnable_token_top20", "token_topk20"}:
            requested_mode = "token_top20"
        if requested_mode in {"fixed", "fixed_k"}:
            requested_mode = "fixed_prefix"
        if not self.enable_text_gating:
            requested_mode = "fixed_prefix"
        self.selection_mode = requested_mode
        self._validate_selection_config()

        if self.selection_mode == "dynamic_prefix":
            self.text_relevance_head = nn.Sequential(
                nn.Linear(config.text_token_dim, config.GATING_MID_DIM),
                nn.ReLU(),
                nn.Linear(config.GATING_MID_DIM, 1)
            )
            nn.init.constant_(self.text_relevance_head[-1].bias, 0.0)

            self.k_budget_head = nn.Sequential(
                nn.Linear(config.vision_token_dim + config.text_token_dim + 2, config.GATING_MID_DIM),
                nn.ReLU(),
                nn.Linear(config.GATING_MID_DIM, 1)
            )
            nn.init.constant_(self.k_budget_head[-1].bias, -1.5)  # 原来 -0.5 太激进
            self.group_selection_head = None
        elif self.selection_mode == "group_top4":
            self.text_relevance_head = None
            self.k_budget_head = None
            self.group_selection_head = nn.Sequential(
                nn.Linear(config.vision_token_dim + config.text_token_dim + 2, config.GATING_MID_DIM),
                nn.ReLU(),
                nn.Linear(config.GATING_MID_DIM, self.num_groups)
            )
            nn.init.constant_(self.group_selection_head[-1].bias, 0.0)
            self.token_selection_head = None
        elif self.selection_mode == "token_top20":
            self.text_relevance_head = None
            self.k_budget_head = None
            self.group_selection_head = None
            self.token_selection_head = nn.Sequential(
                nn.Linear(config.vision_token_dim + config.text_token_dim + 2, config.GATING_MID_DIM),
                nn.ReLU(),
                nn.Linear(config.GATING_MID_DIM, self.total_rep_num)
            )
            nn.init.constant_(self.token_selection_head[-1].bias, 0.0)
        else:
            self.text_relevance_head = None
            self.k_budget_head = None
            self.group_selection_head = None
            self.token_selection_head = None

        self.hard_concrete = HardConcreteGate(temperature)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.debug_context = {}

        self.lambda_general = 1.0
        self.lambda_expert = 0.25
        self.k_cap_expert = 14.0
        self.k_min_expert = 2.0
        self.last_selected_token_indices = None
        self.last_soft_group_scores = None
        self.last_hard_group_mask = None
        init_usage = torch.full((self.num_groups,), 1.0 / max(self.num_groups, 1), dtype=torch.float32)
        self.register_buffer("group_usage_ema", init_usage, persistent=False)

    def _validate_selection_config(self):
        supported_modes = {"dynamic_prefix", "fixed_prefix", "fixed_group20", "group_top4", "token_top20"}
        if self.selection_mode not in supported_modes:
            raise ValueError(
                f"Unsupported TEXT_GATE_SELECTION_MODE={self.selection_mode!r}, "
                f"supported={sorted(supported_modes)}"
            )
        if self.group_size <= 0 or self.num_groups <= 0:
            raise ValueError("TEXT_GATE_GROUP_SIZE and TEXT_GATE_NUM_GROUPS must be positive")
        if self.group_size * self.num_groups != self.total_rep_num:
            raise ValueError(
                f"Group layout mismatch: group_size({self.group_size}) * num_groups({self.num_groups}) "
                f"!= total_rep_num({self.total_rep_num})"
            )
        if len(set(self.selected_groups)) != len(self.selected_groups):
            raise ValueError(f"TEXT_GATE_SELECTED_GROUPS contains duplicates: {self.selected_groups}")
        for g in self.selected_groups:
            if g < 0 or g >= self.num_groups:
                raise ValueError(
                    f"TEXT_GATE_SELECTED_GROUPS out of range: {self.selected_groups}, num_groups={self.num_groups}"
                )
        if self.selection_mode == "fixed_group20" and len(self.selected_groups) * self.group_size != 20:
            raise ValueError(
                "fixed_group20 requires exactly 20 active text tokens, "
                f"but got len(selected_groups)={len(self.selected_groups)}, group_size={self.group_size}"
            )
        if self.selection_mode in {"fixed_group20", "group_top4", "token_top20"} and self.active_rep_token_count != 20:
            raise ValueError(
                f"{self.selection_mode} requires ACTIVE_REP_TOKEN_COUNT=20, got {self.active_rep_token_count}"
            )
        if self.selection_mode == "group_top4" and self.selected_group_count <= 0:
            raise ValueError("TEXT_GATE_SELECTED_GROUP_COUNT must be positive for group_top4")

    def _build_token_topk_gate(self,
                               pooled_vision: torch.Tensor,
                               text_embedding: torch.Tensor,
                               batch_alpha: torch.Tensor,
                               has_image: torch.Tensor):
        feat = torch.cat([pooled_vision, text_embedding, batch_alpha, has_image], dim=-1)
        token_logits = self.token_selection_head(feat)
        soft_token_scores = torch.sigmoid(token_logits)
        topk_idx = torch.topk(token_logits, k=self.active_rep_token_count, dim=-1).indices
        hard_token_mask = torch.zeros_like(token_logits)
        hard_token_mask.scatter_(1, topk_idx, 1.0)
        st_token_mask = hard_token_mask + soft_token_scores - soft_token_scores.detach()
        raw_budget = hard_token_mask.sum(dim=-1, keepdim=True)
        k_budget = soft_token_scores.sum(dim=-1, keepdim=True)
        text_relevance_prob = torch.ones(
            pooled_vision.shape[0], 1, device=pooled_vision.device, dtype=pooled_vision.dtype
        )
        return st_token_mask, text_relevance_prob, raw_budget, k_budget, soft_token_scores, hard_token_mask, topk_idx

    def _build_fixed_k_gate(self,
                            batch_size: int,
                            device: torch.device,
                            dtype: torch.dtype,
                            temperature_override: Optional[float] = None) -> torch.Tensor:
        fixed_k = max(0.0, min(float(self.fixed_k_when_text_gating_disabled), float(self.total_rep_num)))
        slot_idx = torch.arange(self.total_rep_num, device=device, dtype=dtype).view(1, -1)
        fixed_budget = torch.full((batch_size, 1), fixed_k, device=device, dtype=dtype)
        logits = fixed_budget - slot_idx
        return self.hard_concrete(logits, temperature_override)

    def _build_fixed_group_gate(self,
                                batch_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> torch.Tensor:
        gate = torch.zeros(batch_size, self.total_rep_num, device=device, dtype=dtype)
        for group_idx in self.selected_groups:
            start = group_idx * self.group_size
            end = start + self.group_size
            gate[:, start:end] = 1.0
        return gate

    def _build_fixed_group_token_indices(self,
                                         batch_size: int,
                                         device: torch.device) -> torch.Tensor:
        token_indices = []
        for group_idx in self.selected_groups:
            start = group_idx * self.group_size
            token_indices.extend(range(start, start + self.group_size))
        base = torch.tensor(token_indices, device=device, dtype=torch.long)
        return base.unsqueeze(0).expand(batch_size, -1)

    def _build_dynamic_prefix_gate(self,
                                   pooled_vision: torch.Tensor,
                                   text_embedding: torch.Tensor,
                                   batch_alpha: torch.Tensor,
                                   has_image: torch.Tensor,
                                   temperature_override: Optional[float] = None):
        text_relevance_prob = torch.sigmoid(self.text_relevance_head(text_embedding))  # [B, 1]
        feat = torch.cat([pooled_vision, text_embedding, batch_alpha, has_image], dim=-1)
        raw_budget = self.total_rep_num * torch.sigmoid(self.k_budget_head(feat))
        image_scale = 0.5 + 0.5 * has_image
        k_budget = raw_budget * image_scale * (0.25 + 0.75 * text_relevance_prob)
        slot_idx = torch.arange(
            self.total_rep_num, device=k_budget.device, dtype=k_budget.dtype
        ).view(1, -1)
        k_logits = k_budget - slot_idx
        hard_k_logits = self.hard_concrete(k_logits, temperature_override)
        return hard_k_logits, text_relevance_prob, raw_budget, k_budget

    def _build_group_topk_gate(self,
                               pooled_vision: torch.Tensor,
                               text_embedding: torch.Tensor,
                               batch_alpha: torch.Tensor,
                               has_image: torch.Tensor):
        feat = torch.cat([pooled_vision, text_embedding, batch_alpha, has_image], dim=-1)
        group_logits = self.group_selection_head(feat)
        soft_group_scores = torch.sigmoid(group_logits)
        topk_idx = torch.topk(group_logits, k=self.selected_group_count, dim=-1).indices
        hard_group_mask = torch.zeros_like(group_logits)
        hard_group_mask.scatter_(1, topk_idx, 1.0)
        st_group_mask = hard_group_mask + soft_group_scores - soft_group_scores.detach()
        token_gate = st_group_mask.repeat_interleave(self.group_size, dim=-1)
        token_offsets = torch.arange(self.group_size, device=group_logits.device, dtype=torch.long).view(1, 1, -1)
        selected_token_indices = (topk_idx.unsqueeze(-1) * self.group_size + token_offsets).reshape(topk_idx.shape[0], -1)
        raw_budget = hard_group_mask.sum(dim=-1, keepdim=True) * self.group_size
        k_budget = soft_group_scores.sum(dim=-1, keepdim=True) * self.group_size
        text_relevance_prob = torch.ones(
            pooled_vision.shape[0], 1, device=pooled_vision.device, dtype=pooled_vision.dtype
        )
        return token_gate, text_relevance_prob, raw_budget, k_budget, soft_group_scores, hard_group_mask, selected_token_indices

    def _normalize_group_usage(self, usage: torch.Tensor) -> torch.Tensor:
        denom = usage.sum(dim=-1, keepdim=True).clamp_min(self.group_usage_ema_eps)
        return usage / denom

    def compute_group_anti_collapse_loss(self):
        device = self.group_usage_ema.device
        zero = torch.tensor(0.0, device=device, dtype=self.group_usage_ema.dtype)
        soft_group_scores = self.last_soft_group_scores
        if self.selection_mode != "group_top4" or soft_group_scores is None:
            return zero, {}

        batch_usage = self._normalize_group_usage(soft_group_scores).mean(dim=0)
        momentum = min(max(float(self.group_usage_ema_momentum), 0.0), 0.9999)
        ema_prev = self.group_usage_ema.detach().to(device=device, dtype=batch_usage.dtype)
        ema_curr = momentum * ema_prev + (1.0 - momentum) * batch_usage
        ema_curr = self._normalize_group_usage(ema_curr)

        with torch.no_grad():
            self.group_usage_ema.copy_(ema_curr.detach().to(dtype=self.group_usage_ema.dtype))

        uniform = torch.full_like(ema_curr, 1.0 / max(self.num_groups, 1))
        weight = float(self.group_anti_collapse_weight) if self.enable_group_anti_collapse else 0.0
        anti_collapse_loss = zero.to(dtype=batch_usage.dtype)
        if weight > 0.0:
            anti_collapse_loss = weight * (ema_curr - uniform).pow(2).mean()

        entropy = -(ema_curr.clamp_min(self.group_usage_ema_eps) * ema_curr.clamp_min(self.group_usage_ema_eps).log()).sum()
        entropy = entropy / torch.log(torch.tensor(float(max(self.num_groups, 2)), device=entropy.device, dtype=entropy.dtype))
        dead_threshold = 0.25 / max(self.num_groups, 1)
        debug = {
            "anti_collapse_loss": anti_collapse_loss.detach().float(),
            "group_usage_ema_max": ema_curr.max().detach().float(),
            "group_usage_ema_min": ema_curr.min().detach().float(),
            "group_usage_ema_entropy": entropy.detach().float(),
            "group_usage_dead_ratio": (ema_curr < dead_threshold).float().mean().detach(),
            "anti_collapse_weight": torch.tensor(weight, device=ema_curr.device, dtype=ema_curr.dtype).detach().float(),
        }
        for gi in range(int(ema_curr.shape[0])):
            debug[f"group_usage_ema_{gi}"] = ema_curr[gi].detach().float()
        return anti_collapse_loss, debug

    def forward(self,
                delta_vision_token: torch.Tensor,
                alpha: torch.Tensor,
                batch_indices_token: torch.Tensor,
                batch_indices_img: torch.Tensor,
                batch_size: int,
                text_embedding: torch.Tensor,
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

        if self.selection_mode == "dynamic_prefix":
            hard_k_logits, text_relevance_prob, raw_budget, k_budget = self._build_dynamic_prefix_gate(
                pooled_vision=pooled_vision,
                text_embedding=text_embedding,
                batch_alpha=batch_alpha,
                has_image=has_image,
                temperature_override=temperature_override,
            )
            selected_token_indices = None
        elif self.selection_mode == "fixed_group20":
            text_relevance_prob = torch.ones(batch_size, 1, device=text_embedding.device, dtype=text_embedding.dtype)
            raw_budget = torch.full(
                (batch_size, 1),
                float(len(self.selected_groups) * self.group_size),
                device=text_embedding.device,
                dtype=text_embedding.dtype,
            )
            k_budget = raw_budget
            hard_k_logits = self._build_fixed_group_gate(
                batch_size=batch_size,
                device=text_embedding.device,
                dtype=text_embedding.dtype,
            )
            selected_token_indices = self._build_fixed_group_token_indices(
                batch_size=batch_size,
                device=text_embedding.device,
            )
            soft_group_scores = None
            hard_group_mask = None
        elif self.selection_mode == "group_top4":
            hard_k_logits, text_relevance_prob, raw_budget, k_budget, soft_group_scores, hard_group_mask, selected_token_indices = self._build_group_topk_gate(
                pooled_vision=pooled_vision,
                text_embedding=text_embedding,
                batch_alpha=batch_alpha,
                has_image=has_image,
            )
            soft_token_scores = None
            hard_token_mask = None
        elif self.selection_mode == "token_top20":
            hard_k_logits, text_relevance_prob, raw_budget, k_budget, soft_token_scores, hard_token_mask, selected_token_indices = self._build_token_topk_gate(
                pooled_vision=pooled_vision,
                text_embedding=text_embedding,
                batch_alpha=batch_alpha,
                has_image=has_image,
            )
            soft_group_scores = None
            hard_group_mask = None
        else:
            text_relevance_prob = torch.ones(batch_size, 1, device=text_embedding.device, dtype=text_embedding.dtype)
            raw_budget = torch.full(
                (batch_size, 1),
                min(float(self.fixed_k_when_text_gating_disabled), float(self.total_rep_num)),
                device=text_embedding.device,
                dtype=text_embedding.dtype,
            )
            k_budget = raw_budget
            hard_k_logits = self._build_fixed_k_gate(
                batch_size=batch_size,
                device=text_embedding.device,
                dtype=text_embedding.dtype,
                temperature_override=temperature_override,
            )
            selected_token_indices = None
            soft_group_scores = None
            hard_group_mask = None
            soft_token_scores = None
            hard_token_mask = None

        self.last_selected_token_indices = selected_token_indices
        self.last_soft_group_scores = soft_group_scores if self.selection_mode == "group_top4" else None
        self.last_hard_group_mask = hard_group_mask if self.selection_mode == "group_top4" else None

        # alpha 只负责放行/抑制
        if self.training:
            hard_k_logits = hard_k_logits * batch_alpha
        else:
            alpha_hard_mask = (batch_alpha > 0.5).to(dtype=hard_k_logits.dtype)
            hard_k_logits = hard_k_logits * alpha_hard_mask

        self.debug_context = {
            "enable_text_gating": self.enable_text_gating,
            "selection_mode": self.selection_mode,
            "selected_groups": list(self.selected_groups),
            "group_size": self.group_size,
            "selected_group_count": self.selected_group_count,
            "text_rel_mean": text_relevance_prob.detach().float().mean(),
            "raw_budget_mean": raw_budget.detach().float().mean(),
            "k_budget_mean": k_budget.detach().float().mean(),
            "active_token_count_mean": hard_k_logits.detach().float().sum(dim=-1).mean(),
            "batch_alpha_mean": batch_alpha.detach().float().mean(),
        }

        if self.training:
            k_soft = hard_k_logits.sum(dim=-1)  # [B]
            if self.selection_mode == "group_top4":
                tax_loss = hard_k_logits.new_tensor(0.0)
                if soft_group_scores is not None:
                    target_group_count = hard_k_logits.new_tensor(float(self.selected_group_count))
                    count_loss = ((soft_group_scores.sum(dim=-1) - target_group_count) / float(self.num_groups)).pow(2).mean()
                    tax_loss = tax_loss + 0.1 * count_loss
                    self.debug_context["group_count_loss"] = count_loss.detach().float().mean()
            elif self.selection_mode == "token_top20":
                tax_loss = hard_k_logits.new_tensor(0.0)
                if soft_token_scores is not None:
                    target_token_count = hard_k_logits.new_tensor(float(self.active_rep_token_count))
                    count_loss = ((soft_token_scores.sum(dim=-1) - target_token_count) / float(self.total_rep_num)).pow(2).mean()
                    tax_loss = tax_loss + 0.1 * count_loss

                    weight = batch_alpha.detach().clamp(min=0.0, max=1.0)
                    weight_sum = weight.sum().clamp(min=1e-6)
                    per_token_usage = (soft_token_scores * weight).sum(dim=0) / weight_sum.squeeze(-1)
                    target_usage = hard_k_logits.new_full((self.total_rep_num,), float(self.active_rep_token_count) / float(self.total_rep_num))
                    balance_loss = (per_token_usage - target_usage).pow(2).mean()
                    tax_loss = tax_loss + 0.1 * balance_loss
                    self.debug_context["token_count_loss"] = count_loss.detach().float().mean()
                    self.debug_context["token_balance_loss"] = balance_loss.detach().float().mean()
                    self.debug_context["token_usage_mean"] = per_token_usage.detach().float().mean()
            elif self.enable_tax_loss:
                batch_alpha_prob = batch_alpha.detach()
                penalty_mask = torch.where(
                    batch_alpha_prob > 0.5,
                    torch.zeros_like(batch_alpha_prob),
                    1.0 - batch_alpha_prob
                )
                dynamic_lambda = self.lambda_ * penalty_mask
                raw_loss = dynamic_lambda.squeeze(-1) * (k_soft / self.total_rep_num)
                tax_loss = raw_loss.mean()
            else:
                tax_loss = hard_k_logits.new_tensor(0.0)

            if self.enable_collapse_loss and self.selection_mode == "dynamic_prefix":
                target_var = hard_k_logits.new_tensor(4.0)
                k_var = k_soft.var(unbiased=False)
                collapse_loss = 0.05 * torch.relu(target_var - k_var)
                tax_loss = tax_loss + collapse_loss

            return hard_k_logits, tax_loss

        return hard_k_logits


