from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import Cache, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs, TransformersKwargs

import MMRL
import config as cfg
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl
import VisionModelWithMMRL as vmmrl


def _cfg_attr(config, key, default):
    if hasattr(config, key):
        return getattr(config, key)
    return default


class TextResidualAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextAdapterRouter(nn.Module):
    def __init__(self, text_dim: int, vision_dim: int, hidden_dim: int, adapter_count: int):
        super().__init__()
        self.adapter_count = int(max(adapter_count, 1))
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.alpha_proj = nn.Linear(1, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.route_head = nn.Linear(hidden_dim, self.adapter_count)
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        self.relu = nn.ReLU()

    def forward(
        self,
        text_pooled: torch.Tensor,
        visual_pooled: torch.Tensor,
        alpha_state: torch.Tensor,
    ):
        text_feat = self.relu(self.text_proj(text_pooled))
        vision_feat = self.relu(self.vision_proj(visual_pooled.to(dtype=text_pooled.dtype)))
        alpha_feat = self.relu(self.alpha_proj(alpha_state.to(dtype=text_pooled.dtype)))
        fused = self.relu(self.fusion(torch.cat([text_feat, vision_feat, alpha_feat], dim=-1)))
        route_probs = torch.softmax(self.route_head(fused), dim=-1)
        return route_probs


class QWen3WithMMRL(qwen3_vl.Qwen3VLModel):
    def __init__(self,
                 config,
                 tokenizer=None,
                 ):
        # todo：研究一下处理训练保存与加载
        vision_dim = getattr(config.vision_config, "hidden_size")  # 默认值防报错
        text_dim = getattr(config.text_config, "hidden_size")
        # if not hasattr(config, "mmrl_config"):
        config.mmrl_config = {
            "USE_MMRL": _cfg_attr(config, "USE_MMRL", cfg.USE_MMRL),
            "INSERT_LAYER": list(_cfg_attr(config, "INSERT_LAYER", cfg.INSERT_LAYER)),
            "POOLING_DIM": _cfg_attr(config, "POOLING_DIM", cfg.POOLING_DIM),
            "RP_SPACE_LENGTH": _cfg_attr(config, "RP_SPACE_LENGTH", cfg.RP_SPACE_LENGTH),
            "RP_SPACE_DIM": _cfg_attr(config, "RP_SPACE_DIM", cfg.RP_SPACE_DIM),
            "INSERT_METHOD": _cfg_attr(config, "INSERT_METHOD", cfg.INSERT_METHOD),
            "GATING_MID_DIM": _cfg_attr(config, "GATING_MID_DIM", cfg.GATING_MID_DIM),
            "stretching_length": _cfg_attr(config, "stretching_length", cfg.stretching_length),
            "gating_temperature": _cfg_attr(config, "gating_temperature", cfg.gating_temperature),
            "text_gating_epsilon": _cfg_attr(config, "text_gating_epsilon", cfg.text_gating_epsilon),
            "insert_method": _cfg_attr(config, "INSERT_METHOD", cfg.INSERT_METHOD),
            "TEXT_GATE_SELECTION_MODE": _cfg_attr(config, "TEXT_GATE_SELECTION_MODE", "group_threshold_prior"),
            "TEXT_GATE_GROUP_COUNT": _cfg_attr(config, "TEXT_GATE_GROUP_COUNT", 8),
            "TOP4_GROUP_COUNT": _cfg_attr(config, "TOP4_GROUP_COUNT", 4),
            "ACTIVE_REP_TOKEN_COUNT": _cfg_attr(config, "ACTIVE_REP_TOKEN_COUNT", cfg.ACTIVE_REP_TOKEN_COUNT),
            "ABLATE_VISUAL_GATE": _cfg_attr(config, "ABLATE_VISUAL_GATE", False),
            "ABLATE_DIRECT_LEARNABLE_REP": _cfg_attr(config, "ABLATE_DIRECT_LEARNABLE_REP", False),
            "DISABLE_TEXT_GATE": _cfg_attr(config, "DISABLE_TEXT_GATE", False),
            "DISABLE_TEXT_PROMPT_INSERT": _cfg_attr(config, "DISABLE_TEXT_PROMPT_INSERT", False),
            "MASK_REP_PLACEHOLDERS_WHEN_TEXT_DISABLED": _cfg_attr(config, "MASK_REP_PLACEHOLDERS_WHEN_TEXT_DISABLED", True),
            "VISUAL_RESIDUAL_ADAPTER_COUNT": _cfg_attr(config, "VISUAL_RESIDUAL_ADAPTER_COUNT", 4),
            "TEXT_ADAPTER_TOKEN_COUNT": _cfg_attr(config, "TEXT_ADAPTER_TOKEN_COUNT", 5),
            "TEXT_RESIDUAL_ADAPTER_COUNT": _cfg_attr(config, "TEXT_RESIDUAL_ADAPTER_COUNT", 4),
            "TEXT_COMMON_MODE_LOSS_WEIGHT": _cfg_attr(config, "TEXT_COMMON_MODE_LOSS_WEIGHT", 0.0),
            "TEXT_COMMON_MODE_TARGET": _cfg_attr(config, "TEXT_COMMON_MODE_TARGET", 0.85),
            "TEXT_ADAPTER_BALANCE_LOSS_WEIGHT": _cfg_attr(config, "TEXT_ADAPTER_BALANCE_LOSS_WEIGHT", 0.0),
            "TEXT_ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT": _cfg_attr(config, "TEXT_ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT", 0.0),
            "TEXT_ADAPTER_SAMPLE_ENTROPY_TARGET": _cfg_attr(config, "TEXT_ADAPTER_SAMPLE_ENTROPY_TARGET", 1.0),
            "ADAPTER_USAGE_BALANCE_LOSS_WEIGHT": _cfg_attr(config, "ADAPTER_USAGE_BALANCE_LOSS_WEIGHT", 0.0),
            "ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT": _cfg_attr(config, "ADAPTER_SAMPLE_ENTROPY_LOSS_WEIGHT", 0.0),
            "ADAPTER_COMMON_MODE_LOSS_WEIGHT": _cfg_attr(config, "ADAPTER_COMMON_MODE_LOSS_WEIGHT", 0.0),
            "ADAPTER_SAMPLE_ENTROPY_TARGET": _cfg_attr(config, "ADAPTER_SAMPLE_ENTROPY_TARGET", 0.40),
            "ADAPTER_COMMON_MODE_TARGET": _cfg_attr(config, "ADAPTER_COMMON_MODE_TARGET", 0.85),
            "vision_token_dim": vision_dim,
            "text_token_dim": text_dim
        }

        super().__init__(config)
        if tokenizer is not None:
            rep_token_count = int(config.mmrl_config.get("ACTIVE_REP_TOKEN_COUNT", cfg.ACTIVE_REP_TOKEN_COUNT))
            self.vision_end_token_id = (
                tokenizer.vision_end_token_id
                if getattr(tokenizer, "vision_end_token_id", None)
                else tokenizer.convert_tokens_to_ids("<|vision_end|>")
            )
            self.rep_placeholder_ids = [
                tokenizer.convert_tokens_to_ids(f"<|REP_placeholder{i}|>") for i in range(rep_token_count)
            ]
        else:
            raise ValueError("tokenizer must be specified")
        ###################
        vision_config = getattr(config, "vision_config", config)
        if not hasattr(vision_config, "mmrl_config"):
            vision_config.mmrl_config = config.mmrl_config
        original_visual = self.visual
        self.visual = vmmrl.VisionWithMMRL(vision_config)
        self.visual.load_state_dict(original_visual.state_dict(), strict=False)
        del original_visual
        torch.cuda.empty_cache()
        self.MMRL = MMRL.MMRL(config)
        self.post_init()
        if tokenizer is not None:
            vocab_size = len(tokenizer)
            curr_embedding_size = self.get_input_embeddings().weight.shape[0]
            print(f"[Info] tokenizer={vocab_size}, embedding={curr_embedding_size}")
        self.tokenizer = tokenizer
        self.use_mmrl = config.mmrl_config["USE_MMRL"]
        self.tax_loss = None
        self.capacity_prior_loss = None
        self.top4_group_balance_loss = None
        self.text_common_mode_loss = None
        self.text_common_mode_ratio_train = None
        self.text_adapter_balance_loss = None
        self.text_adapter_sample_entropy_loss = None
        self.temperature_override = None
        self.k_results = None
        self.disable_text_prompt_insert = bool(config.mmrl_config.get("DISABLE_TEXT_PROMPT_INSERT", False))
        self.mask_rep_placeholders_when_text_disabled = bool(config.mmrl_config.get("MASK_REP_PLACEHOLDERS_WHEN_TEXT_DISABLED", True))
        self.debug_context = {}
        self.text_rep_scale = nn.Parameter(torch.tensor(float(_cfg_attr(config, "TEXT_REP_INIT_SCALE", 1e-3))))
        self.text_adapter_token_count = int(config.mmrl_config.get("TEXT_ADAPTER_TOKEN_COUNT", 5))
        self.text_residual_adapter_count = int(max(config.mmrl_config.get("TEXT_RESIDUAL_ADAPTER_COUNT", 4), 1))
        self.text_adapter_router = TextAdapterRouter(
            text_dim,
            vision_dim,
            int(config.mmrl_config.get("GATING_MID_DIM", cfg.GATING_MID_DIM)),
            self.text_residual_adapter_count,
        )

        self.text_residual_adapters = nn.ModuleList([
            TextResidualAdapter(text_dim) for _ in range(self.text_residual_adapter_count)
        ])
        ###################

    def _text_insert_debug_defaults(self, device, dtype):
        nan = torch.tensor(float("nan"), device=device, dtype=dtype)
        return {
            "text_insert_delta_norm_mean": nan,
            "text_insert_delta_norm_p95": nan,
            "text_insert_to_base_embed_ratio": nan,
            "text_insert_pool_common_mode_ratio": nan,
            "text_insert_pool_specificity_ratio": nan,
            "text_insert_pool_pairwise_cos_mean": nan,
            "text_rep_scale_abs": self.text_rep_scale.detach().abs().to(device=device, dtype=dtype),
        }

    def _compute_text_common_mode_loss(
        self,
        text_insert_delta: torch.Tensor,
        is_placeholder: torch.Tensor,
        target: float = 0.85,
    ):
        zero = text_insert_delta.new_tensor(0.0)
        ratio_zero = text_insert_delta.new_tensor(0.0)
        if text_insert_delta.numel() == 0 or not is_placeholder.any():
            return zero, ratio_zero

        sample_mask = is_placeholder.to(device=text_insert_delta.device, dtype=text_insert_delta.dtype)
        sample_count = sample_mask.sum(dim=1, keepdim=True)
        valid_samples = sample_count.squeeze(-1) > 0
        if not valid_samples.any():
            return zero, ratio_zero

        pooled_delta = (text_insert_delta.float() * sample_mask.unsqueeze(-1).float()).sum(dim=1)
        pooled_delta = pooled_delta / sample_count.clamp_min(1.0).float()
        pooled_delta = pooled_delta[valid_samples]
        if pooled_delta.shape[0] < 2:
            return zero, ratio_zero

        pooled_norm = pooled_delta.norm(dim=-1)
        valid_pool = pooled_norm > 1e-8
        if valid_pool.sum() < 2:
            return zero, ratio_zero

        pooled_delta = pooled_delta[valid_pool]
        pooled_norm = pooled_norm[valid_pool]
        common_ratio = pooled_delta.mean(dim=0).norm() / pooled_norm.mean().clamp_min(1e-8)
        loss = torch.relu(common_ratio - float(target)).pow(2)
        return loss.to(dtype=text_insert_delta.dtype), common_ratio.to(dtype=text_insert_delta.dtype)

    def _compute_text_adapter_aux_losses(
        self,
        route_probs: torch.Tensor,
        entropy_target: float = 0.55,
    ):
        zero = route_probs.new_tensor(0.0)
        if route_probs.numel() == 0 or route_probs.shape[-1] <= 1:
            return zero, zero

        probs = route_probs.float().clamp_min(1e-8)
        usage = probs.mean(dim=0)
        uniform = torch.full_like(usage, 1.0 / usage.numel())
        balance_loss = (usage - uniform).pow(2).mean()

        entropy = -(probs * probs.log()).sum(dim=-1)
        entropy_norm = entropy / torch.log(probs.new_tensor(float(probs.shape[-1])))
        entropy_loss = (entropy_norm.mean() - float(entropy_target)).pow(2)
        return balance_loss.to(dtype=route_probs.dtype), entropy_loss.to(dtype=route_probs.dtype)

    def _compute_text_insert_debug_metrics(
        self,
        base_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        gate_soft_mask: torch.Tensor,
        is_placeholder: torch.Tensor,
    ):
        metrics = self._text_insert_debug_defaults(base_embeds.device, base_embeds.dtype)
        with torch.no_grad():
            if base_embeds.numel() == 0 or target_embeds.numel() == 0:
                return metrics

            placeholder_mask = is_placeholder.unsqueeze(-1)
            text_insert_delta = target_embeds * gate_soft_mask
            placeholder_delta = text_insert_delta[placeholder_mask.expand_as(text_insert_delta)].view(-1, text_insert_delta.shape[-1]) if is_placeholder.any() else None
            placeholder_base = base_embeds[placeholder_mask.expand_as(base_embeds)].view(-1, base_embeds.shape[-1]) if is_placeholder.any() else None

            if placeholder_delta is None or placeholder_delta.numel() == 0:
                return metrics

            delta_norm = placeholder_delta.detach().float().norm(dim=-1)
            base_norm = placeholder_base.detach().float().norm(dim=-1)
            metrics["text_insert_delta_norm_mean"] = delta_norm.mean().to(device=base_embeds.device, dtype=base_embeds.dtype)
            metrics["text_insert_delta_norm_p95"] = torch.quantile(delta_norm, 0.95).to(device=base_embeds.device, dtype=base_embeds.dtype)
            metrics["text_insert_to_base_embed_ratio"] = (
                delta_norm.mean() / base_norm.mean().clamp_min(1e-8)
            ).to(device=base_embeds.device, dtype=base_embeds.dtype)

            sample_mask = is_placeholder.detach().float()
            sample_count = sample_mask.sum(dim=1, keepdim=True)
            valid_samples = sample_count.squeeze(-1) > 0
            if valid_samples.any():
                pooled_delta = (text_insert_delta.detach().float() * sample_mask.unsqueeze(-1)).sum(dim=1)
                pooled_delta = pooled_delta / sample_count.clamp_min(1.0).unsqueeze(-1)
                pooled_delta = pooled_delta[valid_samples]
                pooled_norm = pooled_delta.norm(dim=-1)
                valid_pool = pooled_norm > 1e-8
                if valid_pool.any():
                    pooled_delta = pooled_delta[valid_pool]
                    pooled_norm = pooled_norm[valid_pool]
                    pooled_mean_norm = pooled_norm.mean().clamp_min(1e-8)
                    pooled_common = pooled_delta.mean(dim=0)
                    pooled_specific = pooled_delta - pooled_common
                    metrics["text_insert_pool_common_mode_ratio"] = (
                        pooled_common.norm() / pooled_mean_norm
                    ).to(device=base_embeds.device, dtype=base_embeds.dtype)
                    metrics["text_insert_pool_specificity_ratio"] = (
                        pooled_specific.norm(dim=-1).mean() / pooled_mean_norm
                    ).to(device=base_embeds.device, dtype=base_embeds.dtype)

                    if pooled_delta.shape[0] >= 2:
                        pooled_unit = pooled_delta / pooled_delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        pairwise = pooled_unit @ pooled_unit.transpose(0, 1)
                        tri = torch.triu_indices(pairwise.shape[0], pairwise.shape[1], offset=1, device=pairwise.device)
                        pairwise_vals = pairwise[tri[0], tri[1]]
                        if pairwise_vals.numel() > 0:
                            metrics["text_insert_pool_pairwise_cos_mean"] = pairwise_vals.mean().to(
                                device=base_embeds.device, dtype=base_embeds.dtype
                            )

        return metrics

    def _pool_text_for_adapter(self, embedding: torch.Tensor, mask: Optional[torch.Tensor]):
        if torch.is_tensor(mask):
            m = mask.to(device=embedding.device, dtype=embedding.dtype).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            return (embedding * m).sum(dim=1) / denom
        return embedding.mean(dim=1)
    def _batch_visual_alpha(self, batch_size: int, images_per_sample, device, dtype):
        alpha = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        alpha_list = getattr(self.visual, "alpha_list", None)
        if not torch.is_tensor(alpha_list):
            return alpha

        alpha_prob = torch.sigmoid(alpha_list.detach().to(device=device, dtype=dtype)).view(-1, 1)
        if images_per_sample is None:
            n = min(batch_size, alpha_prob.shape[0])
            alpha[:n] = alpha_prob[:n]
            return alpha

        cursor = 0
        for sample_idx, count in enumerate(images_per_sample):
            count = int(count)
            if count > 0:
                alpha[sample_idx] = alpha_prob[cursor:cursor + count].mean(dim=0)
            cursor += count
        return alpha

    # def _batch_visual_state(self, batch_size: int, images_per_sample, device, dtype):
    #     alpha = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    #     gate = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    #     alpha_list = getattr(self.visual, "alpha_list", None)
    #     gate_list = getattr(self.visual, "G_list", None)
    #     if not torch.is_tensor(alpha_list):
    #         return torch.cat([alpha, gate], dim=-1)
    #     alpha_prob = torch.sigmoid(alpha_list.detach().to(device=device, dtype=dtype)).view(-1, 1)
    #     gate_prob = gate_list.detach().to(device=device, dtype=dtype).view(-1, 1) if torch.is_tensor(gate_list) else alpha_prob.new_zeros(alpha_prob.shape)
    #     if images_per_sample is None:
    #         n = min(batch_size, alpha_prob.shape[0])
    #         alpha[:n] = alpha_prob[:n]
    #         gate[:n] = gate_prob[:n]
    #         return torch.cat([alpha, gate], dim=-1)
    #     cursor = 0
    #     for sample_idx, count in enumerate(images_per_sample):
    #         count = int(count)
    #         if count > 0:
    #             alpha[sample_idx] = alpha_prob[cursor:cursor + count].mean(dim=0)
    #             gate[sample_idx] = gate_prob[cursor:cursor + count].mean(dim=0)
    #         cursor += count
    #     return torch.cat([alpha, gate], dim=-1)

    def _batch_visual_gate_soft(self, batch_size: int, images_per_sample, device, dtype):
        gate = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        gate_list = getattr(self.visual, "G_list", None)
        if not torch.is_tensor(gate_list):
            return gate
        gate_prob = gate_list.to(device=device, dtype=dtype).view(-1, 1)
        if images_per_sample is None:
            n = min(batch_size, gate_prob.shape[0])
            gate[:n] = gate_prob[:n]
            return gate
        cursor = 0
        for sample_idx, count in enumerate(images_per_sample):
            count = int(count)
            if count > 0:
                gate[sample_idx] = gate_prob[cursor:cursor + count].mean(dim=0)
            cursor += count
        return gate

    def _apply_text_adapter_router(
        self,
        text_tokens: torch.Tensor,
        text_pooled: torch.Tensor,
        visual_pooled: torch.Tensor,
        alpha_state: torch.Tensor,
    ):
        route_probs = self.text_adapter_router(text_pooled, visual_pooled, alpha_state)
        adapter_outputs = torch.stack([adapter(text_tokens) for adapter in self.text_residual_adapters], dim=2)
        mixed_delta = (adapter_outputs * route_probs[:, None, :, None].to(dtype=text_tokens.dtype)).sum(dim=2)
        expert_tokens = text_tokens + mixed_delta
        return expert_tokens, route_probs, mixed_delta



    def get_image_features(self,
                        pixel_values: torch.FloatTensor,
                        image_grid_thw: Optional[torch.LongTensor] = None,
                        v_r_token_list: Optional[list[torch.Tensor]] = None,
                        embedding: Optional[torch.nn.Module] = None,
                        images_per_sample: Optional[list[int]] = None,
                        text_pooling_mask: Optional[torch.Tensor] = None,):
        if self.use_mmrl and v_r_token_list is None:
            raise ValueError("v_r_token_list must be specified")
        elif self.use_mmrl and v_r_token_list is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, deepstack_image_embeds, k = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                v_r_token_list=v_r_token_list,
                embedding=embedding,
                text_pooling_mask=text_pooling_mask,
                gating_temperature_override=self.temperature_override,
                images_per_sample=images_per_sample,
            )
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
            image_embeds = torch.split(image_embeds, split_sizes)
            return image_embeds, deepstack_image_embeds, k
        elif not self.use_mmrl:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, deepstack_image_embeds = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                text_pooling_mask=text_pooling_mask
            )
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
            image_embeds = torch.split(image_embeds, split_sizes)
            return image_embeds, deepstack_image_embeds
        else:
            return None

    @auto_docstring
    @check_model_inputs()
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            img_path: Optional[list[str]] = None,
            images_per_sample=None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        temp_from_kwargs = kwargs.pop("gating_temperature_override", None)
        if kwargs.pop("gating_temperature_overied", None) is not None:
            raise RuntimeError
        if temp_from_kwargs is not None:
            self.temperature_override = temp_from_kwargs
        
        mmrl_gating_mask = kwargs.pop("mmrl_gating_mask", None)
        self.tax_loss = torch.tensor(0.0, device=inputs_embeds.device)
        self.capacity_prior_loss = torch.tensor(0.0, device=inputs_embeds.device)
        self.top4_group_balance_loss = torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        self.text_common_mode_loss = torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        self.text_common_mode_ratio_train = torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        self.text_adapter_balance_loss = torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        self.text_adapter_sample_entropy_loss = torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        self.debug_context = {}
        if self.use_mmrl and input_ids is None:
            raise ValueError("MMRL path currently requires input_ids for placeholder detection.")
        placeholder_ids_tensor = torch.tensor(self.rep_placeholder_ids, device=inputs_embeds.device)
        is_placeholder = (input_ids.unsqueeze(-1) == placeholder_ids_tensor).any(dim=-1)
        # ---- 构造显式 text pooling mask ----
        base_text_mask = None
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            base_text_mask = attention_mask.bool()
        elif isinstance(attention_mask, dict):
            full_attention = attention_mask.get("full_attention", None)
            if torch.is_tensor(full_attention) and full_attention.ndim == 2:
                base_text_mask = full_attention.bool()
        if base_text_mask is None:
            text_pooling_mask = torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=torch.bool
            )
        else:
            text_pooling_mask = base_text_mask.to(inputs_embeds.device)
        if mmrl_gating_mask is not None:
            if mmrl_gating_mask.dim() == 3 and mmrl_gating_mask.size(-1) == 1:
                mmrl_gating_mask = mmrl_gating_mask.squeeze(-1)
            if mmrl_gating_mask.dim() != 2:
                raise ValueError("mmrl_gating_mask must be [B, S] or [B, S, 1]")
            text_pooling_mask = text_pooling_mask & mmrl_gating_mask.to(inputs_embeds.device).bool()
        # placeholder 不能参与 text gating pooling
        text_pooling_mask = text_pooling_mask & (~is_placeholder)
        embedding_for_gating = inputs_embeds
        
        v_r_token_list = None
        t_r_tokens = None
        if self.use_mmrl:
            v_r_token_list, t_r_token_list = self.MMRL()
            t_r_tokens = torch.cat(t_r_token_list, dim=0)
        visual_pos_masks = None
        deepstack_visual_embeds = None
        k_results = None
        has_images = pixel_values is not None and image_grid_thw is not None and image_grid_thw.numel() > 0
        
        if has_images:
            # 仅在有图像时计算 images_per_sample
            if images_per_sample is None:
                images_per_sample = []
                for seq_input_ids in input_ids:
                    count = (seq_input_ids == self.vision_end_token_id).sum().item()
                    images_per_sample.append(count)
            assert sum(images_per_sample) == image_grid_thw.shape[0], \
                f"but got sum {sum(images_per_sample)} and image_grid_thw.shape[0] {image_grid_thw.shape[0]}"
            
            if self.use_mmrl:
                image_embeds_raw, deepstack_image_embeds, k_results = self.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    v_r_token_list=v_r_token_list,
                    embedding=embedding_for_gating,
                    images_per_sample=images_per_sample,
                    text_pooling_mask=text_pooling_mask,
                )
            else:
                image_embeds_raw, deepstack_image_embeds = self.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    v_r_token_list=v_r_token_list,
                    embedding=inputs_embeds,
                    images_per_sample=images_per_sample
                )
            image_embeds = torch.cat(image_embeds_raw, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif self.use_mmrl:
            k_results = self.visual.compute_text_only_gating(
                embedding=embedding_for_gating,
                text_pooling_mask=text_pooling_mask,
                gating_temperature_override=self.temperature_override,
            )
        if self.use_mmrl:
            self.capacity_prior_loss = torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            self.tax_loss = self.capacity_prior_loss
        ######## text gating ########
        if self.use_mmrl and k_results is not None:
            k_sums = k_results
            self.k_results = k_sums
            placeholder_cumsum = is_placeholder.cumsum(dim=-1)
            placeholder_idx = (placeholder_cumsum - 1).clamp(min=0)
            self.debug_context.update(self._text_insert_debug_defaults(inputs_embeds.device, inputs_embeds.dtype))

            if self.disable_text_prompt_insert:
                if self.mask_rep_placeholders_when_text_disabled:
                    attention_mask = attention_mask & (~is_placeholder)
                    inputs_embeds = inputs_embeds.masked_fill(is_placeholder.unsqueeze(-1), 0.0)
                rep_attn = attention_mask[is_placeholder].detach().float().mean() if is_placeholder.any() else inputs_embeds.new_tensor(0.0)
                self.debug_context.update({
                    "text_prompt_insert_disabled_flag": inputs_embeds.new_tensor(1.0),
                    "mask_rep_placeholders_when_text_disabled_flag": inputs_embeds.new_tensor(1.0 if self.mask_rep_placeholders_when_text_disabled else 0.0),
                    "rep_placeholder_attention_ratio": rep_attn.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                })
            else:
                placeholder_base_embeds = inputs_embeds.detach().clone()
                token_count = min(int(self.text_adapter_token_count), int(t_r_tokens.shape[0]))
                selected_t_tokens = t_r_tokens[:token_count].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                selected_t_tokens = selected_t_tokens.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1)
                text_pooled_for_adapter = self._pool_text_for_adapter(embedding_for_gating, text_pooling_mask)

                visual_pooled_for_adapter = getattr(self.visual, "text_router_visual_pooled", None)
                if visual_pooled_for_adapter is None:
                    visual_pooled_for_adapter = torch.zeros(
                        inputs_embeds.shape[0],
                        self.config.mmrl_config["vision_token_dim"],
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                else:
                    visual_pooled_for_adapter = visual_pooled_for_adapter.to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )

                alpha_state = self._batch_visual_alpha(
                    inputs_embeds.shape[0],
                    images_per_sample,
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                )

                selected_t_tokens, text_route_probs, text_mixed_delta = self._apply_text_adapter_router(
                    selected_t_tokens,
                    text_pooled_for_adapter,
                    visual_pooled_for_adapter,
                    alpha_state,
                )

                self.text_adapter_balance_loss, self.text_adapter_sample_entropy_loss = self._compute_text_adapter_aux_losses(
                    text_route_probs,
                    entropy_target=float(self.config.mmrl_config.get("TEXT_ADAPTER_SAMPLE_ENTROPY_TARGET", 1.0)),
                )
                text_gate_soft = self._batch_visual_gate_soft(
                    inputs_embeds.shape[0],
                    images_per_sample,
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                )
                sample_active_mask = (text_gate_soft > 0.5).squeeze(-1)
                selected_count = selected_t_tokens.shape[1]
                gather_idx = placeholder_idx.clamp(max=selected_count - 1).unsqueeze(-1).expand(-1, -1, selected_t_tokens.shape[-1])
                target_embeds = selected_t_tokens.gather(1, gather_idx)
                active_placeholder_mask = is_placeholder & sample_active_mask.unsqueeze(-1)
                inactive_placeholder_mask = is_placeholder & (~sample_active_mask.unsqueeze(-1))
                text_insert_mask = active_placeholder_mask.unsqueeze(-1).to(inputs_embeds.dtype)
                text_insert_delta = target_embeds * text_insert_mask
                self.text_common_mode_loss, self.text_common_mode_ratio_train = self._compute_text_common_mode_loss(
                    text_insert_delta=text_insert_delta,
                    is_placeholder=active_placeholder_mask,
                    target=float(self.config.mmrl_config.get("TEXT_COMMON_MODE_TARGET", 0.85)),
                )
                text_insert_debug = self._compute_text_insert_debug_metrics(
                    base_embeds=placeholder_base_embeds,
                    target_embeds=target_embeds,
                    gate_soft_mask=text_insert_mask,
                    is_placeholder=is_placeholder,
                )
                self.debug_context.update(text_insert_debug)
                inputs_embeds = torch.where(active_placeholder_mask.unsqueeze(-1), target_embeds, inputs_embeds)
                inputs_embeds = inputs_embeds.masked_fill(inactive_placeholder_mask.unsqueeze(-1), 0.0)
                attention_mask = attention_mask & (~inactive_placeholder_mask)
                attention_mask = attention_mask | active_placeholder_mask.to(dtype=attention_mask.dtype)
                rep_attn = attention_mask[is_placeholder].detach().float().mean() if is_placeholder.any() else inputs_embeds.new_tensor(0.0)
                per_sample_insert_count = active_placeholder_mask.detach().float().sum(dim=-1)
                text_usage = text_route_probs.detach().float().mean(dim=0)
                text_usage = text_usage / text_usage.sum().clamp_min(1e-8)
                text_route_entropy = -(text_usage * text_usage.clamp_min(1e-8).log()).sum()
                if text_usage.numel() > 1:
                    text_route_entropy_norm = text_route_entropy / torch.log(text_usage.new_tensor(float(text_usage.numel())))
                else:
                    text_route_entropy_norm = text_usage.new_tensor(0.0)
                self.debug_context.update({
                    "text_prompt_insert_disabled_flag": inputs_embeds.new_tensor(0.0),
                    "mask_rep_placeholders_when_text_disabled_flag": inputs_embeds.new_tensor(0.0),
                    "rep_placeholder_attention_ratio": rep_attn.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_insert_token_count_mean": per_sample_insert_count.mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_active_sample_ratio": sample_active_mask.detach().float().mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_inactive_placeholder_ratio": inactive_placeholder_mask.detach().float().mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_token_count": inputs_embeds.new_tensor(float(selected_count)),
                    "text_adapter_count": inputs_embeds.new_tensor(float(self.text_residual_adapter_count)),
                    "text_adapter_route_entropy_norm": text_route_entropy_norm.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_route_confidence": text_route_probs.detach().float().max(dim=-1).values.mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_usage_max": text_usage.max().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_usage_min": text_usage.min().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_mixed_delta_norm_mean": text_mixed_delta.detach().float().norm(dim=-1).mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_balance_loss": self.text_adapter_balance_loss.detach().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_sample_entropy_loss": self.text_adapter_sample_entropy_loss.detach().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_adapter_sample_entropy_target": inputs_embeds.new_tensor(float(self.config.mmrl_config.get("TEXT_ADAPTER_SAMPLE_ENTROPY_TARGET", 1.0))),
                    "text_common_mode_loss_raw": self.text_common_mode_loss.detach().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_common_mode_ratio_train": self.text_common_mode_ratio_train.detach().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    "text_common_mode_target": inputs_embeds.new_tensor(float(self.config.mmrl_config.get("TEXT_COMMON_MODE_TARGET", 0.85))),
                })
        ######## text gating ########
        
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                    (input_ids is not None and input_ids.shape[1] != 1)
                    or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                    (cache_position is not None and cache_position[0] == 0)
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                # ========== 修复：纯文本时 image_grid_thw 传 None ==========
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw if has_images else None,
                    None,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )