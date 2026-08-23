from typing import Optional, Union

import torch
from transformers import Cache
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
            "MMRL_MEMORY_QUERY_COUNT": _cfg_attr(
                config,
                "MMRL_MEMORY_QUERY_COUNT",
                cfg.MMRL_MEMORY_QUERY_COUNT,
            ),
            "MMRL_MEMORY_ATTENTION_DIM": _cfg_attr(
                config,
                "MMRL_MEMORY_ATTENTION_DIM",
                cfg.MMRL_MEMORY_ATTENTION_DIM,
            ),
            "MMRL_PROJECTOR_HIDDEN_DIM": _cfg_attr(
                config,
                "MMRL_PROJECTOR_HIDDEN_DIM",
                vision_dim,
            ),
            "MMRL_CROSS_ATTENTION_HEADS": _cfg_attr(
                config,
                "MMRL_CROSS_ATTENTION_HEADS",
                cfg.MMRL_CROSS_ATTENTION_HEADS,
            ),
            "INSERT_METHOD": _cfg_attr(config, "INSERT_METHOD", cfg.INSERT_METHOD),
            "GATING_MID_DIM": _cfg_attr(config, "GATING_MID_DIM", cfg.GATING_MID_DIM),
            "stretching_length": _cfg_attr(config, "stretching_length", cfg.stretching_length),
            "gating_temperature": _cfg_attr(config, "gating_temperature", cfg.gating_temperature),
            "insert_method": _cfg_attr(config, "INSERT_METHOD", cfg.INSERT_METHOD),
            "MMRL_SAME_INIT_LAYER_PROJECTORS": _cfg_attr(
                config,
                "MMRL_SAME_INIT_LAYER_PROJECTORS",
                cfg.MMRL_SAME_INIT_LAYER_PROJECTORS,
            ),
            "MMRL_RELATION_MAX_TOKENS": _cfg_attr(config, "MMRL_RELATION_MAX_TOKENS", 64),
            "MMRL_VARIANCE_FLOOR_RATIO": _cfg_attr(config, "MMRL_VARIANCE_FLOOR_RATIO", 0.50),
            "MMRL_VARIANCE_FLOOR_WEIGHT": _cfg_attr(config, "MMRL_VARIANCE_FLOOR_WEIGHT", 0.10),
            "vision_token_dim": vision_dim,
            "text_token_dim": text_dim
        }

        super().__init__(config)
        if tokenizer is not None:
            self.mmrl_visual_template_token_ids = {
                "image_pad": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                "vision_start": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                "vision_end": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            }
            invalid_visual_tokens = {
                name: token_id
                for name, token_id in self.mmrl_visual_template_token_ids.items()
                if token_id is None
                or token_id < 0
                or token_id == getattr(tokenizer, "unk_token_id", None)
            }
            if invalid_visual_tokens:
                raise RuntimeError(
                    "Failed to resolve visual template token IDs for MMRL text pooling: "
                    f"{invalid_visual_tokens}"
                )
            self.vision_end_token_id = self.mmrl_visual_template_token_ids["vision_end"]
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
        self.temperature_override = None
        self.debug_context = {}
        self._text_pooling_audit_contexts = set()
        ###################

    def build_mmrl_text_pooling_mask(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            mmrl_gating_mask: Optional[torch.Tensor] = None,
            audit_context: str = "model_forward",
    ) -> torch.Tensor:
        """Keep textual context only; visual placeholders carry no text semantics."""
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("MMRL text pooling requires input_ids with shape [B, S]")
        if attention_mask is None or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "MMRL text pooling requires attention_mask to match input_ids, "
                f"got input_ids={tuple(input_ids.shape)} "
                f"attention_mask={None if attention_mask is None else tuple(attention_mask.shape)}"
            )

        pooling_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if mmrl_gating_mask is not None:
            if mmrl_gating_mask.dim() == 3 and mmrl_gating_mask.size(-1) == 1:
                mmrl_gating_mask = mmrl_gating_mask.squeeze(-1)
            if mmrl_gating_mask.shape != input_ids.shape:
                raise ValueError(
                    "mmrl_gating_mask must match input_ids, "
                    f"got input_ids={tuple(input_ids.shape)} "
                    f"mmrl_gating_mask={tuple(mmrl_gating_mask.shape)}"
                )
            pooling_mask = pooling_mask & mmrl_gating_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )

        eligible_before_exclusion = pooling_mask.sum(dim=1)
        removed_counts = {}
        for name, token_id in self.mmrl_visual_template_token_ids.items():
            token_mask = input_ids.eq(token_id)
            removed_counts[name] = int((pooling_mask & token_mask).sum().item())
            pooling_mask = pooling_mask & ~token_mask

        remaining_per_sample = pooling_mask.sum(dim=1)
        if bool((remaining_per_sample == 0).any()):
            empty_indices = (remaining_per_sample == 0).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                "MMRL text pooling has no textual tokens after excluding visual "
                f"template tokens; empty sample indices={empty_indices}"
            )

        audit_kind = "visual" if sum(removed_counts.values()) > 0 else "text_only"
        audit_key = f"{audit_context}:{audit_kind}"
        if audit_key not in self._text_pooling_audit_contexts:
            print(
                "[MMRL_TEXT_POOLING_AUDIT] "
                f"context={audit_context} kind={audit_kind} batch={input_ids.shape[0]} "
                f"eligible_before={int(eligible_before_exclusion.sum().item())} "
                f"removed_image_pad={removed_counts['image_pad']} "
                f"removed_vision_start={removed_counts['vision_start']} "
                f"removed_vision_end={removed_counts['vision_end']} "
                f"text_after={int(remaining_per_sample.sum().item())} "
                f"text_per_sample_min={int(remaining_per_sample.min().item())} "
                f"text_per_sample_max={int(remaining_per_sample.max().item())}"
            )
            self._text_pooling_audit_contexts.add(audit_key)

        return pooling_mask

    def get_image_features(self,
                        pixel_values: torch.FloatTensor,
                        image_grid_thw: Optional[torch.LongTensor] = None,
                        rep_generator: Optional[torch.nn.Module] = None,
                        embedding: Optional[torch.Tensor] = None,
                        images_per_sample: Optional[list[int]] = None,
                        text_pooling_mask: Optional[torch.Tensor] = None,):
        if self.use_mmrl and rep_generator is None:
            raise ValueError("rep_generator must be specified")
        elif self.use_mmrl:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, deepstack_image_embeds = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                rep_generator=rep_generator,
                embedding=embedding,
                text_pooling_mask=text_pooling_mask,
                gating_temperature_override=self.temperature_override,
                images_per_sample=images_per_sample,
            )
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
            image_embeds = torch.split(image_embeds, split_sizes)
            return image_embeds, deepstack_image_embeds
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
        self.debug_context = {}
        
        visual_pos_masks = None
        deepstack_visual_embeds = None
        is_prefill = True
        if cache_position is not None and cache_position.numel() > 0:
            is_prefill = bool(cache_position[0].item() == 0)
        elif past_key_values is not None:
            is_prefill = past_key_values.get_seq_length() == 0
        decode_attention_mask = (
            attention_mask.get("full_attention", None)
            if isinstance(attention_mask, dict)
            else attention_mask
        )
        if (
            input_ids is not None
            and torch.is_tensor(decode_attention_mask)
            and decode_attention_mask.ndim == 2
            and decode_attention_mask.shape != input_ids.shape
        ):
            is_prefill = False
        has_images = (
            is_prefill
            and pixel_values is not None
            and image_grid_thw is not None
            and image_grid_thw.numel() > 0
        )
        
        if has_images:
            base_text_mask = None
            if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
                base_text_mask = attention_mask.bool()
            elif isinstance(attention_mask, dict):
                full_attention = attention_mask.get("full_attention", None)
                if torch.is_tensor(full_attention) and full_attention.ndim == 2:
                    base_text_mask = full_attention.bool()
            if base_text_mask is None:
                raise ValueError(
                    "MMRL text pooling requires a 2D attention mask or "
                    "attention_mask['full_attention'] during image prefill"
                )
            text_pooling_mask = self.build_mmrl_text_pooling_mask(
                input_ids=input_ids,
                attention_mask=base_text_mask.to(inputs_embeds.device),
                mmrl_gating_mask=mmrl_gating_mask,
                audit_context="model_forward",
            )

            # 仅在有图像时计算 images_per_sample
            if images_per_sample is None:
                images_per_sample = []
                for seq_input_ids in input_ids:
                    count = (seq_input_ids == self.vision_end_token_id).sum().item()
                    images_per_sample.append(count)
            assert sum(images_per_sample) == image_grid_thw.shape[0], \
                f"but got sum {sum(images_per_sample)} and image_grid_thw.shape[0] {image_grid_thw.shape[0]}"
            
            if self.use_mmrl:
                image_embeds_raw, deepstack_image_embeds = self.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    rep_generator=self.MMRL,
                    embedding=inputs_embeds,
                    images_per_sample=images_per_sample,
                    text_pooling_mask=text_pooling_mask,
                )
            else:
                image_embeds_raw, deepstack_image_embeds = self.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    embedding=inputs_embeds,
                    images_per_sample=images_per_sample,
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
