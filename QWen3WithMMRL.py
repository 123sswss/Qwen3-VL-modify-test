from typing import Optional, Union

import torch
from transformers import Cache, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs, TransformersKwargs

import MMRL
import config as cfg
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl
import VisionModelWithMMRL as vmmrl

class QWen3WithMMRL(qwen3_vl.Qwen3VLModel):
    def __init__(self,
                 config,
                 tokenizer=None,
                 ):
        # todo：研究一下处理训练保存与加载
        vision_dim = getattr(config.vision_config, "hidden_size")  # 默认值防报错
        text_dim = getattr(config.text_config, "hidden_size")
        if not hasattr(config, "mmrl_config"):
            config.mmrl_config = {
                "USE_MMRL": cfg.USE_MMRL,
                "INSERT_LAYER": list(cfg.INSERT_LAYER),
                "POOLING_DIM": cfg.POOLING_DIM,
                "RP_SPACE_LENGTH": cfg.RP_SPACE_LENGTH,
                "RP_SPACE_DIM": cfg.RP_SPACE_DIM,
                "INSERT_METHOD": cfg.INSERT_METHOD,
                "GATING_MID_DIM": cfg.GATING_MID_DIM,
                "stretching_length": cfg.stretching_length,
                "gating_temperature": cfg.gating_temperature,
                "text_gating_epsilon": cfg.text_gating_epsilon,
                "insert_method": cfg.INSERT_METHOD,
                "vision_token_dim": vision_dim,
                "text_token_dim": text_dim
            }

        super().__init__(config)
        if tokenizer is not None:
            self.vision_end_token_id = (
                tokenizer.vision_end_token_id
                if getattr(tokenizer, "vision_end_token_id", None)
                else tokenizer.convert_tokens_to_ids("<|vision_end|>")
            )
            self.rep_placeholder_ids = [
                tokenizer.convert_tokens_to_ids(f"<|REP_placeholder{i}|>") for i in range(40)
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
            if vocab_size != curr_embedding_size:
                print(f"[Warning] Resizing token embeddings from {curr_embedding_size} to {vocab_size}")
                self.resize_token_embeddings(vocab_size)

        self.use_mmrl = config.mmrl_config["USE_MMRL"]
        self.tax_loss = None
        self.temperature_override = None
        ###################

    def get_image_features(self,
                           pixel_values: torch.FloatTensor,
                           image_grid_thw: Optional[torch.LongTensor] = None,
                           v_r_token_list: Optional[list[torch.Tensor]] = None,
                           embedding: Optional[torch.nn.Module] = None,
                           images_per_sample: Optional[list[int]] = None,):
        if self.use_mmrl and v_r_token_list is None:
            raise ValueError("v_r_token_list must be specified")
        elif self.use_mmrl and v_r_token_list is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, deepstack_image_embeds, k = self.visual(pixel_values,
                                                                              grid_thw=image_grid_thw,
                                                                              v_r_token_list=v_r_token_list,
                                                                              embedding=embedding,
                                                                              gating_temperature_overied=self.temperature_override,
                                                                              images_per_sample=images_per_sample)
            # self.alpha_loss = alpha_loss
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
            image_embeds = torch.split(image_embeds, split_sizes)
            return image_embeds, deepstack_image_embeds, k
        elif not self.use_mmrl:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
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
        
        # 为门控准备无答案的embedding
        mmrl_gating_mask = kwargs.pop('mmrl_gating_mask', None)
        
        if self.training and mmrl_gating_mask is not None:
            if mmrl_gating_mask.dim() == 2:
                mask_expanded = mmrl_gating_mask.unsqueeze(-1)
            else:
                mask_expanded = mmrl_gating_mask
            embedding_for_gating = inputs_embeds * mask_expanded
        else:
            embedding_for_gating = inputs_embeds

        self.tax_loss = torch.tensor(0.0, device=inputs_embeds.device)
        # self.alpha_loss = torch.tensor(0.0, device=inputs_embeds.device)

        placeholder_ids_tensor = torch.tensor(self.rep_placeholder_ids, device=input_ids.device)
        # [Batch, Seq]
        is_placeholder = (input_ids.unsqueeze(-1) == placeholder_ids_tensor).any(dim=-1)
        #### MMRL ####
        v_r_token_list = None
        t_r_tokens = None
        if self.use_mmrl:
            v_r_token_list, t_r_token_list = self.MMRL()
            t_r_tokens = torch.cat(t_r_token_list, dim=0)
        self.tax_loss = 0.0
        # self.alpha_loss = 0.0
        #### MMRL ####
        images_per_sample = []
        if input_ids is not None:
            for seq_input_ids in input_ids:
                count = (seq_input_ids == self.vision_end_token_id).sum().item()
                images_per_sample.append(count)
        visual_pos_masks = None
        deepstack_visual_embeds = None
        k_results = None

        if images_per_sample is None and input_ids is not None:
            images_per_sample = []
            for seq_input_ids in input_ids:
                count = (seq_input_ids == self.vision_end_token_id).sum().item()
                images_per_sample.append(count)

        if pixel_values is not None:
            if self.use_mmrl:
                image_embeds_raw, deepstack_image_embeds, k_results = self.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    v_r_token_list=v_r_token_list,
                    embedding = embedding_for_gating,
                    images_per_sample = images_per_sample
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
                embedding=embedding_for_gating
            )
            # self.alpha_loss = alpha_loss
        ######## text gating ########
        if self.use_mmrl and k_results is not None:
            if self.training:
                k_sums, tax_loss = k_results #todo:删掉tax loss
                self.tax_loss = tax_loss
            else:
                k_sums = k_results

            placeholder_cumsum = is_placeholder.cumsum(dim=-1)
            placeholder_idx = (placeholder_cumsum - 1).clamp(min=0)
            target_embeds = t_r_tokens[placeholder_idx]
            gate_soft_mask = torch.clamp(k_sums.unsqueeze(-1) - placeholder_idx.to(inputs_embeds.dtype), min=0,
                                         max=1).unsqueeze(-1)
            gate_soft_mask = gate_soft_mask.to(inputs_embeds.dtype)
            inputs_embeds = torch.where(is_placeholder.unsqueeze(-1), target_embeds * gate_soft_mask, inputs_embeds)
            dynamic_gate_mask = (gate_soft_mask.squeeze(-1) > 1e-3)
            attention_mask = attention_mask & ((~is_placeholder) | dynamic_gate_mask)
            if not self.training:
                hard_gate = (gate_soft_mask.squeeze(-1) > 0.5)
                attention_mask = attention_mask & ((~is_placeholder) | hard_gate)
                embed_mask = torch.ones_like(attention_mask, dtype=inputs_embeds.dtype)
                tokens_to_zero = is_placeholder & (~hard_gate)
                embed_mask = embed_mask.masked_fill(tokens_to_zero, 0.0)
                inputs_embeds = inputs_embeds * embed_mask.unsqueeze(-1)
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
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
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

        # if self.use_mmrl:
        #     shift = is_placeholder.cumsum(dim=-1)  # [Batch, Seq]
        #     new_position_ids = position_ids.clone()
        #     new_position_ids[0] = position_ids[0] - shift
        #     position_ids = new_position_ids

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