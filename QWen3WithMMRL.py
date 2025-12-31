from typing import Optional, Union

import torch
from transformers import Cache, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs, TransformersKwargs

import MMRL
import config as cfg
import Vpatch

from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

class QWen3WithMMRL(qwen3_vl.Qwen3VLModel):
    def __init__(self,
                 config,
                 MMRL_mode: Optional[str] = None,
                 precomputed_path: Optional[str] = None,
                 tokenizer = None,
                 ):
        super().__init__(config)
        if MMRL_mode is None:
            raise ValueError("mode must be specified")
        self.MMRL_mode = MMRL_mode
        self.precomputed_path = precomputed_path
        if tokenizer is not None:
            self.mmrl_token_id = tokenizer.convert_tokens_to_ids("<|text_R_token_placeholder|>")
        else:
            raise ValueError("tokenizer must be specified")
        self.MMRL = MMRL.MMRL(insert_layer_num=len(config.INSERT_LAYER),
                              vision_token_dim=cfg.vision_token_dim,
                              text_token_dim=cfg.text_token_dim,
                              mode=self.MMRL_mode,
                              precomputed_path=self.precomputed_path)
        self.Vpatch = Vpatch.Vpatch()

    def get_image_features(self,
                           pixel_values: torch.FloatTensor,
                           image_grid_thw: Optional[torch.LongTensor] = None,
                           v_r_token_list: Optional[list[torch.Tensor]] = None):
        if v_r_token_list is None:
            raise ValueError("v_r_token_list must be specified")
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values,
                                                           grid_thw=image_grid_thw,
                                                           v_r_token_list=v_r_token_list)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def _prepare_mmrl_embeddings(self,
                                 input_ids:torch.Tensor,
                                 inputs_embeds:torch.Tensor):
        placeholder_id = self.mmrl_token_id
        mmrl_mask = (input_ids == placeholder_id)
        actual_count = mmrl_mask.sum().item()
        if actual_count == 0:
            return inputs_embeds, None
        v_r_token_list, t_r_token_list = self.MMRL()
        t_r_embeds = torch.cat(t_r_token_list, dim=0).to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        if actual_count % 40 != 0:
            raise ValueError(f"占位符数量异常: {actual_count}")
        inputs_embeds = inputs_embeds.masked_scatter(mmrl_mask.unsqueeze(-1), t_r_embeds)
        return inputs_embeds, v_r_token_list

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
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        #todo:在预处理阶段就进行是否为第一轮的判定
        #todo:解耦vpatch使其变成非必须调用。或者可以使其自动化检测是否为局部图？
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        #### MMRL ####
        placeholder_id = self.mmrl_token_id
        mmrl_mask = (input_ids == placeholder_id)
        has_mmrl_placeholders = mmrl_mask.any()
        v_r_token_list = None
        t_r_token_list = None
        if pixel_values is not None or has_mmrl_placeholders:
             v_r_token_list, t_r_token_list = self.MMRL()
        #### MMRL ####

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            #todo: 这里的v_r_token_list是可能会为空，需要处理
            image_embeds_raw, deepstack_image_embeds = self.get_image_features(pixel_values,
                                                                               image_grid_thw,
                                                                               v_r_token_list)
            image_embeds_flatten = torch.cat(image_embeds_raw, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            #todo:不一定会用到vpatch
            merged_hidden, merged_deepstack, new_grid_thw = self.vpatch(
                image_hidden_states=image_embeds_flatten,
                deepstack_feature_lists=deepstack_image_embeds,
                input_embeds=inputs_embeds,
                grid_thw=image_grid_thw,
                spatial_merge_size=self.visual.spatial_merge_size
            )

            original_vision_len = image_embeds_flatten.shape[0]
            new_vision_len = merged_hidden.shape[0]

            if original_vision_len != new_vision_len:
                num_drop = original_vision_len - new_vision_len
                image_mask_original, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_flatten
                )
                img_indices = torch.nonzero(image_mask_original.squeeze(-1), as_tuple=False)
                keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
                if num_drop > 0:
                    indices_to_drop = img_indices[-num_drop:]
                    keep_mask[indices_to_drop[:, 0], indices_to_drop[:, 1]] = False
                input_ids = input_ids[keep_mask].view(input_ids.shape[0], -1)
                mmrl_mask = mmrl_mask[keep_mask].view(input_ids.shape[0], -1)
                if attention_mask is not None:
                    attention_mask = attention_mask[keep_mask].view(attention_mask.shape[0], -1)
                inputs_embeds = inputs_embeds[keep_mask].view(input_ids.shape[0], -1, inputs_embeds.shape[-1])
                image_grid_thw = new_grid_thw

            image_embeds = merged_hidden.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_visual_embeds = merged_deepstack

            new_image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(new_image_mask, image_embeds)
            visual_pos_masks = new_image_mask[..., 0]

        #### MMRL ####
        if has_mmrl_placeholders and t_r_token_list is not None:
            t_r_embeds = torch.cat(t_r_token_list, dim=0).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            actual_count = mmrl_mask.sum().item()
            generated_count = t_r_embeds.shape[0]
            if generated_count != actual_count:
                raise ValueError(f"R-Token 数量不足: 生成 {generated_count} vs 坑位 {actual_count}")
            inputs_embeds = inputs_embeds.masked_scatter(mmrl_mask.unsqueeze(-1), t_r_embeds)
        #### MMRL ####
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