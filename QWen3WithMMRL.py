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
        super().__init__(config)
        if tokenizer is not None:
            self.vision_end_token_id = (
                tokenizer.vision_end_token_id
                if getattr(tokenizer, "vision_end_token_id", None)
                else tokenizer.convert_tokens_to_ids(self.vision_end_token)
            )
            self.rep_placeholder_ids = [
                tokenizer.convert_tokens_to_ids(f"<|REP_placeholder{i}|>") for i in range(40)
            ]
        else:
            raise ValueError("tokenizer must be specified")
        ###################
        self.MMRL = MMRL.MMRL()
        self.use_mmrl = cfg.USE_MMRL
        self.visual = vmmrl.VisionWithMMRL(config, self.use_mmrl)
        self.tax_loss = None
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
                                                               images_per_sample=images_per_sample)
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
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        #### MMRL ####
        v_r_token_list = None
        t_r_token_list = None
        k_remainder = None
        if pixel_values is not None and self.use_mmrl:
            v_r_token_list, t_r_token_list = self.MMRL()
        #### MMRL ####
        images_per_sample = []
        if input_ids is not None:
            # 遍历 Batch 中的每一条数据
            for seq_input_ids in input_ids:
                # 统计等于 image_token_id 的数量
                count = (seq_input_ids == self.config.image_token_id).sum().item()
                images_per_sample.append(count)
        # todo：优化无mmrl流程
        # todo：假设只有文字没有图片？
        # todo：多轮对话逻辑？
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            if self.use_mmrl:
                image_embeds_raw, deepstack_image_embeds, k = self.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    v_r_token_list=v_r_token_list,
                    embedding = inputs_embeds,
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
            # todo：适配batch维度
            if self.training:
                k, k_remainder, tax_loss = k
                self.tax_loss = tax_loss
            else:
                k = k
            assert k is not None, "k is None"
            if self.training:
                real_rep_num = k + 1 if k < 40 else k
            else:
                real_rep_num = k
            gate_mask = torch.cat([torch.ones([real_rep_num]),
                                   torch.zeros([40 - real_rep_num])]).to(dtype=inputs_embeds.dtype)
            inputs_embeds[:, :40, :] = t_r_token_list * gate_mask
            attention_mask[:, :40] = gate_mask.to(dtype=attention_mask.dtype)
            if self.training:
                inputs_embeds = inputs_embeds[-1] * k_remainder
            inputs_embeds = inputs_embeds.masked_scatter(rep_ph_mask, rep_embeds)

            image_embeds = torch.cat(image_embeds_raw, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
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