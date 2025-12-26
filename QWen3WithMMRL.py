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
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None

        v_r_token_list, t_r_token_list = self.MMRL()

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values,
                                                                           image_grid_thw,
                                                                           v_r_token_list)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        t_r_embeds = torch.cat(t_r_token_list, dim=0).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        placeholder_id = self.mmrl_token_id
        mmrl_mask = (input_ids == placeholder_id)
        expected_count = input_ids.shape[0] * 40  # Batch_Size * 40
        actual_count = mmrl_mask.sum().item()

        if actual_count != expected_count:
            raise ValueError(
                f"占位符数量错误！期望 {expected_count} (Batch * 40), "
                f"实际 input_ids 中找到 {actual_count}。"
                f"请检查 Processor 是否正确插入了 40 个 <|text_R_token_placeholder|>。"
            )
        if t_r_embeds.shape[0] != actual_count:
            raise ValueError(f"生成的 R-Token 维度与占位符不匹配: {t_r_embeds.shape[0]} vs {actual_count}")
        inputs_embeds = inputs_embeds.masked_scatter(mmrl_mask.unsqueeze(-1), t_r_embeds)

        # 暂无支持视频推理的计划

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
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
                    # video_grid_thw,
                    None,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        #todo:插入vpatch
        # qwenvl的压缩比例 self.visual.spatial_merge_size
        # https://aistudio.google.com/prompts/1_PmmU9ZeCG9_dMmEtPtxiQkqqpqoTSqW

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