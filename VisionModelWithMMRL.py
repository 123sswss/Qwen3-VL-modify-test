from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jieba.lac_small.predict import batch_size

from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

import config as cfg
import MMRLGating
import utils

from itertools import accumulate


class MMRLVitBlock(qwen3_vl.Qwen3VLVisionBlock):
    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                r_token: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[torch.Tensor] = None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                first_insert: Optional[bool] = False,
                **kwargs):
        assert r_token is not None
        num_r_tokens = r_token.shape[0]
        full_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
        if first_insert:
            split_lengths = [l - num_r_tokens for l in full_seq_lengths]
            hidden_states_split = torch.split(hidden_states, split_lengths, dim=0)
            new_hidden_states_list = [torch.cat([r_token, s], dim=0) for s in hidden_states_split]
            hidden_states = torch.cat(new_hidden_states_list, dim=0)
        else:
            hidden_states_split = torch.split(hidden_states, full_seq_lengths, dim=0)
            new_hidden_states_list = []
            updated_seq = None
            for seq_hidden in hidden_states_split:
                if cfg.INSERT_METHOD == "replace":
                    updated_seq = torch.cat([r_token, seq_hidden[num_r_tokens:]], dim=0)
                elif cfg.INSERT_METHOD == "add":
                    prefix = seq_hidden[:num_r_tokens] + r_token
                    suffix = seq_hidden[num_r_tokens:]
                    updated_seq = torch.cat([prefix, suffix], dim=0)
                new_hidden_states_list.append(updated_seq)
            hidden_states = torch.cat(new_hidden_states_list, dim=0)

        hidden_states = hidden_states + self.attn(self.norm1(hidden_states),
                                                  cu_seqlens=cu_seqlens,
                                                  rotary_pos_emb=rotary_pos_emb,
                                                  position_embeddings=position_embeddings,
                                                  **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

def _resize_cu_and_pos(r_token, cu_seqlens, position_embeddings):
    num_r_tokens = r_token.shape[0]
    total_pic_num = cu_seqlens.shape[0] - 1
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()

    original_cos, original_sin = position_embeddings
    dtype = original_cos.dtype
    device = original_cos.device
    pos_emb_dim = original_cos.shape[-1]

    r_token_cos = torch.ones(num_r_tokens, pos_emb_dim, device=device, dtype=dtype)
    r_token_sin = torch.zeros(num_r_tokens, pos_emb_dim, device=device, dtype=dtype)

    cos_split = torch.split(original_cos, seq_lengths, dim=0)
    sin_split = torch.split(original_sin, seq_lengths, dim=0)

    new_cos_list = []
    new_sin_list = []

    for i in range(total_pic_num):
        new_cos_list.append(torch.cat([r_token_cos, cos_split[i]], dim=0))
        new_sin_list.append(torch.cat([r_token_sin, sin_split[i]], dim=0))

    new_cos = torch.cat(new_cos_list, dim=0)
    new_sin = torch.cat(new_sin_list, dim=0)
    updated_position_embeddings = (new_cos, new_sin)
    # todo:用极大位置编码替换0位置编码，或者做实验看结果

    if total_pic_num > 0:
        offsets = torch.arange(total_pic_num + 1, device=cu_seqlens.device,
                               dtype=cu_seqlens.dtype) * num_r_tokens
        updated_cu_seqlens = cu_seqlens + offsets
    else:
        updated_cu_seqlens = cu_seqlens

    return updated_position_embeddings, updated_cu_seqlens

def _strip_r_token(hidden_states, original_lens, num_r_token):
    current_lens = [l + num_r_token for l in original_lens]
    hidden_states_split = torch.split(hidden_states, current_lens, dim=0)
    clean_list = []
    for seq in hidden_states_split:
        clean_list.append(seq[num_r_token:])
    return torch.cat(clean_list, dim=0)

class zeroInit(nn.Module):
    def __init__(self, dim):
        super(zeroInit, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 4),
                                 nn.ReLU(),
                                 nn.Linear(dim // 4, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class VisionWithMMRL(qwen3_vl.Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.blocks = nn.ModuleList([qwen3_vl.Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.blocks_with_rep = nn.ModuleList([MMRLVitBlock(config) for _ in cfg.INSERT_LAYER])
        self.embedding_pooling = utils.attention_pooling(cfg.vision_token_dim, cfg.POOLING_DIM)
        self.Task_classifier = MMRLGating.Task_classifier()
        self.visionGating = MMRLGating.HardConcreteGate(cfg.gating_temperature)
        self.text_gating = MMRLGating.textGating(cfg.text_gating_epsilon, cfg.gating_temperature)
        self.zero_init_layer = zeroInit(cfg.vision_token_dim)
        self.alpha_list = []
        self.G_list = []

    def forward(self,
                hidden_states: torch.Tensor,
                grid_thw: torch.Tensor,
                v_r_token_list: Optional[list[torch.Tensor]] = None,
                embedding: Optional[torch.Tensor] = None,
                gating_temperature_overied: float = None,
                images_per_sample: Optional[list[int]] = None,
                **kwargs):
        assert len(images_per_sample) == embedding.shape[0]
        batch_size = embedding.shape[0]
        pic_seqlens = [0] + list(accumulate(images_per_sample))

        self.alpha_list = []
        self.G_list = []
        embedding_after_pooling = self.embedding_pooling(embedding)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        original_seq_lens_list = (grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0]).tolist()

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        first_insert = True
        current_num_r_token = cfg.RP_SPACE_LENGTH  # 记录 r_token 的长度

        deepstack_feature_lists = []
        cu_seqlens_with_rep, position_embeddings_with_rep = None, None
        hidden_states_with_rep, org_hidden_states= None, None
        total_pic_num = cu_seqlens.size(0) - 1
        for layer_num, blk in enumerate(self.blocks):
            if layer_num not in cfg.INSERT_LAYER:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    rotary_pos_emb=rotary_pos_emb,
                    **kwargs,
                )
            else:
                ############ N图切分+门控 ############
                if first_insert:
                    # cu_seqlens: [0, len0, len0+len1, ...]
                    img_seqlens = cu_seqlens[1:] - cu_seqlens[:-1]  # [Total_Images]
                    img_indices = torch.repeat_interleave(
                        torch.arange(total_pic_num, device=hidden_states.device),
                        img_seqlens
                    )
                    pooled_vision_states = torch.zeros(
                        total_pic_num,
                        hidden_states.shape[-1],
                        dtype=hidden_states.dtype,
                        device=hidden_states.device
                    )
                    pooled_vision_states.index_add_(0, img_indices, hidden_states)
                    pooled_vision_states = pooled_vision_states / img_seqlens.unsqueeze(-1)
                    images_per_sample_tensor = torch.tensor(images_per_sample, device=hidden_states.device)
                    expanded_text_embedding = torch.repeat_interleave(
                        embedding_after_pooling,
                        images_per_sample_tensor,
                        dim=0
                    )
                    self.alpha_list = self.Task_classifier(pooled_vision_states, expanded_text_embedding)
                    # G_list: [Total_Images, 1]
                    if self.training and gating_temperature_overied is not None:
                        self.G_list = self.visionGating(self.alpha_list, gating_temperature_overied)
                    else:
                        self.G_list = self.visionGating(self.alpha_list)

                ############ N图切分+门控 ############
                idx = cfg.INSERT_LAYER.index(layer_num)
                assert v_r_token_list[idx] is not None
                r_tokens_input = v_r_token_list[idx]
                assert r_tokens_input is not None
                # todo:训练时无论如何都是双支路，全体hidden_states都要参与计算，但是推理时会关闭G小于阈值的支路
                #  此处先强制双支路，以后有空再做支路关闭
                org_hidden_states = blk(hidden_states if first_insert else org_hidden_states,
                                        cu_seqlens=cu_seqlens,
                                        position_embeddings=position_embeddings,
                                        **kwargs)
                if first_insert:
                    cu_seqlens_with_rep, position_embeddings_with_rep = _resize_cu_and_pos(r_tokens_input,
                                                                                           cu_seqlens,
                                                                                           position_embeddings)
                else:
                    assert cu_seqlens_with_rep is not None
                    assert position_embeddings_with_rep is not None
                if self.training:
                    hidden_states_with_rep = self.blocks_with_rep[idx](
                        hidden_states if first_insert else hidden_states_with_rep,
                        cu_seqlens=cu_seqlens_with_rep,
                        position_embeddings=position_embeddings_with_rep,
                        r_token=r_tokens_input,
                        r_token_idx=idx,
                        rotary_pos_emb=rotary_pos_emb,
                        **kwargs
                    )
                first_insert = False
                # 规定这里输出的都是没移除rep的

            if layer_num in self.deepstack_visual_indexes:
                feature_to_save = org_hidden_states
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](feature_to_save)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states_with_rep = _strip_r_token(hidden_states_with_rep,
                                                original_seq_lens_list,
                                                current_num_r_token)
        # for i in range(cu_seqlens.size(0) - 1):
        #     hidden_states_with_rep = hidden_states_with_rep[cu_seqlens[i]:cu_seqlens[i+1]] * self.G_list[i]
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        G_stack = self.G_list  # [Batch, 1]
        G_mask = torch.repeat_interleave(G_stack, seqlens, dim=0)  # [Total_Tokens, 1]
        hidden_states_with_rep = hidden_states_with_rep * G_mask

        delta = hidden_states_with_rep - org_hidden_states
        G_mask = torch.repeat_interleave(self.G_list, seqlens, dim=0)
        gated_delta = delta * G_mask
        final_delta = self.zero_init_layer(gated_delta)
        hidden_states = org_hidden_states + final_delta

        hidden_states = self.merger(hidden_states)
        ########### text gating ###########
        img_counts = torch.tensor(images_per_sample, device=hidden_states.device)
        batch_indices_img = torch.repeat_interleave(
            torch.arange(batch_size, device=hidden_states.device),
            img_counts
        )
        batch_indices_token = torch.repeat_interleave(
            batch_indices_img,
            img_seqlens
        )
        out = self.text_gating(
            final_delta,  # [Total_Tokens, Dim]
            self.alpha_list,  # [Total_Images, 1]
            batch_indices_token,
            batch_indices_img,
            batch_size,
            gating_temperature_overied
        )
        k_results = []
        if self.training:
            hard_k_logits, tax_loss = out  # hard_k_logits: [Batch, 40]

            k_sums = hard_k_logits.sum(dim=-1)
            k_ints = torch.floor(k_sums)
            k_remainders = k_sums - k_ints
            for i in range(batch_size):
                k_results.append((k_ints[i], k_remainders[i], tax_loss))
        else:
            k_sums = out.sum(dim=-1)
            k_results = k_sums.round().int().tolist()
        return hidden_states, deepstack_feature_lists, k_results

