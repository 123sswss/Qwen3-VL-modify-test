from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

import config as cfg


class VisionBlockWithMMRL(qwen3_vl.Qwen3VLVisionBlock):
    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                r_token: Optional[torch.Tensor] = None,
                r_token_idx: Optional[int] = None,
                rotary_pos_emb: Optional[torch.Tensor] = None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs):
        if r_token is not None:
            num_r_tokens = r_token.shape[0]
            batch_size = cu_seqlens.shape[0] - 1
            seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
            hidden_states_split = torch.split(hidden_states, seq_lengths, dim=0)

            if r_token_idx == 0:
                new_hidden_states_list = []
                for seq_hidden in hidden_states_split:
                    new_hidden_states_list.append(torch.cat([r_token, seq_hidden], dim=0))
                hidden_states = torch.cat(new_hidden_states_list, dim=0)

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

                for i in range(batch_size):
                    new_cos_list.append(torch.cat([r_token_cos, cos_split[i]], dim=0))
                    new_sin_list.append(torch.cat([r_token_sin, sin_split[i]], dim=0))

                new_cos = torch.cat(new_cos_list, dim=0)
                new_sin = torch.cat(new_sin_list, dim=0)
                updated_position_embeddings = (new_cos, new_sin)

                if batch_size > 0:
                    offsets = torch.arange(batch_size + 1, device=cu_seqlens.device,
                                           dtype=cu_seqlens.dtype) * num_r_tokens
                    updated_cu_seqlens = cu_seqlens + offsets
                else:
                    updated_cu_seqlens = cu_seqlens

                hidden_states = hidden_states + self.attn(self.norm1(hidden_states),
                                                          cu_seqlens=updated_cu_seqlens,
                                                          rotary_pos_emb=rotary_pos_emb,
                                                          position_embeddings=updated_position_embeddings,
                                                          **kwargs)
                hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
                return hidden_states, updated_cu_seqlens, updated_position_embeddings

            else:
                new_hidden_states_list = []
                updated_seq = None
                for seq_hidden in hidden_states_split:
                    if cfg.INSERT_METHOD == "replace":
                        updated_seq = torch.cat([r_token, seq_hidden[num_r_tokens:]], dim=0)
                    elif cfg.INSERT_METHOD == "add":
                        prefix = (seq_hidden[:num_r_tokens] + r_token) / 2
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
        return hidden_states  # (196, 1024) or (r_token_num + 196 , 1024) flash attention下的视觉编码器没有batch维度

class VisionWithMMRL(qwen3_vl.Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.blocks = nn.ModuleList([VisionBlockWithMMRL(config) for _ in range(config.depth)])

    def forward(self,
                hidden_states: torch.Tensor,
                grid_thw: torch.Tensor,
                v_r_token_list: Optional[list[torch.Tensor]] = None,
                **kwargs):

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

        has_r_token = False  # 标记当前 hidden_states 是否包含 r_token
        current_num_r_token = 0  # 记录 r_token 的长度

        deepstack_feature_lists = []

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
                idx = cfg.INSERT_LAYER.index(layer_num)
                r_tokens_input = v_r_token_list[idx] if v_r_token_list is not None else None

                if not has_r_token and r_tokens_input is not None:
                    current_num_r_token = r_tokens_input.shape[0]

                out = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    r_token=r_tokens_input,
                    r_token_idx=idx,
                    rotary_pos_emb=rotary_pos_emb,
                    **kwargs,
                )

                if isinstance(out, tuple):
                    hidden_states, cu_seqlens, position_embeddings = out
                    if idx == 0:
                        has_r_token = True
                elif isinstance(out, torch.Tensor):
                    hidden_states = out

            if layer_num in self.deepstack_visual_indexes:
                feature_to_save = hidden_states
                if has_r_token and current_num_r_token > 0:
                    feature_to_save = self._strip_r_token(
                        feature_to_save,
                        original_seq_lens_list,
                        current_num_r_token
                    )
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](feature_to_save)
                deepstack_feature_lists.append(deepstack_feature)

        if has_r_token and current_num_r_token > 0:
            hidden_states = self._strip_r_token(
                hidden_states,
                original_seq_lens_list,
                current_num_r_token
            )

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def _strip_r_token(self, hidden_states, original_lens, num_r_token):
        current_lens = [l + num_r_token for l in original_lens]
        hidden_states_split = torch.split(hidden_states, current_lens, dim=0)
        clean_list = []
        for seq in hidden_states_split:
            clean_list.append(seq[num_r_token:])
        return torch.cat(clean_list, dim=0)

