"""Approximate GRASP reproduction for frozen Qwen3-VL models."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
from torch import nn


GRASP_CONFIG_NAME = "grasp_config.json"
GRASP_WEIGHTS_NAME = "grasp_prompt.pt"


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Closed-form alpha=1.5 Entmax."""
    values = logits / 2
    values = values - values.max(dim=dim, keepdim=True).values
    ordered, _ = torch.sort(values, dim=dim, descending=True)
    shape = [1] * values.ndim
    shape[dim] = values.shape[dim]
    rho = torch.arange(
        1, values.shape[dim] + 1, device=values.device, dtype=values.dtype
    ).view(shape)
    mean = ordered.cumsum(dim) / rho
    mean_sq = ordered.square().cumsum(dim) / rho
    delta = (1 - rho * (mean_sq - mean.square())) / rho
    taus = mean - delta.clamp_min(0).sqrt()
    support_size = (taus <= ordered).sum(dim=dim, keepdim=True).clamp_min(1)
    tau = taus.gather(dim, support_size - 1)
    probabilities = (values - tau).clamp_min(0).square()
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )


def fixed_2d_sincos_encoding(block_count: int, hidden_size: int) -> torch.Tensor:
    """Create deterministic position encodings for square spatial blocks."""
    side = math.isqrt(block_count)
    if side * side != block_count:
        raise ValueError("GRASP block_count must be a perfect square")
    if hidden_size < 4:
        raise ValueError("GRASP hidden_size must be at least 4")
    coordinates = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, side),
            torch.linspace(0, 1, side),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(block_count, 2)
    axis_size = hidden_size // 2
    frequencies = torch.exp(
        torch.arange(max(1, axis_size // 2), dtype=torch.float32)
        * (-math.log(10_000.0) / max(1, axis_size // 2 - 1))
    )
    encoded_axes = []
    for axis in range(2):
        angles = coordinates[:, axis : axis + 1] * frequencies.unsqueeze(0)
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
        encoded_axes.append(torch.nn.functional.pad(encoded, (0, axis_size))[:, :axis_size])
    result = torch.cat(encoded_axes, dim=-1)
    return torch.nn.functional.pad(result, (0, hidden_size))[:, :hidden_size]


def pool_spatial_blocks(
    visual_tokens: torch.Tensor,
    grid_height: int,
    grid_width: int,
    block_count: int,
) -> torch.Tensor:
    """Losslessly average a visual grid into non-overlapping blocks."""
    side = math.isqrt(block_count)
    if side * side != block_count:
        raise ValueError("GRASP block_count must be a perfect square")
    if visual_tokens.ndim != 2 or visual_tokens.shape[0] != grid_height * grid_width:
        raise ValueError("Visual token count does not match the post-merger grid")
    if grid_height < side or grid_width < side:
        raise ValueError("Visual grid is smaller than the requested GRASP block grid")
    grid = visual_tokens.reshape(grid_height, grid_width, -1)
    pooled = []
    for row in range(side):
        r0, r1 = row * grid_height // side, (row + 1) * grid_height // side
        for column in range(side):
            c0, c1 = column * grid_width // side, (column + 1) * grid_width // side
            pooled.append(grid[r0:r1, c0:c1].reshape(-1, grid.shape[-1]).mean(0))
    return torch.stack(pooled)


class GRASPPromptTuningModel(nn.Module):
    """Question-guided sparse routing over spatial prompt prototypes."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        block_count: int = 4,
        bottleneck_dim: int = 512,
        prompt_init_std: float = 0.02,
        init_seed: int = 44,
    ) -> None:
        super().__init__()
        if bottleneck_dim < 1 or prompt_init_std <= 0:
            raise ValueError("GRASP dimensions and initialization must be positive")
        self.base_model = base_model
        for parameter in base_model.parameters():
            parameter.requires_grad = False
        embeddings = base_model.get_input_embeddings().weight
        model_device = embeddings.device
        self.hidden_size = int(embeddings.shape[-1])
        self.block_count = int(block_count)
        self.bottleneck_dim = int(bottleneck_dim)
        self.prompt_init_std = float(prompt_init_std)
        self.init_seed = int(init_seed)
        self.prompt_length = 1
        generator = torch.Generator(device="cpu").manual_seed(self.init_seed)
        prompt_init = (
            torch.randn(
                self.block_count,
                self.hidden_size,
                generator=generator,
                dtype=torch.float32,
            )
            * self.prompt_init_std
        ).to(device=model_device)
        self.prompt_prototypes = nn.Parameter(prompt_init)
        self.visual_key_projection = nn.Linear(
            self.hidden_size, bottleneck_dim
        ).to(device=model_device)
        self.text_query_projection = nn.Linear(
            self.hidden_size, bottleneck_dim
        ).to(device=model_device)
        self.register_buffer(
            "block_position_encoding",
            fixed_2d_sincos_encoding(self.block_count, self.hidden_size),
        )
        self.visual_token_ids = tuple(
            int(tokenizer.convert_tokens_to_ids(token))
            for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
        )
        if any(token_id < 0 for token_id in self.visual_token_ids):
            raise RuntimeError("GRASP could not resolve Qwen visual token ids")
        visual = getattr(getattr(base_model, "model", None), "visual", None)
        visual_config = getattr(visual, "config", None)
        self.spatial_merge_size = int(
            getattr(visual, "spatial_merge_size", None)
            or getattr(visual_config, "spatial_merge_size", 2)
        )
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.debug_context: Dict[str, torch.Tensor] = {}
        self._encoding_question = False
        self._forward_audited = False

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def trainable_parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        return {
            "prompt_prototypes": [self.prompt_prototypes],
            "projections": list(self.visual_key_projection.parameters())
            + list(self.text_query_projection.parameters()),
        }

    def _expand_inputs(self, batch: Dict[str, Any]):
        batch = dict(batch)
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        labels = batch.get("labels")
        context = batch.pop("mmrl_gating_mask", None)
        question_ids = batch.pop("grasp_question_input_ids", None)
        question_mask = batch.pop("grasp_question_attention_mask", None)
        if question_ids is None or question_mask is None:
            raise RuntimeError("GRASP requires separately tokenized raw-question inputs")
        if question_ids.shape != question_mask.shape:
            raise ValueError("GRASP question ids and attention mask must share shape")
        if not bool(question_mask.bool().any(dim=1).all()):
            raise RuntimeError("GRASP found a sample without raw-question tokens")
        if context is not None and context.shape != input_ids.shape:
            raise ValueError("mmrl_gating_mask must match input_ids")
        batch_size = input_ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        vision_end_id = self.visual_token_ids[2]
        prompt_positions = []
        expanded_ids = torch.full(
            (batch_size, input_ids.shape[1] + 1),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        expanded_attention = torch.zeros(
            expanded_ids.shape,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        expanded_labels = None
        if labels is not None:
            expanded_labels = torch.full(
                expanded_ids.shape, -100, dtype=labels.dtype, device=labels.device
            )
        for index in range(batch_size):
            vision_ends = torch.nonzero(
                input_ids[index].eq(vision_end_id) & attention_mask[index].bool(),
                as_tuple=False,
            ).flatten()
            if vision_ends.numel() != 1:
                raise RuntimeError("GRASP requires exactly one visual segment per sample")
            position = int(vision_ends.item()) + 1
            prompt_positions.append(position)
            expanded_ids[index, :position] = input_ids[index, :position]
            expanded_ids[index, position + 1 :] = input_ids[index, position:]
            expanded_attention[index, :position] = attention_mask[index, :position]
            expanded_attention[index, position] = 1
            expanded_attention[index, position + 1 :] = attention_mask[index, position:]
            if expanded_labels is not None:
                expanded_labels[index, :position] = labels[index, :position]
                expanded_labels[index, position + 1 :] = labels[index, position:]
        expanded = {
            **batch,
            "input_ids": expanded_ids,
            "attention_mask": expanded_attention,
        }
        if expanded_labels is not None:
            expanded["labels"] = expanded_labels
        grid_thw = expanded.get("image_grid_thw")
        if grid_thw is None:
            raise RuntimeError("GRASP requires image_grid_thw")
        return (
            expanded,
            expanded_ids,
            torch.tensor(prompt_positions, dtype=torch.long, device=input_ids.device),
            question_ids,
            question_mask,
            grid_thw,
        )

    @staticmethod
    def _pack_tokens(embeddings: torch.Tensor, mask: torch.Tensor):
        lengths = mask.sum(1)
        if bool((lengths == 0).any()):
            raise RuntimeError("GRASP found a sample without question tokens")
        packed = embeddings.new_zeros(embeddings.shape[0], int(lengths.max()), embeddings.shape[-1])
        packed_mask = torch.zeros(packed.shape[:2], dtype=torch.long, device=embeddings.device)
        for index, length_tensor in enumerate(lengths):
            length = int(length_tensor)
            packed[index, :length] = embeddings[index][mask[index]]
            packed_mask[index, :length] = 1
        return packed, packed_mask

    def _encode_question(self, language_model, question_ids, question_mask):
        question_embeddings = self.get_input_embeddings()(question_ids)
        packed, packed_mask = self._pack_tokens(
            question_embeddings, question_mask.bool()
        )
        self._encoding_question = True
        try:
            with torch.no_grad():
                output = language_model(
                    inputs_embeds=packed,
                    attention_mask=packed_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            self._encoding_question = False
        hidden = getattr(output, "last_hidden_state", output[0] if isinstance(output, tuple) else None)
        if hidden is None:
            raise RuntimeError("Frozen LLM did not return last_hidden_state")
        weights = packed_mask.to(hidden.dtype).unsqueeze(-1)
        return ((hidden * weights).sum(1) / weights.sum(1).clamp_min(1)).detach()

    def _pool_visual(self, embeddings, mask, grid_thw):
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        if grid_thw.shape[0] != embeddings.shape[0]:
            raise RuntimeError("GRASP reproduction supports one image per VQA sample")
        pooled = []
        merge = self.spatial_merge_size
        for index in range(embeddings.shape[0]):
            temporal, height, width = map(int, grid_thw[index].tolist())
            if temporal != 1 or height % merge or width % merge:
                raise RuntimeError("GRASP received an unsupported visual grid")
            pooled.append(
                pool_spatial_blocks(
                    embeddings[index][mask[index]],
                    height // merge,
                    width // merge,
                    self.block_count,
                )
            )
        blocks = torch.stack(pooled)
        return blocks + self.block_position_encoding.to(blocks).unsqueeze(0)

    @contextmanager
    def _inject_prompt(
        self, ids, prompt_positions, question_ids, question_mask, grid_thw
    ) -> Iterator[None]:
        language_model = getattr(getattr(self.base_model, "model", None), "language_model", None)
        if language_model is None:
            raise RuntimeError("GRASP requires base_model.model.language_model")

        def hook(_module, args, kwargs):
            if self._encoding_question:
                return args, kwargs
            embeddings = kwargs.get("inputs_embeds")
            if embeddings is None or embeddings.shape[1] != context.shape[1]:
                return args, kwargs
            visual_mask = kwargs.get("visual_pos_masks")
            if visual_mask is None:
                visual_mask = ids.eq(self.visual_token_ids[0])
            visual_mask = visual_mask.to(device=embeddings.device, dtype=torch.bool)
            question = self._encode_question(
                language_model,
                question_ids.to(embeddings.device),
                question_mask.to(embeddings.device),
            )
            blocks = self._pool_visual(embeddings, visual_mask, grid_thw)
            keys = self.visual_key_projection(blocks.float())
            query = self.text_query_projection(question.float())
            scores = torch.einsum("bnh,bh->bn", keys, query) / math.sqrt(self.bottleneck_dim)
            weights = entmax15(scores)
            prompt = torch.einsum("bn,nd->bd", weights, self.prompt_prototypes).to(embeddings)
            updated_embeddings = embeddings.clone()
            batch_indices = torch.arange(embeddings.shape[0], device=embeddings.device)
            updated_embeddings[
                batch_indices, prompt_positions.to(embeddings.device)
            ] = prompt
            kwargs["inputs_embeds"] = updated_embeddings
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(1)
            self.debug_context = {
                "grasp_zero_weight_fraction": weights.eq(0).float().mean().detach(),
                "grasp_weight_entropy_norm": (entropy / math.log(self.block_count)).mean().detach(),
                "grasp_max_region_weight": weights.max(1).values.mean().detach(),
                "grasp_global_prompt_norm": prompt.float().norm(dim=-1).mean().detach(),
                "grasp_question_norm": question.float().norm(dim=-1).mean().detach(),
                "grasp_visual_block_norm": blocks.float().norm(dim=-1).mean().detach(),
            }
            if not self._forward_audited:
                print(
                    "[GRASP_FORWARD_AUDIT] blocks=%d bottleneck=%d alpha=1.5 "
                    "question=raw_question_only_frozen_llm_last_hidden_mean "
                    "visual=post_merger prompt_placement=after_visual_segment "
                    "prompt_tokens=1 pass=True"
                    % (self.block_count, self.bottleneck_dim)
                )
                self._forward_audited = True
            return args, kwargs

        handle = language_model.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    def forward(self, **kwargs):
        expanded, ids, positions, question_ids, question_mask, grid = self._expand_inputs(kwargs)
        with self._inject_prompt(ids, positions, question_ids, question_mask, grid):
            return self.base_model(**expanded)

    def generate(self, **kwargs):
        expanded, ids, positions, question_ids, question_mask, grid = self._expand_inputs(kwargs)
        with self._inject_prompt(ids, positions, question_ids, question_mask, grid):
            return self.base_model.generate(**expanded)

    def save_grasp(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "grasp_reimplementation",
            "source": "arXiv:2601.17089v1",
            "block_count": self.block_count,
            "bottleneck_dim": self.bottleneck_dim,
            "entmax_alpha": 1.5,
            "prompt_init_std": self.prompt_init_std,
            "init_seed": self.init_seed,
            "hidden_size": self.hidden_size,
            "question_source": "raw_question_only",
            "prompt_placement": "after_visual_segment",
            "prompt_length": 1,
            "question_encoder": "raw_question_only_frozen_llm_last_hidden_mean",
            "visual_source": "post_merger_grid",
            "position_encoding": "fixed_2d_sincos",
            "approximation_notes": [
                "last hidden layer selected because the paper does not identify a layer",
                "prompt initialization std=0.02 because the paper leaves sigma unspecified",
            ],
        }
        with (output / GRASP_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        torch.save(
            {
                "prompt_prototypes": self.prompt_prototypes.detach().cpu(),
                "visual_key_projection": self.visual_key_projection.state_dict(),
                "text_query_projection": self.text_query_projection.state_dict(),
            },
            output / GRASP_WEIGHTS_NAME,
        )

    def load_grasp(self, checkpoint_dir: str | Path) -> None:
        state = torch.load(Path(checkpoint_dir) / GRASP_WEIGHTS_NAME, map_location="cpu", weights_only=True)
        if tuple(state["prompt_prototypes"].shape) != tuple(self.prompt_prototypes.shape):
            raise ValueError("GRASP prompt prototype shape mismatch")
        self.prompt_prototypes.data.copy_(state["prompt_prototypes"].to(self.prompt_prototypes.device))
        self.visual_key_projection.load_state_dict(state["visual_key_projection"], strict=True)
        self.text_query_projection.load_state_dict(state["text_query_projection"], strict=True)
        self._forward_audited = True


__all__ = ["GRASPPromptTuningModel", "entmax15", "fixed_2d_sincos_encoding", "pool_spatial_blocks"]
