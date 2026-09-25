"""V3: the unchanged V1 conditional P20 is placed after the image segment."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from slake.visual_selection_prefix import (
    EXPECTED_TRAINABLE, VisualSelectionPrefixModel,
)


CONFIG_NAME = "visual_selection_postvisual_config.json"
WEIGHTS_NAME = "visual_selection_postvisual.pt"
PROMPT_LENGTH = 20


class VisualSelectionPostvisualModel(VisualSelectionPrefixModel):
    """Reuse V1's parameters/condition, changing only where its P20 is read."""

    def _prepare(self, batch: dict[str, Any]):
        """Consume processor-reserved slots; never insert tokens in the model."""
        batch = dict(batch)
        source_ids = batch.pop("question_source_ids")
        source_mask = batch.pop("question_source_mask")
        prompt_mask = batch.pop("prompt_mask").bool()
        ids, attention = batch["input_ids"], batch["attention_mask"]
        if ids.ndim != 2 or attention.shape != ids.shape or prompt_mask.shape != ids.shape:
            raise ValueError("V3 processor IDs, attention and prompt mask shapes differ")
        if source_ids.ndim != 2 or source_ids.shape != source_mask.shape or source_ids.shape[0] != ids.shape[0]:
            raise ValueError("V3 standalone raw-question IDs/mask mismatch")
        if not bool(source_mask.bool().any(dim=1).all()):
            raise ValueError("V3 requires nonempty standalone raw-question tokens")
        if not bool((prompt_mask.sum(dim=1) == PROMPT_LENGTH).all()):
            raise ValueError("V3 processor must reserve exactly 20 prompt positions per sample")
        positions = torch.nonzero(prompt_mask, as_tuple=False)[:, 1].reshape(ids.shape[0], PROMPT_LENGTH)
        if not torch.equal(positions, positions[:, :1] + torch.arange(PROMPT_LENGTH, device=ids.device)[None, :]):
            raise ValueError("V3 prompt positions must be contiguous")
        row = torch.arange(ids.shape[0], device=ids.device)
        if not bool(ids[row, positions[:, 0] - 1].eq(int(self.config.vision_end_token_id)).all()):
            raise ValueError("V3 prompt positions must immediately follow vision_end")
        if not bool(attention[prompt_mask].eq(1).all()):
            raise ValueError("V3 processor masked out a prompt position")
        if "labels" in batch and not bool(batch["labels"][prompt_mask].eq(-100).all()):
            raise ValueError("V3 prompt positions must have labels=-100")
        if any(batch.get(name) is not None for name in ("position_ids", "cache_position", "inputs_embeds")):
            raise ValueError("V3 requires fresh Qwen mRoPE/cache positions from processor IDs")
        return batch, source_ids, source_mask, positions, prompt_mask

    @contextmanager
    def _injection(self, ids: torch.Tensor, grid: torch.Tensor, source_ids: torch.Tensor,
                   source_mask: torch.Tensor, prompt_positions: torch.Tensor,
                   prompt_mask: torch.Tensor) -> Iterator[None]:
        if self._active:
            raise RuntimeError("V3 injection is not reentrant")
        self._active = True
        self._features = {}
        self._expected_visual_segments = int(grid[:, 0].sum())
        self._expected_visual_patches = int(grid.prod(dim=1).sum())
        question, valid = self._question_embeddings(source_ids, source_mask)
        embeddings = self.get_input_embeddings()
        language = self.base_model.model.language_model
        batch_index = torch.arange(ids.shape[0], device=ids.device)[:, None]
        prefill_done = False

        def replace_p20(_module, _inputs, output):
            if prefill_done or output.ndim != 3 or output.shape[:2] != ids.shape:
                return output
            replaced = output.clone()
            prompt = self.p20.to(output.dtype)[None, :, :].expand(ids.shape[0], -1, -1)
            replaced[batch_index, prompt_positions] = prompt
            if not torch.equal(replaced[~prompt_mask], output[~prompt_mask]):
                raise RuntimeError("V3 static prompt modified native embeddings")
            return replaced

        def condition_p20(_module, args, kwargs):
            nonlocal prefill_done
            if prefill_done:
                return args, kwargs
            inputs = kwargs.get("inputs_embeds")
            if inputs is None or inputs.shape[:2] != ids.shape:
                raise RuntimeError("V3 prefill inputs_embeds shape mismatch")
            vision_mask = kwargs.get("visual_pos_masks")
            expected_mask = ids.eq(int(self.config.image_token_id))
            if vision_mask is None or not torch.equal(vision_mask, expected_mask):
                raise RuntimeError("V3 visual position mask lost image-placeholder alignment")
            deepstack = kwargs.get("deepstack_visual_embeds")
            if deepstack is None or any(value.shape[0] != int(expected_mask.sum()) for value in deepstack):
                raise RuntimeError("V3 native DeepStack length/order does not match visual placeholders")
            rope = kwargs.get("position_ids")
            if rope is None or rope.shape[-2:] != ids.shape or rope.shape[0] not in (3, 4):
                raise RuntimeError("V3 Qwen3-VL mRoPE position shape mismatch")
            verified_modes = getattr(self, "_mrope_verified_modes", set())
            checked_mrope = int(rope.shape[0]) not in verified_modes
            if checked_mrope:
                expected_rope, _ = self.base_model.model.get_rope_index(
                    ids, image_grid_thw=grid, attention_mask=kwargs.get("attention_mask"),
                )
                if not torch.equal(rope[-3:], expected_rope):
                    raise RuntimeError("V3 mRoPE does not match the expanded multimodal sequence")
                self._mrope_verified_modes = verified_modes | {int(rope.shape[0])}
            cache = kwargs.get("cache_position")
            if cache is not None and (cache.numel() != ids.shape[1] or int(cache[0]) != 0):
                raise RuntimeError("V3 prefill cache_position does not cover the expanded sequence")
            condition, map_debug = self._condition(question, valid, grid)
            shift = self._prefix_shift(condition)
            if shift.shape != (ids.shape[0], inputs.shape[-1]) or not bool(torch.isfinite(shift).all()):
                raise RuntimeError("V3 conditional P20 shift invalid")
            updated = inputs.clone()
            updated[batch_index, prompt_positions] = inputs[batch_index, prompt_positions] + shift[:, None, :].to(inputs.dtype)
            if not torch.equal(updated[~prompt_mask], inputs[~prompt_mask]):
                raise RuntimeError("V3 dynamic prompt modified native visual/text embeddings")
            kwargs = dict(kwargs)
            kwargs["inputs_embeds"] = updated
            p_rms = self.p20.square().mean().sqrt().clamp_min(1e-8)
            s_rms = shift.square().mean().sqrt()
            self.debug_context = {
                **map_debug,
                "question_rms": question[valid].square().mean().sqrt().detach(),
                "condition_rms": condition.square().mean().sqrt().detach(),
                "native_value_rms": self._features["value"].float().square().mean().sqrt().detach(),
                "offset_rms": s_rms.detach(), "p20_rms": p_rms.detach(),
                "offset_to_p20_rms": (s_rms / p_rms).detach(),
                "question_tokens": valid.sum().float().detach(),
            }
            if self.training and self.first_batch_diagnostics is None:
                self.first_batch_diagnostics = {key: float(value.float()) for key, value in self.debug_context.items()}
            self.last_injection_audit = {
                "prefix_only": True, "positions": PROMPT_LENGTH,
                "prompt_positions": prompt_positions.detach().cpu().tolist(),
                "native_embeddings_unchanged": True,
                "vision_mask_aligned": True, "deepstack_aligned": True,
                "mrope_checked": checked_mrope,
                "single_prefill": True,
            }
            prefill_done = True
            self.diagnostic_prefill_calls += 1
            return args, kwargs

        embedding_hook = embeddings.register_forward_hook(replace_p20)
        language_hook = language.register_forward_pre_hook(condition_p20, with_kwargs=True)
        try:
            yield
            if not prefill_done:
                raise RuntimeError("V3 prefill did not reach the LLM")
        finally:
            language_hook.remove()
            embedding_hook.remove()
            self._features = {}
            self._active = False

    def forward(self, **kwargs: Any):
        prepared, source_ids, source_mask, positions, prompt_mask = self._prepare(kwargs)
        with self._injection(prepared["input_ids"], prepared["image_grid_thw"],
                             source_ids, source_mask, positions, prompt_mask):
            return self.base_model(**prepared)

    def generate(self, **kwargs: Any):
        prepared, source_ids, source_mask, positions, prompt_mask = self._prepare(kwargs)
        with self._injection(prepared["input_ids"], prepared["image_grid_thw"],
                             source_ids, source_mask, positions, prompt_mask):
            return self.base_model.generate(**prepared)

    def save_v3(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "visual_selection_postvisual_p20_v3", "init_seed": self.init_seed,
            "prompt_tokens": PROMPT_LENGTH, "insertion": "after_vision_end_before_question",
            "trainable_parameters": self._audit_parameters(),
        }
        with (path / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save({name: p.detach().cpu() for name, p in self.named_parameters()
                    if p.requires_grad}, path / WEIGHTS_NAME)

    def load_v3(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        with (path / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("method") != "visual_selection_postvisual_p20_v3" or int(config["init_seed"]) != self.init_seed:
            raise ValueError("V3 checkpoint architecture/seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters) or sum(p.numel() for p in parameters.values()) != EXPECTED_TRAINABLE:
            raise ValueError("V3 checkpoint tensor set/parameter count mismatch")
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"V3 checkpoint tensor shape mismatch: {name}")
            parameter.data.copy_(state[name].to(parameter.device))
