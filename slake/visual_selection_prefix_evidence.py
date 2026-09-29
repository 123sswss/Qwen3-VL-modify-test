"""V1B: keep the full V1 head and inject its output as one evidence token."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from slake.visual_selection_offset import LAYERS
from slake.visual_selection_prefix import (
    CONFIG_NAME, EXPECTED_TRAINABLE, WEIGHTS_NAME, VisualSelectionPrefixModel,
)


STATIC_PROMPT_TOKENS = 20
EVIDENCE_TOKENS = 1
TOTAL_PREFIX_TOKENS = STATIC_PROMPT_TOKENS + EVIDENCE_TOKENS


class VisualSelectionPrefixEvidenceModel(VisualSelectionPrefixModel):
    """Sequence layout ``[P20; b(I,Q); native chat]`` with the original V1 head."""

    method_name = "visual_selection_prefix_evidence_token_v1b"

    def _expand(self, batch: dict[str, Any]):
        batch = dict(batch)
        ids = batch.pop("input_ids")
        attention = batch.pop("attention_mask")
        source_ids = batch.pop("question_source_ids")
        source_mask = batch.pop("question_source_mask")
        if ids.ndim != 2 or ids.shape != attention.shape:
            raise ValueError("V1B input and attention shape mismatch")
        if source_ids.ndim != 2 or source_ids.shape != source_mask.shape or source_ids.shape[0] != ids.shape[0]:
            raise ValueError("V1B independent question IDs/mask mismatch")
        if not bool(source_mask.any(dim=1).all()):
            raise ValueError("V1B requires nonempty standalone raw-question tokens")
        if any(batch.get(name) is not None for name in ("position_ids", "cache_position", "inputs_embeds")):
            raise ValueError("V1B requires fresh Qwen mRoPE/cache positions for its 21-token prefix")
        labels = batch.pop("labels", None)
        batch_size = ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        batch["input_ids"] = torch.cat(
            (ids.new_full((batch_size, TOTAL_PREFIX_TOKENS), pad_id), ids), dim=1,
        )
        batch["attention_mask"] = torch.cat(
            (attention.new_ones((batch_size, TOTAL_PREFIX_TOKENS)), attention), dim=1,
        )
        if labels is not None:
            batch["labels"] = torch.cat(
                (labels.new_full((batch_size, TOTAL_PREFIX_TOKENS), -100), labels), dim=1,
            )
        return batch, source_ids, source_mask

    @contextmanager
    def _injection(
        self, ids: torch.Tensor, grid: torch.Tensor,
        source_ids: torch.Tensor, source_mask: torch.Tensor,
    ) -> Iterator[None]:
        if self._active:
            raise RuntimeError("V1B injection is not reentrant")
        self._active = True
        self._features = {}
        self._expected_visual_segments = int(grid[:, 0].sum())
        self._expected_visual_patches = int(grid.prod(dim=1).sum())
        question, valid = self._question_embeddings(source_ids, source_mask)
        embeddings = self.get_input_embeddings()
        language = self.base_model.model.language_model
        prefill_done = False

        def replace_p20(_module, _inputs, output):
            if prefill_done or output.ndim != 3 or output.shape[:2] != ids.shape:
                return output
            replaced = output.clone()
            replaced[:, :STATIC_PROMPT_TOKENS] = self.p20.to(output.dtype)[None, :, :]
            if not torch.equal(replaced[:, TOTAL_PREFIX_TOKENS:], output[:, TOTAL_PREFIX_TOKENS:]):
                raise RuntimeError("V1B static P20 replacement modified the native chat")
            return replaced

        def inject_evidence(_module, args, kwargs):
            nonlocal prefill_done
            if prefill_done:
                return args, kwargs
            inputs = kwargs.get("inputs_embeds")
            if inputs is None or inputs.shape[:2] != ids.shape or inputs.shape[1] < TOTAL_PREFIX_TOKENS:
                raise RuntimeError("V1B prefill inputs_embeds shape mismatch")
            vision_mask = kwargs.get("visual_pos_masks")
            expected_mask = ids.eq(int(self.config.image_token_id))
            if vision_mask is None or not torch.equal(vision_mask, expected_mask):
                raise RuntimeError("V1B visual mask lost native image-placeholder alignment")
            deepstack = kwargs.get("deepstack_visual_embeds")
            if deepstack is None or any(value.shape[0] != int(expected_mask.sum()) for value in deepstack):
                raise RuntimeError("V1B DeepStack length/order differs from native visual placeholders")
            rope = kwargs.get("position_ids")
            if rope is None or rope.shape[-2:] != ids.shape or rope.shape[0] not in (3, 4):
                raise RuntimeError("V1B Qwen3-VL mRoPE shape does not cover the expanded sequence")
            verified_modes = getattr(self, "_mrope_verified_modes", set())
            checked_mrope = int(rope.shape[0]) not in verified_modes
            if checked_mrope:
                expected_rope, _ = self.base_model.model.get_rope_index(
                    ids, image_grid_thw=grid, attention_mask=kwargs.get("attention_mask"),
                )
                if not torch.equal(rope[-3:], expected_rope):
                    raise RuntimeError("V1B mRoPE differs from the expanded multimodal sequence")
                self._mrope_verified_modes = verified_modes | {int(rope.shape[0])}
            cache = kwargs.get("cache_position")
            if cache is not None and (cache.numel() != ids.shape[1] or int(cache[0]) != 0):
                raise RuntimeError("V1B prefill cache_position does not cover all 21 prefix positions")
            condition, map_debug = self._condition(question, valid, grid)
            evidence = self._prefix_shift(condition)
            if evidence.shape != (ids.shape[0], inputs.shape[-1]) or not bool(torch.isfinite(evidence).all()):
                raise RuntimeError("V1B evidence token is missing, mis-shaped, or nonfinite")
            updated = inputs.clone()
            updated[:, STATIC_PROMPT_TOKENS] = evidence.to(inputs.dtype)
            if not torch.equal(updated[:, :STATIC_PROMPT_TOKENS], inputs[:, :STATIC_PROMPT_TOKENS]):
                raise RuntimeError("V1B added the dynamic shift to P20")
            if not torch.equal(updated[:, TOTAL_PREFIX_TOKENS:], inputs[:, TOTAL_PREFIX_TOKENS:]):
                raise RuntimeError("V1B modified native visual/question embeddings")
            kwargs = dict(kwargs)
            kwargs["inputs_embeds"] = updated
            evidence_rms = evidence.square().mean().sqrt()
            p20_rms = self.p20.square().mean().sqrt().clamp_min(1e-8)
            self.debug_context = {
                **map_debug,
                "question_rms": question[valid].square().mean().sqrt().detach(),
                "condition_rms": condition.square().mean().sqrt().detach(),
                "native_value_rms": self._features["value"].float().square().mean().sqrt().detach(),
                "evidence_token_rms": evidence_rms.detach(),
                "p20_rms": p20_rms.detach(),
                "evidence_to_p20_rms": (evidence_rms / p20_rms).detach(),
                "question_tokens": valid.sum().float().detach(),
            }
            if self.training and self.first_batch_diagnostics is None:
                self.first_batch_diagnostics = {
                    key: float(value.float()) for key, value in self.debug_context.items()
                }
            self.last_injection_audit = {
                "layout": "P20_then_one_evidence_then_native_chat",
                "prefix_tokens": TOTAL_PREFIX_TOKENS,
                "p20_unchanged_by_dynamic_branch": True,
                "evidence_replaces_placeholder_not_added_to_base": True,
                "native_embeddings_unchanged": True,
                "vision_mask_aligned": True,
                "deepstack_aligned": True,
                "mrope_checked": checked_mrope,
                "single_prefill": True,
            }
            prefill_done = True
            self.diagnostic_prefill_calls += 1
            return args, kwargs

        embedding_hook = embeddings.register_forward_hook(replace_p20)
        language_hook = language.register_forward_pre_hook(inject_evidence, with_kwargs=True)
        try:
            yield
            if not prefill_done:
                raise RuntimeError("V1B prefill did not reach the language model")
        finally:
            language_hook.remove()
            embedding_hook.remove()
            self._features = {}
            self._active = False

    def save_v1(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": self.method_name,
            "init_seed": self.init_seed,
            "layers": list(LAYERS),
            "static_prompt_tokens": STATIC_PROMPT_TOKENS,
            "evidence_tokens": EVIDENCE_TOKENS,
            "prefix_tokens": TOTAL_PREFIX_TOKENS,
            "layout": "P20_then_b_then_native_chat",
            "trainable_parameters": self._audit_parameters(),
        }
        with (path / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(
            {name: parameter.detach().cpu() for name, parameter in self.named_parameters()
             if parameter.requires_grad},
            path / WEIGHTS_NAME,
        )

    def load_v1(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        with (path / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if (
            config.get("method") != self.method_name
            or int(config.get("prefix_tokens", -1)) != TOTAL_PREFIX_TOKENS
            or int(config["init_seed"]) != self.init_seed
        ):
            raise ValueError("V1B checkpoint architecture/seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {name: parameter for name, parameter in self.named_parameters()
                      if parameter.requires_grad}
        if set(state) != set(parameters) or sum(p.numel() for p in parameters.values()) != EXPECTED_TRAINABLE:
            raise ValueError("V1B checkpoint tensor set/parameter count mismatch")
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"V1B checkpoint tensor shape mismatch: {name}")
            parameter.data.copy_(state[name].to(parameter.device))
