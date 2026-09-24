"""V1: V0 visual selection conditions the existing P20, not question tokens."""

from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from slake.visual_selection_offset import (
    BLOCK_WIDTH, LAYERS, OUTPUT_INIT_STD, QUESTION_WIDTH,
    VisualSelectionOffsetModel, _Visual18Block,
)


CONFIG_NAME = "visual_selection_prefix_config.json"
WEIGHTS_NAME = "visual_selection_prefix.pt"
EXPECTED_TRAINABLE = 1_864_963
EXPECTED_GROUP_COUNTS = {
    "p20": 51_200, "visual_s8": 8_192, "visual_av10": 10_240,
    "question_context": 345_088, "maps": 448_896,
    "layer_condition": 507_267, "prefix_output": 494_080,
}


class VisualSelectionPrefixModel(nn.Module):
    """Frozen Qwen3-VL, native Visual18, and one shared conditional P20 shift."""

    # Reuse the audited V0 selection algorithm verbatim; neither V0's offset
    # modules nor its write-to-question path are instantiated by this class.
    _install_capture_hooks = VisualSelectionOffsetModel._install_capture_hooks
    _install_first_backward_hooks = VisualSelectionOffsetModel._install_first_backward_hooks
    _question_embeddings = VisualSelectionOffsetModel._question_embeddings
    _condition = VisualSelectionOffsetModel._condition

    def __init__(self, base_model: nn.Module, init_seed: int = 44) -> None:
        super().__init__()
        self.base_model = base_model
        self.init_seed = int(init_seed)
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        embedding = base_model.get_input_embeddings().weight
        visual = base_model.model.visual
        if embedding.shape[1] != 2560 or visual.config.hidden_size != 1024:
            raise ValueError("V1 requires text width 2560 and visual width 1024")
        if len(visual.blocks) <= 17 or int(visual.spatial_merge_size) != 2:
            raise ValueError("V1 requires visual blocks 5/11/17 and 2x2 merger")
        try:
            vision_source = inspect.getsource(type(visual).forward)
        except (OSError, TypeError):
            vision_source = ""
        if "window_index" in vision_source or "reverse_indices" in vision_source:
            raise RuntimeError("V1 requires the V0-audited block-major visual ordering")
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)

        # Same construction and RNG-consumption order as V0 through value_blocks.
        generator = torch.Generator(device="cpu").manual_seed(self.init_seed)
        rows = torch.randint(embedding.shape[0], (20,), generator=generator)
        self.p20 = nn.Parameter(embedding.detach()[rows.to(embedding.device)].float().clone())
        self.visual_s8 = nn.Parameter(torch.empty(8, 1024, device=embedding.device))
        self.visual_av10 = nn.Parameter(torch.empty(10, 1024, device=embedding.device))
        nn.init.normal_(self.visual_s8, std=0.02)
        nn.init.normal_(self.visual_av10, std=0.02)
        self.text_projection = nn.Linear(2560, QUESTION_WIDTH, bias=False)
        self.question_depthwise = nn.Conv1d(
            QUESTION_WIDTH, QUESTION_WIDTH, 3, padding=1, groups=QUESTION_WIDTH,
        )
        self.question_pointwise = nn.Conv1d(QUESTION_WIDTH, QUESTION_WIDTH, 1)
        self.question_norm = nn.LayerNorm(QUESTION_WIDTH)
        self.question_pool = nn.Linear(QUESTION_WIDTH, 1, bias=False)
        self.query_heads = nn.ModuleList(nn.Linear(QUESTION_WIDTH, QUESTION_WIDTH) for _ in LAYERS)
        self.key_norms = nn.ModuleList(nn.LayerNorm(1024) for _ in LAYERS)
        self.key_heads = nn.ModuleList(nn.Linear(1024, QUESTION_WIDTH, bias=False) for _ in LAYERS)
        self.layer_gate = nn.Linear(QUESTION_WIDTH, 3)
        self.value_norms = nn.ModuleList(nn.LayerNorm(2560) for _ in LAYERS)
        self.value_blocks = nn.ModuleList(nn.Linear(2560, BLOCK_WIDTH, bias=False) for _ in LAYERS)
        self.prefix_output = nn.Linear(192, 2560)
        nn.init.normal_(self.prefix_output.weight, mean=0.0, std=OUTPUT_INIT_STD)
        nn.init.zeros_(self.prefix_output.bias)
        for module in (
            self.text_projection, self.question_depthwise, self.question_pointwise,
            self.question_norm, self.question_pool, self.query_heads,
            self.key_norms, self.key_heads, self.layer_gate, self.value_norms,
            self.value_blocks, self.prefix_output,
        ):
            module.to(device=embedding.device, dtype=torch.float32)
        print(
            "[V1_INIT_AUDIT] common_order=V0_through_value_blocks "
            "visual18_order=S8_then_Av10 visual18_std=0.02 "
            f"s8_actual_std={float(self.visual_s8.detach().std()):.8f} "
            f"av10_actual_std={float(self.visual_av10.detach().std()):.8f} "
            f"output_std={float(self.prefix_output.weight.detach().std()):.8f} "
            "output_bias_zero=True"
        )
        visual.blocks[17] = _Visual18Block(visual.blocks[17], self)
        self._active = False
        self._features: dict[int | str, torch.Tensor] = {}
        self._expected_visual_segments = 0
        self._expected_visual_patches = 0
        self.debug_context: dict[str, torch.Tensor] = {}
        self.first_batch_diagnostics: dict[str, float] | None = None
        self.first_backward_gradients: dict[str, float] = {}
        self._first_grad_handles = []
        self.last_injection_audit: dict[str, Any] | None = None
        self._install_capture_hooks()
        self._install_first_backward_hooks()
        self._audit_parameters()

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "p20": [self.p20], "visual_s8": [self.visual_s8],
            "visual_av10": [self.visual_av10],
            "question_context": list(self.text_projection.parameters())
            + list(self.question_depthwise.parameters()) + list(self.question_pointwise.parameters())
            + list(self.question_norm.parameters()) + list(self.question_pool.parameters()),
            "maps": list(self.query_heads.parameters()) + list(self.key_norms.parameters())
            + list(self.key_heads.parameters()),
            "layer_condition": list(self.layer_gate.parameters()) + list(self.value_norms.parameters())
            + list(self.value_blocks.parameters()),
            "prefix_output": list(self.prefix_output.parameters()),
        }

    def _audit_parameters(self) -> dict[str, int]:
        groups = self.trainable_parameter_groups()
        grouped = [p for values in groups.values() for p in values]
        active = [p for p in self.parameters() if p.requires_grad]
        if len(grouped) != len({id(p) for p in grouped}) or {id(p) for p in grouped} != {id(p) for p in active}:
            raise RuntimeError("V1 parameter groups are not a unique complete partition")
        counts = {name: sum(p.numel() for p in values) for name, values in groups.items()}
        if counts != EXPECTED_GROUP_COUNTS or sum(counts.values()) != EXPECTED_TRAINABLE:
            raise RuntimeError(f"V1 parameter budget mismatch: {counts}")
        if any(p.requires_grad for p in self.base_model.parameters()):
            raise RuntimeError("V1 backbone is not frozen")
        if hasattr(self, "offset_down") or hasattr(self, "offset_up"):
            raise RuntimeError("V0 question-offset modules leaked into V1")
        print(f"[V1_PARAMETER_AUDIT] {json.dumps(counts, sort_keys=True)} total={sum(counts.values())}")
        return counts

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def _expand(self, batch: dict[str, Any]):
        batch = dict(batch)
        ids = batch.pop("input_ids")
        attention = batch.pop("attention_mask")
        source_ids = batch.pop("question_source_ids")
        source_mask = batch.pop("question_source_mask")
        if ids.ndim != 2 or ids.shape != attention.shape:
            raise ValueError("V1 input and attention shape mismatch")
        if source_ids.ndim != 2 or source_ids.shape != source_mask.shape or source_ids.shape[0] != ids.shape[0]:
            raise ValueError("V1 independent question IDs/mask mismatch")
        if not bool(source_mask.any(dim=1).all()):
            raise ValueError("V1 requires nonempty standalone raw question tokens")
        labels = batch.pop("labels", None)
        bsz = ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        batch["input_ids"] = torch.cat((ids.new_full((bsz, 20), pad_id), ids), dim=1)
        batch["attention_mask"] = torch.cat((attention.new_ones((bsz, 20)), attention), dim=1)
        if labels is not None:
            batch["labels"] = torch.cat((labels.new_full((bsz, 20), -100), labels), dim=1)
        return batch, source_ids, source_mask

    @contextmanager
    def _injection(self, ids: torch.Tensor, grid: torch.Tensor,
                   source_ids: torch.Tensor, source_mask: torch.Tensor) -> Iterator[None]:
        if self._active:
            raise RuntimeError("V1 injection is not reentrant")
        self._active = True
        self._features = {}
        self._expected_visual_segments = int(grid[:, 0].sum())
        self._expected_visual_patches = int(grid.prod(dim=1).sum())
        question, valid = self._question_embeddings(source_ids, source_mask)
        embeddings = self.get_input_embeddings()
        language = self.base_model.model.language_model
        prefill_done = False

        def replace_p20(_module, _inputs, output):
            if output.ndim != 3 or output.shape[1] != ids.shape[1] or prefill_done:
                return output
            prompt = self.p20.to(output.dtype).unsqueeze(0).expand(output.shape[0], -1, -1)
            return torch.cat((prompt, output[:, 20:]), dim=1)

        def condition_p20(_module, args, kwargs):
            nonlocal prefill_done
            if prefill_done:
                return args, kwargs
            inputs = kwargs.get("inputs_embeds")
            if inputs is None or inputs.shape[:2] != ids.shape or inputs.shape[1] < 20:
                raise RuntimeError("V1 prefill inputs_embeds shape mismatch")
            condition, map_debug = self._condition(question, valid, grid)
            shift = self.prefix_output(torch.relu(condition))
            prompt = inputs[:, :20] + shift[:, None, :].to(inputs.dtype)
            kwargs = dict(kwargs)
            kwargs["inputs_embeds"] = torch.cat((prompt, inputs[:, 20:]), dim=1)
            if not torch.equal(kwargs["inputs_embeds"][:, 20:], inputs[:, 20:]):
                raise RuntimeError("V1 modified native chat embeddings")
            if shift.shape != (inputs.shape[0], inputs.shape[-1]) or not bool(torch.isfinite(shift).all()):
                raise RuntimeError("V1 shared P20 shift is malformed or nonfinite")
            actual_delta = prompt.float() - inputs[:, :20].float()
            shared_delta_error = (actual_delta - actual_delta[:, :1]).abs().max()
            # The same shift is broadcast to all 20 positions above. In bf16,
            # subtracting different rounded P20 bases does not recover exactly
            # the same increment; this is a diagnostic, never a failure gate.
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
                self.first_batch_diagnostics = {k: float(v.float()) for k, v in self.debug_context.items()}
            self.last_injection_audit = {
                "prefix_only": True, "positions": 20,
                "max_effective_delta_spread_bf16": float(shared_delta_error),
                "native_embeddings_unchanged": True,
            }
            prefill_done = True
            return args, kwargs

        embedding_hook = embeddings.register_forward_hook(replace_p20)
        language_hook = language.register_forward_pre_hook(condition_p20, with_kwargs=True)
        try:
            yield
            if not prefill_done:
                raise RuntimeError("V1 prefill did not reach the LLM")
        finally:
            language_hook.remove()
            embedding_hook.remove()
            self._features = {}
            self._active = False

    def forward(self, **kwargs: Any):
        expanded, source_ids, source_mask = self._expand(kwargs)
        with self._injection(expanded["input_ids"], expanded["image_grid_thw"], source_ids, source_mask):
            return self.base_model(**expanded)

    def generate(self, **kwargs: Any):
        expanded, source_ids, source_mask = self._expand(kwargs)
        with self._injection(expanded["input_ids"], expanded["image_grid_thw"], source_ids, source_mask):
            return self.base_model.generate(**expanded)

    def save_v1(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "visual_selection_prefix_p20_v1", "init_seed": self.init_seed,
            "layers": list(LAYERS), "question_width": QUESTION_WIDTH,
            "block_width": BLOCK_WIDTH, "condition_width": 192,
            "visual_tokens": [8, 10], "prefix_tokens": 20,
            "output_init_distribution": "normal", "output_init_std": OUTPUT_INIT_STD,
            "trainable_parameters": self._audit_parameters(),
        }
        with (path / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(
            {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad},
            path / WEIGHTS_NAME,
        )

    def load_v1(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        with (path / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("method") != "visual_selection_prefix_p20_v1" or tuple(config.get("layers", ())) != LAYERS:
            raise ValueError("V1 checkpoint architecture mismatch")
        if int(config["init_seed"]) != self.init_seed:
            raise ValueError("V1 checkpoint seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("V1 checkpoint trainable tensor set mismatch")
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise ValueError(f"V1 checkpoint tensor shape mismatch: {name}")
            parameter.data.copy_(state[name].to(parameter.device))
