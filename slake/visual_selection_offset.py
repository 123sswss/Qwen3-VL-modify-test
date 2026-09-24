"""V0: question-guided three-layer visual selection and token-shared offset.

This is intentionally independent of the legacy QDPT/MMRL modules. The frozen
Qwen3-VL performs its normal vision, DeepStack and language passes exactly once.
"""

from __future__ import annotations

import json
import inspect
import math
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn


CONFIG_NAME = "visual_selection_offset_config.json"
WEIGHTS_NAME = "visual_selection_offset.pt"
LAYERS = (5, 11, 17)
QUESTION_WIDTH = 128
OFFSET_WIDTH = 192
BLOCK_WIDTH = 64
OUTPUT_INIT_STD = 1e-4
EXPECTED_TRAINABLE = 2_356_675
EXPECTED_GROUP_COUNTS = {
    "p20": 51_200,
    "visual_s8": 8_192,
    "visual_av10": 10_240,
    "question_context": 345_088,
    "maps": 448_896,
    "layer_condition": 507_267,
    "offset": 985_792,
}


def locate_question_mask(
    input_ids: torch.Tensor,
    question: str,
    tokenizer: Any,
    *,
    context_mask: torch.Tensor | None = None,
    prompt_text: str | None = None,
) -> torch.Tensor:
    """Locate the exact raw-question token span; never infer it from labels.

    A non-unique or tokenizer-dependent boundary is a hard error, not an excuse
    to include chat instructions, image markers, or teacher-forcing answers.
    """
    if input_ids.ndim != 1:
        raise ValueError("question locator expects one unpadded token sequence")
    question_ids = tokenizer.encode(str(question).strip(), add_special_tokens=False)
    if not question_ids:
        raise ValueError("raw question tokenization is empty")
    tokens = input_ids.tolist()
    valid = context_mask.tolist() if context_mask is not None else [True] * len(tokens)
    length = len(question_ids)
    matches = [
        start
        for start in range(len(tokens) - length + 1)
        if tokens[start : start + length] == question_ids
        and all(valid[start : start + length])
    ]
    if len(matches) != 1 and prompt_text is not None:
        # Evaluation appends a short-answer instruction after the raw question.
        # A BPE token may straddle the question/newline boundary. Match the
        # complete user text, then retain tokens that overlap the raw-question
        # character range. A tokenizer may attach the following newline to the
        # final question token; requiring its end offset to remain inside the
        # question would incorrectly drop that training-time question token.
        if not prompt_text.startswith(str(question).strip()):
            raise ValueError("evaluation prompt does not start with the raw question")
        encoded = tokenizer(
            prompt_text, add_special_tokens=False, return_offsets_mapping=True,
        )
        prompt_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        spans = [
            start for start in range(len(tokens) - len(prompt_ids) + 1)
            if tokens[start : start + len(prompt_ids)] == prompt_ids
            and all(valid[start : start + len(prompt_ids)])
        ]
        if len(spans) != 1:
            raise ValueError(f"full user text has no unique token span; matches={spans}")
        indices = [
            spans[0] + index for index, (start, end) in enumerate(offsets)
            if start < len(str(question).strip()) and end > 0
        ]
        if not indices:
            raise ValueError("no raw-question-overlapping tokens after BPE alignment")
        if len(indices) != len(question_ids):
            raise ValueError(
                "prefill question positions cannot align one-to-one with the "
                f"training question tokens: training_count={len(question_ids)} "
                f"prefill_count={len(indices)}"
            )
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[indices] = True
        return mask
    if len(matches) != 1:
        raise ValueError(
            "raw question must have exactly one exact token span in the user "
            f"context; matches={matches} question={question!r}"
        )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[matches[0] : matches[0] + length] = True
    return mask


class _Visual18Block(nn.Module):
    """Insert S8 + A_v10 before block 17 and strip them after one block call."""

    def __init__(self, block: nn.Module, owner: "VisualSelectionOffsetModel") -> None:
        super().__init__()
        self.block = block
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))

    @staticmethod
    def _prefix(tensor: torch.Tensor, lengths: list[int], count: int, fill: float):
        chunks = torch.split(tensor, lengths, dim=0)
        return torch.cat(
            [
                torch.cat((tensor.new_full((count, *tensor.shape[1:]), fill), chunk))
                for chunk in chunks
            ],
            dim=0,
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        owner = self._owner_ref()
        if owner is None or not owner._active:
            return self.block(*args, **kwargs)
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cu = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        if hidden is None or cu is None:
            raise RuntimeError("Visual18 requires hidden_states and cu_seqlens")
        lengths = [int(x) for x in (cu[1:] - cu[:-1]).tolist()]
        if not lengths or min(lengths) < 1 or sum(lengths) != hidden.shape[0]:
            raise RuntimeError("Visual18 segment lengths do not match ViT tokens")
        if len(lengths) != owner._expected_visual_segments or sum(lengths) != owner._expected_visual_patches:
            raise RuntimeError("Visual18 segments are not one native sequence per image/frame")
        prompt = torch.cat((owner.visual_s8, owner.visual_av10), dim=0).to(
            device=hidden.device, dtype=hidden.dtype
        )
        count = prompt.shape[0]
        expanded = torch.cat(
            [torch.cat((prompt, chunk), dim=0) for chunk in torch.split(hidden, lengths)],
            dim=0,
        )
        new_args, new_kwargs = list(args), dict(kwargs)

        def replace(name: str, position: int, value: Any) -> None:
            if name in new_kwargs:
                new_kwargs[name] = value
            elif len(new_args) > position:
                new_args[position] = value
            else:
                new_kwargs[name] = value

        replace("hidden_states", 0, expanded)
        replace("cu_seqlens", 1, cu + torch.arange(cu.numel(), device=cu.device, dtype=cu.dtype) * count)
        rotary = kwargs.get("rotary_pos_emb", args[2] if len(args) > 2 else None)
        if rotary is not None:
            replace("rotary_pos_emb", 2, self._prefix(rotary, lengths, count, 0.0))
        position = kwargs.get("position_embeddings", args[3] if len(args) > 3 else None)
        if position is not None:
            cos, sin = position
            replace(
                "position_embeddings", 3,
                (self._prefix(cos, lengths, count, 1.0), self._prefix(sin, lengths, count, 0.0)),
            )
        if "max_seqlen" in new_kwargs:
            new_kwargs["max_seqlen"] = int(new_kwargs["max_seqlen"]) + count
        output = self.block(*new_args, **new_kwargs)
        if not torch.is_tensor(output):
            raise TypeError("Visual18 block must return a tensor")
        chunks = torch.split(output, [length + count for length in lengths], dim=0)
        return torch.cat([chunk[count:] for chunk in chunks], dim=0)


class VisualSelectionOffsetModel(nn.Module):
    """Frozen Qwen3-VL with P20, Visual18 and the V0 conditional offset."""

    def __init__(self, base_model: nn.Module, init_seed: int = 44) -> None:
        super().__init__()
        self.base_model = base_model
        self.init_seed = int(init_seed)
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        embedding = base_model.get_input_embeddings().weight
        visual = base_model.model.visual
        if embedding.shape[1] != 2560 or visual.config.hidden_size != 1024:
            raise ValueError("V0 is fixed to text width 2560 and vision width 1024")
        if len(visual.blocks) <= 17:
            raise ValueError("V0 requires visual code indexes 5, 11, 17")
        if int(visual.spatial_merge_size) != 2:
            raise ValueError("V0 map geometry currently requires spatial_merge_size=2")
        try:
            vision_source = inspect.getsource(type(visual).forward)
        except (OSError, TypeError):
            vision_source = ""
        if "window_index" in vision_source or "reverse_indices" in vision_source:
            raise RuntimeError(
                "V0 fixed 2x2 map requires block-major vision ordering; this "
                "vision implementation reorders windows and needs an explicit inverse map"
            )
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)

        generator = torch.Generator(device="cpu").manual_seed(self.init_seed)
        rows = torch.randint(embedding.shape[0], (20,), generator=generator)
        self.p20 = nn.Parameter(embedding.detach()[rows.to(embedding.device)].float().clone())
        # Match the old Dense Sandwich Visual18 rule: S8 and A_v10 are separate
        # 1024-wide tables at code index 17, each initialized N(0, 0.02^2).
        self.visual_s8 = nn.Parameter(torch.empty(8, 1024, device=embedding.device))
        self.visual_av10 = nn.Parameter(torch.empty(10, 1024, device=embedding.device))
        nn.init.normal_(self.visual_s8, std=0.02)
        nn.init.normal_(self.visual_av10, std=0.02)
        print(
            "[V0_VISUAL18_INIT_AUDIT] layer=17 order=S8_then_Av10 "
            "baseline_distribution=Normal(0,0.02) "
            f"s8_shape={tuple(self.visual_s8.shape)} s8_std={float(self.visual_s8.detach().std()):.8f} "
            f"av10_shape={tuple(self.visual_av10.shape)} av10_std={float(self.visual_av10.detach().std()):.8f}"
        )

        self.text_projection = nn.Linear(2560, QUESTION_WIDTH, bias=False)
        self.question_depthwise = nn.Conv1d(
            QUESTION_WIDTH, QUESTION_WIDTH, 3, padding=1, groups=QUESTION_WIDTH
        )
        self.question_pointwise = nn.Conv1d(QUESTION_WIDTH, QUESTION_WIDTH, 1)
        self.question_norm = nn.LayerNorm(QUESTION_WIDTH)
        self.question_pool = nn.Linear(QUESTION_WIDTH, 1, bias=False)
        self.query_heads = nn.ModuleList(
            nn.Linear(QUESTION_WIDTH, QUESTION_WIDTH) for _ in LAYERS
        )
        self.key_norms = nn.ModuleList(nn.LayerNorm(1024) for _ in LAYERS)
        self.key_heads = nn.ModuleList(
            nn.Linear(1024, QUESTION_WIDTH, bias=False) for _ in LAYERS
        )
        self.layer_gate = nn.Linear(QUESTION_WIDTH, 3)
        self.value_norms = nn.ModuleList(nn.LayerNorm(2560) for _ in LAYERS)
        self.value_blocks = nn.ModuleList(
            nn.Linear(2560, BLOCK_WIDTH, bias=False) for _ in LAYERS
        )
        self.offset_down = nn.Linear(2560, OFFSET_WIDTH)
        self.offset_up = nn.Linear(OFFSET_WIDTH, 2560)
        # Small *nonzero* terminal weights give Q/K, convolution and layer gate
        # a gradient on the first backward pass. No zero gate/projection follows.
        nn.init.normal_(self.offset_up.weight, mean=0.0, std=OUTPUT_INIT_STD)
        nn.init.zeros_(self.offset_up.bias)
        print(
            "[V0_OFFSET_INIT_AUDIT] distribution=Normal(0,1e-4) "
            f"actual_std={float(self.offset_up.weight.detach().std()):.8f} bias_zero=True"
        )
        # device_map may already have placed the frozen backbone on CUDA. Move
        # only the small new modules; moving self would relocate the 4B backbone.
        for module in (
            self.text_projection, self.question_depthwise, self.question_pointwise,
            self.question_norm, self.question_pool, self.query_heads,
            self.key_norms, self.key_heads, self.layer_gate, self.value_norms,
            self.value_blocks, self.offset_down, self.offset_up,
        ):
            module.to(device=embedding.device, dtype=torch.float32)

        visual.blocks[17] = _Visual18Block(visual.blocks[17], self)
        self._active = False
        self._features: dict[int | str, torch.Tensor] = {}
        self._expected_visual_segments = 0
        self._expected_visual_patches = 0
        self.debug_context: dict[str, torch.Tensor] = {}
        # Inference-only, non-persistent diagnostics. Never included in V0 checkpoints.
        self.inference_intervention = "normal"
        self.diagnostic_capture = False
        self.diagnostic_alternative_question_ids: torch.Tensor | None = None
        self.last_forward_probe: dict[str, Any] | None = None
        self.first_batch_diagnostics: dict[str, float] | None = None
        self.first_backward_gradients: dict[str, float] = {}
        self._first_grad_handles = []
        self._install_capture_hooks()
        self._install_first_backward_hooks()
        self._audit_parameters()

    def _install_capture_hooks(self) -> None:
        visual = self.base_model.model.visual
        for index in LAYERS:
            def capture(_module, _args, output, layer=index):
                if self._active:
                    if not torch.is_tensor(output):
                        raise TypeError(f"ViT layer {layer} did not return a tensor")
                    self._features[layer] = output
            visual.blocks[index].register_forward_hook(capture)

        def capture_value(_module, _args, output):
            if self._active:
                if not torch.is_tensor(output):
                    raise TypeError("native visual merger did not return a tensor")
                self._features["value"] = output
        visual.merger.register_forward_hook(capture_value)

    def _install_first_backward_hooks(self) -> None:
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            def capture(gradient, key=name):
                if key not in self.first_backward_gradients:
                    self.first_backward_gradients[key] = float(gradient.detach().float().norm())
                return gradient
            self._first_grad_handles.append(parameter.register_hook(capture))

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "p20": [self.p20],
            "visual_s8": [self.visual_s8],
            "visual_av10": [self.visual_av10],
            "question_context": list(self.text_projection.parameters())
            + list(self.question_depthwise.parameters())
            + list(self.question_pointwise.parameters())
            + list(self.question_norm.parameters())
            + list(self.question_pool.parameters()),
            "maps": list(self.query_heads.parameters())
            + list(self.key_norms.parameters())
            + list(self.key_heads.parameters()),
            "layer_condition": list(self.layer_gate.parameters())
            + list(self.value_norms.parameters())
            + list(self.value_blocks.parameters()),
            "offset": list(self.offset_down.parameters()) + list(self.offset_up.parameters()),
        }

    def _audit_parameters(self) -> dict[str, int]:
        groups = self.trainable_parameter_groups()
        grouped = [p for values in groups.values() for p in values]
        active = [p for p in self.parameters() if p.requires_grad]
        if len(grouped) != len({id(p) for p in grouped}) or {id(p) for p in grouped} != {id(p) for p in active}:
            raise RuntimeError("V0 trainable parameters must belong to exactly one group")
        counts = {key: sum(p.numel() for p in values) for key, values in groups.items()}
        if counts != EXPECTED_GROUP_COUNTS:
            raise RuntimeError(f"V0 per-module parameter budget mismatch: {counts}")
        if sum(counts.values()) != EXPECTED_TRAINABLE:
            raise RuntimeError(f"V0 parameter budget mismatch: {counts}")
        if any(p.requires_grad for p in self.base_model.parameters()):
            raise RuntimeError("V0 backbone is not frozen")
        print(f"[V0_PARAMETER_AUDIT] {json.dumps(counts, sort_keys=True)} total={sum(counts.values())}")
        return counts

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def _expand(self, batch: dict[str, Any]):
        batch = dict(batch)
        ids = batch.pop("input_ids")
        attention = batch.pop("attention_mask")
        question_mask = batch.pop("question_mask")
        question_source_ids = batch.pop("question_source_ids", ids)
        question_source_mask = batch.pop("question_source_mask", question_mask)
        if ids.shape != attention.shape or ids.shape != question_mask.shape:
            raise ValueError("input, attention and question masks must align")
        if not bool((question_mask & attention.bool()).any(dim=1).all()):
            raise ValueError("each V0 sample requires real question tokens")
        if bool((question_mask & ~attention.bool()).any()):
            raise ValueError("question mask overlaps padding")
        if (question_source_ids.ndim != 2
            or question_source_ids.shape != question_source_mask.shape
            or question_source_ids.shape[0] != ids.shape[0]):
            raise ValueError("question source IDs and mask must align by batch")
        target_counts = question_mask.sum(dim=1)
        source_counts = question_source_mask.sum(dim=1)
        if not torch.equal(target_counts.to(source_counts.device), source_counts):
            raise ValueError(
                "standalone question source tokens must align one-to-one with "
                f"prefill target positions: source={source_counts.tolist()} "
                f"target={target_counts.tolist()}"
            )
        labels = batch.pop("labels", None)
        if labels is not None and bool((question_mask & labels.ne(-100)).any()):
            raise ValueError("question mask overlaps teacher-forcing answers")
        bsz = ids.shape[0]
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        expanded_ids = torch.cat((ids.new_full((bsz, 20), pad_id), ids), dim=1)
        expanded_attention = torch.cat((attention.new_ones((bsz, 20)), attention), dim=1)
        expanded_question = torch.cat((question_mask.new_zeros((bsz, 20)), question_mask), dim=1)
        batch.update(input_ids=expanded_ids, attention_mask=expanded_attention)
        if labels is not None:
            batch["labels"] = torch.cat((labels.new_full((bsz, 20), -100), labels), dim=1)
        return batch, expanded_question, question_source_ids, question_source_mask

    def _question_embeddings(self, ids: torch.Tensor, mask: torch.Tensor):
        weight = self.get_input_embeddings().weight
        vectors = [weight[ids[b, mask[b]].to(weight.device)].float() for b in range(ids.shape[0])]
        max_len = max(x.shape[0] for x in vectors)
        padded = weight.new_zeros((len(vectors), max_len, weight.shape[1]), dtype=torch.float32)
        valid = torch.zeros((len(vectors), max_len), dtype=torch.bool, device=weight.device)
        for b, vector in enumerate(vectors):
            padded[b, :vector.shape[0]] = vector
            valid[b, :vector.shape[0]] = True
        return padded, valid

    def _condition(self, question: torch.Tensor, valid: torch.Tensor, grid: torch.Tensor,
                   probe: dict[str, Any] | None = None):
        features = getattr(self, "diagnostic_condition_features", None) or self._features
        if set(features) != {*LAYERS, "value"}:
            raise RuntimeError(f"V0 expected layers {LAYERS} and native Value, got {set(features)}")
        x = self.text_projection(question)
        x = x * valid.unsqueeze(-1)
        conv = self.question_pointwise(torch.nn.functional.gelu(self.question_depthwise(x.transpose(1, 2))))
        x = (x + conv.transpose(1, 2)) * valid.unsqueeze(-1)
        pool_logits = self.question_pool(self.question_norm(x)).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~valid, -torch.inf)
        u = (pool_logits.softmax(dim=-1).unsqueeze(-1) * x).sum(dim=1)
        beta = self.layer_gate(u).softmax(dim=-1)
        if probe is not None:
            probe["layer_weights"] = beta.detach().float().cpu().clone()
        grid = grid.to(device=question.device)
        if grid.ndim != 2 or grid.shape[0] != question.shape[0]:
            raise RuntimeError("V0 currently requires exactly one image per sample")
        patches = [int(t * h * w) for t, h, w in grid.tolist()]
        if any(int(h) % 2 or int(w) % 2 for _, h, w in grid.tolist()):
            raise RuntimeError("visual height and width must each be divisible by spatial merge size")
        value_counts = [n // 4 for n in patches]
        if features["value"].shape[0] != sum(value_counts):
            raise RuntimeError("native Value length does not match post-merger image geometry")
        values = torch.split(
            features["value"].to(device=question.device, dtype=torch.float32),
            value_counts, dim=0,
        )
        blocks = []
        diagnostics: dict[str, torch.Tensor] = {}
        for li, layer in enumerate(LAYERS):
            h_all = features[layer]
            if h_all.shape[0] != sum(patches):
                raise RuntimeError(f"ViT layer {layer} contains unexpected prompt/padding tokens")
            segments = torch.split(h_all.to(device=question.device, dtype=torch.float32), patches, dim=0)
            q = self.query_heads[li](u)
            summaries = []
            entropies, top_mass = [], []
            merged_maps = []
            for b, (h, value) in enumerate(zip(segments, values)):
                keys = self.key_heads[li](self.key_norms[li](h))
                logits = (keys * q[b]).sum(dim=-1) / math.sqrt(QUESTION_WIDTH)
                patch_prob = logits.softmax(dim=0)
                merged_prob = patch_prob.reshape(-1, 4).sum(dim=1)
                if getattr(self, "diagnostic_uniform_maps", False):
                    merged_prob = torch.full_like(merged_prob, 1.0 / merged_prob.numel())
                if merged_prob.shape[0] != value.shape[0] or not torch.allclose(
                    merged_prob.sum(), merged_prob.new_tensor(1.0), atol=1e-4
                ):
                    raise RuntimeError("V0 patch-to-Value map lost probability or image alignment")
                summaries.append((merged_prob[:, None] * value).sum(dim=0))
                if probe is not None:
                    merged_maps.append(merged_prob)
                entropies.append(-(patch_prob * patch_prob.clamp_min(1e-12).log()).sum() / math.log(patch_prob.numel()))
                top_mass.append(merged_prob.max())
            z = torch.stack(summaries)
            if probe is not None:
                probe[f"map{layer}"] = [
                    p.detach().float().cpu().clone() for p in merged_maps
                ]
                probe[f"summary{layer}"] = z.detach().float().cpu().clone()
            block = self.value_blocks[li](self.value_norms[li](z)) * beta[:, li, None]
            blocks.append(block)
            diagnostics[f"map{layer}_entropy_norm"] = torch.stack(entropies).mean().detach()
            diagnostics[f"map{layer}_top_mass"] = torch.stack(top_mass).mean().detach()
            diagnostics[f"layer{layer}_weight"] = beta[:, li].mean().detach()
            diagnostics[f"layer{layer}_summary_rms"] = z.square().mean().sqrt().detach()
        return torch.cat(blocks, dim=-1), diagnostics

    @contextmanager
    def _injection(self, ids: torch.Tensor, question_mask: torch.Tensor,
                   grid: torch.Tensor, question_source_ids: torch.Tensor,
                   question_source_mask: torch.Tensor) -> Iterator[None]:
        if self._active:
            raise RuntimeError("V0 injection context is not reentrant")
        self._active = True
        self._features = {}
        self._expected_visual_segments = int(grid[:, 0].sum())
        self._expected_visual_patches = int(grid.prod(dim=1).sum())
        question, valid = self._question_embeddings(
            question_source_ids, question_source_mask,
        )
        embeddings = self.get_input_embeddings()
        language = self.base_model.model.language_model
        prefill_done = False

        def replace_p20(_module, _inputs, output):
            if output.ndim != 3 or output.shape[1] != ids.shape[1] or prefill_done:
                return output
            prompt = self.p20.to(output.dtype).unsqueeze(0).expand(output.shape[0], -1, -1)
            return torch.cat((prompt, output[:, 20:]), dim=1)

        def replace_question(_module, args, kwargs):
            nonlocal prefill_done
            if prefill_done:
                return args, kwargs  # generated answer uses the native KV cache
            inputs = kwargs.get("inputs_embeds")
            if inputs is None or inputs.shape[:2] != question_mask.shape:
                raise RuntimeError("V0 prefill inputs_embeds do not align with question mask")
            if self.inference_intervention not in {"normal", "offset_off", "condition_off"}:
                raise ValueError(f"Unknown V0 inference intervention: {self.inference_intervention}")
            if self.training and self.inference_intervention != "normal":
                raise RuntimeError("V0 interventions are inference-only")
            probe = {} if self.diagnostic_capture else None
            condition, map_debug = self._condition(question, valid, grid, probe=probe)
            down = self.offset_down(question)
            applied_condition = (
                torch.zeros_like(condition)
                if self.inference_intervention == "condition_off" else condition
            )
            delta = self.offset_up(torch.relu(down + applied_condition[:, None, :]))
            delta = delta * valid.unsqueeze(-1)
            if probe is not None:
                text = torch.nn.functional.linear(question, self.offset_down.weight, None)
                bias = self.offset_down.bias
                if bias is None:
                    raise RuntimeError("V0 offset_down bias unexpectedly absent")
                alt = self.diagnostic_alternative_question_ids
                if alt is not None:
                    if ids.shape[0] != 1:
                        raise RuntimeError("V0 alternative question probe requires one image")
                    alt = alt.to(self.get_input_embeddings().weight.device)
                    if alt.ndim != 1 or alt.numel() < 1:
                        raise ValueError("V0 alternative question IDs must be nonempty 1-D")
                    alt_question = self.get_input_embeddings().weight[alt].float().unsqueeze(0)
                    alt_valid = torch.ones(alt_question.shape[:2], device=alt_question.device, dtype=torch.bool)
                    alt_probe: dict[str, Any] = {}
                    self._condition(alt_question, alt_valid, grid, probe=alt_probe)
                    probe["alternative"] = alt_probe
                probe.update({
                    "question": question.detach().float().cpu().clone(),
                    "valid": valid.detach().cpu().clone(),
                    "text": text.detach().float().cpu().clone(),
                    "condition": condition.detach().float().cpu().clone(),
                    "bias": bias.detach().float().cpu().clone(),
                    "preactivation": (down + condition[:, None, :]).detach().float().cpu().clone(),
                    "offset": self.offset_up(torch.relu(down + condition[:, None, :])).detach().float().cpu().clone(),
                    "offset_condition_off": self.offset_up(torch.relu(down)).detach().float().cpu().clone(),
                })
                self.last_forward_probe = probe
            if self.inference_intervention == "offset_off":
                delta = torch.zeros_like(delta)
            full_delta = inputs.new_zeros(inputs.shape)
            for b in range(ids.shape[0]):
                full_delta[b, question_mask[b]] = delta[b, valid[b]].to(inputs.dtype)
            kwargs = dict(kwargs)
            kwargs["inputs_embeds"] = inputs + full_delta
            q_rms = question[valid].square().mean().sqrt().clamp_min(1e-8)
            d_rms = delta[valid].square().mean().sqrt()
            self.debug_context = {
                **map_debug,
                "question_rms": q_rms.detach(),
                "condition_rms": condition.square().mean().sqrt().detach(),
                "down_rms": down[valid].square().mean().sqrt().detach(),
                "native_value_rms": self._features["value"].float().square().mean().sqrt().detach(),
                "offset_rms": d_rms.detach(),
                "offset_to_question_rms": (d_rms / q_rms).detach(),
                "question_tokens": valid.sum().float().detach(),
            }
            if self.training and self.first_batch_diagnostics is None:
                self.first_batch_diagnostics = {
                    key: float(value.float()) for key, value in self.debug_context.items()
                }
            prefill_done = True
            return args, kwargs

        embedding_hook = embeddings.register_forward_hook(replace_p20)
        language_hook = language.register_forward_pre_hook(replace_question, with_kwargs=True)
        try:
            yield
            if not prefill_done:
                raise RuntimeError("V0 prefill never reached language model")
        finally:
            language_hook.remove()
            embedding_hook.remove()
            self._features = {}
            self._active = False

    def forward(self, **kwargs: Any):
        expanded, mask, source_ids, source_mask = self._expand(kwargs)
        with self._injection(
            expanded["input_ids"], mask, expanded["image_grid_thw"],
            source_ids, source_mask,
        ):
            return self.base_model(**expanded)

    def generate(self, **kwargs: Any):
        expanded, mask, source_ids, source_mask = self._expand(kwargs)
        with self._injection(
            expanded["input_ids"], mask, expanded["image_grid_thw"],
            source_ids, source_mask,
        ):
            return self.base_model.generate(**expanded)

    def save_v0(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "method": "visual_selection_offset_v0", "init_seed": self.init_seed,
            "layers": list(LAYERS), "question_width": QUESTION_WIDTH,
            "offset_width": OFFSET_WIDTH, "block_width": BLOCK_WIDTH,
            "visual_tokens": [8, 10], "output_init_distribution": "normal",
            "output_init_std": OUTPUT_INIT_STD,
            "trainable_parameters": self._audit_parameters(),
        }
        with (path / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        torch.save(
            {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad},
            path / WEIGHTS_NAME,
        )

    def load_v0(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        with (path / CONFIG_NAME).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("method") != "visual_selection_offset_v0" or tuple(config.get("layers", ())) != LAYERS:
            raise ValueError("V0 checkpoint architecture mismatch")
        if int(config["init_seed"]) != self.init_seed:
            raise ValueError("V0 checkpoint seed mismatch")
        state = torch.load(path / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("V0 checkpoint trainable tensor set mismatch")
        for name, p in parameters.items():
            if tuple(state[name].shape) != tuple(p.shape):
                raise ValueError(f"V0 checkpoint tensor shape mismatch: {name}")
            p.data.copy_(state[name].to(p.device))
