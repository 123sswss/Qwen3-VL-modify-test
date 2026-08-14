"""Offline matched-vs-shuffled memory causality test for dynamic Rep.

The model is never updated. For each SLAKE batch, the script repeats the same
teacher-forced forward while swapping only visual memory, only text memory, or
both between samples at the Cross-Attention boundary.
"""

from __future__ import annotations

import argparse
import gc
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dissect_dynamic_rep import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    IndexedCollator,
    IndexedDataset,
    _to_device,
    checkpoint_label,
    numeric_summary,
    per_sample_ce,
    resolve_checkpoint,
    write_json,
    write_jsonl,
)
from inferEngine import ModelInterface
from slake.data_pipeline import SLAKEDataCollator, SLAKEDataset


MEMORY_SHUFFLE_MODES = ("visual", "text", "both")
KEY_SHUFFLE_MODES = ("key_visual", "key_text", "key_both")
VALUE_SHUFFLE_MODES = ("value_visual", "value_text", "value_both")
KEY_STRUCTURE_MODES = ("key_modality_mean", "key_zero")
STATIC_ROUTING_MODES = ("attention_static_modality",)
SHUFFLE_MODES = (
    *MEMORY_SHUFFLE_MODES,
    *KEY_SHUFFLE_MODES,
    *VALUE_SHUFFLE_MODES,
    *KEY_STRUCTURE_MODES,
    *STATIC_ROUTING_MODES,
)
SAMPLE_METRICS = (
    "baseline_ce",
    "shuffled_ce",
    "ce_delta",
    "supervised_token_flip_rate",
    "baseline_token_accuracy",
    "shuffled_token_accuracy",
    "visual_memory_change_ratio",
    "text_memory_change_ratio",
    "key_change_ratio",
    "value_change_ratio",
    "attention_tv_mean",
    "attention_js_mean",
    "context_change_ratio_mean",
    "delta_change_ratio_mean",
    "dynamic_rep_change_ratio_mean",
    "delta_cosine_mean",
)
LAYER_METRICS = (
    "attention_tv",
    "attention_js",
    "attention_change_ratio",
    "context_change_ratio",
    "context_cosine",
    "delta_change_ratio",
    "delta_cosine",
    "dynamic_rep_change_ratio",
    "dynamic_rep_cosine",
)


def _get_call_value(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    position: int,
    name: str,
) -> Any:
    return kwargs.get(name, args[position] if len(args) > position else None)


def _replace_call_value(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    position: int,
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if len(new_args) > position:
        new_args[position] = value
    else:
        new_kwargs[name] = value
    return tuple(new_args), new_kwargs


class MemoryShuffleController:
    """Swap raw memory or projected K/V between samples at the CA boundary."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.mode = "matched"
        self.last_permutation: torch.Tensor | None = None
        self.memory_split_index: int | None = None
        self.static_visual_mass: torch.Tensor | None = None
        self.suspend_projection_hooks = False
        self.pre_handle = module.register_forward_pre_hook(
            self._hook,
            with_kwargs=True,
        )
        self.key_handle = module.key_projection.register_forward_hook(
            lambda _module, _args, output: self._projected_hook("key", output)
        )
        self.value_handle = module.value_projection.register_forward_hook(
            lambda _module, _args, output: self._projected_hook("value", output)
        )
        self.output_handle = module.register_forward_hook(
            self._static_routing_hook,
            with_kwargs=True,
        )

    def close(self) -> None:
        self.pre_handle.remove()
        self.key_handle.remove()
        self.value_handle.remove()
        self.output_handle.remove()

    def set_static_visual_mass(self, visual_mass: torch.Tensor) -> None:
        if visual_mass.ndim != 3:
            raise ValueError(
                "Static visual mass must be [heads,layers,slots], got "
                f"{tuple(visual_mass.shape)}"
            )
        self.static_visual_mass = visual_mass.detach().float().clamp(0.0, 1.0)

    def static_attention_weights(
        self,
        batch_size: int,
        memory_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.static_visual_mass is None or self.memory_split_index is None:
            raise RuntimeError("Static modality routing has not been calibrated")
        split = self.memory_split_index
        text_count = memory_count - split
        if split < 1 or text_count < 1:
            raise RuntimeError(f"Invalid static routing split {split}/{memory_count}")
        visual_mass = self.static_visual_mass.to(device=device)
        visual_weights = visual_mass.unsqueeze(-1).expand(
            *visual_mass.shape,
            split,
        ) / float(split)
        text_weights = (1.0 - visual_mass).unsqueeze(-1).expand(
            *visual_mass.shape,
            text_count,
        ) / float(text_count)
        weights = torch.cat((visual_weights, text_weights), dim=-1)
        return weights.reshape(
            1,
            visual_mass.shape[0],
            visual_mass.shape[1] * visual_mass.shape[2],
            memory_count,
        ).expand(batch_size, -1, -1, -1)

    def _static_routing_hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.mode != "attention_static_modality":
            return None
        queries = _get_call_value(args, kwargs, 0, "queries")
        memory = _get_call_value(args, kwargs, 1, "memory")
        group_shape = kwargs.get("query_group_shape")
        if queries is None or memory is None or group_shape is None:
            raise RuntimeError("Static routing hook did not receive CA inputs")
        with torch.no_grad():
            normalized_memory = module.memory_norm(memory)
            value_states = module._split_heads(
                module.value_projection(normalized_memory)
            )
            weights = self.static_attention_weights(
                queries.shape[0],
                memory.shape[1],
                queries.device,
            ).to(value_states.dtype)
            context = torch.matmul(weights, value_states)
            context = context.transpose(1, 2).contiguous().view_as(queries)
            delta = module.output_projection(context)
            if module.layer_lora_target == "output":
                grouped_context = module._group_layers(context, group_shape)
                delta = delta + module.layer_lora(grouped_context).reshape_as(delta)
        return queries + delta, delta

    def _projected_hook(
        self,
        target: str,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.suspend_projection_hooks:
            return None
        if not self.mode.startswith(f"{target}_"):
            return None
        return self.apply_projected_intervention(target, output)

    def apply_projected_intervention(
        self,
        target: str,
        projected: torch.Tensor,
    ) -> torch.Tensor:
        expected_prefix = f"{target}_"
        if not self.mode.startswith(expected_prefix):
            return projected
        if self.last_permutation is None or self.memory_split_index is None:
            raise RuntimeError("Projected K/V shuffle ran before CA metadata capture")
        split = self.memory_split_index
        if target == "key" and self.mode == "key_zero":
            return torch.zeros_like(projected)
        if target == "key" and self.mode == "key_modality_mean":
            visual = projected[:, :split]
            text = projected[:, split:]
            visual = visual.mean(dim=1, keepdim=True).expand_as(visual)
            text = text.mean(dim=1, keepdim=True).expand_as(text)
            return torch.cat((visual, text), dim=1)
        modality = self.mode.removeprefix(expected_prefix)
        if modality not in MEMORY_SHUFFLE_MODES:
            raise RuntimeError(f"Unsupported projected shuffle mode: {self.mode}")
        permutation = self.last_permutation.to(projected.device)
        visual = projected[:, :split]
        text = projected[:, split:]
        if modality in {"visual", "both"}:
            visual = visual.index_select(0, permutation)
        if modality in {"text", "both"}:
            text = text.index_select(0, permutation)
        return torch.cat((visual, text), dim=1)

    def _hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        memory = _get_call_value(args, kwargs, 1, "memory")
        split_value = kwargs.get("memory_split_index")
        if memory is None or split_value is None:
            raise RuntimeError("Memory shuffle hook did not receive CA memory metadata")
        batch_size, memory_count = memory.shape[:2]
        if batch_size < 2:
            raise RuntimeError("Memory shuffle requires at least two samples per batch")
        split = int(split_value)
        if not 0 < split < memory_count:
            raise RuntimeError(f"Invalid memory split {split}/{memory_count}")
        self.memory_split_index = split

        permutation = torch.roll(
            torch.arange(batch_size, device=memory.device),
            shifts=1,
        )
        self.last_permutation = permutation.detach().cpu()
        if self.mode == "matched" or self.mode not in MEMORY_SHUFFLE_MODES:
            if self.mode != "matched" and self.mode not in SHUFFLE_MODES:
                raise RuntimeError(f"Unsupported shuffle mode: {self.mode}")
            return None

        visual = memory[:, :split]
        text = memory[:, split:]
        if self.mode in {"visual", "both"}:
            visual = visual.index_select(0, permutation)
        if self.mode in {"text", "both"}:
            text = text.index_select(0, permutation)
        shuffled_memory = torch.cat((visual, text), dim=1)
        new_args, new_kwargs = _replace_call_value(
            args,
            kwargs,
            1,
            "memory",
            shuffled_memory,
        )

        logit_bias = kwargs.get("memory_logit_bias")
        if logit_bias is not None:
            visual_bias = logit_bias[:, :split]
            text_bias = logit_bias[:, split:]
            if self.mode in {"visual", "both"}:
                visual_bias = visual_bias.index_select(0, permutation)
            if self.mode in {"text", "both"}:
                text_bias = text_bias.index_select(0, permutation)
            new_kwargs["memory_logit_bias"] = torch.cat(
                (visual_bias, text_bias),
                dim=1,
            )
        return new_args, new_kwargs


class CrossAttentionStateCapture:
    """Capture exact CA outputs and inexpensive attention intermediates."""

    def __init__(
        self,
        module: torch.nn.Module,
        controller: MemoryShuffleController,
    ) -> None:
        self.latest: dict[str, torch.Tensor] | None = None
        self.controller = controller
        self.handle = module.register_forward_hook(self._hook, with_kwargs=True)

    def close(self) -> None:
        self.handle.remove()

    def _hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        queries = _get_call_value(args, kwargs, 0, "queries")
        memory = _get_call_value(args, kwargs, 1, "memory")
        group_shape = kwargs.get("query_group_shape")
        split_value = kwargs.get("memory_split_index")
        logit_bias = kwargs.get("memory_logit_bias")
        if queries is None or memory is None or group_shape is None:
            raise RuntimeError("CA capture did not receive required inputs")

        layer_count, slot_count = map(int, group_shape)
        batch_size, query_count, hidden_dim = queries.shape
        split = int(split_value)
        if layer_count * slot_count != query_count:
            raise RuntimeError(
                f"Invalid CA grouping: queries={query_count} groups={group_shape}"
            )

        with torch.no_grad(), torch.autocast(
            device_type=queries.device.type,
            enabled=False,
        ):
            self.controller.suspend_projection_hooks = True
            try:
                normalized_queries = module.query_norm(queries)
                projected_queries = module.query_projection(normalized_queries)
                if module.layer_lora_target == "query":
                    grouped = module._group_layers(normalized_queries, group_shape)
                    projected_queries = projected_queries + module.layer_lora(
                        grouped
                    ).reshape_as(projected_queries)
                normalized_memory = module.memory_norm(memory)
                projected_keys = module.key_projection(normalized_memory)
                projected_values = module.value_projection(normalized_memory)
            finally:
                self.controller.suspend_projection_hooks = False
            projected_keys = self.controller.apply_projected_intervention(
                "key",
                projected_keys,
            )
            projected_values = self.controller.apply_projected_intervention(
                "value",
                projected_values,
            )
            query_heads = module._split_heads(projected_queries)
            key_heads = module._split_heads(projected_keys)
            value_heads = module._split_heads(projected_values)
            if self.controller.mode == "attention_static_modality":
                weights = self.controller.static_attention_weights(
                    batch_size,
                    memory.shape[1],
                    queries.device,
                )
            else:
                scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
                scores = scores.float() / math.sqrt(float(module.head_dim))
                if logit_bias is not None:
                    scores = scores + logit_bias.to(
                        device=scores.device,
                        dtype=scores.dtype,
                    )[:, None, None, :]
                weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights.to(value_heads.dtype), value_heads)
            context = context.transpose(1, 2).contiguous().view(
                batch_size,
                query_count,
                hidden_dim,
            )

            self.latest = {
                "visual_memory": memory[:, :split].detach().float(),
                "text_memory": memory[:, split:].detach().float(),
                "key": projected_keys.detach().float(),
                "value": projected_values.detach().float(),
                "attention": weights.mean(dim=1).reshape(
                    batch_size,
                    layer_count,
                    slot_count,
                    memory.shape[1],
                ).detach(),
                "head_visual_mass": weights[..., :split].sum(dim=-1).reshape(
                    batch_size,
                    module.num_heads,
                    layer_count,
                    slot_count,
                ).detach().float(),
                "context": context.reshape(
                    batch_size,
                    layer_count,
                    slot_count,
                    hidden_dim,
                ).detach().float(),
                "delta": output[1].reshape(
                    batch_size,
                    layer_count,
                    slot_count,
                    hidden_dim,
                ).detach().float(),
                "dynamic_rep": output[0].reshape(
                    batch_size,
                    layer_count,
                    slot_count,
                    hidden_dim,
                ).detach().float(),
            }


def _relative_change(base: torch.Tensor, changed: torch.Tensor) -> float:
    numerator = (changed.float() - base.float()).norm()
    denominator = base.float().norm().clamp_min(1e-8)
    return float((numerator / denominator).item())


def _cosine(base: torch.Tensor, changed: torch.Tensor) -> float:
    return float(F.cosine_similarity(
        base.float().reshape(1, -1),
        changed.float().reshape(1, -1),
        dim=-1,
        eps=1e-8,
    ).item())


def _attention_distance(
    base: torch.Tensor,
    changed: torch.Tensor,
) -> tuple[float, float, float]:
    base = base.float().clamp_min(1e-8)
    changed = changed.float().clamp_min(1e-8)
    total_variation = 0.5 * (changed - base).abs().sum(dim=-1).mean()
    midpoint = 0.5 * (base + changed)
    js = 0.5 * (
        (base * (base.log() - midpoint.log())).sum(dim=-1)
        + (changed * (changed.log() - midpoint.log())).sum(dim=-1)
    ).mean()
    return (
        float(total_variation.item()),
        float(js.item()),
        _relative_change(base, changed),
    )


def supervised_token_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[torch.Tensor], list[float]]:
    predictions = []
    accuracies = []
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    for sample_index in range(labels.shape[0]):
        valid = shifted_labels[sample_index] != -100
        selected_labels = shifted_labels[sample_index][valid]
        if selected_labels.numel() == 0:
            predictions.append(torch.empty(0, dtype=torch.long))
            accuracies.append(float("nan"))
            continue
        selected_predictions = shifted_logits[sample_index][valid].argmax(dim=-1)
        predictions.append(selected_predictions.detach().cpu())
        accuracies.append(float(
            (selected_predictions == selected_labels).float().mean().item()
        ))
    return predictions, accuracies


def _summarize_fields(
    rows: Sequence[Mapping[str, Any]],
    fields: Iterable[str],
) -> dict[str, Any]:
    return {
        field: numeric_summary(row.get(field) for row in rows)
        for field in fields
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _build_model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    model_inputs = {
        key: _to_device(batch[key], device)
        for key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mmrl_gating_mask",
        )
    }
    model_inputs["images_per_sample"] = batch["images_per_sample"]
    return model_inputs


def calibrate_static_modality_routing(
    model: torch.nn.Module,
    controller: MemoryShuffleController,
    capture: CrossAttentionStateCapture,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    mass_sum: torch.Tensor | None = None
    sample_count = 0
    controller.mode = "matched"
    for batch_number, (_dataset_indices, batch) in enumerate(loader, start=1):
        image_counts = [int(value) for value in batch["images_per_sample"]]
        if any(value != 1 for value in image_counts):
            raise RuntimeError(
                "Static routing calibration requires one image per sample; "
                f"got images_per_sample={image_counts}"
            )
        capture.latest = None
        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None
        with torch.inference_mode():
            outputs = model(
                **_build_model_inputs(batch, device),
                use_cache=False,
                return_dict=True,
            )
        if capture.latest is None:
            raise RuntimeError("No CA state captured during static routing calibration")
        visual_mass = capture.latest["head_visual_mass"]
        batch_sum = visual_mass.sum(dim=0).cpu()
        mass_sum = batch_sum if mass_sum is None else mass_sum + batch_sum
        sample_count += visual_mass.shape[0]
        del outputs
        if batch_number == len(loader) or batch_number % 10 == 0:
            print(
                "[STATIC_ROUTING_CALIBRATION] "
                f"batches={batch_number}/{len(loader)} samples={sample_count}"
            )
    if mass_sum is None or sample_count < 1:
        raise RuntimeError("Static routing calibration produced no samples")
    visual_mass = mass_sum / float(sample_count)
    controller.set_static_visual_mass(visual_mass)
    return {
        "sample_count": sample_count,
        "visual_mass_mean": float(visual_mass.mean().item()),
        "visual_mass_std": float(visual_mass.std(unbiased=False).item()),
        "visual_mass_min": float(visual_mass.min().item()),
        "visual_mass_max": float(visual_mass.max().item()),
        "shape": list(visual_mass.shape),
    }


def report_markdown(summary: Mapping[str, Any]) -> str:
    def mean(mode: str, metric: str) -> float:
        value = summary["modes"][mode]["sample_metrics"].get(metric, {})
        return float(value.get("mean", float("nan")))

    lines = [
        f"# Memory Causality: {summary['checkpoint_label']}",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Samples: {summary['sample_count']}",
        f"- Force G=1: {summary['force_g_one']}",
        "",
        "| Intervention | Visual memory change | Text memory change | K change | V change | Attention TV | Context change | Delta change | Rep change | CE delta | Token flip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in summary["shuffle_modes"]:
        lines.append(
            f"| `{mode}` | "
            f"{mean(mode, 'visual_memory_change_ratio'):.6f} | "
            f"{mean(mode, 'text_memory_change_ratio'):.6f} | "
            f"{mean(mode, 'key_change_ratio'):.6f} | "
            f"{mean(mode, 'value_change_ratio'):.6f} | "
            f"{mean(mode, 'attention_tv_mean'):.6f} | "
            f"{mean(mode, 'context_change_ratio_mean'):.6f} | "
            f"{mean(mode, 'delta_change_ratio_mean'):.6f} | "
            f"{mean(mode, 'dynamic_rep_change_ratio_mean'):.6f} | "
            f"{mean(mode, 'ce_delta'):+.6f} | "
            f"{mean(mode, 'supervised_token_flip_rate'):.6f} |"
        )
    lines.extend([
        "",
        "`CE delta > 0` means the intervention hurts the supervised answer. "
        "Low output change is only evidence of CA insensitivity when the corresponding "
        "memory change is itself non-trivial.",
    ])
    return "\n".join(lines) + "\n"


def analyze_checkpoint(
    checkpoint: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    label = checkpoint_label(checkpoint)
    run_output = output_dir / label
    run_output.mkdir(parents=True, exist_ok=True)
    print(f"\n[MEMORY_CAUSALITY] loading={checkpoint}")

    interface = ModelInterface(str(checkpoint), str(args.base_model))
    model = interface.model
    processor = interface.processor
    model.eval()
    visual = model.model.visual
    original_ablation = bool(visual.ablate_visual_gate)
    if not args.respect_gate:
        visual.ablate_visual_gate = True

    static_routing_requested = any(
        mode in STATIC_ROUTING_MODES for mode in args.modes
    )
    calibration_limit = (
        args.static_routing_calibration_samples
        if static_routing_requested
        else 0
    )
    dataset = SLAKEDataset(
        processor=processor,
        image_root=str(args.image_root),
        questions_path=str(args.questions),
        total_limit=args.limit + calibration_limit,
        languages=None if args.language == "all" else (args.language,),
        base_types=None,
        splits=None,
        ce_enabled=True,
        seed=args.seed,
        deterministic_sampling=True,
        max_length=args.max_length,
    )
    if len(dataset) < args.limit + calibration_limit:
        raise RuntimeError(
            "Dataset returned fewer samples than requested for causality test: "
            f"actual={len(dataset)} requested={args.limit + calibration_limit}"
        )
    indexed_dataset = IndexedDataset(dataset)
    collator = IndexedCollator(SLAKEDataCollator(processor))
    evaluation_indices = range(calibration_limit, calibration_limit + args.limit)
    loader = DataLoader(
        Subset(indexed_dataset, evaluation_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.pair_seed),
        collate_fn=collator,
    )

    cross_attention = model.model.MMRL.cross_attention
    controller = MemoryShuffleController(cross_attention)
    capture = CrossAttentionStateCapture(cross_attention, controller)
    device = next(model.parameters()).device
    static_routing_calibration: dict[str, Any] | None = None
    if static_routing_requested:
        calibration_loader = DataLoader(
            Subset(indexed_dataset, range(calibration_limit)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collator,
        )
        static_routing_calibration = calibrate_static_modality_routing(
            model,
            controller,
            capture,
            calibration_loader,
            device,
        )
        print(
            "[STATIC_ROUTING_CALIBRATION_DONE] "
            f"{static_routing_calibration}"
        )
    insert_layers = list(visual.insert_layers)
    sample_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    processed_samples = 0

    try:
        for batch_number, (dataset_indices, batch) in enumerate(loader, start=1):
            image_counts = [int(value) for value in batch["images_per_sample"]]
            if any(value != 1 for value in image_counts):
                raise RuntimeError(
                    "This diagnostic currently requires one image per sample; "
                    f"got images_per_sample={image_counts}"
                )
            model_inputs = _build_model_inputs(batch, device)
            labels = batch["labels"].to(device)
            mode_payloads: dict[str, dict[str, Any]] = {}

            for mode in ("matched", *args.modes):
                controller.mode = mode
                capture.latest = None
                if hasattr(model.model, "rope_deltas"):
                    model.model.rope_deltas = None
                with torch.inference_mode():
                    outputs = model(
                        **model_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                if capture.latest is None:
                    raise RuntimeError(f"No CA state captured for mode={mode}")
                ce_values = per_sample_ce(outputs.logits, labels)
                token_predictions, token_accuracies = supervised_token_predictions(
                    outputs.logits,
                    labels,
                )
                mode_payloads[mode] = {
                    "states": capture.latest,
                    "ce": ce_values,
                    "token_predictions": token_predictions,
                    "token_accuracies": token_accuracies,
                }
                del outputs

            baseline = mode_payloads["matched"]
            if controller.last_permutation is None:
                raise RuntimeError("Memory shuffle did not expose its permutation")
            partner_offsets = controller.last_permutation.tolist()
            for batch_offset, dataset_index in enumerate(dataset_indices):
                sample = dataset.data[dataset_index]
                qid = str(sample.get("question_id", dataset_index))
                partner_index = dataset_indices[partner_offsets[batch_offset]]
                partner_sample = dataset.data[partner_index]
                baseline_predictions = baseline["token_predictions"][batch_offset]
                for mode in args.modes:
                    changed = mode_payloads[mode]
                    changed_predictions = changed["token_predictions"][batch_offset]
                    if baseline_predictions.numel() == 0:
                        token_flip_rate = float("nan")
                    else:
                        token_flip_rate = float(
                            (baseline_predictions != changed_predictions)
                            .float()
                            .mean()
                            .item()
                        )

                    base_states = baseline["states"]
                    changed_states = changed["states"]
                    sample_record: dict[str, Any] = {
                        "dataset_index": int(dataset_index),
                        "question_id": qid,
                        "question": str(sample.get("question", "")),
                        "answer": str(sample.get("answer", "")),
                        "mismatch_dataset_index": int(partner_index),
                        "mismatch_question_id": str(
                            partner_sample.get("question_id", partner_index)
                        ),
                        "shuffle_mode": mode,
                        "baseline_ce": baseline["ce"][batch_offset],
                        "shuffled_ce": changed["ce"][batch_offset],
                        "ce_delta": (
                            changed["ce"][batch_offset]
                            - baseline["ce"][batch_offset]
                        ),
                        "supervised_token_count": int(
                            baseline_predictions.numel()
                        ),
                        "supervised_token_flip_rate": token_flip_rate,
                        "baseline_token_accuracy": baseline[
                            "token_accuracies"
                        ][batch_offset],
                        "shuffled_token_accuracy": changed[
                            "token_accuracies"
                        ][batch_offset],
                        "visual_memory_change_ratio": _relative_change(
                            base_states["visual_memory"][batch_offset],
                            changed_states["visual_memory"][batch_offset],
                        ),
                        "text_memory_change_ratio": _relative_change(
                            base_states["text_memory"][batch_offset],
                            changed_states["text_memory"][batch_offset],
                        ),
                        "key_change_ratio": _relative_change(
                            base_states["key"][batch_offset],
                            changed_states["key"][batch_offset],
                        ),
                        "value_change_ratio": _relative_change(
                            base_states["value"][batch_offset],
                            changed_states["value"][batch_offset],
                        ),
                    }
                    per_sample_layer_metrics: dict[str, list[float]] = defaultdict(list)
                    for layer_offset, layer_index in enumerate(insert_layers):
                        base_attention = base_states["attention"][
                            batch_offset,
                            layer_offset,
                        ]
                        changed_attention = changed_states["attention"][
                            batch_offset,
                            layer_offset,
                        ]
                        attention_tv, attention_js, attention_change = (
                            _attention_distance(base_attention, changed_attention)
                        )
                        row = {
                            "dataset_index": int(dataset_index),
                            "question_id": qid,
                            "shuffle_mode": mode,
                            "layer_offset": int(layer_offset),
                            "layer_0based": int(layer_index),
                            "layer_natural": int(layer_index) + 1,
                            "attention_tv": attention_tv,
                            "attention_js": attention_js,
                            "attention_change_ratio": attention_change,
                        }
                        for state_name in ("context", "delta", "dynamic_rep"):
                            base_value = base_states[state_name][
                                batch_offset,
                                layer_offset,
                            ]
                            changed_value = changed_states[state_name][
                                batch_offset,
                                layer_offset,
                            ]
                            row[f"{state_name}_change_ratio"] = _relative_change(
                                base_value,
                                changed_value,
                            )
                            row[f"{state_name}_cosine"] = _cosine(
                                base_value,
                                changed_value,
                            )
                        layer_rows.append(row)
                        for metric in LAYER_METRICS:
                            per_sample_layer_metrics[metric].append(row[metric])

                    for metric, values in per_sample_layer_metrics.items():
                        sample_record[f"{metric}_mean"] = _mean(values)
                    sample_rows.append(sample_record)

            processed_samples += len(dataset_indices)
            del mode_payloads
            if batch_number % args.log_every == 0 or batch_number == len(loader):
                print(
                    f"[MEMORY_CAUSALITY] checkpoint={label} "
                    f"batches={batch_number}/{len(loader)} "
                    f"samples={processed_samples}"
                )
    finally:
        capture.close()
        controller.close()
        visual.ablate_visual_gate = original_ablation

    modes = {}
    for mode in args.modes:
        selected_samples = [
            row for row in sample_rows if row["shuffle_mode"] == mode
        ]
        selected_layers = [
            row for row in layer_rows if row["shuffle_mode"] == mode
        ]
        modes[mode] = {
            "sample_count": len(selected_samples),
            "sample_metrics": _summarize_fields(
                selected_samples,
                SAMPLE_METRICS,
            ),
            "layer_metrics": _summarize_fields(
                selected_layers,
                LAYER_METRICS,
            ),
            "per_layer": {
                str(layer_index + 1): _summarize_fields(
                    [
                        row
                        for row in selected_layers
                        if row["layer_0based"] == layer_index
                    ],
                    LAYER_METRICS,
                )
                for layer_index in insert_layers
            },
        }

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_label": label,
        "base_model": str(args.base_model),
        "questions": str(args.questions),
        "image_root": str(args.image_root),
        "requested_limit": args.limit,
        "sample_count": processed_samples,
        "batch_size": args.batch_size,
        "pair_seed": args.pair_seed,
        "static_routing_calibration": static_routing_calibration,
        "shuffle_modes": list(args.modes),
        "force_g_one": not args.respect_gate,
        "insert_layers_0based": insert_layers,
        "insert_layers_natural": [index + 1 for index in insert_layers],
        "modes": modes,
    }
    write_jsonl(run_output / "memory_causality_samples.jsonl", sample_rows)
    write_jsonl(run_output / "memory_causality_layers.jsonl", layer_rows)
    write_json(run_output / "memory_causality_summary.json", summary)
    (run_output / "memory_causality_report.md").write_text(
        report_markdown(summary),
        encoding="utf-8",
    )

    del model, interface
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def comparison_markdown(summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Matched vs Shuffled Memory Comparison",
        "",
        "| Checkpoint | Intervention | Visual memory change | Text memory change | K change | V change | Attention TV | Context change | Delta change | Rep change | CE delta | Token flip |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = (
        "visual_memory_change_ratio",
        "text_memory_change_ratio",
        "key_change_ratio",
        "value_change_ratio",
        "attention_tv_mean",
        "context_change_ratio_mean",
        "delta_change_ratio_mean",
        "dynamic_rep_change_ratio_mean",
        "ce_delta",
        "supervised_token_flip_rate",
    )
    for summary in summaries:
        for mode in summary["shuffle_modes"]:
            values = summary["modes"][mode]["sample_metrics"]
            formatted = []
            for metric in metrics:
                value = float(values.get(metric, {}).get("mean", float("nan")))
                formatted.append(
                    f"{value:+.6f}" if metric == "ce_delta" else f"{value:.6f}"
                )
            lines.append(
                f"| `{summary['checkpoint_label']}` | `{mode}` | "
                + " | ".join(formatted)
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure causal sensitivity to matched vs shuffled CA memory."
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-seed", type=int, default=20260814)
    parser.add_argument("--static-routing-calibration-samples", type=int, default=64)
    parser.add_argument("--language", choices=("all", "en", "zh"), default="all")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SHUFFLE_MODES,
        default=list(SHUFFLE_MODES),
    )
    parser.add_argument("--respect-gate", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    args.base_model = args.base_model.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.questions = (
        args.questions.expanduser().resolve()
        if args.questions is not None
        else (args.data_root / "validation.json").resolve()
    )
    args.image_root = (
        args.image_root.expanduser().resolve()
        if args.image_root is not None
        else (args.data_root / "imgs").resolve()
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / f"memory_causality_{timestamp}").resolve()
    )

    if args.limit < args.batch_size:
        parser.error("--limit must be at least --batch-size")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for derangement")
    if (
        any(mode in STATIC_ROUTING_MODES for mode in args.modes)
        and args.static_routing_calibration_samples < args.batch_size
    ):
        parser.error(
            "--static-routing-calibration-samples must be at least --batch-size"
        )
    if args.workers < 0 or args.log_every < 1:
        parser.error("--workers must be non-negative and --log-every positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    for path, label in (
        (args.base_model, "base model"),
        (args.questions, "questions"),
        (args.image_root, "image root"),
    ):
        if not path.exists():
            parser.error(f"{label} path does not exist: {path}")
    return args


def main() -> int:
    args = parse_args()
    checkpoints = [resolve_checkpoint(path) for path in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise RuntimeError(f"Duplicate checkpoints resolved from {args.checkpoints}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[MEMORY_CAUSALITY_CONFIG]")
    print(f"  checkpoints={[str(path) for path in checkpoints]}")
    print(f"  modes={args.modes}")
    print(f"  questions={args.questions}")
    print(f"  images={args.image_root}")
    print(f"  limit={args.limit} batch_size={args.batch_size} workers={args.workers}")
    print(f"  sample_seed={args.seed} pair_seed={args.pair_seed}")
    print(
        "  static_routing_calibration_samples="
        f"{args.static_routing_calibration_samples}"
    )
    print(f"  force_g_one={not args.respect_gate}")
    print(f"  output={args.output_dir}")

    summaries = [
        analyze_checkpoint(checkpoint, args, args.output_dir)
        for checkpoint in checkpoints
    ]
    write_json(args.output_dir / "memory_causality_comparison.json", summaries)
    comparison_path = args.output_dir / "memory_causality_comparison.md"
    comparison_path.write_text(
        comparison_markdown(summaries),
        encoding="utf-8",
    )
    print(f"[MEMORY_CAUSALITY_DONE] comparison={comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
