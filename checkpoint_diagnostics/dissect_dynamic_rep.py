"""Offline slot-level dissection for dynamic MMRL Rep checkpoints.

The script never updates model weights. It hooks the trained Cross-Attention
module during teacher-forced SLAKE forwards and localizes slot collapse across
Q, K/V memory, logits, attention maps, retrieved context, and output delta.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEST_DIR = REPO_ROOT / "test"
for import_root in (REPO_ROOT, TEST_DIR):
    import_text = str(import_root)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from inferEngine import ModelInterface
from mmrl_checkpoint import MANIFEST_NAME
from slake.data_pipeline import SLAKEDataCollator, SLAKEDataset
from slake.slake_official_eval import load_json, unwrap_question_records
from slake.slake_vqa_metric import normalize_vqa_answer, record_id


DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/model")
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/dataset/slake")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_CHECKPOINT_ROOT = Path(
    "/root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl"
)
PRIMARY_SAMPLE_METRICS = (
    "visual_memory_specificity_ratio",
    "text_memory_specificity_ratio",
    "visual_key_specificity_ratio",
    "text_key_specificity_ratio",
    "visual_value_specificity_ratio",
    "text_value_specificity_ratio",
)
CORRELATION_METRICS = (
    "layer_mean_query_specificity_ratio",
    "layer_mean_logit_map_specificity_ratio",
    "layer_mean_attention_map_specificity_ratio",
    "layer_mean_context_specificity_ratio",
    "layer_mean_delta_specificity_ratio",
    "layer_mean_attention_entropy_norm",
    "layer_mean_logit_slot_to_memory_std_ratio",
    *PRIMARY_SAMPLE_METRICS,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, Any]:
        return index, self.dataset[index]


class IndexedCollator:
    def __init__(self, collator: SLAKEDataCollator) -> None:
        self.collator = collator

    def __call__(self, items: Sequence[tuple[int, Any]]) -> tuple[list[int], Any]:
        indices, features = zip(*items)
        return list(indices), self.collator(list(features))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def resolve_checkpoint(raw_path: Path) -> Path:
    candidates = [raw_path.expanduser()]
    if not raw_path.is_absolute():
        candidates.append(DEFAULT_CHECKPOINT_ROOT / raw_path)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.name == MANIFEST_NAME:
            return candidate.parent
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
        if candidate.is_dir():
            manifests = sorted(candidate.rglob(MANIFEST_NAME))
            if len(manifests) == 1:
                return manifests[0].parent
            if len(manifests) > 1:
                final_manifests = [
                    path for path in manifests if path.parent.name == "final"
                ]
                if len(final_manifests) == 1:
                    return final_manifests[0].parent
                raise RuntimeError(
                    f"Checkpoint path is ambiguous; manifests={manifests}"
                )
    raise FileNotFoundError(
        f"Cannot find {MANIFEST_NAME} from checkpoint argument {raw_path}"
    )


def checkpoint_label(checkpoint: Path) -> str:
    if checkpoint.name == "final":
        parent = checkpoint.parent
        if parent.name == "stage3":
            parent = parent.parent
        return parent.name
    return checkpoint.name


def _geometry(states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return geometry over the penultimate token dimension."""

    states = states.detach().float()
    token_count = states.shape[-2]
    if token_count < 2:
        raise ValueError(f"Geometry requires at least two tokens: {states.shape}")

    norms = states.norm(dim=-1)
    unit = states / norms.unsqueeze(-1).clamp_min(1e-8)
    cosine = torch.matmul(unit, unit.transpose(-2, -1))
    diagonal = torch.diagonal(cosine, dim1=-2, dim2=-1)
    pair_count = float(token_count * (token_count - 1))
    pair_cos_mean = (cosine.sum(dim=(-2, -1)) - diagonal.sum(dim=-1)) / pair_count
    diagonal_mask = torch.eye(
        token_count,
        device=states.device,
        dtype=torch.bool,
    )
    pair_cos_max = cosine.masked_fill(diagonal_mask, float("-inf")).amax(
        dim=(-2, -1)
    )

    def rank_stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gram = torch.matmul(values, values.transpose(-2, -1))
        singular = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
        singular_sum = singular.sum(dim=-1)
        distribution = singular / singular_sum.unsqueeze(-1).clamp_min(1e-8)
        effective_rank = torch.exp(
            -(distribution * distribution.clamp_min(1e-8).log()).sum(dim=-1)
        )
        effective_rank = effective_rank * (singular_sum > 1e-8)
        descending = singular.flip(-1)
        top1 = descending[..., 0] / singular_sum.clamp_min(1e-8)
        top5 = descending[..., : min(5, token_count)].sum(dim=-1)
        top5 = top5 / singular_sum.clamp_min(1e-8)
        return effective_rank, top1, top5

    effective_rank, singular_top1, singular_top5 = rank_stats(states)
    centered = states - states.mean(dim=-2, keepdim=True)
    centered_rank, _, _ = rank_stats(centered)
    raw_rms = states.square().sum(dim=-1).mean(dim=-1).sqrt()
    centered_rms = centered.square().sum(dim=-1).mean(dim=-1).sqrt()

    return {
        "pair_cos_mean": pair_cos_mean,
        "pair_cos_max": pair_cos_max,
        "effective_rank": effective_rank,
        "centered_effective_rank": centered_rank,
        "singular_top1_ratio": singular_top1,
        "singular_top5_ratio": singular_top5,
        "common_mode_ratio": states.mean(dim=-2).norm(dim=-1)
        / norms.mean(dim=-1).clamp_min(1e-8),
        "specificity_ratio": centered_rms / raw_rms.clamp_min(1e-8),
        "token_norm_mean": norms.mean(dim=-1),
    }


def _add_geometry(
    target: dict[str, torch.Tensor],
    prefix: str,
    states: torch.Tensor,
) -> None:
    for key, value in _geometry(states).items():
        target[f"{prefix}_{key}"] = value


class CrossAttentionCapture:
    """Recompute cheap CA internals and retain only metrics/common vectors."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.latest: dict[str, Any] | None = None
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
        queries = kwargs.get("queries", args[0] if args else None)
        memory = kwargs.get("memory", args[1] if len(args) > 1 else None)
        group_shape = kwargs.get("query_group_shape")
        memory_split = kwargs.get("memory_split_index")
        if queries is None or memory is None or group_shape is None:
            raise RuntimeError("Cross-Attention hook did not receive required inputs")

        layer_count, slot_count = map(int, group_shape)
        batch_size, query_count, hidden_dim = queries.shape
        if layer_count * slot_count != query_count:
            raise RuntimeError(
                f"Invalid query grouping: queries={query_count} groups={group_shape}"
            )
        split = int(memory_split)
        memory_count = memory.shape[1]
        if not 0 < split < memory_count:
            raise RuntimeError(
                f"Invalid visual/text memory split {split}/{memory_count}"
            )

        with torch.no_grad(), torch.autocast(
            device_type=queries.device.type,
            enabled=False,
        ):
            query_heads = module._split_heads(
                module.query_projection(module.query_norm(queries))
            )
            normalized_memory = module.memory_norm(memory)
            key_heads = module._split_heads(
                module.key_projection(normalized_memory)
            )
            value_heads = module._split_heads(
                module.value_projection(normalized_memory)
            )
            scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
            scores = scores.float() / math.sqrt(module.head_dim)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights.to(value_heads.dtype), value_heads)
            context = context.transpose(1, 2).contiguous().view(
                batch_size,
                query_count,
                hidden_dim,
            )

            query_projected = query_heads.transpose(1, 2).contiguous().view(
                batch_size,
                query_count,
                hidden_dim,
            )
            key_projected = key_heads.transpose(1, 2).contiguous().view(
                batch_size,
                memory_count,
                hidden_dim,
            )
            value_projected = value_heads.transpose(1, 2).contiguous().view(
                batch_size,
                memory_count,
                hidden_dim,
            )

            query_grouped = query_projected.reshape(
                batch_size, layer_count, slot_count, hidden_dim
            )
            score_grouped = scores.reshape(
                batch_size,
                module.num_heads,
                layer_count,
                slot_count,
                memory_count,
            )
            weight_grouped = weights.reshape_as(score_grouped)
            context_grouped = context.reshape(
                batch_size, layer_count, slot_count, hidden_dim
            )
            dynamic_grouped = output[0].reshape(
                batch_size, layer_count, slot_count, hidden_dim
            )
            delta_grouped = output[1].reshape(
                batch_size, layer_count, slot_count, hidden_dim
            )

            layer_metrics: dict[str, torch.Tensor] = {}
            _add_geometry(layer_metrics, "query", query_grouped)

            # Memory-centering removes per-query offsets that Softmax ignores.
            logit_maps = score_grouped.mean(dim=1)
            logit_maps = logit_maps - logit_maps.mean(dim=-1, keepdim=True)
            _add_geometry(layer_metrics, "logit_map", logit_maps)
            layer_metrics["logit_memory_std_mean"] = score_grouped.std(
                dim=-1, unbiased=False
            ).mean(dim=(1, 3))
            layer_metrics["logit_slot_std_mean"] = score_grouped.std(
                dim=3, unbiased=False
            ).mean(dim=(1, 3))
            layer_metrics["logit_slot_to_memory_std_ratio"] = (
                layer_metrics["logit_slot_std_mean"]
                / layer_metrics["logit_memory_std_mean"].clamp_min(1e-8)
            )

            attention_maps = weight_grouped.mean(dim=1)
            _add_geometry(layer_metrics, "attention_map", attention_maps)
            entropy = -(
                weight_grouped * weight_grouped.clamp_min(1e-8).log()
            ).sum(dim=-1)
            layer_metrics["attention_entropy_norm"] = (
                entropy / math.log(float(memory_count))
            ).mean(dim=(1, 3))
            layer_metrics["attention_effective_support"] = torch.exp(entropy).mean(
                dim=(1, 3)
            )
            layer_metrics["attention_peak_mean"] = weight_grouped.amax(
                dim=-1
            ).mean(dim=(1, 3))
            visual_mass = weight_grouped[..., :split].sum(dim=-1)
            layer_metrics["attention_visual_mass_mean"] = visual_mass.mean(
                dim=(1, 3)
            )
            layer_metrics["attention_visual_mass_slot_std"] = visual_mass.mean(
                dim=1
            ).std(dim=-1, unbiased=False)

            _add_geometry(layer_metrics, "context", context_grouped)
            _add_geometry(layer_metrics, "delta", delta_grouped)
            _add_geometry(layer_metrics, "dynamic_rep", dynamic_grouped)
            query_norm = queries.reshape(
                batch_size, layer_count, slot_count, hidden_dim
            ).float().norm(dim=-1).mean(dim=-1)
            delta_norm = delta_grouped.float().norm(dim=-1).mean(dim=-1)
            layer_metrics["delta_to_query_norm_ratio"] = (
                delta_norm / query_norm.clamp_min(1e-8)
            )

            sample_metrics: dict[str, torch.Tensor] = {}
            memory_parts = {
                "visual_memory": memory[:, :split],
                "text_memory": memory[:, split:],
                "visual_key": key_projected[:, :split],
                "text_key": key_projected[:, split:],
                "visual_value": value_projected[:, :split],
                "text_value": value_projected[:, split:],
            }
            for prefix, states in memory_parts.items():
                _add_geometry(sample_metrics, prefix, states)

            self.latest = {
                "layer_metrics": {
                    key: value.detach().float().cpu()
                    for key, value in layer_metrics.items()
                },
                "sample_metrics": {
                    key: value.detach().float().cpu()
                    for key, value in sample_metrics.items()
                },
                "common_vectors": {
                    "query_common": query_grouped.mean(dim=2).detach().float().cpu(),
                    "attention_common": attention_maps.mean(dim=2).detach().float().cpu(),
                    "context_common": context_grouped.mean(dim=2).detach().float().cpu(),
                    "delta_common": delta_grouped.mean(dim=2).detach().float().cpu(),
                },
            }


def per_sample_ce(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    values = []
    shifted_labels = labels[:, 1:]
    shifted_logits = logits[:, :-1]
    for sample_index in range(labels.shape[0]):
        valid = shifted_labels[sample_index] != -100
        if not bool(valid.any()):
            values.append(float("nan"))
            continue
        selected_logits = shifted_logits[sample_index][valid].float()
        selected_labels = shifted_labels[sample_index][valid]
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="mean")
        values.append(float(loss.item()))
    return values


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    return value


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    finite = [value for raw in values if (value := _finite_float(raw)) is not None]
    if not finite:
        return {"count": 0}
    tensor = torch.tensor(finite, dtype=torch.float64)
    return {
        "count": len(finite),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "max": float(tensor.max().item()),
    }


def summarize_numeric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: summary
        for key in keys
        if (summary := numeric_summary(row.get(key) for row in rows))["count"] > 0
    }


def _rank_values(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 3:
        return None
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = left.norm() * right.norm()
    if float(denominator.item()) <= 1e-12:
        return None
    return float((left * right).sum().div(denominator).item())


def correlation(
    rows: Sequence[Mapping[str, Any]],
    left_key: str,
    right_key: str,
) -> dict[str, Any]:
    pairs = []
    for row in rows:
        left = _finite_float(row.get(left_key))
        right = _finite_float(row.get(right_key))
        if left is not None and right is not None:
            pairs.append((left, right))
    if len(pairs) < 3:
        return {"count": len(pairs), "pearson": None, "spearman": None}
    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.float64)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.float64)
    return {
        "count": len(pairs),
        "pearson": _pearson(left, right),
        "spearman": _pearson(_rank_values(left), _rank_values(right)),
    }


def cross_sample_common_summary(
    batches: Mapping[str, Sequence[torch.Tensor]],
    insert_layers: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parts in batches.items():
        states = torch.cat(list(parts), dim=0).float()
        per_layer = []
        for layer_offset, layer_index in enumerate(insert_layers):
            vectors = states[:, layer_offset]
            norms = vectors.norm(dim=-1)
            raw_rms = vectors.square().sum(dim=-1).mean().sqrt()
            centered = vectors - vectors.mean(dim=0, keepdim=True)
            centered_rms = centered.square().sum(dim=-1).mean().sqrt()
            unit = vectors / norms.unsqueeze(-1).clamp_min(1e-8)
            cosine = torch.matmul(unit, unit.transpose(0, 1))
            sample_count = vectors.shape[0]
            pair_cos = (
                cosine.sum() - torch.diagonal(cosine).sum()
            ) / float(max(sample_count * (sample_count - 1), 1))
            per_layer.append(
                {
                    "layer_0based": int(layer_index),
                    "layer_natural": int(layer_index) + 1,
                    "sample_count": int(sample_count),
                    "pair_cos_mean": float(pair_cos.item()),
                    "common_mode_ratio": float(
                        vectors.mean(dim=0).norm().div(norms.mean().clamp_min(1e-8)).item()
                    ),
                    "input_variation_ratio": float(
                        centered_rms.div(raw_rms.clamp_min(1e-8)).item()
                    ),
                }
            )
        result[name] = {
            "per_layer": per_layer,
            "overall_pair_cos_mean": numeric_summary(
                item["pair_cos_mean"] for item in per_layer
            )["mean"],
            "overall_input_variation_ratio": numeric_summary(
                item["input_variation_ratio"] for item in per_layer
            )["mean"],
        }
    return result


def load_question_metadata(questions_path: Path) -> dict[str, dict[str, Any]]:
    records = unwrap_question_records(load_json(questions_path))
    metadata = {}
    for index, record in enumerate(records):
        qid = str(record_id(record, fallback=index))
        metadata[qid] = {
            "language": str(
                record.get("q_lang", record.get("language", ""))
            ).strip().lower(),
            "answer_type": str(record.get("answer_type", "")).strip(),
            "question_type": str(record.get("question_type", "")).strip(),
            "base_type": str(record.get("base_type", "")).strip(),
        }
    return metadata


def load_predictions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = load_json(path)
    if isinstance(payload, dict):
        for key in ("predictions", "results", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Predictions must be a JSON list: {path}")
    result = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        qid = str(record_id(item, fallback=index))
        answer = item.get("answer", item.get("prediction", ""))
        result[qid] = str(answer)
    return result


def group_summary(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = ("ce_loss", *CORRELATION_METRICS)
    for group_key in (
        "correct",
        "language",
        "answer_type",
        "question_type",
        "base_type",
    ):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in sample_rows:
            value = row.get(group_key)
            if value is None or value == "":
                continue
            groups[str(value)].append(row)
        if not groups:
            continue
        result[group_key] = {
            group: {
                "count": len(rows),
                "metrics": {
                    metric: summary
                    for metric in metrics
                    if (summary := numeric_summary(row.get(metric) for row in rows))[
                        "count"
                    ]
                    > 0
                },
            }
            for group, rows in sorted(groups.items())
        }
    return result


def markdown_report(summary: Mapping[str, Any]) -> str:
    def mean(section: str, metric: str) -> float | None:
        value = summary.get(section, {}).get(metric, {}).get("mean")
        return _finite_float(value)

    def fmt(value: Any, digits: int = 5) -> str:
        number = _finite_float(value)
        return "--" if number is None else f"{number:.{digits}f}"

    lines = [
        f"# Dynamic Rep Dissection: {summary['checkpoint_label']}",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Questions: `{summary['questions']}`",
        f"- Samples: {summary['sample_count']}",
        f"- Force G=1: `{summary['force_g_one']}`",
        "",
        "## Slot Geometry",
        "",
        "| Metric | Mean | Std | P50 |",
        "|---|---:|---:|---:|",
    ]
    layer_summary = summary["layer_metrics"]
    displayed = (
        "query_pair_cos_mean",
        "query_centered_effective_rank",
        "query_specificity_ratio",
        "logit_slot_to_memory_std_ratio",
        "attention_entropy_norm",
        "attention_effective_support",
        "attention_peak_mean",
        "attention_map_pair_cos_mean",
        "attention_map_centered_effective_rank",
        "attention_map_specificity_ratio",
        "context_specificity_ratio",
        "delta_pair_cos_mean",
        "delta_centered_effective_rank",
        "delta_specificity_ratio",
        "dynamic_rep_specificity_ratio",
    )
    for metric in displayed:
        stats = layer_summary.get(metric, {})
        lines.append(
            f"| `{metric}` | {fmt(stats.get('mean'))} | "
            f"{fmt(stats.get('std'))} | {fmt(stats.get('p50'))} |"
        )

    lines.extend(
        [
            "",
            "## Layer Detail",
            "",
            "| Layer | Q spec. | Logit slot/memory | Attn entropy | Attn spec. | Delta spec. |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer, metrics in summary["per_layer"].items():
        lines.append(
            f"| {layer} | {fmt(metrics.get('query_specificity_ratio', {}).get('mean'))} | "
            f"{fmt(metrics.get('logit_slot_to_memory_std_ratio', {}).get('mean'))} | "
            f"{fmt(metrics.get('attention_entropy_norm', {}).get('mean'))} | "
            f"{fmt(metrics.get('attention_map_specificity_ratio', {}).get('mean'))} | "
            f"{fmt(metrics.get('delta_specificity_ratio', {}).get('mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Input Sensitivity",
            "",
            "| Common vector | Cross-sample cosine | Input variation ratio |",
            "|---|---:|---:|",
        ]
    )
    for name, stats in summary["cross_sample_common"].items():
        lines.append(
            f"| `{name}` | {fmt(stats['overall_pair_cos_mean'])} | "
            f"{fmt(stats['overall_input_variation_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## Correlation With CE",
            "",
            "| Metric | Pearson | Spearman | N |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, stats in summary["correlations_with_ce"].items():
        lines.append(
            f"| `{metric}` | {fmt(stats.get('pearson'))} | "
            f"{fmt(stats.get('spearman'))} | {stats.get('count', 0)} |"
        )

    attention_cos = mean("layer_metrics", "attention_map_pair_cos_mean")
    attention_spec = mean("layer_metrics", "attention_map_specificity_ratio")
    q_spec = mean("layer_metrics", "query_specificity_ratio")
    lines.extend(["", "## Automatic Checks", ""])
    if attention_cos is not None and attention_cos > 0.995:
        lines.append(
            "- Same-layer attention maps remain strongly common-mode dominated "
            f"(mean cosine {attention_cos:.5f})."
        )
    if attention_spec is not None:
        lines.append(
            "- Attention slot-specific RMS is "
            f"{attention_spec:.4f} of total attention-map RMS."
        )
    if q_spec is not None:
        lines.append(
            f"- Projected-Q slot-specific RMS ratio is {q_spec:.4f}; compare it "
            "with logits and attention to locate where specificity is lost."
        )
    lines.append(
        "- This report is diagnostic only. It does not justify a slot-diversity "
        "loss until Q/K/logit/context localization is complete."
    )
    return "\n".join(lines) + "\n"


def analyze_checkpoint(
    checkpoint: Path,
    args: argparse.Namespace,
    output_dir: Path,
    question_metadata: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, str],
) -> dict[str, Any]:
    label = checkpoint_label(checkpoint)
    run_output = output_dir / label
    run_output.mkdir(parents=True, exist_ok=True)
    print(f"\n[DISSECTION] loading={checkpoint}")

    interface = ModelInterface(str(checkpoint), str(args.base_model))
    model = interface.model
    processor = interface.processor
    model.eval()
    visual = model.model.visual
    original_ablation = bool(visual.ablate_visual_gate)
    if not args.respect_gate:
        visual.ablate_visual_gate = True

    dataset = SLAKEDataset(
        processor=processor,
        image_root=str(args.image_root),
        questions_path=str(args.questions),
        total_limit=args.limit,
        languages=None if args.language == "all" else (args.language,),
        base_types=None,
        splits=None,
        ce_enabled=True,
        seed=args.seed,
        deterministic_sampling=True,
        max_length=args.max_length,
    )
    indexed_collator = IndexedCollator(SLAKEDataCollator(processor))
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=indexed_collator,
    )

    mmrl = model.model.MMRL
    capture = CrossAttentionCapture(mmrl.cross_attention)
    device = next(model.parameters()).device
    insert_layers = list(visual.insert_layers)
    layer_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    common_batches: dict[str, list[torch.Tensor]] = defaultdict(list)
    skipped_no_branch = 0

    try:
        for batch_number, (dataset_indices, batch) in enumerate(loader, start=1):
            capture.latest = None
            if hasattr(model.model, "rope_deltas"):
                model.model.rope_deltas = None
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
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    return_dict=True,
                )
            ce_values = per_sample_ce(outputs.logits, batch["labels"].to(device))
            payload = capture.latest
            if payload is None:
                skipped_no_branch += len(dataset_indices)
                print(
                    "[DISSECTION][SKIP] batch produced no MMRL branch; "
                    f"indices={dataset_indices}"
                )
                del outputs
                continue

            gate_tensor = getattr(visual, "G_list", None)
            if torch.is_tensor(gate_tensor):
                gate_values = gate_tensor.detach().float().reshape(-1).cpu().tolist()
            else:
                gate_values = [None] * len(dataset_indices)

            for key, value in payload["common_vectors"].items():
                common_batches[key].append(value)

            batch_layer_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for batch_offset, dataset_index in enumerate(dataset_indices):
                sample = dataset.data[dataset_index]
                qid = str(sample.get("question_id", dataset_index))
                metadata = dict(question_metadata.get(qid, {}))
                base_record: dict[str, Any] = {
                    "dataset_index": int(dataset_index),
                    "question_id": qid,
                    "question": str(sample.get("question", "")),
                    "answer": str(sample.get("answer", "")),
                    "ce_loss": ce_values[batch_offset],
                    "gate": gate_values[batch_offset]
                    if batch_offset < len(gate_values)
                    else None,
                    **metadata,
                }
                if qid in predictions:
                    prediction = predictions[qid]
                    base_record["prediction"] = prediction
                    base_record["correct"] = (
                        normalize_vqa_answer(prediction)
                        == normalize_vqa_answer(base_record["answer"])
                    )

                sample_record = dict(base_record)
                for metric, values in payload["sample_metrics"].items():
                    sample_record[metric] = float(values[batch_offset].item())

                for layer_offset, layer_index in enumerate(insert_layers):
                    row = {
                        **base_record,
                        "layer_offset": int(layer_offset),
                        "layer_0based": int(layer_index),
                        "layer_natural": int(layer_index) + 1,
                    }
                    for metric, values in payload["layer_metrics"].items():
                        row[metric] = float(values[batch_offset, layer_offset].item())
                    layer_rows.append(row)
                    batch_layer_rows[dataset_index].append(row)

                for metric in payload["layer_metrics"]:
                    values = [row[metric] for row in batch_layer_rows[dataset_index]]
                    sample_record[f"layer_mean_{metric}"] = sum(values) / len(values)
                sample_rows.append(sample_record)

            del outputs
            if batch_number % args.log_every == 0 or batch_number == len(loader):
                print(
                    f"[DISSECTION] checkpoint={label} "
                    f"batches={batch_number}/{len(loader)} "
                    f"samples={len(sample_rows)}"
                )
    finally:
        capture.close()
        visual.ablate_visual_gate = original_ablation

    per_layer = {}
    for layer_index in insert_layers:
        natural = str(layer_index + 1)
        selected = [row for row in layer_rows if row["layer_0based"] == layer_index]
        per_layer[natural] = summarize_numeric_rows(selected)

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_label": label,
        "base_model": str(args.base_model),
        "questions": str(args.questions),
        "image_root": str(args.image_root),
        "sample_limit": args.limit,
        "sample_count": len(sample_rows),
        "skipped_no_branch": skipped_no_branch,
        "force_g_one": not args.respect_gate,
        "insert_layers_0based": insert_layers,
        "insert_layers_natural": [index + 1 for index in insert_layers],
        "slot_count": int(mmrl.rp_space_length),
        "memory_query_count_per_modality": int(mmrl.memory_query_count),
        "layer_metrics": summarize_numeric_rows(layer_rows),
        "sample_metrics": summarize_numeric_rows(sample_rows),
        "per_layer": per_layer,
        "cross_sample_common": cross_sample_common_summary(
            common_batches,
            insert_layers,
        ),
        "correlations_with_ce": {
            metric: correlation(sample_rows, metric, "ce_loss")
            for metric in CORRELATION_METRICS
        },
        "groups": group_summary(sample_rows),
    }

    write_jsonl(run_output / "samples.jsonl", sample_rows)
    write_jsonl(run_output / "layer_metrics.jsonl", layer_rows)
    write_json(run_output / "summary.json", summary)
    (run_output / "report.md").write_text(
        markdown_report(summary),
        encoding="utf-8",
    )
    print(f"[DISSECTION] report={run_output / 'report.md'}")

    del loader, dataset, model, interface, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def comparison_markdown(summaries: Sequence[Mapping[str, Any]]) -> str:
    metrics = (
        "query_specificity_ratio",
        "logit_slot_to_memory_std_ratio",
        "attention_map_specificity_ratio",
        "context_specificity_ratio",
        "delta_specificity_ratio",
        "attention_entropy_norm",
    )
    lines = [
        "# Dynamic Rep Checkpoint Comparison",
        "",
        "| Checkpoint | " + " | ".join(metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for summary in summaries:
        values = []
        for metric in metrics:
            value = summary.get("layer_metrics", {}).get(metric, {}).get("mean")
            values.append("--" if value is None else f"{float(value):.5f}")
        lines.append(
            f"| `{summary['checkpoint_label']}` | " + " | ".join(values) + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dissect dynamic MMRL Rep slots without modifying a checkpoint."
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", choices=("all", "en", "zh"), default="all")
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
    args.predictions = (
        args.predictions.expanduser().resolve()
        if args.predictions is not None
        else None
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / timestamp).resolve()
    )

    if args.limit < 2 or args.batch_size < 1 or args.workers < 0:
        parser.error("--limit must be >=2; batch size positive; workers non-negative")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    for path, label in (
        (args.base_model, "base model"),
        (args.questions, "questions"),
        (args.image_root, "image root"),
    ):
        if not path.exists():
            parser.error(f"{label} path does not exist: {path}")
    if args.predictions is not None and not args.predictions.is_file():
        parser.error(f"predictions file does not exist: {args.predictions}")
    if args.predictions is not None and len(args.checkpoints) != 1:
        parser.error("--predictions currently requires exactly one checkpoint")
    return args


def main() -> int:
    args = parse_args()
    checkpoints = [resolve_checkpoint(path) for path in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise RuntimeError(f"Duplicate checkpoints resolved from {args.checkpoints}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_question_metadata(args.questions)
    predictions = load_predictions(args.predictions)

    print("[DISSECTION_CONFIG]")
    print(f"  checkpoints={[str(path) for path in checkpoints]}")
    print(f"  questions={args.questions}")
    print(f"  images={args.image_root}")
    print(f"  limit={args.limit} batch_size={args.batch_size} workers={args.workers}")
    print(f"  force_g_one={not args.respect_gate}")
    print(f"  output={args.output_dir}")

    summaries = [
        analyze_checkpoint(
            checkpoint,
            args,
            args.output_dir,
            metadata,
            predictions,
        )
        for checkpoint in checkpoints
    ]
    write_json(args.output_dir / "comparison.json", summaries)
    (args.output_dir / "comparison.md").write_text(
        comparison_markdown(summaries),
        encoding="utf-8",
    )
    print(f"[DISSECTION_DONE] comparison={args.output_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
