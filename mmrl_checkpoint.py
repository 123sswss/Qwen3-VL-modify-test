"""Compact checkpoint I/O for FROST-VL/MMRL models.

Only project-specific delta modules are serialized. Base-model weights and the
frozen Rep-Token vision blocks are reconstructed from the configured base model.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file


FORMAT_NAME = "frost-vl-mmrl-delta"
FORMAT_VERSION = 3
WEIGHTS_NAME = "mmrl_delta.safetensors"
MANIFEST_NAME = "mmrl_manifest.json"
LEGACY_WEIGHT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

# This is the union of modules trained in Stage 1 and Stage 3. Stage 1 modules
# are frozen by the end of Stage 3, so final requires_grad flags are insufficient.
DELTA_PREFIXES = (
    "model.MMRL.",
    "model.visual.hidden_state_pooling.",
    "model.visual.embedding_pooling.",
    "model.visual.Task_classifier.",
)

MMRL_COMPONENT_SELECTORS = {
    "layer_projectors": (
        "model.MMRL.v_r_token_projector.",
    ),
    "query_generator": (
        "model.MMRL.shared_represent_space",
        "model.MMRL.layer_embeddings",
        "model.MMRL.v_r_token_projector.",
    ),
}


def _is_delta_key(key: str) -> bool:
    return key.startswith(DELTA_PREFIXES)


def _matches_component(key: str, selectors: tuple[str, ...]) -> bool:
    return any(
        key.startswith(selector) if selector.endswith(".") else key == selector
        for selector in selectors
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _config_signature(config: Any) -> dict[str, Any]:
    text_config = config.text_config
    vision_config = config.vision_config
    return {
        "model_type": str(getattr(config, "model_type", "")),
        "text_hidden_size": int(text_config.hidden_size),
        "vision_hidden_size": int(vision_config.hidden_size),
        "vision_depth": int(vision_config.depth),
        "vocab_size": int(getattr(config, "vocab_size", text_config.vocab_size)),
    }


def _collect_delta_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if _is_delta_key(key)
    }
    if not state:
        raise RuntimeError("No FROST-VL delta tensors matched the checkpoint allowlist")

    active_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    outside_allowlist = sorted(
        name for name in active_trainable if not _is_delta_key(name)
    )
    if outside_allowlist:
        raise RuntimeError(
            "Trainable parameters fall outside the compact checkpoint allowlist: "
            f"{outside_allowlist}"
        )
    return state


def save_mmrl_checkpoint(
    model: torch.nn.Module,
    processor: Any,
    output_dir: str | Path,
    base_model_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a standalone delta checkpoint without any frozen base weights."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_files = [name for name in LEGACY_WEIGHT_NAMES if (output_dir / name).exists()]
    if legacy_files:
        raise RuntimeError(
            "Refusing to mix compact and legacy full-model weights in one directory: "
            f"{legacy_files}"
        )

    delta_state = _collect_delta_state(model)
    weights_path = output_dir / WEIGHTS_NAME
    temporary_weights = output_dir / f".{WEIGHTS_NAME}.tmp"
    save_file(delta_state, str(temporary_weights))
    os.replace(temporary_weights, weights_path)

    model.config.save_pretrained(output_dir)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    tensor_manifest = {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "numel": int(tensor.numel()),
        }
        for key, tensor in sorted(delta_state.items())
    }
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "weights_file": WEIGHTS_NAME,
        "weights_sha256": _sha256(weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "base_model": str(Path(base_model_path).expanduser()),
        "config_signature": _config_signature(model.config),
        "tensor_count": len(delta_state),
        "parameter_count": sum(tensor.numel() for tensor in delta_state.values()),
        "tensors": tensor_manifest,
        "metadata": dict(metadata or {}),
    }
    _atomic_write_json(output_dir / MANIFEST_NAME, manifest)
    print(
        "[MMRL_COMPACT_CHECKPOINT] "
        f"saved={output_dir} tensors={manifest['tensor_count']} "
        f"parameters={manifest['parameter_count']} "
        f"bytes={weights_path.stat().st_size}"
    )
    return manifest


def load_mmrl_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    manifest_path = checkpoint_dir / MANIFEST_NAME
    weights_path = checkpoint_dir / WEIGHTS_NAME
    if not manifest_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Compact FROST-VL checkpoint requires {MANIFEST_NAME} and "
            f"{WEIGHTS_NAME} under {checkpoint_dir}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != FORMAT_NAME:
        raise RuntimeError(f"Unsupported checkpoint format: {manifest.get('format')!r}")
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported checkpoint format version: "
            f"{manifest.get('format_version')!r}"
        )
    if manifest.get("weights_file") != WEIGHTS_NAME:
        raise RuntimeError(f"Unexpected compact weights filename: {manifest.get('weights_file')!r}")
    actual_hash = _sha256(weights_path)
    if actual_hash != manifest.get("weights_sha256"):
        raise RuntimeError(
            f"Compact checkpoint SHA256 mismatch: expected={manifest.get('weights_sha256')} "
            f"actual={actual_hash}"
        )
    return manifest


def validate_base_config(manifest: Mapping[str, Any], base_config: Any) -> None:
    expected = dict(manifest["config_signature"])
    actual = _config_signature(base_config)
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Base model is incompatible with compact checkpoint: {mismatches}")


def initialize_mmrl_from_base(model: torch.nn.Module, base_model: torch.nn.Module) -> None:
    """Restore all frozen weights and Rep blocks from the base model."""

    base_result = model.model.load_state_dict(base_model.model.state_dict(), strict=False)
    if base_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected base-model keys while rebuilding FROST-VL: {base_result.unexpected_keys}"
        )
    model.lm_head.load_state_dict(base_model.lm_head.state_dict(), strict=True)

    visual = model.model.visual
    if len(visual.blocks_with_rep) != len(visual.insert_layers):
        raise RuntimeError(
            "Rep block count does not match insert layers: "
            f"{len(visual.blocks_with_rep)} != {len(visual.insert_layers)}"
        )
    for rep_index, layer_index in enumerate(visual.insert_layers):
        visual.blocks_with_rep[rep_index].load_state_dict(
            visual.blocks[layer_index].state_dict(),
            strict=True,
        )
    print(
        "[MMRL_BASE_REBUILD] "
        f"rep_blocks={len(visual.blocks_with_rep)} insert_layers={list(visual.insert_layers)}"
    )


def load_mmrl_delta(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Strictly load every and only expected delta tensor into a rebuilt model."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    manifest = load_mmrl_manifest(checkpoint_dir)
    state = load_file(str(checkpoint_dir / WEIGHTS_NAME), device="cpu")
    expected_keys = {
        key for key in model.state_dict() if _is_delta_key(key)
    }
    saved_keys = set(state)
    manifest_keys = set(manifest["tensors"])
    if saved_keys != manifest_keys:
        raise RuntimeError(
            "Compact checkpoint manifest does not match its tensor file: "
            f"missing={sorted(manifest_keys - saved_keys)} "
            f"unexpected={sorted(saved_keys - manifest_keys)}"
        )
    if saved_keys != expected_keys:
        raise RuntimeError(
            "Compact checkpoint does not exactly match the configured FROST-VL modules: "
            f"missing={sorted(expected_keys - saved_keys)} "
            f"unexpected={sorted(saved_keys - expected_keys)}"
        )

    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected delta keys: {result.unexpected_keys}")
    mmrl = getattr(getattr(model, "model", None), "MMRL", None)
    if mmrl is not None and hasattr(mmrl, "cached_base_queries"):
        mmrl.cached_base_queries = None
    print(
        "[MMRL_COMPACT_CHECKPOINT] "
        f"loaded={checkpoint_dir} tensors={len(state)} "
        f"parameters={sum(tensor.numel() for tensor in state.values())}"
    )
    return manifest


def transplant_mmrl_component(
    model: torch.nn.Module,
    donor_checkpoint_dir: str | Path,
    component: str,
) -> dict[str, Any]:
    """Replace one inference component with tensors from a donor checkpoint."""

    component = str(component).strip().lower()
    if component not in MMRL_COMPONENT_SELECTORS:
        raise ValueError(
            f"Unsupported MMRL component={component!r}; "
            f"choices={sorted(MMRL_COMPONENT_SELECTORS)}"
        )
    donor_checkpoint_dir = Path(donor_checkpoint_dir).expanduser().resolve()
    manifest = load_mmrl_manifest(donor_checkpoint_dir)
    donor_state = load_file(
        str(donor_checkpoint_dir / WEIGHTS_NAME),
        device="cpu",
    )
    saved_keys = set(donor_state)
    manifest_keys = set(manifest["tensors"])
    if saved_keys != manifest_keys:
        raise RuntimeError(
            "Donor checkpoint manifest does not match its tensor file: "
            f"missing={sorted(manifest_keys - saved_keys)} "
            f"unexpected={sorted(saved_keys - manifest_keys)}"
        )

    selectors = MMRL_COMPONENT_SELECTORS[component]
    target_state = model.state_dict()
    expected_keys = {
        key for key in target_state if _matches_component(key, selectors)
    }
    donor_keys = {
        key for key in donor_state if _matches_component(key, selectors)
    }
    if not expected_keys:
        raise RuntimeError(
            f"Recipient model exposes no tensors for component={component!r}"
        )
    if donor_keys != expected_keys:
        raise RuntimeError(
            f"Donor component={component!r} does not match the recipient: "
            f"missing={sorted(expected_keys - donor_keys)} "
            f"unexpected={sorted(donor_keys - expected_keys)}"
        )
    shape_mismatches = {
        key: {
            "recipient": tuple(target_state[key].shape),
            "donor": tuple(donor_state[key].shape),
        }
        for key in sorted(expected_keys)
        if tuple(target_state[key].shape) != tuple(donor_state[key].shape)
    }
    if shape_mismatches:
        raise RuntimeError(
            f"Donor component={component!r} shape mismatch: {shape_mismatches}"
        )

    component_state = {key: donor_state[key] for key in sorted(donor_keys)}
    result = model.load_state_dict(component_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected donor component keys: {result.unexpected_keys}"
        )
    mmrl = getattr(getattr(model, "model", None), "MMRL", None)
    if mmrl is not None and hasattr(mmrl, "cached_base_queries"):
        mmrl.cached_base_queries = None
    audit = {
        "component": component,
        "donor": str(donor_checkpoint_dir),
        "tensor_count": len(component_state),
        "parameter_count": sum(
            tensor.numel() for tensor in component_state.values()
        ),
        "tensor_keys": sorted(component_state),
    }
    print(
        "[MMRL_COMPONENT_TRANSPLANT] "
        f"component={component} donor={donor_checkpoint_dir} "
        f"tensors={audit['tensor_count']} parameters={audit['parameter_count']}"
    )
    return audit
