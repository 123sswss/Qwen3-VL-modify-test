"""Evaluate a fully trained MMRL checkpoint with the shared consistency protocol."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    Qwen3VLForConditionalGeneration,
)

from common import BASE_MODEL_PATH, MODEL_KWARGS, parse_args, run_experiment

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Both the repository and eval_consistency contain config.py. Root MMRL modules
# must capture the repository config, while common.py keeps its already-imported
# evaluation constants.
_evaluation_config_module = sys.modules.get("config")
if _evaluation_config_module is None:
    raise RuntimeError("eval_consistency config was not loaded by common.py")
sys.modules.pop("config")
try:
    import QWen3WithMMRL
finally:
    sys.modules["config"] = _evaluation_config_module


class Qwen3VLMMRLForConsistency(Qwen3VLForConditionalGeneration):
    """Inference wrapper matching the parameter layout saved by Stage 3."""

    def __init__(self, config: Any, tokenizer: Any) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = QWen3WithMMRL.QWen3WithMMRL(config, tokenizer=tokenizer)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            len(tokenizer),
            bias=False,
        )
        self.generation_config = GenerationConfig.from_model_config(config)
        if tokenizer.pad_token_id is not None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id
        self.post_init()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings


REQUIRED_MMRL_CONFIG_KEYS = (
    "USE_MMRL",
    "INSERT_LAYER",
    "POOLING_DIM",
    "RP_SPACE_LENGTH",
    "RP_SPACE_DIM",
    "INSERT_METHOD",
    "GATING_MID_DIM",
    "stretching_length",
    "gating_temperature",
    "VISUAL_RESIDUAL_ADAPTER_COUNT",
    "RANDOM_INIT_ADAPTER_OUTPUT_COUNT",
)


def _hydrate_nested_mmrl_config(config: Any) -> None:
    saved_mmrl_config = getattr(config, "mmrl_config", None)
    if hasattr(saved_mmrl_config, "to_dict"):
        saved_mmrl_config = saved_mmrl_config.to_dict()
    if not isinstance(saved_mmrl_config, dict):
        raise RuntimeError(
            "Checkpoint config does not contain a serialized mmrl_config mapping"
        )

    hydrated = []
    for key, value in saved_mmrl_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
            hydrated.append(key)
    print(f"[MMRL_CONFIG_AUDIT] hydrated_from_nested={sorted(hydrated)}")


def _validate_checkpoint_config(config: Any, checkpoint: Path) -> None:
    missing = [
        key for key in REQUIRED_MMRL_CONFIG_KEYS
        if not hasattr(config, key)
    ]
    if missing:
        raise RuntimeError(
            f"Checkpoint config is missing MMRL architecture keys {missing}: "
            f"{checkpoint / 'config.json'}"
        )
    if not bool(config.USE_MMRL):
        raise RuntimeError(f"Checkpoint has USE_MMRL=False: {checkpoint}")
    if int(config.VISUAL_RESIDUAL_ADAPTER_COUNT) < 1:
        raise RuntimeError(
            "VISUAL_RESIDUAL_ADAPTER_COUNT must be positive, got "
            f"{config.VISUAL_RESIDUAL_ADAPTER_COUNT}"
        )


def _apply_shared_attention_backend(config: Any) -> None:
    implementation = str(MODEL_KWARGS["attn_implementation"])
    config._attn_implementation = implementation
    config.text_config._attn_implementation = implementation
    config.vision_config._attn_implementation = implementation
    print(f"[MMRL_BACKEND_AUDIT] attn_implementation={implementation}")


def _load_pytorch_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"PyTorch checkpoint shard must contain a state dict: {path}"
        )
    return state_dict


def _load_sharded_checkpoint_strict(
    model: Qwen3VLMMRLForConsistency,
    checkpoint: Path,
    index_path: Path,
    use_safetensors: bool,
) -> int:
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(
            f"Checkpoint shard index has no non-empty weight_map: {index_path}"
        )
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise RuntimeError(
            f"Checkpoint shard index contains invalid weight_map entries: {index_path}"
        )

    model_keys = set(model.state_dict())
    checkpoint_keys = set(weight_map)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Strict sharded checkpoint index does not match the model: "
            f"missing={missing_keys} unexpected={unexpected_keys}"
        )

    checkpoint_root = checkpoint.resolve()
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        raise RuntimeError(f"Checkpoint shard index names no shard files: {index_path}")

    tensor_count = 0
    loaded_keys = set()
    for shard_number, shard_name in enumerate(shard_names, start=1):
        shard_path = (checkpoint / shard_name).resolve()
        if shard_path.parent != checkpoint_root:
            raise RuntimeError(
                f"Checkpoint shard must be directly inside {checkpoint}: {shard_name}"
            )
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint shard listed by {index_path.name} is missing: {shard_path}"
            )

        expected_shard_keys = {
            key for key, mapped_shard in weight_map.items()
            if mapped_shard == shard_name
        }
        state_dict = (
            load_file(str(shard_path), device="cpu")
            if use_safetensors
            else _load_pytorch_state_dict(shard_path)
        )
        actual_shard_keys = set(state_dict)
        if actual_shard_keys != expected_shard_keys:
            raise RuntimeError(
                f"Checkpoint shard keys do not match {index_path.name}: "
                f"shard={shard_name} "
                f"missing={sorted(expected_shard_keys - actual_shard_keys)} "
                f"unexpected={sorted(actual_shard_keys - expected_shard_keys)}"
            )
        duplicate_keys = loaded_keys & actual_shard_keys
        if duplicate_keys:
            raise RuntimeError(
                f"Checkpoint keys appear in multiple shards: {sorted(duplicate_keys)}"
            )

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Shard load returned unexpected model keys: "
                f"shard={shard_name} unexpected={incompatible.unexpected_keys}"
            )
        loaded_keys.update(actual_shard_keys)
        tensor_count += len(state_dict)
        del state_dict
        gc.collect()
        print(
            "[MMRL_CHECKPOINT_SHARD] "
            f"loaded={shard_number}/{len(shard_names)} "
            f"file={shard_name} tensors={len(actual_shard_keys)}"
        )

    if loaded_keys != model_keys:
        raise RuntimeError(
            "Sharded checkpoint load did not cover the complete model: "
            f"missing={sorted(model_keys - loaded_keys)} "
            f"unexpected={sorted(loaded_keys - model_keys)}"
        )
    return tensor_count


def _load_full_checkpoint(
    model: Qwen3VLMMRLForConsistency,
    checkpoint: Path,
) -> None:
    safetensors_path = checkpoint / "model.safetensors"
    pytorch_path = checkpoint / "pytorch_model.bin"
    safe_index = checkpoint / "model.safetensors.index.json"
    pytorch_index = checkpoint / "pytorch_model.bin.index.json"

    if safetensors_path.is_file():
        state_dict = load_file(str(safetensors_path), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        tensor_count = len(state_dict)
        del state_dict
        print(
            "[MMRL_CHECKPOINT_AUDIT] "
            f"format=safetensors tensors={tensor_count} strict=True"
        )
    elif pytorch_path.is_file():
        state_dict = _load_pytorch_state_dict(pytorch_path)
        model.load_state_dict(state_dict, strict=True)
        tensor_count = len(state_dict)
        del state_dict
        print(
            "[MMRL_CHECKPOINT_AUDIT] "
            f"format=pytorch tensors={tensor_count} strict=True"
        )
    elif safe_index.is_file() or pytorch_index.is_file():
        use_safetensors = safe_index.is_file()
        index_path = safe_index if use_safetensors else pytorch_index
        tensor_count = _load_sharded_checkpoint_strict(
            model,
            checkpoint,
            index_path,
            use_safetensors=use_safetensors,
        )
        print(
            "[MMRL_CHECKPOINT_AUDIT] "
            f"format={'sharded_safetensors' if use_safetensors else 'sharded_pytorch'} "
            f"shards={len(set(json.loads(index_path.read_text(encoding='utf-8'))['weight_map'].values()))} "
            f"tensors={tensor_count} strict=True"
        )
    else:
        raise FileNotFoundError(
            "No supported full-model checkpoint found in "
            f"{checkpoint}; expected model.safetensors, pytorch_model.bin, "
            "or a Transformers shard index."
        )

    gc.collect()


def _audit_loaded_model(
    model: Qwen3VLMMRLForConsistency,
    checkpoint: Path,
) -> None:
    visual = model.model.visual
    parameter_device = next(model.parameters()).device
    if parameter_device.type != "cuda":
        raise RuntimeError(f"MMRL model was not loaded on CUDA: {parameter_device}")
    custom_parameter_names = [
        name
        for name, _ in model.named_parameters()
        if (
            ".MMRL." in name
            or ".residual_adapters." in name
            or ".adapter_router." in name
        )
    ]
    if not custom_parameter_names:
        raise RuntimeError("Loaded model contains no MMRL/router/adapter parameters")
    if len(visual.residual_adapters) != int(
        model.config.VISUAL_RESIDUAL_ADAPTER_COUNT
    ):
        raise RuntimeError(
            "Loaded adapter count does not match checkpoint config: "
            f"model={len(visual.residual_adapters)} "
            f"config={model.config.VISUAL_RESIDUAL_ADAPTER_COUNT}"
        )
    expected_vocab_size = int(model.config.text_config.vocab_size)
    if model.lm_head.weight.shape[0] != expected_vocab_size:
        raise RuntimeError(
            "MMRL output vocabulary does not match checkpoint config and cannot "
            "be paired with Base logits: "
            f"lm_head={model.lm_head.weight.shape[0]} "
            f"config={expected_vocab_size}"
        )

    print(
        "[MMRL_MODEL_AUDIT] "
        f"checkpoint={checkpoint} "
        f"insert_layers={list(visual.insert_layers)} "
        f"adapter_count={len(visual.residual_adapters)} "
        f"custom_parameter_tensors={len(custom_parameter_names)} "
        f"device={parameter_device} "
        f"lm_head_shape={tuple(model.lm_head.weight.shape)}"
    )


def load_ours(checkpoint_path: str) -> Tuple[Any, Any]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"MMRL checkpoint directory not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("MMRL consistency evaluation requires a CUDA device")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    _hydrate_nested_mmrl_config(config)
    _validate_checkpoint_config(config, checkpoint)
    _apply_shared_attention_backend(config)

    with torch.device("cuda"):
        model = Qwen3VLMMRLForConsistency(config, tokenizer)
    model.to(dtype=torch.bfloat16)
    _load_full_checkpoint(model, checkpoint)
    model.eval()
    _audit_loaded_model(model, checkpoint)

    return model, processor


def main() -> None:
    args = parse_args(
        "MMRL output-disruption consistency evaluation",
        "ours",
        include_checkpoint=True,
    )
    if not args.checkpoint:
        raise SystemExit(
            "Provide --checkpoint /path/to/final or set "
            "MMRL_CONSISTENCY_CHECKPOINT."
        )
    model, processor = load_ours(args.checkpoint)
    run_experiment(
        experiment_name="Ours (MMRL)",
        method="ours_mmrl",
        model=model,
        processor=processor,
        args=args,
        checkpoint_path=str(Path(args.checkpoint).expanduser().resolve()),
    )


if __name__ == "__main__":
    main()
