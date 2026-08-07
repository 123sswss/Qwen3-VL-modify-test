"""Evaluate a compact FROST-VL checkpoint with the consistency protocol."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
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

from mmrl_checkpoint import (
    initialize_mmrl_from_base,
    load_mmrl_delta,
    load_mmrl_manifest,
    validate_base_config,
)

# eval_consistency and the repository both contain config.py. Root MMRL modules
# must capture the repository config while common.py keeps its own constants.
_evaluation_config_module = sys.modules.get("config")
if _evaluation_config_module is None:
    raise RuntimeError("eval_consistency config was not loaded by common.py")
sys.modules.pop("config")
try:
    import QWen3WithMMRL
finally:
    sys.modules["config"] = _evaluation_config_module


class Qwen3VLMMRLForConsistency(Qwen3VLForConditionalGeneration):
    """Inference wrapper matching the compact Stage 3 parameter layout."""

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
    "MMRL_MEMORY_QUERY_COUNT",
    "MMRL_MEMORY_ATTENTION_DIM",
    "MMRL_PROJECTOR_HIDDEN_DIM",
    "MMRL_CROSS_ATTENTION_HEADS",
    "INSERT_METHOD",
    "GATING_MID_DIM",
    "stretching_length",
    "gating_temperature",
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
    missing = [key for key in REQUIRED_MMRL_CONFIG_KEYS if not hasattr(config, key)]
    if missing:
        raise RuntimeError(
            f"Checkpoint config is missing MMRL architecture keys {missing}: "
            f"{checkpoint / 'config.json'}"
        )
    if not bool(config.USE_MMRL):
        raise RuntimeError(f"Checkpoint has USE_MMRL=False: {checkpoint}")


def _apply_shared_attention_backend(config: Any) -> None:
    implementation = str(MODEL_KWARGS["attn_implementation"])
    config._attn_implementation = implementation
    config.text_config._attn_implementation = implementation
    config.vision_config._attn_implementation = implementation
    print(f"[MMRL_BACKEND_AUDIT] attn_implementation={implementation}")


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
        if ".MMRL." in name
    ]
    if not custom_parameter_names:
        raise RuntimeError("Loaded model contains no MMRL parameters")
    expected_vocab_size = int(model.config.text_config.vocab_size)
    if model.lm_head.weight.shape[0] != expected_vocab_size:
        raise RuntimeError(
            "MMRL output vocabulary does not match checkpoint config: "
            f"lm_head={model.lm_head.weight.shape[0]} "
            f"config={expected_vocab_size}"
        )

    print(
        "[MMRL_MODEL_AUDIT] "
        f"checkpoint={checkpoint} "
        f"insert_layers={list(visual.insert_layers)} "
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

    manifest = load_mmrl_manifest(checkpoint)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    base_config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    validate_base_config(manifest, base_config)
    _hydrate_nested_mmrl_config(config)
    _validate_checkpoint_config(config, checkpoint)
    _apply_shared_attention_backend(config)

    with torch.device("cuda"):
        model = Qwen3VLMMRLForConsistency(config, tokenizer)
    model.to(dtype=torch.bfloat16)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    initialize_mmrl_from_base(model, base_model)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    load_mmrl_delta(model, checkpoint)
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
            "Provide --checkpoint /path/to/final or set MMRL_CONSISTENCY_CHECKPOINT."
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
