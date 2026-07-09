"""Inference and JSON storage shared by every experiment entry point."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from config import (
    BASE_MODEL_PATH,
    DATASET_JSON,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    IMAGE_DIR,
    MODEL_KWARGS,
)


def parse_args(description: str, default_run_id: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-id", default=default_run_id)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--dataset-json", default=DATASET_JSON)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_model(model_path: str = BASE_MODEL_PATH) -> Any:
    kwargs = dict(MODEL_KWARGS)
    kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
    loaders = []
    if Qwen3VLForConditionalGeneration is not None:
        loaders.append(Qwen3VLForConditionalGeneration)
    if AutoModelForImageTextToText is not None:
        loaders.append(AutoModelForImageTextToText)
    loaders.append(AutoModelForCausalLM)

    last_error: Optional[Exception] = None
    for loader in loaders:
        try:
            return loader.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            last_error = exc
            print(f"[model] {loader.__name__} failed: {exc}")
    raise RuntimeError(f"Cannot load base model: {model_path}") from last_error


def load_peft_model(adapter_path: str) -> Tuple[Any, Any]:
    from peft import PeftModel

    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"PEFT checkpoint not found: {path}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = PeftModel.from_pretrained(load_base_model(), str(path))
    return model, processor


def _conversation_text(item: Dict[str, Any]) -> str:
    conversations = item.get("conversations") or item.get("conversation") or []
    if conversations:
        first = conversations[0]
        text = first.get("value", first.get("content", ""))
    else:
        text = item.get("question", item.get("prompt", item.get("text", "")))
    return str(text).replace("<image>", "").strip()


def _image_path(item: Dict[str, Any], image_dir: Path) -> Optional[Path]:
    value = item.get("image") or item.get("image_path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else image_dir / path


def load_manifest(dataset_json: str, image_dir: str, limit: int) -> List[Dict[str, Any]]:
    dataset_path = Path(dataset_json)
    with dataset_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("samples", []))
    if not isinstance(raw, list):
        raise TypeError("Evaluation dataset must be a JSON list or contain data/samples.")

    root = Path(image_dir)
    samples = []
    scanned = 0
    missing_image_field = 0
    missing_image_file = 0
    empty_prompt = 0
    for index, item in enumerate(raw):
        scanned += 1
        prompt = _conversation_text(item)
        image_path = _image_path(item, root)
        if not prompt:
            empty_prompt += 1
            continue
        if image_path is None:
            missing_image_field += 1
            continue
        if not image_path.is_file():
            missing_image_file += 1
            continue
        stable_id = str(item.get("id", item.get("sample_id", index)))
        samples.append(
            {
                "sample_id": stable_id,
                "source_index": index,
                "prompt": prompt,
                "image_path": str(image_path),
            }
        )
        if len(samples) >= limit:
            break

    print(
        "[dataset] "
        f"scanned={scanned}, selected={len(samples)}, "
        f"missing_image_file={missing_image_file}, "
        f"missing_image_field={missing_image_field}, "
        f"empty_prompt={empty_prompt}"
    )
    if len(samples) < limit:
        raise RuntimeError(
            f"Only found {len(samples)} usable image-text samples after scanning "
            f"{scanned} records; requested {limit}. Missing image files: "
            f"{missing_image_file}. Check IMAGE_DIR if most or all images are missing."
        )
    return samples


def move_inputs(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            result[key] = value
        elif value.is_floating_point():
            result[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            result[key] = value.to(device=device)
    return result


def encode_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    array = tensor.detach().cpu().to(torch.float16).contiguous().numpy()
    return {
        "encoding": "base64",
        "dtype": "float16",
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def infer_one(
    model: Any,
    processor: Any,
    sample: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    image_path = Path(sample["image_path"])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ],
        }]
        rendered_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=rendered_prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = move_inputs(inputs, device)
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits[0, -1, :].float()
        probabilities = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probabilities, k=50)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = generated[:, input_length:]
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    return {
        "sample_id": sample["sample_id"],
        "source_index": sample["source_index"],
        "prompt": sample["prompt"],
        "rendered_prompt": rendered_prompt,
        "image_path": sample["image_path"],
        "image_sha256": file_sha256(image_path),
        "input_ids": input_ids,
        "input_ids_sha256": hashlib.sha256(
            np.asarray(input_ids, dtype=np.int64).tobytes()
        ).hexdigest(),
        "first_token_logits": encode_tensor(logits),
        "top50_token_ids": top_indices.cpu().tolist(),
        "top50_probabilities": top_values.cpu().tolist(),
        "generated_token_ids": generated_ids[0].detach().cpu().tolist(),
        "generated_text": generated_text,
    }


def _environment() -> Dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": gpu_names,
        "pid": os.getpid(),
    }


def run_experiment(
    experiment_name: str,
    method: str,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    checkpoint_path: str,
) -> Path:
    seed_everything(args.seed)
    model.eval()
    samples = load_manifest(args.dataset_json, args.image_dir, args.num_samples)
    output_dir = Path(args.output_root) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_results.json"
    metadata = {
        "experiment_name": experiment_name,
        "run_id": args.run_id,
        "method": method,
        "base_model_path": BASE_MODEL_PATH,
        "checkpoint_path": checkpoint_path,
        "dataset_json": args.dataset_json,
        "dataset_sha256": file_sha256(Path(args.dataset_json)),
        "image_dir": args.image_dir,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "environment": _environment(),
    }

    # Stream a single valid JSON document so full-vocabulary logits do not accumulate in RAM.
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write('{\n  "conclusion": ')
        json.dump(
            {
                "status": "raw_data_complete",
                "statement": "Raw evidence only; run analyze.py for the disruption conclusion.",
            },
            handle,
            ensure_ascii=False,
        )
        handle.write(',\n  "metadata": ')
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write(',\n  "samples": [\n')
        completed = 0
        for sample in tqdm(samples, desc=args.run_id):
            result = infer_one(model, processor, sample, args.max_new_tokens)
            if completed:
                handle.write(",\n")
            json.dump(result, handle, ensure_ascii=False)
            handle.flush()
            completed += 1
        handle.write("\n  ]\n}\n")
    print(f"[done] complete JSON evidence: {output_path}")
    return output_path


def run_base_entry(args: argparse.Namespace) -> Path:
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    return run_experiment(
        experiment_name="Original Qwen3-VL",
        method="base",
        model=load_base_model(),
        processor=processor,
        args=args,
        checkpoint_path=BASE_MODEL_PATH,
    )


def run_peft_entry(
    args: argparse.Namespace,
    experiment_name: str,
    method: str,
    checkpoint_path: str,
) -> Path:
    model, processor = load_peft_model(checkpoint_path)
    return run_experiment(
        experiment_name, method, model, processor, args, checkpoint_path
    )
