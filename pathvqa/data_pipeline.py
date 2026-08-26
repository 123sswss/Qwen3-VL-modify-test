"""PathVQA Parquet loading and Qwen3-VL training collation."""

from __future__ import annotations

import os
import random
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from data_pipeline import build_target_supervision_masks


TASK_TYPE_VQA_ID = 1
OFFICIAL_SPLIT_SIZES = {
    "train": 19654,
    "validation": 6259,
    "test": 6719,
}
SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}
SHARD_PATTERN = re.compile(r"-(\d+)-of-(\d+)-[0-9a-f]+\.parquet$")


def normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    normalized = SPLIT_ALIASES.get(normalized, normalized)
    if normalized not in OFFICIAL_SPLIT_SIZES:
        raise ValueError(
            f"Unsupported PathVQA split={split!r}; "
            f"choices={sorted(OFFICIAL_SPLIT_SIZES)}"
        )
    return normalized


def _require_hf_datasets():
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "PathVQA Parquet support requires datasets and pyarrow. Install "
            "them with: python -m pip install -r pathvqa/requirements.txt"
        ) from exc
    return HFImage, load_dataset


def discover_split_shards(data_root: Path, split: str) -> List[Path]:
    split = normalize_split(split)
    shards = sorted(data_root.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No PathVQA {split} Parquet shards found under {data_root}"
        )

    parsed = []
    for path in shards:
        match = SHARD_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Unexpected PathVQA shard name: {path.name}")
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    expected_total = parsed[0][1]
    if any(total != expected_total for _, total, _ in parsed):
        raise ValueError(f"Inconsistent PathVQA shard totals: {parsed}")
    indices = [index for index, _, _ in parsed]
    if indices != list(range(expected_total)):
        raise FileNotFoundError(
            f"Incomplete PathVQA {split} shards: expected 0..{expected_total - 1}, "
            f"found={indices}"
        )
    return [path.resolve() for _, _, path in parsed]


class PathVQAParquetStore:
    """Random-access PathVQA records backed by Hugging Face Arrow caching."""

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        split: str,
        cache_dir: str | os.PathLike[str] | None = None,
        verify_official_size: bool = True,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"PathVQA data root not found: {self.data_root}")
        self.split = normalize_split(split)
        self.shards = discover_split_shards(self.data_root, self.split)
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else (self.data_root / ".hf_cache").resolve()
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        HFImage, load_dataset = _require_hf_datasets()
        dataset = load_dataset(
            "parquet",
            data_files={self.split: [str(path) for path in self.shards]},
            split=self.split,
            cache_dir=str(self.cache_dir),
        )
        required_columns = {"image", "question", "answer"}
        missing_columns = sorted(required_columns - set(dataset.column_names))
        if missing_columns:
            raise ValueError(
                f"PathVQA Parquet is missing columns={missing_columns}; "
                f"found={dataset.column_names}"
            )
        if not isinstance(dataset.features["image"], HFImage):
            dataset = dataset.cast_column("image", HFImage(decode=True))
        if verify_official_size and len(dataset) != OFFICIAL_SPLIT_SIZES[self.split]:
            raise ValueError(
                f"PathVQA {self.split} row count mismatch: "
                f"expected={OFFICIAL_SPLIT_SIZES[self.split]} actual={len(dataset)}"
            )
        self.dataset = dataset

        questions = dataset["question"]
        answers = dataset["answer"]
        self.samples = []
        invalid = []
        for row_index, (question, answer) in enumerate(zip(questions, answers)):
            normalized_question = str(question or "").strip()
            normalized_answer = str(answer or "").strip()
            if not normalized_question or not normalized_answer:
                invalid.append(row_index)
                continue
            self.samples.append(
                {
                    "row_index": row_index,
                    "question_id": f"pathvqa:{self.split}:{row_index}",
                    "question": normalized_question,
                    "answer": normalized_answer,
                    "split": self.split,
                }
            )
        if invalid:
            raise ValueError(
                f"PathVQA {self.split} contains empty question/answer rows: "
                f"count={len(invalid)} examples={invalid[:5]}"
            )
        print(
            "[PATHVQA_PARQUET] "
            f"split={self.split} shards={len(self.shards)} rows={len(self.samples)} "
            f"cache={self.cache_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, sample: Dict[str, Any]) -> Image.Image:
        value = self.dataset[int(sample["row_index"])]["image"]
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            raw_bytes = value.get("bytes")
            raw_path = value.get("path")
            if raw_bytes is not None:
                with Image.open(BytesIO(raw_bytes)) as source:
                    return source.convert("RGB")
            if raw_path:
                with Image.open(raw_path) as source:
                    return source.convert("RGB")
        raise TypeError(
            "Unsupported decoded PathVQA image value: "
            f"type={type(value).__name__} row={sample['row_index']}"
        )


class PathVQADataset(Dataset):
    def __init__(
        self,
        processor,
        data_root: str | os.PathLike[str],
        split: str = "train",
        cache_dir: str | os.PathLike[str] | None = None,
        total_limit: Optional[int] = None,
        ce_enabled: bool = True,
        seed: int = 42,
        deterministic_sampling: bool = True,
        max_length: int = 2048,
    ):
        if total_limit is not None and total_limit < 1:
            raise ValueError("total_limit must be positive or None")
        self.processor = processor
        self.store = PathVQAParquetStore(data_root, split, cache_dir=cache_dir)
        self.total_limit = total_limit
        self.ce_enabled = ce_enabled
        self.seed = seed
        self.deterministic_sampling = deterministic_sampling
        self.max_length = max_length
        self.resample_round = 0
        self.raw_samples = self.store.samples

        tokenizer = self.processor.tokenizer
        self.assistant_header_ids = tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        self.assistant_label_prefix_ids = tokenizer.encode(
            "<|im_start|>assistant",
            add_special_tokens=False,
        )
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            not self.assistant_header_ids
            or not self.assistant_label_prefix_ids
            or self.im_end_token_id is None
        ):
            raise RuntimeError("Failed to resolve Qwen assistant boundary tokens")
        self.data: List[Dict[str, Any]] = []
        self._build()

    def _build(self) -> None:
        rng = (
            random.Random(self.seed + self.resample_round)
            if self.deterministic_sampling
            else random
        )
        if self.total_limit is None or self.total_limit >= len(self.raw_samples):
            selected = list(self.raw_samples)
        else:
            selected = rng.sample(self.raw_samples, self.total_limit)
        rng.shuffle(selected)
        self.data = selected
        print(
            "[PathVQADataset] "
            f"split={self.store.split} resample_round={self.resample_round} "
            f"selected_samples={len(self.data)}"
        )

    def resample_data(self) -> None:
        self.resample_round += 1
        self._build()

    def __len__(self) -> int:
        return len(self.data)

    def load_image(self, sample: Dict[str, Any]) -> Image.Image:
        return self.store.load_image(sample)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.data[index]
        image = self.load_image(sample)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
        text_inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            images=image,
            text=text_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if input_ids.numel() > self.max_length:
            raise ValueError(
                "PathVQA sample exceeds the complete-sequence limit; "
                f"tokens={input_ids.numel()} max_length={self.max_length} "
                f"question_id={sample['question_id']}"
            )
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = inputs["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        if self.ce_enabled:
            labels, mmrl_gating_mask = build_target_supervision_masks(
                input_ids=input_ids,
                attention_mask=attention_mask,
                assistant_header_ids=self.assistant_header_ids,
                assistant_label_prefix_ids=self.assistant_label_prefix_ids,
                im_end_token_id=self.im_end_token_id,
                target_assistant_ordinal=1,
            )
        else:
            labels = torch.full_like(input_ids, -100)
            mmrl_gating_mask = attention_mask.bool()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "mmrl_gating_mask": mmrl_gating_mask,
            "alpha_labels": 1.0,
            "task_type_id": TASK_TYPE_VQA_ID,
            "task_type_name": "vqa",
            "source_name": "pathvqa",
            "images_per_sample": 1,
            "is_mm": 1,
        }


class PathVQAStage1Dataset(Dataset):
    """Expose each PathVQA question as multimodal and text-only gate views."""

    def __init__(self, dataset: PathVQADataset):
        self.dataset = dataset
        self.processor = dataset.processor
        self.max_length = dataset.max_length
        self.data = [
            (sample_index, view_type)
            for sample_index in range(len(dataset))
            for view_type in ("mm", "text")
        ]
        random.Random(dataset.seed).shuffle(self.data)
        print(
            "[PathVQAStage1Dataset] "
            f"raw_samples={len(dataset)} positive_views={len(self.data)} "
            "views=('mm', 'text') prompt_only=True"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_index, view_type = self.data[index]
        sample = self.dataset.data[sample_index]
        image = self.dataset.load_image(sample) if view_type == "mm" else None
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": sample["question"]})
        conversation = [{"role": "user", "content": content}]
        text_inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {
            "text": text_inputs,
            "padding": False,
            "truncation": False,
            "return_tensors": "pt",
        }
        if image is not None:
            processor_kwargs["images"] = image
        inputs = self.processor(**processor_kwargs)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if input_ids.numel() > self.max_length:
            raise ValueError(
                "PathVQA Stage 1 sample exceeds the complete-sequence limit; "
                f"tokens={input_ids.numel()} max_length={self.max_length} "
                f"question_id={sample['question_id']} view={view_type}"
            )

        if view_type == "mm":
            pixel_values = inputs["pixel_values"]
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.squeeze(0)
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            images_per_sample = 1
            is_mm = 1
        else:
            pixel_values = None
            image_grid_thw = None
            images_per_sample = 0
            is_mm = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": torch.full_like(input_ids, -100),
            "mmrl_gating_mask": attention_mask.bool(),
            "alpha_labels": 1.0,
            "task_type_id": TASK_TYPE_VQA_ID,
            "task_type_name": "vqa",
            "source_name": "pathvqa",
            "images_per_sample": images_per_sample,
            "is_mm": is_mm,
        }


class PathVQADataCollator:
    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("PathVQADataCollator received an empty feature list")
        features = [dict(feature) for feature in features]
        alpha_labels = [feature.pop("alpha_labels") for feature in features]
        task_type_ids = [feature.pop("task_type_id") for feature in features]
        task_type_names = [feature.pop("task_type_name") for feature in features]
        source_names = [feature.pop("source_name") for feature in features]
        images_per_sample = [feature.pop("images_per_sample") for feature in features]
        is_mm = [feature.pop("is_mm") for feature in features]
        pixel_values = [feature.pop("pixel_values") for feature in features]
        image_grids = [feature.pop("image_grid_thw") for feature in features]

        padding = {
            "input_ids": int(self.processor.tokenizer.pad_token_id),
            "attention_mask": 0,
            "labels": -100,
            "mmrl_gating_mask": False,
        }
        batch = {}
        for key in features[0]:
            values = [feature[key] for feature in features]
            if key in padding:
                batch[key] = pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=padding[key],
                )
            else:
                batch[key] = (
                    torch.stack(values, dim=0)
                    if torch.is_tensor(values[0])
                    else values
                )

        mm_pixels = [value for value in pixel_values if value is not None]
        mm_grids = [value for value in image_grids if value is not None]
        batch["pixel_values"] = torch.cat(mm_pixels, dim=0) if mm_pixels else None
        batch["image_grid_thw"] = torch.cat(mm_grids, dim=0) if mm_grids else None
        batch["alpha_labels"] = torch.tensor(alpha_labels, dtype=torch.float32)
        batch["task_type_ids"] = torch.tensor(task_type_ids, dtype=torch.long)
        batch["task_type_names"] = task_type_names
        batch["source_names"] = source_names
        batch["images_per_sample"] = images_per_sample
        batch["is_mm"] = torch.tensor(is_mm, dtype=torch.long)
        return batch
