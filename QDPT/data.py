"""PathVQA Parquet loading and Qwen3-VL collation."""

from __future__ import annotations

import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import DATA_SEED, MAX_LENGTH


OFFICIAL_SPLIT_SIZES = {
    "train": 19_654,
    "validation": 6_259,
    "test": 6_719,
}
SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}
SHARD_PATTERN = re.compile(r"-(\d+)-of-(\d+)-[0-9a-f]+\.parquet$")


def normalize_split(split: str) -> str:
    split = SPLIT_ALIASES.get(split.strip().lower(), split.strip().lower())
    if split not in OFFICIAL_SPLIT_SIZES:
        raise ValueError(f"Unsupported PathVQA split: {split}")
    return split


def discover_split_shards(data_root: Path, split: str) -> list[Path]:
    split = normalize_split(split)
    shards = sorted(data_root.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No PathVQA {split} Parquet shards under {data_root}")

    parsed = []
    for path in shards:
        match = SHARD_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Unexpected PathVQA shard name: {path.name}")
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    expected_total = parsed[0][1]
    indices = [index for index, total, _ in parsed if total == expected_total]
    if len(indices) != len(parsed) or indices != list(range(expected_total)):
        raise FileNotFoundError(f"Incomplete PathVQA {split} shards: {indices}")
    return [path.resolve() for _, _, path in parsed]


class PathVQAParquetStore:
    def __init__(self, data_root: str | Path, split: str):
        from datasets import Image as HFImage
        from datasets import load_dataset

        self.data_root = Path(data_root).expanduser().resolve()
        self.split = normalize_split(split)
        shards = discover_split_shards(self.data_root, self.split)
        dataset = load_dataset(
            "parquet",
            data_files={self.split: [str(path) for path in shards]},
            split=self.split,
            cache_dir=str(self.data_root / ".hf_cache"),
        )
        if not isinstance(dataset.features["image"], HFImage):
            dataset = dataset.cast_column("image", HFImage(decode=True))
        if len(dataset) != OFFICIAL_SPLIT_SIZES[self.split]:
            raise ValueError(
                f"PathVQA {self.split} size mismatch: "
                f"expected={OFFICIAL_SPLIT_SIZES[self.split]} actual={len(dataset)}"
            )
        self.dataset = dataset
        self.samples = [
            {
                "row_index": index,
                "question_id": f"pathvqa:{self.split}:{index}",
                "question": str(question).strip(),
                "answer": str(answer).strip(),
            }
            for index, (question, answer) in enumerate(
                zip(dataset["question"], dataset["answer"])
            )
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, sample: dict[str, Any]) -> Image.Image:
        value = self.dataset[int(sample["row_index"])]["image"]
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if value.get("bytes") is not None:
            with Image.open(BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
        with Image.open(value["path"]) as image:
            return image.convert("RGB")


def _find_subsequence_positions(
    input_ids: torch.Tensor,
    pattern_ids: list[int],
    valid_length: int,
) -> list[int]:
    first_id = int(pattern_ids[0])
    candidates = (input_ids[:valid_length] == first_id).nonzero(as_tuple=True)[0].tolist()
    return [
        start
        for start in candidates
        if input_ids[start : start + len(pattern_ids)].tolist() == pattern_ids
    ]


def build_target_supervision_masks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_header_ids: list[int],
    assistant_label_prefix_ids: list[int],
    im_end_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_length = int((attention_mask != 0).nonzero(as_tuple=True)[0][-1].item()) + 1
    header_positions = _find_subsequence_positions(
        input_ids, assistant_header_ids, valid_length
    )
    if len(header_positions) != 1:
        raise ValueError("PathVQA sample must contain one assistant response")
    target_start = header_positions[0] + len(assistant_label_prefix_ids)
    if not bool((input_ids[target_start:valid_length] == im_end_token_id).any()):
        raise ValueError("PathVQA assistant response is incomplete")

    labels = torch.full_like(input_ids, -100)
    labels[target_start:valid_length] = input_ids[target_start:valid_length]
    mmrl_gating_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    mmrl_gating_mask[:target_start] = attention_mask[:target_start].bool()
    return labels, mmrl_gating_mask


class PathVQATrainDataset(Dataset):
    def __init__(self, processor, data_root: str | Path):
        self.processor = processor
        self.store = PathVQAParquetStore(data_root, "train")
        self.samples = list(self.store.samples)
        random.Random(DATA_SEED).shuffle(self.samples)
        tokenizer = processor.tokenizer
        self.assistant_header_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_label_prefix_ids = tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = self.store.load_image(sample)
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
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            images=image,
            text=text,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        image.close()
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if input_ids.numel() > MAX_LENGTH:
            raise ValueError(
                f"PathVQA sample exceeds {MAX_LENGTH} tokens: {sample['question_id']}"
            )
        labels, gating_mask = build_target_supervision_masks(
            input_ids,
            attention_mask,
            self.assistant_header_ids,
            self.assistant_label_prefix_ids,
            self.im_end_token_id,
        )
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = inputs["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "mmrl_gating_mask": gating_mask,
        }


class PathVQACollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
        pixel_values = [feature.pop("pixel_values") for feature in features]
        image_grids = [feature.pop("image_grid_thw") for feature in features]
        padding = {
            "input_ids": int(self.processor.tokenizer.pad_token_id),
            "attention_mask": 0,
            "labels": -100,
            "mmrl_gating_mask": False,
        }
        batch = {
            key: pad_sequence(
                [feature[key] for feature in features],
                batch_first=True,
                padding_value=padding[key],
            )
            for key in features[0]
        }
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        batch["image_grid_thw"] = torch.cat(image_grids, dim=0)
        return batch

