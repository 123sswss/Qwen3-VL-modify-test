import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import torch
from PIL import Image
from torch.utils.data import Dataset


TASK_TYPE_VQA_ID = 1
XMLAB_DIR_PATTERN = re.compile(r"^xmlab(\d+)$")


def _normalize_filter(values: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        values = (values,)
    return {str(value).strip().lower() for value in values}


def _xmlab_sort_key(path: str):
    match = XMLAB_DIR_PATTERN.fullmatch(os.path.basename(path))
    return int(match.group(1)) if match else float("inf")


class SLAKEDataset(Dataset):
    """Load per-image SLAKE annotations as independent VQA samples.

    ``image_root`` must contain ``xmlabN`` directories. Each directory is
    expected to contain ``source.jpg`` and ``question.json``. Every object in
    a question file becomes one dataset sample, even when several objects
    reference the same image.
    """

    def __init__(
        self,
        processor,
        image_root: str,
        total_limit: Optional[int] = None,
        languages: Optional[Sequence[str]] = None,
        base_types: Optional[Sequence[str]] = None,
        ce_enabled: bool = True,
        seed: int = 42,
        deterministic_sampling: bool = True,
        max_length: int = 1024,
    ):
        self.processor = processor
        self.image_root = os.path.abspath(os.fspath(image_root))
        self.total_limit = total_limit
        self.languages = _normalize_filter(languages)
        self.base_types = _normalize_filter(base_types)
        self.ce_enabled = ce_enabled
        self.seed = seed
        self.deterministic_sampling = deterministic_sampling
        self.max_length = max_length
        self.resample_round = 0

        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(f"SLAKE image root does not exist: {self.image_root}")
        if total_limit is not None and total_limit < 0:
            raise ValueError("total_limit must be non-negative or None")

        self.raw_samples = self._scan_samples()
        if not self.raw_samples:
            raise ValueError(
                "No valid SLAKE conversations were found. Expected "
                "image_root/xmlabN/{source.jpg,question.json}."
            )

        self.data: List[Dict[str, Any]] = []
        self._build()

    def _scan_samples(self) -> List[Dict[str, Any]]:
        xmlab_dirs = []
        for entry in os.scandir(self.image_root):
            if entry.is_dir() and XMLAB_DIR_PATTERN.fullmatch(entry.name):
                xmlab_dirs.append(entry.path)
        xmlab_dirs.sort(key=_xmlab_sort_key)

        samples = []
        skipped_missing_files = 0
        skipped_invalid_records = 0
        skipped_by_filter = 0

        for xmlab_dir in xmlab_dirs:
            question_path = os.path.join(xmlab_dir, "question.json")
            fallback_image_path = os.path.join(xmlab_dir, "source.jpg")
            if not os.path.isfile(question_path) or not os.path.isfile(fallback_image_path):
                skipped_missing_files += 1
                continue

            with open(question_path, "r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                raise ValueError(f"SLAKE question file must contain a JSON list: {question_path}")

            for record_index, record in enumerate(records):
                if not isinstance(record, dict):
                    skipped_invalid_records += 1
                    continue

                question = str(record.get("question", "")).strip()
                answer = str(record.get("answer", "")).strip()
                if not question or not answer:
                    skipped_invalid_records += 1
                    continue

                language = str(record.get("q_lang", "")).strip().lower()
                base_type = str(record.get("base_type", "")).strip().lower()
                if self.languages is not None and language not in self.languages:
                    skipped_by_filter += 1
                    continue
                if self.base_types is not None and base_type not in self.base_types:
                    skipped_by_filter += 1
                    continue

                image_path = self._resolve_image_path(record, fallback_image_path)
                if not os.path.isfile(image_path):
                    skipped_missing_files += 1
                    continue

                samples.append(
                    {
                        "image_path": image_path,
                        "question_path": question_path,
                        "record_index": record_index,
                        "question": question,
                        "answer": answer,
                    }
                )

        print(
            "[SLAKEDataset] "
            f"root={self.image_root} xmlab_dirs={len(xmlab_dirs)} "
            f"valid_conversations={len(samples)} "
            f"skipped_by_filter={skipped_by_filter} "
            f"skipped_missing_files={skipped_missing_files} "
            f"skipped_invalid_records={skipped_invalid_records}"
        )
        return samples

    def _resolve_image_path(self, record: Dict[str, Any], fallback_path: str) -> str:
        relative_path = str(record.get("img_name", "")).strip()
        if not relative_path:
            return fallback_path

        normalized = os.path.normpath(relative_path.replace("/", os.sep))
        candidate = os.path.abspath(os.path.join(self.image_root, normalized))
        try:
            if os.path.commonpath((self.image_root, candidate)) != self.image_root:
                return fallback_path
        except ValueError:
            return fallback_path
        return candidate

    def _build(self):
        if self.deterministic_sampling:
            rng = random.Random(self.seed + self.resample_round)
        else:
            rng = random

        if self.total_limit is None or self.total_limit >= len(self.raw_samples):
            selected = list(self.raw_samples)
        else:
            selected = rng.sample(self.raw_samples, self.total_limit)
        rng.shuffle(selected)
        self.data = selected

        print(
            "[SLAKEDataset] "
            f"resample_round={self.resample_round} selected_samples={len(self.data)}"
        )

    def resample_data(self):
        self.resample_round += 1
        self._build()

    def __len__(self):
        return len(self.data)

    def _build_qwen_conversation(self, sample: Dict[str, Any], image: Image.Image):
        return [
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

    def __getitem__(self, idx):
        sample = self.data[idx]
        with Image.open(sample["image_path"]) as source_image:
            image = source_image.convert("RGB")

        conversation = self._build_qwen_conversation(sample, image)
        text_inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            images=image,
            text=text_inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.squeeze(0)

        image_grid_thw = inputs["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        labels = self._build_labels(input_ids, attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "alpha_labels": 1.0,
            "task_type_id": TASK_TYPE_VQA_ID,
            "task_type_name": "vqa",
            "source_name": "slake",
            "images_per_sample": 1,
            "is_mm": 1,
        }

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        labels = input_ids.clone()
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
        if len(im_start_positions) < 2:
            raise ValueError(
                "The rendered SLAKE conversation has no assistant turn marker. "
                "Check the processor chat template."
            )

        assistant_start = im_start_positions[1].item() + 2
        labels[:assistant_start] = -100
        labels[attention_mask == 0] = -100
        if not self.ce_enabled:
            labels[:] = -100
        return labels


class SLAKEDataCollator:
    """Collate SLAKE samples into the same batch keys as MMRLDataCollator."""

    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("SLAKEDataCollator received an empty feature list")

        features = [dict(feature) for feature in features]
        alpha_labels = [feature.pop("alpha_labels") for feature in features]
        task_type_ids = [feature.pop("task_type_id") for feature in features]
        task_type_names = [feature.pop("task_type_name") for feature in features]
        source_names = [feature.pop("source_name") for feature in features]
        images_per_sample = [feature.pop("images_per_sample") for feature in features]
        is_mm = [feature.pop("is_mm") for feature in features]
        pixel_values = [feature.pop("pixel_values") for feature in features]
        image_grids = [feature.pop("image_grid_thw") for feature in features]

        batch = {}
        for key in features[0]:
            values = [feature[key] for feature in features]
            batch[key] = torch.stack(values, dim=0) if torch.is_tensor(values[0]) else values

        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        batch["image_grid_thw"] = torch.cat(image_grids, dim=0)
        batch["alpha_labels"] = torch.tensor(alpha_labels, dtype=torch.float32)
        batch["task_type_ids"] = torch.tensor(task_type_ids, dtype=torch.long)
        batch["task_type_names"] = task_type_names
        batch["source_names"] = source_names
        batch["images_per_sample"] = images_per_sample
        batch["is_mm"] = torch.tensor(is_mm, dtype=torch.long)
        return batch
