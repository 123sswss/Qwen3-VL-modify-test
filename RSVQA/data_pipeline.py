"""Qwen training adapter for the normalized RSVQA-LR record contract."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset

from slake.data_pipeline import (
    SLAKEDataCollator,
    TASK_TYPE_VQA_ID,
    build_target_supervision_masks,
)

from .data import load_rsvqa_lr_split
from .prompts import build_prompt


class RSVQALRDataset(Dataset):
    """Build complete Qwen image/question/answer sequences from official splits.

    ``data`` deliberately retains the raw question.  The V1 wrapper uses it as
    the independent condition source, while the LLM conversation uses the
    shared type-specific short-answer prompt from :mod:`RSVQA.prompts`.
    """

    def __init__(
        self,
        processor,
        data_root: Path | str,
        split: str = "train",
        *,
        ce_enabled: bool = True,
        seed: int = 42,
        deterministic_sampling: bool = True,
        max_length: int = 2048,
        enforce_official_counts: bool = True,
    ) -> None:
        self.processor = processor
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.ce_enabled = bool(ce_enabled)
        self.seed = int(seed)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.max_length = int(max_length)
        self.resample_round = 0
        if self.max_length < 1:
            raise ValueError("max_length must be positive")

        records, self.manifest = load_rsvqa_lr_split(
            self.data_root,
            split,
            enforce_official_counts=enforce_official_counts,
            require_images=True,
        )
        self.raw_samples = records
        self.data: list[Dict[str, Any]] = []

        tokenizer = self.processor.tokenizer
        self.assistant_header_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_label_prefix_ids = tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            not self.assistant_header_ids
            or not self.assistant_label_prefix_ids
            or self.im_end_token_id is None
        ):
            raise RuntimeError("Failed to resolve Qwen assistant boundary tokens")
        self._build()

    def _build(self) -> None:
        rows = list(self.raw_samples)
        rng = (
            random.Random(self.seed + self.resample_round)
            if self.deterministic_sampling
            else random
        )
        rng.shuffle(rows)
        self.data = rows
        print(
            "[RSVQALRDataset] "
            f"split={self.split} samples={len(rows)} images={self.manifest['active_images']} "
            f"types={self.manifest['question_type_counts']} data_seed={self.seed} "
            "answer_supervision=raw_release_answer prompt_policy=question_type_short_answer_v1"
        )

    def resample_data(self) -> None:
        self.resample_round += 1
        self._build()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.data[index]
        with Image.open(sample["image_path"]) as source_image:
            image = source_image.convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": build_prompt(sample)},
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
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if input_ids.numel() > self.max_length:
            raise ValueError(
                "RSVQA-LR sample exceeds the complete-sequence limit; "
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
            "source_name": "rsvqa_lr",
            "images_per_sample": 1,
            "is_mm": 1,
        }


RSVQADataCollator = SLAKEDataCollator

__all__ = ["RSVQADataCollator", "RSVQALRDataset"]
