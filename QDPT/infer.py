#!/usr/bin/env python3
"""Run single-image or full-split PathVQA inference with QDPT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import INSTRUCTION, MAX_NEW_TOKENS, MODEL_NAME
from data import PathVQAParquetStore
from evaluate import evaluate_records
from model import QDPTModel


class QDPTInference:
    def __init__(self, model_path: str, checkpoint: Path) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = QDPTModel(base_model, self.processor.tokenizer)
        self.model.load_adapter(checkpoint)
        self.model.eval()
        self.device = next(base_model.parameters()).device

    def infer(self, image: Image.Image, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question.strip()}\n{INSTRUCTION}"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(
                device=self.device,
                dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
            )
        original_length = int(moved["input_ids"].shape[-1])
        with torch.inference_mode():
            output_ids = self.model.generate(
                **moved,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        generated = output_ids[:, original_length + self.model.prompt_length :]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QDPT inference")
    parser.add_argument("--model-path", default=MODEL_NAME)
    parser.add_argument("--checkpoint", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--image", type=Path, required=True)
    single.add_argument("--question", required=True)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--data-root", type=Path, required=True)
    dataset.add_argument("--output-dir", type=Path, required=True)
    dataset.add_argument(
        "--split", choices=("validation", "test"), default="test"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    interface = QDPTInference(args.model_path, args.checkpoint)
    if args.mode == "single":
        with Image.open(args.image) as image:
            print(interface.infer(image.convert("RGB"), args.question))
        return 0

    store = PathVQAParquetStore(args.data_root, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for index, record in enumerate(store.samples, start=1):
        image = store.load_image(record)
        try:
            answer = interface.infer(image, record["question"])
        finally:
            image.close()
        predictions.append(
            {"question_id": record["question_id"], "answer": answer}
        )
        print(f"[{index}/{len(store)}] {record['question_id']}: {answer}")

    predictions_path = args.output_dir / "pathvqa_predictions.json"
    summary_path = args.output_dir / "pathvqa_summary.json"
    with predictions_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
    summary = evaluate_records(store.samples, predictions)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

