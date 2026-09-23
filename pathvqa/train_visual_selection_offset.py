#!/usr/bin/env python3
"""Single PathVQA seed44/epoch3 V0 training; no evaluation or server action here."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback,
    TrainingArguments,
)

from pathvqa.data_pipeline import PathVQADataCollator, PathVQADataset
from slake.visual_selection_offset import (
    EXPECTED_TRAINABLE, VisualSelectionOffsetModel, locate_question_mask,
)


class QuestionMaskedDataset(Dataset):
    def __init__(self, source: PathVQADataset, tokenizer) -> None:
        self.source = source
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        row = dict(self.source[index])
        row["question_mask"] = locate_question_mask(
            row["input_ids"], self.source.data[index]["question"], self.tokenizer,
            context_mask=row["mmrl_gating_mask"].bool(),
        )
        return row


class QuestionMaskedCollator:
    def __init__(self, processor) -> None:
        self.base = PathVQADataCollator(processor)

    def __call__(self, features):
        features = [dict(row) for row in features]
        masks = [row.pop("question_mask") for row in features]
        batch = self.base(features)
        batch["question_mask"] = pad_sequence(masks, batch_first=True, padding_value=False)
        return {
            key: batch[key]
            for key in (
                "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                "labels", "question_mask",
            )
        }


class V0Trainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        rates = {
            "p20": 0.3,
            "visual_s8": 3e-5,
            "visual_av10": 1e-4,
            "question_context": 1e-4,
            "maps": 1e-4,
            "layer_condition": 1e-4,
            "offset": 1e-4,
        }
        groups = self.model.trainable_parameter_groups()
        self.model._audit_parameters()
        self.optimizer = torch.optim.AdamW(
            [
                {"params": groups[name], "lr": lr, "weight_decay": 0.0, "group_name": name}
                for name, lr in rates.items()
            ],
            betas=(0.9, 0.999), eps=1e-8,
        )
        print(f"[V0_OPTIMIZER] rates={json.dumps(rates)} warmup_ratio=0.03 scheduler=linear")
        return self.optimizer


class V0AuditCallback(TrainerCallback):
    def __init__(self, output_dir: Path, processor) -> None:
        self.output_dir = output_dir
        self.processor = processor
        self.first_recorded = False
        self.step_path = output_dir / "v0_diagnostics.jsonl"

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        groups = model.trainable_parameter_groups()
        row = {"step": int(state.global_step), "epoch": float(state.epoch or 0)}
        for name, parameters in groups.items():
            grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
            if len(grads) != len(parameters) or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError(f"V0 missing/nonfinite gradient in {name}")
            row[f"{name}_grad_norm"] = math.sqrt(sum(float(g.square().sum()) for g in grads))
        for name, value in model.debug_context.items():
            row[name] = float(value.float())
        if not self.first_recorded:
            if model.first_batch_diagnostics is None:
                raise RuntimeError("V0 first real batch did not produce offset diagnostics")
            first_grad = model.first_backward_gradients
            names = {id(p): key for key, p in model.named_parameters() if p.requires_grad}
            first_groups = {}
            for group, parameters in groups.items():
                values = [first_grad.get(names[id(p)]) for p in parameters]
                if any(v is None or not math.isfinite(v) for v in values):
                    raise RuntimeError(f"V0 first backward did not reach all {group} parameters")
                first_groups[group] = math.sqrt(sum(v * v for v in values))
                if first_groups[group] <= 0:
                    raise RuntimeError(f"V0 first backward {group} branch is inactive")
            required = [
                "p20", "visual_s8", "visual_av10", "text_projection.weight",
                "question_depthwise.weight", "question_pointwise.weight",
                "question_pool.weight", "layer_gate.weight",
                "offset_down.weight", "offset_up.weight",
            ] + [
                f"{prefix}.{index}.weight"
                for prefix in ("query_heads", "key_heads", "value_blocks")
                for index in range(3)
            ]
            inactive = [name for name in required if first_grad.get(name, 0.0) <= 0]
            if inactive:
                raise RuntimeError(f"V0 first backward has inactive condition branches: {inactive}")
            first = {
                "first_batch_forward": model.first_batch_diagnostics,
                "first_batch_backward_group_grad_norm": first_groups,
                "first_batch_backward_parameter_grad_norm": first_grad,
            }
            if first["first_batch_forward"]["offset_to_question_rms"] >= 0.1:
                raise RuntimeError("V0 initial offset exceeds 10% of real-question RMS")
            with (self.output_dir / "v0_first_batch_audit.json").open("w", encoding="utf-8") as handle:
                json.dump(first, handle, indent=2)
            print("[V0_FIRST_BATCH_AUDIT] " + json.dumps(first, ensure_ascii=False))
            self.first_recorded = True
        if int(state.global_step) % 20 == 0 or int(state.global_step) == 0:
            with self.step_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("[V0_DIAGNOSTICS] " + json.dumps(row, ensure_ascii=False))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(float(state.epoch or 0)))
        if state.is_world_process_zero and epoch == 3:
            checkpoint = self.output_dir / "checkpoints" / "epoch_3"
            kwargs["model"].save_v0(checkpoint)
            self.processor.save_pretrained(checkpoint)
            print(f"[V0_EPOCH3_CHECKPOINT] {checkpoint}")
        return control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/dataset/pathVQA"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    seed, data_seed = 44, 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    base.config.use_cache = False
    model = VisualSelectionOffsetModel(base, init_seed=seed)
    counts = model._audit_parameters()
    if sum(counts.values()) != EXPECTED_TRAINABLE:
        raise RuntimeError("V0 trainable parameter total changed")
    dataset = QuestionMaskedDataset(
        PathVQADataset(
            processor=processor, data_root=args.data_root, split="train",
            ce_enabled=True, seed=data_seed, deterministic_sampling=True,
            max_length=2048,
        ),
        processor.tokenizer,
    )
    trainer = V0Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"), num_train_epochs=3,
            per_device_train_batch_size=2, gradient_accumulation_steps=16,
            learning_rate=1e-4, weight_decay=0.0, warmup_ratio=0.03,
            lr_scheduler_type="linear", max_grad_norm=1.0, logging_steps=20,
            save_strategy="no", bf16=True, gradient_checkpointing=False,
            dataloader_num_workers=2, remove_unused_columns=False,
            report_to="none", seed=seed, data_seed=data_seed,
        ),
        train_dataset=dataset,
        data_collator=QuestionMaskedCollator(processor),
        processing_class=processor,
        callbacks=[V0AuditCallback(args.output_dir, processor)],
    )
    result = trainer.train()
    checkpoint = args.output_dir / "checkpoints" / "epoch_3"
    if not checkpoint.is_dir():
        raise RuntimeError("V0 epoch3 checkpoint was not saved")
    report = {
        "experiment": args.experiment_name, "method": "visual_selection_offset_v0",
        "dataset": "PathVQA", "model_seed": seed, "data_seed": data_seed,
        "epochs": 3, "trainable_parameters": counts,
        "total_trainable_parameters": sum(counts.values()),
        "train_metrics": result.metrics,
        "first_batch_audit": str(args.output_dir / "v0_first_batch_audit.json"),
    }
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[V0_TRAIN_DONE] checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
