#!/usr/bin/env python3
"""Single V1 seed44 PathVQA train with a real-batch preflight."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
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
from slake.visual_selection_offset import VisualSelectionOffsetModel
from slake.visual_selection_prefix import EXPECTED_TRAINABLE, VisualSelectionPrefixModel


class RawQuestionDataset(Dataset):
    def __init__(self, source: PathVQADataset, tokenizer) -> None:
        self.source = source
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        row = dict(self.source[index])
        question = str(self.source.data[index]["question"]).strip()
        ids = self.tokenizer.encode(question, add_special_tokens=False)
        if not ids:
            raise ValueError(f"empty independent raw question at row {index}")
        row["question_source_ids"] = torch.tensor(ids, dtype=row["input_ids"].dtype)
        return row


class RawQuestionCollator:
    def __init__(self, processor) -> None:
        self.base = PathVQADataCollator(processor)

    def __call__(self, features):
        features = [dict(row) for row in features]
        source = [row.pop("question_source_ids") for row in features]
        batch = self.base(features)
        batch["question_source_ids"] = pad_sequence(source, batch_first=True, padding_value=0)
        batch["question_source_mask"] = pad_sequence(
            [torch.ones_like(ids, dtype=torch.bool) for ids in source],
            batch_first=True, padding_value=False,
        )
        return {key: batch[key] for key in (
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
            "labels", "question_source_ids", "question_source_mask",
        )}


class V1Trainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        rates = {
            "p20": 0.3, "visual_s8": 3e-5, "visual_av10": 1e-4,
            "question_context": 1e-4, "maps": 1e-4,
            "layer_condition": 1e-4, "prefix_output": 1e-4,
        }
        groups = self.model.trainable_parameter_groups()
        self.model._audit_parameters()
        self.optimizer = torch.optim.AdamW([
            {"params": groups[name], "lr": lr, "weight_decay": 0.0, "group_name": name}
            for name, lr in rates.items()
        ], betas=(0.9, 0.999), eps=1e-8)
        print(f"[V1_OPTIMIZER] rates={json.dumps(rates)} warmup_ratio=0.03 scheduler=linear")
        return self.optimizer


class V1AuditCallback(TrainerCallback):
    def __init__(self, output_dir: Path, processor) -> None:
        self.output_dir = output_dir
        self.processor = processor
        self.step_path = output_dir / "v1_diagnostics.jsonl"

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        groups = model.trainable_parameter_groups()
        row = {"step": int(state.global_step), "epoch": float(state.epoch or 0)}
        for name, parameters in groups.items():
            grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
            if len(grads) != len(parameters) or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError(f"V1 missing/nonfinite gradient in {name}")
            row[f"{name}_grad_norm"] = math.sqrt(sum(float(g.square().sum()) for g in grads))
        row.update({key: float(value.float()) for key, value in model.debug_context.items()})
        if int(state.global_step) % 20 == 0:
            with self.step_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("[V1_DIAGNOSTICS] " + json.dumps(row, ensure_ascii=False))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and int(round(float(state.epoch or 0))) == 3:
            checkpoint = self.output_dir / "checkpoints" / "epoch_3"
            kwargs["model"].save_v1(checkpoint)
            self.processor.save_pretrained(checkpoint)
            print(f"[V1_EPOCH3_CHECKPOINT] {checkpoint}")
        return control


def real_batch_preflight(model, dataset, collator, output_dir: Path) -> None:
    """Fail before training if the actual image/question batch cannot train V1."""
    batch = collator([dataset[0], dataset[1]])
    device = next(model.base_model.parameters()).device
    moved = {
        key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
        for key, value in batch.items()
    }
    model.train()
    output = model(**moved)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise RuntimeError("V1 real batch has missing/nonfinite loss")
    output.loss.backward()
    groups = model.trainable_parameter_groups()
    grad_norms = {}
    for name, parameters in groups.items():
        grads = [p.grad for p in parameters]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError(f"V1 real batch lacks finite {name} gradients")
        grad_norms[name] = math.sqrt(sum(float(g.float().square().sum()) for g in grads))
        if grad_norms[name] <= 0:
            raise RuntimeError(f"V1 real batch has inactive {name} branch")
    required = (
        "text_projection.weight", "question_depthwise.weight", "question_pointwise.weight",
        "question_pool.weight", "layer_gate.weight", "prefix_output.weight",
    ) + tuple(
        f"{prefix}.{i}.weight"
        for prefix in ("query_heads", "key_heads", "value_blocks") for i in range(3)
    )
    inactive = [name for name in required if model.first_backward_gradients.get(name, 0) <= 0]
    if inactive:
        raise RuntimeError(f"V1 real batch inactive parameters: {inactive}")
    audit = {
        "loss": float(output.loss.detach()), "parameter_counts": model._audit_parameters(),
        "group_gradient_norms": grad_norms,
        "parameter_gradient_norms": model.first_backward_gradients,
        "forward": model.first_batch_diagnostics,
        "injection": model.last_injection_audit,
        "question_policy": "independent_raw_question_ids_no_prefill_write_mapping",
    }
    with (output_dir / "v1_real_batch_preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print("[V1_REAL_BATCH_PREFLIGHT] " + json.dumps(audit, ensure_ascii=False))
    model.zero_grad(set_to_none=True)


def construct_with_v0_initialization_audit(base, seed: int, output_dir: Path):
    """Compare every shared trainable tensor against fresh same-seed V0."""
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    reference = VisualSelectionOffsetModel(base, init_seed=seed)
    common = {
        name: parameter.detach().cpu().clone()
        for name, parameter in reference.named_parameters()
        if parameter.requires_grad and not name.startswith(("offset_down.", "offset_up."))
    }
    base.model.visual.blocks[17] = base.model.visual.blocks[17].block
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    model = VisualSelectionPrefixModel(base, init_seed=seed)
    actual = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("prefix_output.")
    }
    if set(common) != set(actual):
        raise RuntimeError(f"V1/V0 common initialization names differ: {set(common) ^ set(actual)}")
    mismatches = [
        name for name, expected in common.items()
        if not torch.equal(expected, actual[name].detach().cpu())
    ]
    if mismatches:
        raise RuntimeError(f"V1/V0 common initial values differ: {mismatches}")
    audit = {"reference": "fresh_V0_seed44", "equal": True, "tensor_count": len(common)}
    with (output_dir / "v1_shared_initialization_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print("[V1_SHARED_INITIALIZATION_AUDIT] " + json.dumps(audit))
    return model


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
    model = construct_with_v0_initialization_audit(base, seed, args.output_dir)
    counts = model._audit_parameters()
    if sum(counts.values()) != EXPECTED_TRAINABLE:
        raise RuntimeError("V1 parameter total changed")
    dataset = RawQuestionDataset(PathVQADataset(
        processor=processor, data_root=args.data_root, split="train",
        ce_enabled=True, seed=data_seed, deterministic_sampling=True,
        max_length=2048,
    ), processor.tokenizer)
    collator = RawQuestionCollator(processor)
    # Preserve training RNG after the mandatory real-batch audit.
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    real_batch_preflight(model, dataset, collator, args.output_dir)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.reset_peak_memory_stats()
    trainer = V1Trainer(
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
        train_dataset=dataset, data_collator=collator,
        processing_class=processor,
        callbacks=[V1AuditCallback(args.output_dir, processor)],
    )
    result = trainer.train()
    checkpoint = args.output_dir / "checkpoints" / "epoch_3"
    if not checkpoint.is_dir():
        raise RuntimeError("V1 epoch3 checkpoint not saved")
    report = {
        "experiment": args.experiment_name, "method": "visual_selection_prefix_p20_v1",
        "dataset": "PathVQA", "model_seed": seed, "data_seed": data_seed,
        "epochs": 3, "trainable_parameters": counts,
        "total_trainable_parameters": sum(counts.values()),
        "train_metrics": result.metrics,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "preflight": str(args.output_dir / "v1_real_batch_preflight.json"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=False,
        ).stdout.strip(),
        "optimizer": {
            "type": "AdamW", "weight_decay": 0.0, "betas": [0.9, 0.999],
            "eps": 1e-8, "scheduler": "linear", "warmup_ratio": 0.03,
            "max_grad_norm": 1.0, "per_device_batch_size": 2,
            "gradient_accumulation_steps": 16,
        },
    }
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[V1_TRAIN_DONE] checkpoint={checkpoint} runtime={result.metrics.get('train_runtime')} peak_gpu_bytes={report['peak_gpu_memory_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
