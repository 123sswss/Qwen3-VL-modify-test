#!/usr/bin/env python3
"""Single V1 train with a real-batch preflight for PathVQA or SLAKE."""

from __future__ import annotations

import argparse
import importlib.metadata
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
from slake.data_pipeline import SLAKEDataCollator, SLAKEDataset
from slake.visual_selection_offset import VisualSelectionOffsetModel
from slake.visual_selection_prefix import EXPECTED_TRAINABLE, VisualSelectionPrefixModel
from slake.visual_selection_prefix_visual20 import (
    EXPECTED_TRAINABLE_VISUAL20,
    VisualSelectionPrefixVisual20Model,
)
from slake.visual_selection_prefix_deep5 import (
    DEEP_VISUAL_LAYERS,
    DEEP_VISUAL_TOKENS_PER_LAYER,
    EXPECTED_TRAINABLE_DEEP5,
    VisualSelectionPrefixDeep5Model,
)
from slake.visual_selection_prefix_deep20_split_lr import (
    DEEP_VISUAL_TOKENS_PER_GROUP,
    EXPECTED_TRAINABLE_DEEP20_SPLIT_LR,
    VisualSelectionPrefixDeep20SplitLRModel,
)


class RawQuestionDataset(Dataset):
    def __init__(self, source: Dataset, tokenizer) -> None:
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
    def __init__(self, processor, base_collator=None) -> None:
        self.base = base_collator or PathVQADataCollator(processor)

    def __call__(self, features):
        features = [dict(row) for row in features]
        source = [row.pop("question_source_ids") for row in features]
        batch = self.base(features)
        batch["question_source_ids"] = pad_sequence(source, batch_first=True, padding_value=0)
        batch["question_source_mask"] = pad_sequence(
            [torch.ones_like(ids, dtype=torch.bool) for ids in source],
            batch_first=True, padding_value=False,
        )
        keys = (
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
            "labels", "question_source_ids", "question_source_mask",
        )
        if "prompt_mask" in batch:
            keys += ("prompt_mask",)
        return {key: batch[key] for key in keys}


class V1Trainer(Trainer):
    def __init__(self, *args, visual_av10_learning_rate: float = 1e-4, **kwargs):
        self.visual_av10_learning_rate = float(visual_av10_learning_rate)
        if self.visual_av10_learning_rate not in (3e-5, 1e-4):
            raise ValueError("V1 Visual18 Av10 learning rate must be 3e-5 or 1e-4")
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.model.trainable_parameter_groups()
        rates = {
            "p20": 0.3,
            "question_context": 1e-4, "maps": 1e-4,
            "layer_condition": 1e-4, "prefix_output": 1e-4,
        }
        if "visual_deep_low_prompts" in groups:
            rates["visual_deep_low_prompts"] = 3e-5
            rates["visual_deep_high_prompts"] = 1e-4
        elif "visual_deep_prompts" in groups:
            rates["visual_deep_prompts"] = 1e-4
        elif "visual_prompt20" in groups:
            rates["visual_prompt20"] = 1e-4
        else:
            rates["visual_s8"] = 3e-5
            rates["visual_av10"] = self.visual_av10_learning_rate
        if set(rates) != set(groups):
            raise RuntimeError(f"V1 optimizer/group mismatch: rates={set(rates)} groups={set(groups)}")
        self.configured_group_learning_rates = dict(rates)
        self.model._audit_parameters()
        self.optimizer = torch.optim.AdamW([
            {"params": groups[name], "lr": lr, "weight_decay": 0.0, "group_name": name}
            for name, lr in rates.items()
        ], betas=(0.9, 0.999), eps=1e-8)
        print(f"[V1_OPTIMIZER] rates={json.dumps(rates)} warmup_ratio=0.03 scheduler=linear")
        return self.optimizer


class V1AuditCallback(TrainerCallback):
    def __init__(self, output_dir: Path, processor, save_epochs: tuple[int, ...]) -> None:
        self.output_dir = output_dir
        self.processor = processor
        self.save_epochs = frozenset(save_epochs)
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
        epoch = int(round(float(state.epoch or 0)))
        if state.is_world_process_zero and epoch in self.save_epochs:
            checkpoint = self.output_dir / "checkpoints" / f"epoch_{epoch}"
            kwargs["model"].save_v1(checkpoint)
            self.processor.save_pretrained(checkpoint)
            print(f"[V1_EPOCH_CHECKPOINT] epoch={epoch} path={checkpoint}")
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


def construct_with_v0_initialization_audit(
    base, seed: int, output_dir: Path, *, visual_prompt_mode: str = "split18",
):
    """Compare every shared trainable tensor against fresh same-seed V0."""
    if visual_prompt_mode in {"deep5_l16_23", "deep20_split_lr_l16_23"} and len(base.model.visual.blocks) != 24:
        raise ValueError(
            "Deep5 experiment requires exactly 24 visual blocks before any "
            f"experiment modules are constructed; found {len(base.model.visual.blocks)}"
        )
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
    model_classes = {
        "split18": VisualSelectionPrefixModel,
        "unified20": VisualSelectionPrefixVisual20Model,
        "deep5_l16_23": VisualSelectionPrefixDeep5Model,
        "deep20_split_lr_l16_23": VisualSelectionPrefixDeep20SplitLRModel,
    }
    model_class = model_classes[visual_prompt_mode]
    model = model_class(base, init_seed=seed)
    actual = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("prefix_output.")
        and name != "visual_prompt20"
        and not name.startswith("visual_deep_prompts.")
        and not name.startswith("visual_deep_low_prompts.")
        and not name.startswith("visual_deep_high_prompts.")
    }
    visual_prompt_audit = None
    if visual_prompt_mode == "unified20":
        expected_visual18 = torch.cat(
            (common.pop("visual_s8"), common.pop("visual_av10")), dim=0,
        )
        if not torch.equal(expected_visual18, model.visual_prompt20[:18].detach().cpu()):
            raise RuntimeError("V1 Visual20 first18 rows differ from baseline S8+Av10 initialization")
        visual_prompt_audit = {
            "first18_equal_to_baseline_s8_av10": True,
            "extra_rows": 2,
            "extra_rng": "private_cpu_generator",
            "global_rng_unchanged": True,
        }
    elif visual_prompt_mode == "deep5_l16_23":
        common.pop("visual_s8")
        common.pop("visual_av10")
        visual_prompt_audit = {
            "backbone_blocks": len(base.model.visual.blocks),
            "layers": list(DEEP_VISUAL_LAYERS),
            "tokens_per_layer": DEEP_VISUAL_TOKENS_PER_LAYER,
            "parameter_count": sum(
                parameter.numel() for parameter in model.visual_deep_prompts
            ),
            "independent_per_layer": True,
            "lifetime": "single_block_insert_then_remove",
            "extra_rng": "private_cpu_generator",
            "global_rng_unchanged": True,
        }
    elif visual_prompt_mode == "deep20_split_lr_l16_23":
        common.pop("visual_s8")
        common.pop("visual_av10")
        visual_prompt_audit = {
            "backbone_blocks": len(base.model.visual.blocks),
            "layers": list(DEEP_VISUAL_LAYERS),
            "tokens_per_group_per_layer": DEEP_VISUAL_TOKENS_PER_GROUP,
            "prompt_order": "low10_then_high10",
            "low_parameter_count": sum(
                parameter.numel() for parameter in model.visual_deep_low_prompts
            ),
            "high_parameter_count": sum(
                parameter.numel() for parameter in model.visual_deep_high_prompts
            ),
            "independent_per_layer_and_group": True,
            "lifetime": "single_block_insert_then_remove",
            "extra_rng": "private_cpu_generator",
            "global_rng_unchanged": True,
        }
    if set(common) != set(actual):
        raise RuntimeError(f"V1/V0 common initialization names differ: {set(common) ^ set(actual)}")
    mismatches = [
        name for name, expected in common.items()
        if not torch.equal(expected, actual[name].detach().cpu())
    ]
    if mismatches:
        raise RuntimeError(f"V1/V0 common initial values differ: {mismatches}")
    audit = {
        "reference": f"fresh_V0_seed{seed}", "equal": True,
        "tensor_count": len(common), "visual_prompt_mode": visual_prompt_mode,
        "visual_prompt": visual_prompt_audit,
    }
    with (output_dir / "v1_shared_initialization_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print("[V1_SHARED_INITIALIZATION_AUDIT] " + json.dumps(audit))
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pathvqa", "slake"), default="pathvqa")
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/dataset/pathVQA"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-epochs", type=int, nargs="+")
    parser.add_argument(
        "--visual-prompt-mode",
        choices=(
            "split18", "unified20", "deep5_l16_23",
            "deep20_split_lr_l16_23",
        ),
        default="split18",
    )
    parser.add_argument(
        "--visual-av10-learning-rate", type=float, choices=(3e-5, 1e-4), default=1e-4,
        help="Av10 optimizer LR for the original split18 layout; S8 remains 3e-5",
    )
    parser.add_argument("--correct-loss-accumulation", action="store_true",
                        help="Use Trainer's equal-microbatch mean loss scaling; guarded by the read-only audit launcher")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    save_epochs = tuple(sorted(set(args.save_epochs or [args.epochs])))
    if any(epoch <= 0 or epoch > args.epochs for epoch in save_epochs):
        raise ValueError(f"save epochs must be within [1,{args.epochs}]: {save_epochs}")
    if args.visual_prompt_mode != "split18" and args.visual_av10_learning_rate != 1e-4:
        raise ValueError("--visual-av10-learning-rate only applies to split18")
    print("[V1_RUNTIME] " + json.dumps({
        "experiment": args.experiment_name,
        "dataset": args.dataset,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=False).stdout.strip(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "correct_loss_accumulation": args.correct_loss_accumulation,
        "epochs": args.epochs,
        "save_epochs": save_epochs,
        "visual_prompt_mode": args.visual_prompt_mode,
        "visual_av10_learning_rate": args.visual_av10_learning_rate,
    }, sort_keys=True), flush=True)
    seed, data_seed = args.model_seed, 42
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
    model = construct_with_v0_initialization_audit(
        base, seed, args.output_dir, visual_prompt_mode=args.visual_prompt_mode,
    )
    counts = model._audit_parameters()
    expected_trainable = {
        "split18": EXPECTED_TRAINABLE,
        "unified20": EXPECTED_TRAINABLE_VISUAL20,
        "deep5_l16_23": EXPECTED_TRAINABLE_DEEP5,
        "deep20_split_lr_l16_23": EXPECTED_TRAINABLE_DEEP20_SPLIT_LR,
    }[args.visual_prompt_mode]
    if sum(counts.values()) != expected_trainable:
        raise RuntimeError("V1 parameter total changed")
    if args.dataset == "pathvqa":
        source_dataset = PathVQADataset(
            processor=processor, data_root=args.data_root, split="train",
            ce_enabled=True, seed=data_seed, deterministic_sampling=True,
            max_length=2048,
        )
        base_collator = PathVQADataCollator(processor)
        dataset_name = "PathVQA"
        split_manifest = None
    else:
        split_manifest = args.data_root / "train.json"
        source_dataset = SLAKEDataset(
            processor=processor,
            questions_path=str(split_manifest),
            image_root=str(args.data_root / "imgs"),
            languages=None,
            base_types=None,
            splits=("train",),
            ce_enabled=True,
            seed=data_seed,
            deterministic_sampling=True,
            max_length=2048,
        )
        base_collator = SLAKEDataCollator(processor)
        dataset_name = "SLAKE"
    dataset = RawQuestionDataset(source_dataset, processor.tokenizer)
    collator = RawQuestionCollator(processor, base_collator=base_collator)
    print(
        f"[V1_DATASET] dataset={dataset_name} split=train languages=all "
        f"samples={len(dataset)} manifest={split_manifest} data_seed={data_seed}",
        flush=True,
    )
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
        visual_av10_learning_rate=args.visual_av10_learning_rate,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer"), num_train_epochs=args.epochs,
            per_device_train_batch_size=2, gradient_accumulation_steps=16,
            learning_rate=1e-4, weight_decay=0.0, warmup_ratio=0.03,
            lr_scheduler_type="linear", max_grad_norm=1.0, logging_steps=20,
            save_strategy="no", bf16=True, gradient_checkpointing=False,
            dataloader_num_workers=2, remove_unused_columns=False,
            report_to="none", seed=seed, data_seed=data_seed,
        ),
        train_dataset=dataset, data_collator=collator,
        processing_class=processor,
        callbacks=[V1AuditCallback(args.output_dir, processor, save_epochs)],
    )
    if args.correct_loss_accumulation:
        # Qwen's mean-token CE ignores num_items_in_batch. Let Trainer divide
        # each microbatch loss by the actual accumulation-window size.
        # Do not also divide manually or change Accelerate's accumulation.
        trainer.model_accepts_loss_kwargs = False
        if trainer.model_accepts_loss_kwargs is not False or trainer.args.gradient_accumulation_steps != 16 or trainer.accelerator.gradient_accumulation_steps != 1:
            raise RuntimeError("V1 corrected accumulation runtime does not match the audited path")
    print(f"[V1_LOSS_ACCUMULATION] corrected={args.correct_loss_accumulation} "
          f"trainer_model_accepts_loss_kwargs={trainer.model_accepts_loss_kwargs} "
          f"trainer_steps={trainer.args.gradient_accumulation_steps} "
          f"accelerate_steps={trainer.accelerator.gradient_accumulation_steps}")
    result = trainer.train()
    missing_checkpoints = [
        str(args.output_dir / "checkpoints" / f"epoch_{epoch}")
        for epoch in save_epochs
        if not (args.output_dir / "checkpoints" / f"epoch_{epoch}").is_dir()
    ]
    if missing_checkpoints:
        raise RuntimeError(f"V1 requested checkpoints not saved: {missing_checkpoints}")
    checkpoint = args.output_dir / "checkpoints" / f"epoch_{args.epochs}"
    report = {
        "experiment": args.experiment_name,
        "method": getattr(model, "method_name", "visual_selection_prefix_p20_v1"),
        "dataset": dataset_name, "model_seed": seed, "data_seed": data_seed,
        "train_split": "train", "languages": "all",
        "train_manifest": str(split_manifest) if split_manifest is not None else None,
        "epochs": args.epochs, "saved_epochs": list(save_epochs),
        "visual_prompt_mode": args.visual_prompt_mode,
        "visual_av10_learning_rate": args.visual_av10_learning_rate,
        "trainable_parameters": counts,
        "total_trainable_parameters": sum(counts.values()),
        "train_metrics": result.metrics,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "preflight": str(args.output_dir / "v1_real_batch_preflight.json"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=False,
        ).stdout.strip(),
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
        },
        "trainer_model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
        "accelerator_gradient_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
        "loss_accumulation_protocol": (
            "equal_microbatch_mean_trainer_normalized" if args.correct_loss_accumulation
            else "original_unmodified_trainer_protocol"
        ),
        "optimizer": {
            "type": "AdamW", "weight_decay": 0.0, "betas": [0.9, 0.999],
            "eps": 1e-8, "scheduler": "linear", "warmup_ratio": 0.03,
            "max_grad_norm": 1.0, "per_device_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "group_learning_rates": trainer.configured_group_learning_rates,
        },
    }
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[V1_TRAIN_DONE] epochs={args.epochs} saved_epochs={save_epochs} "
          f"checkpoint={checkpoint} runtime={result.metrics.get('train_runtime')} "
          f"peak_gpu_bytes={report['peak_gpu_memory_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
