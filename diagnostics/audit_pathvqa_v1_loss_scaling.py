#!/usr/bin/env python3
"""Read-only V1 loss/accumulation audit; run only after the active training ends.

The audit loads a saved V1 checkpoint, replays fixed training microbatches,
never calls optimizer.step(), and writes only to its independent output dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
from accelerate import Accelerator

from pathvqa.data_pipeline import PathVQADataset
from pathvqa.train_visual_selection_prefix import (
    RawQuestionCollator, RawQuestionDataset, V1Trainer,
)
from slake.visual_selection_prefix import VisualSelectionPrefixModel


HISTORICAL_RUNS = {
    "v0": "visual_selection_offset/pathvqa_v0_visual_selection_offset_seed44_20260923",
    "cocoop": "cocoop/pathvqa_cocoop_style_p20_h160_seed44_20260909",
    "static_p20": "prompt_tuning/pathvqa_prompt_tuning_len20_seed44_20260827",
    "qdpt": "dynamic_prompt/pathvqa_qdpt_d768_question_q10_l17_p20_s8_av10_sandwich_seed44_20260909",
    "lora": "lora/pathvqa_lora_full_model_attention_r8_seed44_20260830",
}
HISTORICAL_SOURCE = {
    "v0": ("pathvqa/train_visual_selection_offset.py", "slake/visual_selection_offset.py"),
    "cocoop": ("pathvqa/train_cocoop.py", "slake/cocoop_prompt_tuning.py"),
    "static_p20": ("pathvqa/train_prompt_tuning.py", "slake/prompt_tuning.py"),
    "qdpt": ("pathvqa/train_dynamic_prompt.py", "slake/dynamic_prompt_tuning.py"),
    "lora": ("pathvqa/train_visual_lora.py", "PEFT PeftModel runtime wrapper"),
}


def historical_provenance(output_root: Path) -> dict[str, dict]:
    """Collect only versions explicitly recorded by each historical run."""
    found = {}
    version_pattern = re.compile(r"\b(torch|transformers|accelerate|peft)\s*(?:==|=|:)\s*([0-9][^\s,;]*)", re.I)
    for family, relative in HISTORICAL_RUNS.items():
        root = output_root / relative
        row = {
            "experiment": root.name, "seed": 44, "output": str(root), "exists": root.is_dir(),
            "commit": None, "commit_evidence": None, "recorded_versions": {},
            "version_evidence": {}, "training_entry": HISTORICAL_SOURCE[family][0],
            "model_wrapper": HISTORICAL_SOURCE[family][1],
        }
        report = root / "train_report.json"
        if report.is_file():
            try:
                metadata = json.loads(report.read_text(encoding="utf-8"))
                row["commit"] = metadata.get("git_commit")
                if row["commit"]:
                    row["commit_evidence"] = str(report)
                for name in ("torch", "transformers", "accelerate", "peft"):
                    for key in (f"{name}_version", name):
                        value = metadata.get(key)
                        if isinstance(value, str):
                            row["recorded_versions"][name] = value
                            row["version_evidence"][name] = f"{report}:{key}"
                for key in ("versions", "library_versions"):
                    values = metadata.get(key)
                    if isinstance(values, dict):
                        for name in ("torch", "transformers", "accelerate", "peft"):
                            if name in values:
                                row["recorded_versions"][name] = str(values[name])
                                row["version_evidence"][name] = f"{report}:{key}.{name}"
            except (OSError, ValueError) as exc:
                row["report_read_error"] = str(exc)
        log = root / "train.log"
        if log.is_file():
            with log.open(encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    for match in version_pattern.finditer(line):
                        name = match.group(1).lower()
                        if name not in row["recorded_versions"]:
                            row["recorded_versions"][name] = match.group(2)
                            row["version_evidence"][name] = f"{log}:{line_number}"
        if row["commit"]:
            entry, wrapper = HISTORICAL_SOURCE[family]
            row["source_at_training_commit"] = {}
            for path in (entry, wrapper):
                if path.startswith("PEFT "):
                    continue
                result = subprocess.run(
                    ["git", "show", f"{row['commit']}:{path}"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
                )
                if result.returncode == 0:
                    source = result.stdout
                    row["source_at_training_commit"][path] = {
                        "forward_var_kwargs": bool(re.search(r"def forward\(self,\s*\*\*kwargs", source)),
                        "explicit_accepts_loss_kwargs": bool(re.search(r"accepts_loss_kwargs\s*=", source)),
                        "uses_trainer": "Trainer(" in source,
                        "uses_peft": "get_peft_model(" in source,
                    }
                else:
                    row["source_at_training_commit"][path] = {"unavailable": True}
        row["runtime_version_status"] = (
            "historically_recorded" if row["recorded_versions"] else "not_recorded_in_available_artifacts"
        )
        wrapper_at_commit = row.get("source_at_training_commit", {}).get(HISTORICAL_SOURCE[family][1], {})
        row["forward_signature_evidence"] = (
            "forward(**kwargs)" if wrapper_at_commit.get("forward_var_kwargs")
            else "PEFT wrapper or source unavailable; requires historical runtime inspection"
        )
        row["loss_kwargs_support_at_runtime"] = "unknown_without_recorded_Trainer_attribute_or_exact_runtime_behavior"
        row["num_items_reaches_loss_denominator"] = "unknown_without_historical_runtime_evidence"
        row["trainer_accelerate_normalization"] = "unknown_without_historical_runtime_evidence"
        row["loss_path_conclusion"] = "evidence_insufficient_until_historical_runtime_and_wrapper_behavior_are_confirmed"
        found[family] = row
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/dataset/pathVQA"))
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(
            device=device,
            dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
        )
        for key, value in batch.items()
    }


def gradients(model: VisualSelectionPrefixModel) -> dict[str, torch.Tensor]:
    result = {}
    for group, parameters in model.trainable_parameter_groups().items():
        pieces = []
        for parameter in parameters:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"missing/nonfinite gradient in {group}")
            pieces.append(parameter.grad.detach().float().cpu().reshape(-1))
        result[group] = torch.cat(pieces)
    result["all"] = torch.cat(list(result.values()))
    return result


def parameter_digest(model: VisualSelectionPrefixModel) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_location(symbol) -> dict[str, object]:
    try:
        return {"file": inspect.getsourcefile(symbol), "line": inspect.getsourcelines(symbol)[1]}
    except (OSError, TypeError):
        return {"file": None, "line": None}


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref_norm = float(reference.norm())
    cand_norm = float(candidate.norm())
    if ref_norm <= 0 or cand_norm <= 0:
        raise RuntimeError("zero gradient prevents normalization comparison")
    return {
        "reference_preclip_norm": ref_norm,
        "candidate_preclip_norm": cand_norm,
        "norm_ratio": cand_norm / ref_norm,
        "cosine": float(F.cosine_similarity(reference, candidate, dim=0)),
        "relative_l2_error": float((candidate - reference).norm()) / ref_norm,
    }


def capture_rng() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return (
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def replay(
    model: VisualSelectionPrefixModel,
    trainer: V1Trainer,
    batches: list[dict[str, torch.Tensor]],
    count: int,
    mode: str,
    rng_state: tuple[torch.Tensor, list[torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], list[float], int | None]:
    restore_rng(rng_state)
    model.zero_grad(set_to_none=True)
    model.train()
    trainer.current_gradient_accumulation_steps = count
    trainer.model_accepts_loss_kwargs = mode != "candidate_fixed"
    prepared, items = trainer.get_batch_samples(iter(batches[:count]), count, trainer.args.device)
    if len(prepared) != count:
        raise RuntimeError("Trainer did not collect the requested accumulation window")
    item_value = None if items is None else int(items)
    losses = []
    for batch in prepared:
        if mode == "manual_equal_microbatch_mean":
            moved = trainer._prepare_inputs(dict(batch))
            with trainer.compute_loss_context_manager():
                loss = model(**moved).loss
            (loss / count).backward()
            losses.append(float(loss.detach()))
        else:
            loss = trainer.training_step(model, dict(batch), num_items_in_batch=items)
            losses.append(float(loss.detach()))
    return gradients(model), losses, item_value


def main() -> int:
    args = parse_args()
    if not (args.checkpoint / "visual_selection_prefix.pt").is_file():
        raise FileNotFoundError("V1 epoch3 checkpoint required; do not audit the active training process")
    run_root = args.checkpoint.parent.parent
    train_report_path = run_root / "train_report.json"
    if not train_report_path.is_file() or not (run_root / "eval_validation/epoch_3/pathvqa_summary.json").is_file():
        raise FileNotFoundError("Completed V1 train report and epoch3 Validation are required before this audit")
    training_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    if training_report.get("git_commit") != "4636416ee99768c667377f0c66b5809daf48ffcd":
        raise RuntimeError(f"Unexpected V1 training commit: {training_report.get('git_commit')}")
    if args.output_dir.resolve() == args.checkpoint.resolve() or args.checkpoint.resolve() in args.output_dir.resolve().parents:
        raise ValueError("audit output must not overwrite or sit inside the checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)

    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    base.config.use_cache = False
    model = VisualSelectionPrefixModel(base, init_seed=44)
    model.load_v1(args.checkpoint)
    dataset = RawQuestionDataset(PathVQADataset(
        processor=processor, data_root=args.data_root, split="train",
        ce_enabled=True, seed=42, deterministic_sampling=True,
        max_length=2048,
    ), processor.tokenizer)
    collator = RawQuestionCollator(processor)
    trainer = V1Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer_state"),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=1e-4, weight_decay=0.0, warmup_ratio=0.03,
            lr_scheduler_type="linear", max_grad_norm=1.0,
            bf16=True, gradient_checkpointing=False,
            dataloader_num_workers=2, remove_unused_columns=False,
            report_to="none", seed=44, data_seed=42,
        ),
        train_dataset=dataset, data_collator=collator,
        processing_class=processor,
    )
    loader = trainer.get_train_dataloader()
    # Iterate the prepared DataLoader's batch sampler, not the image batches.
    # This reproduces its batch size, drop_last, shuffling, and final-window size
    # without decoding the other ~19k training images.
    index_batches = [list(map(int, indices)) for indices in loader.batch_sampler]
    microbatches_per_epoch = len(loader)
    if len(index_batches) != microbatches_per_epoch:
        raise RuntimeError("Prepared DataLoader batch sampler length differs from DataLoader length")
    remainder = microbatches_per_epoch % trainer.args.gradient_accumulation_steps
    if not 0 < remainder < 16:
        raise RuntimeError(f"Expected a non-full final accumulation window, got {remainder}")
    selected = {16: index_batches[:16], remainder: index_batches[-remainder:]}
    if any(len(indices) != trainer.args.per_device_train_batch_size for groups in selected.values() for indices in groups):
        raise RuntimeError("Selected prepared DataLoader microbatch size changed")
    selected_batches = {
        count: [collator([dataset[i] for i in indices]) for indices in groups]
        for count, groups in selected.items()
    }
    batches = selected_batches[16]
    device = next(base.parameters()).device

    # One forward supplies both numbers; no separate or stale logits are used.
    model.eval()
    first = move_batch(batches[0], device)
    expanded, _, _ = model._expand(first)
    original_accepts = bool(trainer.model_accepts_loss_kwargs)
    original_window, original_items = trainer.get_batch_samples(
        iter(batches), 16, trainer.args.device,
    )
    if len(original_window) != 16:
        raise RuntimeError("Trainer full-window collection failed")
    with torch.inference_mode():
        outputs = model(**first)
        shifted = F.pad(expanded["labels"], (0, 1), value=-100)[..., 1:]
        valid = shifted.ne(-100)
        if outputs.logits.shape[:2] != shifted.shape or not bool(valid.any()):
            raise RuntimeError("V1 logits/causal labels do not align or contain no valid targets")
        ce_sum = F.cross_entropy(outputs.logits[valid].float(), shifted[valid], reduction="sum")
        manual_ce = ce_sum / valid.sum()
        direct_loss = outputs.loss.float()
        with_items_loss = model(**{**first, "num_items_in_batch": original_items}).loss.float()
        snippets = []
        for sample_index, row in enumerate(expanded["labels"][:2]):
            positions = row.ne(-100).nonzero(as_tuple=True)[0]
            if positions.numel():
                start = int(positions[0])
                target_ids = row[positions[: min(16, positions.numel())]].tolist()
                snippets.append({
                    "sample_index": sample_index,
                    "first_supervised_position_after_p20": start,
                    "preceding_input_ids": expanded["input_ids"][sample_index, max(0, start - 6):start].tolist(),
                    "first_target_ids": target_ids,
                    "decoded_first_targets": processor.tokenizer.decode(target_ids),
                })
        ce_check = {
            "direct_forward_loss": float(direct_loss),
            "with_window_num_items_loss": float(with_items_loss),
            "without_vs_with_num_items_absolute_difference": float((direct_loss - with_items_loss).abs()),
            "window_num_items_in_batch": int(original_items) if original_items is not None else None,
            "manual_shifted_valid_token_mean_ce": float(manual_ce),
            "manual_shifted_valid_token_ce_sum": float(ce_sum),
            "absolute_difference": float((direct_loss - manual_ce).abs()),
            "relative_difference": float((direct_loss - manual_ce).abs() / manual_ce.abs().clamp_min(1e-8)),
            "valid_tokens": int(valid.sum()),
            "expanded_prefix_labels_ignored": bool(expanded["labels"][:, :20].eq(-100).all()),
            "supervision_snippets": snippets,
        }
    del outputs

    accelerator_steps = int(trainer.accelerator.gradient_accumulation_steps)
    if not original_accepts or accelerator_steps != 1:
        raise RuntimeError(
            f"server Trainer path differs: accepts={original_accepts} "
            f"accelerator_steps={accelerator_steps}"
        )
    rng_state = capture_rng()
    parameters_before = parameter_digest(model)
    windows = {}
    for count, window_batches in selected_batches.items():
        manual, manual_losses, manual_items = replay(
            model, trainer, window_batches, count, "manual_equal_microbatch_mean", rng_state,
        )
        original, original_losses, original_items = replay(
            model, trainer, window_batches, count, "original_trainer_path", rng_state,
        )
        fixed, fixed_losses, fixed_items = replay(
            model, trainer, window_batches, count, "candidate_fixed", rng_state,
        )
        if original_items is None or fixed_items is not None:
            raise RuntimeError(
                f"Trainer num_items path unexpected: original={original_items} candidate={fixed_items}"
            )
        group_comparisons = {
            group: {
                "original_vs_manual": compare(manual[group], original[group]),
                "candidate_vs_manual": compare(manual[group], fixed[group]),
            }
            for group in manual
        }
        def clipped_copy(vector: torch.Tensor) -> torch.Tensor:
            return vector * min(1.0, 1.0 / (float(vector.norm()) + 1e-6))

        windows[str(count)] = {
            "microbatches": count,
            "labels_nonignored_across_window": sum(
                int(batch["labels"].ne(-100).sum()) for batch in window_batches
            ),
            "sample_indices_by_microbatch": selected[count],
            "sampler_source": "Trainer.get_train_dataloader().batch_sampler; fresh equivalent sampler, not historical epoch-3 RNG replay",
            "trainer_num_items_in_batch": {"manual": manual_items, "original": original_items, "candidate": fixed_items},
            "manual_microbatch_losses": manual_losses,
            "original_training_step_returned_losses": original_losses,
            "candidate_training_step_returned_losses": fixed_losses,
            "groups": group_comparisons,
            "original_vs_manual": group_comparisons["all"]["original_vs_manual"],
            "candidate_vs_manual": group_comparisons["all"]["candidate_vs_manual"],
            "clipped_gradient_copy_threshold_1": {
                "manual_norm": float(clipped_copy(manual["all"]).norm()),
                "original_norm": float(clipped_copy(original["all"]).norm()),
                "candidate_norm": float(clipped_copy(fixed["all"]).norm()),
                "original_vs_manual": compare(clipped_copy(manual["all"]), clipped_copy(original["all"])),
                "candidate_vs_manual": compare(clipped_copy(manual["all"]), clipped_copy(fixed["all"])),
            },
            "optimizer_step_executed": False,
            "gradient_clipping_executed": False,
        }
        observed = windows[str(count)]["original_vs_manual"]
        corrected = windows[str(count)]["candidate_vs_manual"]
        windows[str(count)]["numeric_checks"] = {
            "original_matches_missing_accumulation_division": (
                abs(observed["norm_ratio"] - count) / count < 0.02
                and observed["cosine"] > 0.999
            ),
            "candidate_matches_equal_microbatch_mean": (
                abs(corrected["norm_ratio"] - 1.0) < 0.02
                and corrected["cosine"] > 0.999
                and corrected["relative_l2_error"] < 0.02
            ),
        }
        print(f"[V1_LOSS_WINDOW_{count}] " + json.dumps(windows[str(count)], ensure_ascii=False))
    model.zero_grad(set_to_none=True)
    parameters_after = parameter_digest(model)
    if parameters_before != parameters_after:
        raise RuntimeError("Trainable parameter tensors changed during read-only gradient audit")
    trainer_loop_source = inspect.getsource(Trainer._inner_training_loop)
    clip_position = trainer_loop_source.find("self.accelerator.clip_grad_norm_(")
    callback_position = trainer_loop_source.find("on_pre_optimizer_step")
    optimizer_position = trainer_loop_source.find("self.optimizer.step()")
    order_verified = 0 <= clip_position < callback_position < optimizer_position
    if not order_verified:
        raise RuntimeError("Installed Trainer clipping/callback/optimizer order differs from audited path")
    base_forward_source = inspect.getsource(type(base).forward)
    source_path = {
        "v1_wrapper_forward": source_location(VisualSelectionPrefixModel.forward),
        "base_forward": source_location(type(base).forward),
        "base_loss_function": source_location(base.loss_function),
        "trainer_init": source_location(Trainer.__init__),
        "trainer_get_batch_samples": source_location(Trainer.get_batch_samples),
        "trainer_compute_loss": source_location(Trainer.compute_loss),
        "trainer_training_step": source_location(Trainer.training_step),
        "trainer_training_loop": source_location(Trainer._inner_training_loop),
        "accelerator_backward": source_location(Accelerator.backward),
    }
    peft_reference = {"scope": "diagnostic_runtime_only_not_historical_evidence"}
    try:
        from peft import PeftModel
        peft_reference.update({
            "version": importlib.metadata.version("peft"),
            "forward_signature": str(inspect.signature(PeftModel.forward)),
            "class_accepts_loss_kwargs": getattr(PeftModel, "accepts_loss_kwargs", None),
            "forward_source": source_location(PeftModel.forward),
        })
    except (ImportError, importlib.metadata.PackageNotFoundError):
        peft_reference["status"] = "not_installed_in_diagnostic_runtime"
    report = {
        "method": "v1_loss_scaling_read_only_audit",
        "checkpoint": str(args.checkpoint),
        "training_run_root": str(run_root),
        "training_commit": training_report["git_commit"],
        "training_report_evidence": str(train_report_path),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "versions": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
        },
        "source_locations": source_path,
        "peft_reference": peft_reference,
        "loss_path": {
            "base_forward_passes_num_items_to_loss_function": "num_items_in_batch" in base_forward_source[base_forward_source.find("self.loss_function("):base_forward_source.find("self.loss_function(") + 250],
            "observed_loss_reduction": "mean_over_nonignored_shifted_target_tokens",
            "observed_loss_denominator": ce_check["valid_tokens"],
            "clip_callback_optimizer_order_verified": order_verified,
            "clip_then_callback_then_optimizer": True,
            "v1_diagnostic_callback_observes": "postclip_gradients",
            "hf_logged_grad_norm_observes": "preclip_norm_returned_by_clip_grad_norm",
        },
        "trainer_model_accepts_loss_kwargs": original_accepts,
        "trainer_gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
        "accelerator_gradient_accumulation_steps": accelerator_steps,
        "prepared_train_dataloader": {
            "dataset_size": len(dataset), "microbatches_per_epoch": microbatches_per_epoch,
            "final_accumulation_microbatches": remainder,
            "batch_sampler": type(loader.batch_sampler).__qualname__,
            "drop_last": trainer.args.dataloader_drop_last,
            "per_device_train_batch_size": trainer.args.per_device_train_batch_size,
            "dataloader_num_workers": trainer.args.dataloader_num_workers,
        },
        "same_logits_ce": ce_check,
        "same_logits_ce_matches": ce_check["relative_difference"] < 1e-4,
        "num_items_is_ignored_by_model_loss": ce_check["without_vs_with_num_items_absolute_difference"] < 1e-3,
        "windows": windows,
        "historical_provenance": historical_provenance(args.checkpoint.parents[3]),
        "trainable_parameter_sha256_before": parameters_before,
        "trainable_parameter_sha256_after": parameters_after,
        "train_parameters_modified": parameters_before != parameters_after,
        "optimizer_step_executed": False,
    }
    passed = report["same_logits_ce_matches"] and report["num_items_is_ignored_by_model_loss"] and all(
        all(row["numeric_checks"].values()) for row in windows.values()
    )
    report["passed"] = passed
    with (args.output_dir / "v1_loss_scaling_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    lines = [
        "# V1 loss/gradient audit (read-only)", "",
        f"- Training commit: `{report['training_commit']}`; audit commit: `{report['git_commit']}`.",
        f"- Checkpoint: `{args.checkpoint}`; no optimizer/scheduler step or checkpoint write.",
        f"- Runtime: PyTorch {torch.__version__}, Transformers {report['versions']['transformers']}, Accelerate {report['versions']['accelerate']}.",
        f"- Same-logits CE: valid tokens {ce_check['valid_tokens']}, sum {ce_check['manual_shifted_valid_token_ce_sum']:.8f}, mean {ce_check['manual_shifted_valid_token_mean_ce']:.8f}, outputs.loss {ce_check['direct_forward_loss']:.8f}, relative error {ce_check['relative_difference']:.3g}.",
        f"- With/without num_items loss absolute difference: {ce_check['without_vs_with_num_items_absolute_difference']:.8g}; num_items={ce_check['window_num_items_in_batch']}.",
        f"- Trainer accepts loss kwargs={original_accepts}; Trainer accumulation={trainer.args.gradient_accumulation_steps}; Accelerate accumulation={accelerator_steps}.",
        "- Each path restores the same CPU/CUDA RNG state and zeroes gradients. Candidate pass criteria allow 2% norm/relative-L2 error and require cosine >0.999 to accommodate BF16 rounding; no expected ratio is substituted for a measurement.",
        "- The Trainer loop clips before on_pre_optimizer_step callback, then optimizer.step; this audit intercepts before clipping and never executes those operations.",
        "", "| K | Current norm/reference | Current cosine | Fixed norm/reference | Fixed cosine | Current num_items | Fixed num_items |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in (16, remainder):
        row = windows[str(count)]
        old = row["original_vs_manual"]
        new = row["candidate_vs_manual"]
        lines.append(
            f"| {count} | {old['norm_ratio']:.5f} | {old['cosine']:.6f} | {new['norm_ratio']:.5f} | {new['cosine']:.6f} | {row['trainer_num_items_in_batch']['original']} | {row['trainer_num_items_in_batch']['candidate']} |"
        )
    lines += ["", f"Numeric checks passed: **{passed}**. Full per-group norms, relative L2 errors, clipped-gradient-copy comparison, sample indices, and source locations are in the JSON.", "", "## Historical seed44 provenance", "", "| Method | Experiment | Commit | Recorded runtime versions | Forward | Loss kwargs / num_items | Trainer / Accelerate | Conclusion |", "|---|---|---|---|---|---|---|---|"]
    for family, row in report["historical_provenance"].items():
        versions = ", ".join(f"{k}={v}" for k, v in row["recorded_versions"].items()) or "unknown"
        lines.append(f"| {family} | {row['experiment']} | {row['commit'] or 'unknown'} | {versions} | {row['forward_signature_evidence']} | {row['loss_kwargs_support_at_runtime']}; {row['num_items_reaches_loss_denominator']} | {row['trainer_accelerate_normalization']} | {row['loss_path_conclusion']} |")
    lines += ["", "Historical versions are reported only when recorded by that run's artifacts; current diagnostic versions are not retroactively assigned to old runs."]
    (args.output_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[V1_LOSS_SCALING_AUDIT] " + json.dumps({
        "output": str(args.output_dir), "ce": ce_check,
        "original_16_ratio": windows["16"]["original_vs_manual"]["norm_ratio"],
        "fixed_16_ratio": windows["16"]["candidate_vs_manual"]["norm_ratio"],
        "original_3_ratio": windows["3"]["original_vs_manual"]["norm_ratio"],
        "fixed_3_ratio": windows["3"]["candidate_vs_manual"]["norm_ratio"],
        "passed": passed,
    }, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
