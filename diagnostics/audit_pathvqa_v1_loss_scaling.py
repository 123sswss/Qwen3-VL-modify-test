#!/usr/bin/env python3
"""Read-only V1 loss/accumulation audit; run only after the active training ends.

The audit loads a saved V1 checkpoint, replays fixed training microbatches,
never calls optimizer.step(), and writes only to its independent output dir.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments

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


def historical_provenance(output_root: Path) -> dict[str, dict]:
    """Collect only versions explicitly recorded by each historical run."""
    found = {}
    version_pattern = re.compile(r"\b(torch|transformers|accelerate|peft)\s*(?:==|=|:)\s*([0-9][^\s,;]*)", re.I)
    for family, relative in HISTORICAL_RUNS.items():
        root = output_root / relative
        row = {"output": str(root), "exists": root.is_dir(), "commit": None, "recorded_versions": {}}
        report = root / "train_report.json"
        if report.is_file():
            try:
                metadata = json.loads(report.read_text(encoding="utf-8"))
                row["commit"] = metadata.get("git_commit")
                for name in ("torch", "transformers", "accelerate", "peft"):
                    for key in (f"{name}_version", name):
                        value = metadata.get(key)
                        if isinstance(value, str):
                            row["recorded_versions"][name] = value
                for key in ("versions", "library_versions"):
                    values = metadata.get(key)
                    if isinstance(values, dict):
                        row["recorded_versions"].update({
                            name: str(values[name]) for name in ("torch", "transformers", "accelerate", "peft")
                            if name in values
                        })
            except (OSError, ValueError) as exc:
                row["report_read_error"] = str(exc)
        log = root / "train.log"
        if log.is_file():
            with log.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    for match in version_pattern.finditer(line):
                        row["recorded_versions"].setdefault(match.group(1).lower(), match.group(2))
        row["runtime_version_status"] = (
            "historically_recorded" if row["recorded_versions"] else "not_recorded_in_available_artifacts"
        )
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


def gradients(model: VisualSelectionPrefixModel) -> torch.Tensor:
    pieces = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"missing/nonfinite gradient in {name}")
        pieces.append(parameter.grad.detach().float().cpu().reshape(-1))
    return torch.cat(pieces)


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
) -> tuple[torch.Tensor, list[float]]:
    restore_rng(rng_state)
    model.zero_grad(set_to_none=True)
    model.train()
    trainer.current_gradient_accumulation_steps = count
    trainer.model_accepts_loss_kwargs = mode != "candidate_fixed"
    token_count = sum(int(batch["labels"].ne(-100).sum()) for batch in batches[:count])
    items = torch.tensor(token_count, device=trainer.args.device)
    losses = []
    for batch in batches[:count]:
        if mode == "manual_equal_microbatch_mean":
            moved = trainer._prepare_inputs(dict(batch))
            with trainer.compute_loss_context_manager():
                loss = model(**moved).loss
            (loss / count).backward()
            losses.append(float(loss.detach()))
        else:
            loss = trainer.training_step(
                model, dict(batch), num_items_in_batch=items,
            )
            losses.append(float(loss.detach()))
    return gradients(model), losses


def main() -> int:
    args = parse_args()
    if not (args.checkpoint / "visual_selection_prefix.pt").is_file():
        raise FileNotFoundError("V1 epoch3 checkpoint required; do not audit the active training process")
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
    batches = [collator([dataset[i], dataset[i + 1]]) for i in range(0, 32, 2)]
    microbatches_per_epoch = (len(dataset) + 1) // 2
    remainder = microbatches_per_epoch % 16
    if remainder != 3 or len(dataset) % 2:
        raise RuntimeError(
            "PathVQA tail-window geometry changed; inspect before replaying "
            f"microbatches_per_epoch={microbatches_per_epoch} remainder={remainder}"
        )
    tail_start = len(dataset) - 2 * remainder
    tail_batches = [
        collator([dataset[i], dataset[i + 1]])
        for i in range(tail_start, len(dataset), 2)
    ]
    device = next(base.parameters()).device

    # One forward supplies both numbers; no separate or stale logits are used.
    model.eval()
    first = move_batch(batches[0], device)
    expanded, _, _ = model._expand(first)
    with torch.inference_mode():
        outputs = model(**first)
        shifted = F.pad(expanded["labels"], (0, 1), value=-100)[..., 1:]
        valid = shifted.ne(-100)
        if outputs.logits.shape[:2] != shifted.shape or not bool(valid.any()):
            raise RuntimeError("V1 logits/causal labels do not align or contain no valid targets")
        manual_ce = F.cross_entropy(
            outputs.logits[valid].float(), shifted[valid], reduction="mean",
        )
        direct_loss = outputs.loss.float()
        ce_check = {
            "direct_forward_loss": float(direct_loss),
            "manual_shifted_valid_token_mean_ce": float(manual_ce),
            "absolute_difference": float((direct_loss - manual_ce).abs()),
            "valid_tokens": int(valid.sum()),
            "expanded_prefix_labels_ignored": bool(expanded["labels"][:, :20].eq(-100).all()),
        }
    del outputs

    trainer = V1Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "trainer_state"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=1e-4, weight_decay=0.0, warmup_ratio=0.03,
            lr_scheduler_type="linear", max_grad_norm=1.0,
            bf16=True, gradient_checkpointing=False,
            remove_unused_columns=False, report_to="none",
            seed=44, data_seed=42,
        ),
        processing_class=processor,
    )
    original_accepts = bool(trainer.model_accepts_loss_kwargs)
    accelerator_steps = int(trainer.accelerator.gradient_accumulation_steps)
    if not original_accepts or accelerator_steps != 1:
        raise RuntimeError(
            f"server Trainer path differs: accepts={original_accepts} "
            f"accelerator_steps={accelerator_steps}"
        )
    rng_state = capture_rng()
    windows = {}
    for count, selected_batches in ((16, batches), (3, tail_batches)):
        manual, manual_losses = replay(
            model, trainer, selected_batches, count, "manual_equal_microbatch_mean", rng_state,
        )
        original, original_losses = replay(
            model, trainer, selected_batches, count, "original_trainer_path", rng_state,
        )
        fixed, fixed_losses = replay(
            model, trainer, selected_batches, count, "candidate_fixed", rng_state,
        )
        windows[str(count)] = {
            "microbatches": count,
            "labels_nonignored_across_window": sum(
                int(batch["labels"].ne(-100).sum()) for batch in selected_batches
            ),
            "sample_indices": (
                list(range(32)) if count == 16
                else list(range(tail_start, len(dataset)))
            ),
            "tail_indices_are_dataset_order_not_trainer_shuffle_order": count == 3,
            "manual_microbatch_losses": manual_losses,
            "original_training_step_returned_losses": original_losses,
            "candidate_training_step_returned_losses": fixed_losses,
            "original_vs_manual": compare(manual, original),
            "candidate_vs_manual": compare(manual, fixed),
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
    report = {
        "method": "v1_loss_scaling_read_only_audit",
        "checkpoint": str(args.checkpoint),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "versions": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
        },
        "trainer_model_accepts_loss_kwargs": original_accepts,
        "accelerator_gradient_accumulation_steps": accelerator_steps,
        "same_logits_ce": ce_check,
        "same_logits_ce_matches": ce_check["absolute_difference"] < 1e-3,
        "windows": windows,
        "historical_provenance": historical_provenance(args.checkpoint.parents[3]),
        "train_parameters_modified": False,
        "optimizer_step_executed": False,
    }
    with (args.output_dir / "v1_loss_scaling_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    passed = report["same_logits_ce_matches"] and all(
        all(row["numeric_checks"].values()) for row in windows.values()
    )
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
