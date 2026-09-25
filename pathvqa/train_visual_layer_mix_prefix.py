#!/usr/bin/env python3
"""Train one normalized V2 seed44 run after V1-equivalence and real-batch gates."""

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
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, TrainerCallback, TrainingArguments,
)

from pathvqa.data_pipeline import PathVQADataset, PathVQAParquetStore
from pathvqa.pathvqa_official_eval import build_prompt
from pathvqa.train_visual_selection_prefix import (
    RawQuestionCollator, RawQuestionDataset, V1Trainer,
)
from slake.visual_selection_layer_mix import EXPECTED_TRAINABLE, VisualSelectionLayerMixModel
from slake.visual_selection_prefix import VisualSelectionPrefixModel
from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface


def _rng_state():
    return (torch.random.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state) -> None:
    torch.random.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def _trainable(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def construct_with_v1_audit(base, output_dir: Path):
    """V1 and V2 must share every seed44 trainable tensor at step zero."""
    state = _rng_state()
    reference = VisualSelectionPrefixModel(base, init_seed=44)
    reference_block = base.model.visual.blocks[17]
    expected = {name: parameter.detach().cpu().clone()
                for name, parameter in _trainable(reference).items()}
    base.model.visual.blocks[17] = reference_block.block
    _restore_rng(state)
    model = VisualSelectionLayerMixModel(base, init_seed=44)
    actual = _trainable(model)
    if set(actual) != set(expected) | {"alpha"}:
        raise RuntimeError(f"V2 trainable tensor set differs from V1+alpha: {set(actual) ^ (set(expected) | {'alpha'})}")
    mismatches = [name for name in expected
                  if not torch.equal(expected[name], actual[name].detach().cpu())]
    if mismatches:
        raise RuntimeError(f"V2 common initialization differs from V1: {mismatches}")
    if not torch.equal(model.alpha.detach(), torch.ones_like(model.alpha)):
        raise RuntimeError("V2 alpha initial values are not all one")
    audit = {"reference": "fresh_normalized_V1_seed44", "shared_tensor_count": len(expected),
             "shared_equal": True, "alpha_all_one": True,
             "trainable_groups": model._audit_parameters()}
    with (output_dir / "v2_initialization_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print("[V2_INITIALIZATION_AUDIT] " + json.dumps(audit), flush=True)
    return reference, reference_block, model


def formula_and_serialization_preflight(model, output_dir: Path):
    state = _rng_state()
    condition = torch.randn(2, 192, device=model.p20.device, dtype=torch.float32)
    with torch.no_grad():
        v1 = model.prefix_output(torch.relu(condition))
        v2 = model._prefix_shift(condition)
        all_one_error = float((v2 - v1[:, None, :]).abs().max())
        if not torch.allclose(v2, v1[:, None, :].expand_as(v2), atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"V2 all-one output differs from fused V1: {all_one_error}")
        components = model.layer_components(condition)
        model.alpha[0, 0] = 1.25
        model.alpha[1, 2] = 0.75
        changed = model._prefix_shift(condition)
        explicit = (model.alpha[None, :, :, None] * components[:, None, :, :]).sum(dim=2)
        explicit = explicit + model.prefix_output.bias[None, None, :]
        explicit_error = float((changed - explicit).abs().max())
        if not torch.allclose(changed, explicit, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"V2 alpha mixture/bias differs from explicit formula: {explicit_error}")
        if not bool((changed[:, 0] - changed[:, 1]).abs().max() > 1e-6):
            raise RuntimeError("V2 nonuniform alpha did not distinguish P20 positions")
        model.alpha.fill_(1.0)
        checkpoint = output_dir / "preflight_v2_roundtrip"
        model.save_v2(checkpoint)
        model.alpha[0, 0] = 2.0
        model.load_v2(checkpoint)
        if not torch.equal(model.alpha, torch.ones_like(model.alpha)):
            raise RuntimeError("V2 save/reload lost alpha")
    _restore_rng(state)
    report = {"alpha_one_v1_max_abs_error": all_one_error,
              "nonuniform_explicit_formula_max_abs_error": explicit_error,
              "equivalence_atol": 2e-5, "equivalence_rtol": 2e-5,
              "nonuniform_positions_differ": True, "save_reload_alpha_equal": True,
              "output_bias_count": 1}
    with (output_dir / "v2_formula_preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[V2_FORMULA_PREFLIGHT] " + json.dumps(report), flush=True)


def _greedy(model, processor, image, question: str):
    interface = VisualSelectionPrefixInterface.__new__(VisualSelectionPrefixInterface)
    interface.processor = processor
    inputs = interface.prepare_inputs(image, build_prompt(question, None), question=question)
    length = int(inputs["input_ids"].shape[1]) + 20
    device = next(model.base_model.parameters()).device
    moved = {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point()
                           else value.dtype) for key, value in inputs.items()}
    before = model.diagnostic_prefill_calls
    with torch.inference_mode():
        output = model.generate(**moved, max_new_tokens=32, do_sample=False, use_cache=True)
    if model.diagnostic_prefill_calls - before != 1:
        raise RuntimeError("V2/V1 KV-cache generation repeated or skipped prefill injection")
    return processor.batch_decode(output[:, length:], skip_special_tokens=True)[0].strip()


def greedy_equivalence_preflight(reference, reference_block, model, processor,
                                 data_root: Path, cache_dir: Path, output_dir: Path):
    """Compare the same fresh shared tensors on fixed Validation examples."""
    store = PathVQAParquetStore(data_root, "validation", cache_dir=cache_dir)
    selected = [dict(store.samples[index]) for index in (0, 1, 2)]
    visual = model.base_model.model.visual
    v2_block = visual.blocks[17]
    state = _rng_state()
    old_cache = model.base_model.config.use_cache
    model.base_model.config.use_cache = True
    reference.eval()
    model.eval()
    rows = []
    try:
        for row in selected:
            image = store.load_image(row)
            try:
                visual.blocks[17] = reference_block
                v1_text = _greedy(reference, processor, image, str(row["question"]))
                visual.blocks[17] = v2_block
                v2_text = _greedy(model, processor, image, str(row["question"]))
                if not rows:
                    roundtrip = output_dir / "preflight_v2_roundtrip"
                    model.save_v2(roundtrip)
                    with torch.no_grad():
                        model.alpha[0, 0] = 2.0
                    model.load_v2(roundtrip)
                    reloaded_text = _greedy(model, processor, image, str(row["question"]))
                    if reloaded_text != v2_text:
                        raise RuntimeError("V2 save/reload changed greedy generation")
            finally:
                image.close()
                visual.blocks[17] = v2_block
            rows.append({"question_id": row["question_id"],
                         "v1_prediction": v1_text, "v2_prediction": v2_text})
            if v1_text != v2_text:
                raise RuntimeError(f"V2 all-one greedy prediction differs from V1 at {row['question_id']}")
    finally:
        visual.blocks[17] = v2_block
        model.base_model.config.use_cache = old_cache
        _restore_rng(state)
    with (output_dir / "v2_greedy_equivalence.json").open("w", encoding="utf-8") as handle:
        json.dump({"source": "fresh_same_seed_V1_and_V2", "rows": rows,
                   "all_match": True, "roundtrip_generation_match": True,
                   "single_prefill_injection_per_generation": True}, handle, indent=2)
    print(f"[V2_GREEDY_PREFLIGHT] matched={len(rows)}/{len(selected)}", flush=True)
    model.train()


def real_batch_preflight(model, dataset, collator, output_dir: Path):
    batch = collator([dataset[0], dataset[1]])
    device = next(model.base_model.parameters()).device
    moved = {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point()
                                  else value.dtype) for key, value in batch.items()}
    model.train()
    output = model(**moved)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise RuntimeError("V2 real batch loss missing/nonfinite")
    output.loss.backward()
    groups = model.trainable_parameter_groups()
    grad_norms = {}
    for name, parameters in groups.items():
        grads = [parameter.grad for parameter in parameters]
        if len(grads) != len(parameters) or any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError(f"V2 real batch lacks finite gradients in {name}")
        norm = math.sqrt(sum(float(g.float().square().sum()) for g in grads))
        if norm <= 0:
            raise RuntimeError(f"V2 real batch has zero {name} gradients")
        grad_norms[name] = norm
    required = (
        "text_projection.weight", "question_depthwise.weight", "question_pointwise.weight",
        "question_pool.weight", "layer_gate.weight", "prefix_output.weight", "alpha",
    ) + tuple(
        f"{prefix}.{index}.weight"
        for prefix in ("query_heads", "key_heads", "value_blocks") for index in range(3)
    )
    inactive = [name for name in required if model.first_backward_gradients.get(name, 0.0) <= 0.0]
    if inactive:
        raise RuntimeError(f"V2 real batch inactive parameters: {inactive}")
    if not model.last_injection_audit or not model.last_injection_audit["native_embeddings_unchanged"]:
        raise RuntimeError("V2 preflight failed native embedding invariance")
    if model.last_injection_audit["positions"] != 20:
        raise RuntimeError("V2 preflight did not limit injection to P20")
    report = {"loss": float(output.loss.detach()), "parameter_groups": model._audit_parameters(),
              "group_grad_norms": grad_norms,
              "first_parameter_grad_norms": model.first_backward_gradients,
              "forward": model.first_batch_diagnostics,
              "injection": model.last_injection_audit,
              "question_policy": "independent_raw_question_ids_no_prefill_write_mapping"}
    with (output_dir / "v2_real_batch_preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[V2_REAL_BATCH_PREFLIGHT] " + json.dumps(report, ensure_ascii=False), flush=True)
    model.zero_grad(set_to_none=True)


class V2Trainer(V1Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        rates = {
            "p20": 0.3, "visual_s8": 3e-5, "visual_av10": 1e-4,
            "question_context": 1e-4, "maps": 1e-4,
            "layer_condition": 1e-4, "prefix_output": 1e-4, "alpha": 1e-4,
        }
        groups = self.model.trainable_parameter_groups()
        self.model._audit_parameters()
        if set(groups) != set(rates):
            raise RuntimeError("V2 optimizer rates do not match trainable parameter groups")
        self.optimizer = torch.optim.AdamW([
            {"params": groups[name], "lr": rate, "weight_decay": 0.0, "group_name": name}
            for name, rate in rates.items()
        ], betas=(0.9, 0.999), eps=1e-8)
        print(f"[V2_OPTIMIZER] rates={json.dumps(rates)} warmup_ratio=0.03 scheduler=linear")
        return self.optimizer


class V2AuditCallback(TrainerCallback):
    def __init__(self, output_dir: Path, processor) -> None:
        self.output_dir = output_dir
        self.processor = processor
        self.step_path = output_dir / "v2_diagnostics.jsonl"

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        groups = model.trainable_parameter_groups()
        row = {"step": int(state.global_step), "epoch": float(state.epoch or 0)}
        for name, parameters in groups.items():
            grads = [parameter.grad.detach().float() for parameter in parameters
                     if parameter.grad is not None]
            if len(grads) != len(parameters) or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError(f"V2 missing/nonfinite gradient in {name}")
            row[f"{name}_grad_norm"] = math.sqrt(sum(float(g.square().sum()) for g in grads))
        row.update({key: float(value.float()) for key, value in model.debug_context.items()})
        alpha = model.alpha.detach().float()
        row["alpha_columns_position_std"] = alpha.std(dim=0, unbiased=False).tolist()
        row["alpha_max_abs"] = float(alpha.abs().max())
        row["alpha_values"] = alpha.tolist()
        if int(state.global_step) % 20 == 0:
            with self.step_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("[V2_DIAGNOSTICS] " + json.dumps(row, ensure_ascii=False), flush=True)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and int(round(float(state.epoch or 0))) == 3:
            checkpoint = self.output_dir / "checkpoints" / "epoch_3"
            kwargs["model"].save_v2(checkpoint)
            self.processor.save_pretrained(checkpoint)
            print(f"[V2_EPOCH3_CHECKPOINT] {checkpoint}", flush=True)
        return control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed, data_seed = 44, 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    versions = {"torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "accelerate": importlib.metadata.version("accelerate")}
    print("[V2_RUNTIME] " + json.dumps({"experiment": args.experiment_name,
                                     "git_commit": commit, "versions": versions,
                                     "model_seed": seed, "data_seed": data_seed}, sort_keys=True), flush=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    base.config.use_cache = False
    reference, reference_block, model = construct_with_v1_audit(base, args.output_dir)
    counts = model._audit_parameters()
    if sum(counts.values()) != EXPECTED_TRAINABLE:
        raise RuntimeError("V2 actual parameter total changed")
    preflight_rng = _rng_state()
    formula_and_serialization_preflight(model, args.output_dir)
    greedy_equivalence_preflight(reference, reference_block, model, processor,
                                 args.data_root, args.cache_dir, args.output_dir)
    _restore_rng(preflight_rng)
    dataset = RawQuestionDataset(PathVQADataset(
        processor=processor, data_root=args.data_root, split="train",
        ce_enabled=True, seed=data_seed, deterministic_sampling=True,
        max_length=2048,
    ), processor.tokenizer)
    collator = RawQuestionCollator(processor)
    preflight_rng = _rng_state()
    real_batch_preflight(model, dataset, collator, args.output_dir)
    _restore_rng(preflight_rng)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = V2Trainer(
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
        callbacks=[V2AuditCallback(args.output_dir, processor)],
    )
    trainer.model_accepts_loss_kwargs = False
    if (trainer.model_accepts_loss_kwargs is not False
        or trainer.args.gradient_accumulation_steps != 16
        or trainer.accelerator.gradient_accumulation_steps != 1):
        raise RuntimeError("V2 normalized accumulation differs from audited V1 path")
    print("[V2_LOSS_ACCUMULATION] trainer_model_accepts_loss_kwargs=False "
          "trainer_steps=16 accelerate_steps=1", flush=True)
    result = trainer.train()
    checkpoint = args.output_dir / "checkpoints" / "epoch_3"
    if not (checkpoint / "visual_selection_layer_mix.pt").is_file():
        raise RuntimeError("V2 epoch3 checkpoint missing")
    report = {
        "experiment": args.experiment_name, "method": "visual_selection_layer_mix_prefix_p20_v2",
        "dataset": "PathVQA", "model_seed": seed, "data_seed": data_seed,
        "epochs": 3, "parameter_groups": counts, "total_trainable_parameters": sum(counts.values()),
        "train_metrics": result.metrics,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "git_commit": commit, "runtime_versions": versions,
        "trainer_model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
        "accelerator_gradient_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
        "loss_accumulation_protocol": "equal_microbatch_mean_trainer_normalized",
        "optimizer": {"type": "AdamW", "weight_decay": 0.0, "betas": [0.9, 0.999],
                      "eps": 1e-8, "scheduler": "linear", "warmup_ratio": 0.03,
                      "max_grad_norm": 1.0, "per_device_batch_size": 2,
                      "gradient_accumulation_steps": 16},
    }
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[V2_TRAIN_DONE] checkpoint={checkpoint} runtime={result.metrics.get('train_runtime')} "
          f"peak_gpu_bytes={report['peak_gpu_memory_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
