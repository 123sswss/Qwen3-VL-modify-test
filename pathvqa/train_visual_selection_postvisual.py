#!/usr/bin/env python3
"""One normalized V3 seed44 train, gated by positional and real-batch audits."""

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
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback, TrainingArguments

from pathvqa.data_pipeline import PathVQADataset, PathVQAParquetStore
from pathvqa.pathvqa_official_eval import build_prompt
from pathvqa.train_visual_selection_prefix import RawQuestionCollator, RawQuestionDataset, V1Trainer
from processingWithMMRL import Qwen3ProcessorWithV3
from slake.visual_selection_postvisual import EXPECTED_TRAINABLE, PROMPT_LENGTH, VisualSelectionPostvisualModel
from slake.visual_selection_prefix import VisualSelectionPrefixModel
from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface


class V3PathVQADataset(PathVQADataset):
    """Apply V1's original 2048-token limit before counting reserved P20."""

    def __getitem__(self, index):
        row = super().__getitem__(index)
        if "prompt_mask" not in row or int(row["prompt_mask"].sum()) != PROMPT_LENGTH:
            raise RuntimeError(f"V3 processor did not reserve 20 slots in dataset row {index}")
        if row["input_ids"].numel() - PROMPT_LENGTH > 2048:
            raise ValueError(f"V3 native PathVQA sequence exceeds V1's 2048-token limit at row {index}")
        return row


def _rng_state():
    return torch.random.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def _restore_rng(state) -> None:
    torch.random.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def construct_with_v1_initialization_audit(base, output_dir: Path):
    state = _rng_state()
    reference = VisualSelectionPrefixModel(base, init_seed=44)
    expected = {name: p.detach().cpu().clone() for name, p in reference.named_parameters() if p.requires_grad}
    base.model.visual.blocks[17] = base.model.visual.blocks[17].block
    _restore_rng(state)
    model = VisualSelectionPostvisualModel(base, init_seed=44)
    actual = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(expected) != set(actual):
        raise RuntimeError(f"V3/V1 trainable parameter names differ: {set(expected) ^ set(actual)}")
    mismatched = [name for name in expected if not torch.equal(expected[name], actual[name].detach().cpu())]
    if mismatched:
        raise RuntimeError(f"V3 does not share fresh V1 seed44 initialization: {mismatched}")
    with torch.no_grad():
        condition = torch.linspace(-0.5, 0.5, 192, device=model.p20.device)[None, :]
        if not torch.equal(reference._prefix_shift(condition), model._prefix_shift(condition)):
            raise RuntimeError("V3 changed the shared V1 conditional output computation")
    report = {"reference": "fresh_normalized_V1_seed44", "shared_equal": True,
              "shared_tensor_count": len(actual), "condition_output_equal": True,
              "groups": model._audit_parameters()}
    with (output_dir / "v3_initialization_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[V3_INITIALIZATION_AUDIT] " + json.dumps(report), flush=True)
    return model


def _select_diverse_batch(dataset, collator):
    first = dataset[0]
    first_grid = tuple(int(x) for x in first["image_grid_thw"].reshape(-1).tolist())
    first_question_length = int(first["question_source_ids"].numel())
    selected = None
    for index in range(1, min(len(dataset), 128)):
        candidate = dataset[index]
        grid = tuple(int(x) for x in candidate["image_grid_thw"].reshape(-1).tolist())
        if grid != first_grid and int(candidate["question_source_ids"].numel()) != first_question_length:
            selected = (index, candidate)
            break
    if selected is None:
        raise RuntimeError("V3 could not find two real training examples with different grid and question length")
    return (0, selected[0]), collator([first, selected[1]])


def _check_sequence_reorder(native, expanded):
    ids, attention, labels = native["input_ids"], native["attention_mask"], native["labels"]
    prompt_mask = expanded["prompt_mask"].bool()
    if ids.shape[0] != expanded["input_ids"].shape[0]:
        raise RuntimeError("V3 native/expanded batch size differs")
    for key in ("image_grid_thw", "pixel_values"):
        if not torch.equal(native[key], expanded[key]):
            raise RuntimeError(f"V3 processor changed native {key}")
    positions = torch.nonzero(prompt_mask, as_tuple=False)[:, 1].reshape(ids.shape[0], PROMPT_LENGTH)
    for row in range(ids.shape[0]):
        pos = int(positions[row, 0])
        # Compare only active native tokens: the two collators may pad in
        # different places after inserting the 20 slots.
        native_len = int(attention[row].sum())
        original_ids = ids[row, :native_len]
        original_labels = labels[row, :native_len]
        original_attention = attention[row, :native_len]
        new_len = native_len + PROMPT_LENGTH
        for name, old, new in (
            ("input_ids", original_ids, expanded["input_ids"][row, :new_len]),
            ("attention_mask", original_attention, expanded["attention_mask"][row, :new_len]),
            ("labels", original_labels, expanded["labels"][row, :new_len]),
        ):
            if not torch.equal(old, torch.cat((new[:pos], new[pos + PROMPT_LENGTH:]))):
                raise RuntimeError(f"V3 changed native {name} order in row {row}")
        if not bool(expanded["labels"][row, pos:pos + PROMPT_LENGTH].eq(-100).all()):
            raise RuntimeError(f"V3 introduced answer supervision in prompt row {row}")
        if not bool(expanded["attention_mask"][row, pos:pos + PROMPT_LENGTH].eq(1).all()):
            raise RuntimeError(f"V3 prompt attention is masked in row {row}")
        if not torch.equal(torch.nonzero(prompt_mask[row], as_tuple=True)[0], positions[row]):
            raise RuntimeError(f"V3 prompt mask/position mismatch in row {row}")
    return True


def real_batch_preflight(model, dataset, collator, native_dataset, native_collator,
                         processor, output_dir: Path) -> None:
    indices, batch = _select_diverse_batch(dataset, collator)
    native_batch = native_collator([native_dataset[index] for index in indices])
    _check_sequence_reorder(native_batch, batch)
    device = next(model.base_model.parameters()).device
    moved = {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point()
                                  else value.dtype) for key, value in batch.items()}
    prepared, _, _, positions, prompt_mask = model._prepare(moved)
    tokenizer = processor.tokenizer
    boundaries = []
    for row in range(moved["input_ids"].shape[0]):
        prompt_start = int(positions[row, 0])
        vision_end = prompt_start - 1
        vision_start = int(torch.nonzero(moved["input_ids"][row].eq(model.config.vision_start_token_id),
                                         as_tuple=True)[0].item())
        boundary = {"row": row, "vision_start": vision_start,
                    "image_tokens": int(moved["image_grid_thw"][row].prod()) // 4,
                    "vision_end": vision_end, "prompt_start": prompt_start,
                    "prompt_end_exclusive": prompt_start + PROMPT_LENGTH}
        first_supervised = torch.nonzero(prepared["labels"][row].ne(-100), as_tuple=True)[0]
        if first_supervised.numel() == 0 or int(first_supervised[0]) <= boundary["prompt_end_exclusive"]:
            raise RuntimeError("V3 answer supervision is not after image, P20 and question")
        original_text = tokenizer.decode(moved["input_ids"][row][moved["attention_mask"][row].bool()],
                                         skip_special_tokens=False)
        source_text = tokenizer.decode(moved["question_source_ids"][row][moved["question_source_mask"][row].bool()],
                                       skip_special_tokens=False)
        if source_text.strip() not in original_text:
            raise RuntimeError("V3 standalone question is not present verbatim in the original chat")
        boundaries.append({**boundary, "dataset_index": indices[row],
                           "vision_end_token": tokenizer.convert_ids_to_tokens(int(moved["input_ids"][row, boundary["vision_end"]])),
                           "prompt_positions": positions[row].tolist(),
                           "post_prompt_chat_preview": tokenizer.decode(moved["input_ids"][row, boundary["prompt_end_exclusive"]:
                                                                           boundary["prompt_end_exclusive"] + 45], skip_special_tokens=False),
                           "standalone_question": source_text,
                           "first_answer_label_position": int(first_supervised[0]),
                           "answer_label_preview": tokenizer.decode(prepared["labels"][row, first_supervised[:12]],
                                                                   skip_special_tokens=False)})
    model.train()
    output = model(**moved)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise RuntimeError("V3 real-batch loss missing/nonfinite")
    output.loss.backward()
    groups = model.trainable_parameter_groups()
    norms = {}
    for name, parameters in groups.items():
        gradients = [p.grad for p in parameters]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f"V3 real-batch gradient missing/nonfinite in {name}")
        norm = math.sqrt(sum(float(g.detach().float().square().sum()) for g in gradients))
        if norm <= 0:
            raise RuntimeError(f"V3 real-batch gradient inactive in {name}")
        norms[name] = norm
    required = ("text_projection.weight", "question_depthwise.weight", "question_pointwise.weight",
                "question_pool.weight", "layer_gate.weight", "prefix_output.weight") + tuple(
        f"{prefix}.{index}.weight" for prefix in ("query_heads", "key_heads", "value_blocks") for index in range(3)
    )
    if any(model.first_backward_gradients.get(name, 0.0) <= 0.0 for name in required):
        raise RuntimeError("V3 condition branch has inactive first-batch parameter gradient")
    if not model.last_injection_audit or not model.last_injection_audit["native_embeddings_unchanged"]:
        raise RuntimeError("V3 preflight did not preserve native embeddings")
    report = {"dataset_indices": list(indices), "loss": float(output.loss.detach()),
              "boundary_examples": boundaries, "parameter_groups": model._audit_parameters(),
              "gradient_group_norms": norms, "first_parameter_gradients": model.first_backward_gradients,
              "forward": model.first_batch_diagnostics, "injection": model.last_injection_audit,
              "question_policy": "independent_raw_question_ids_no_prefill_write_mapping"}
    with (output_dir / "v3_real_batch_preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("[V3_REAL_BATCH_PREFLIGHT] " + json.dumps(report, ensure_ascii=False), flush=True)
    model.zero_grad(set_to_none=True)


def generation_roundtrip_preflight(model, processor, data_root: Path, cache_dir: Path, output_dir: Path):
    store = PathVQAParquetStore(data_root, "validation", cache_dir=cache_dir)
    row = dict(store.samples[0])
    image = store.load_image(row)
    try:
        interface = VisualSelectionPrefixInterface.__new__(VisualSelectionPrefixInterface)
        interface.processor = processor
        inputs = interface.prepare_inputs(image, build_prompt(str(row["question"]), None), question=str(row["question"]))
    finally:
        image.close()
    device = next(model.base_model.parameters()).device
    moved = {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point()
                                  else value.dtype) for key, value in inputs.items()}
    old_cache = model.base_model.config.use_cache
    model.base_model.config.use_cache = True
    model.eval()
    try:
        def greedy():
            before = model.diagnostic_prefill_calls
            with torch.inference_mode():
                output = model.generate(**moved, max_new_tokens=8, do_sample=False, use_cache=True)
            if model.diagnostic_prefill_calls - before != 1:
                raise RuntimeError("V3 generation did not inject exactly once in prefill")
            return output[:, moved["input_ids"].shape[1]:].detach().cpu()
        first = greedy()
        checkpoint = output_dir / "preflight_v3_roundtrip"
        model.save_v3(checkpoint)
        with torch.no_grad():
            model.p20[0, 0].add_(1.0)
        model.load_v3(checkpoint)
        second = greedy()
        if not torch.equal(first, second):
            raise RuntimeError("V3 save/reload changed greedy generation")
    finally:
        model.base_model.config.use_cache = old_cache
        model.train()
    report = {"question_id": row["question_id"], "roundtrip_equal": True,
              "single_prefill_injection": True, "generated_tokens": first[0].tolist()}
    with (output_dir / "v3_generation_preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("[V3_GENERATION_PREFLIGHT] " + json.dumps(report), flush=True)


class V3AuditCallback(TrainerCallback):
    def __init__(self, output_dir: Path, processor):
        self.output_dir = output_dir
        self.processor = processor
        self.path = output_dir / "v3_diagnostics.jsonl"

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        row = {"step": int(state.global_step), "epoch": float(state.epoch or 0),
               "gradient_stage": "after_Trainer_global_clip_before_optimizer"}
        for name, parameters in model.trainable_parameter_groups().items():
            grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
            if len(grads) != len(parameters) or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError(f"V3 missing/nonfinite clipped gradient in {name}")
            row[f"{name}_grad_norm"] = math.sqrt(sum(float(g.square().sum()) for g in grads))
        row.update({key: float(value.float()) for key, value in model.debug_context.items()})
        if int(state.global_step) % 20 == 0:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("[V3_DIAGNOSTICS] " + json.dumps(row, ensure_ascii=False), flush=True)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and int(round(float(state.epoch or 0))) == 3:
            checkpoint = self.output_dir / "checkpoints" / "epoch_3"
            kwargs["model"].save_v3(checkpoint)
            self.processor.save_pretrained(checkpoint)
            print(f"[V3_EPOCH3_CHECKPOINT] {checkpoint}", flush=True)
        return control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    if args.experiment_name != "pathvqa_v3_postvisual_prefix_p20_norm_fixed_seed44":
        raise ValueError("V3 train entry is restricted to the authorized seed44 experiment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed, data_seed = 44, 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    versions = {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "accelerate": importlib.metadata.version("accelerate")}
    print("[V3_RUNTIME] " + json.dumps({"experiment": args.experiment_name, "git_commit": commit,
                                     "versions": versions, "model_seed": seed, "data_seed": data_seed},
                                    sort_keys=True), flush=True)
    native_processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    processor = Qwen3ProcessorWithV3(
        image_processor=native_processor.image_processor, tokenizer=native_processor.tokenizer,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    base.config.use_cache = False
    model = construct_with_v1_initialization_audit(base, args.output_dir)
    counts = model._audit_parameters()
    if sum(counts.values()) != EXPECTED_TRAINABLE:
        raise RuntimeError("V3 trainable parameter count changed")
    dataset = RawQuestionDataset(V3PathVQADataset(
        processor=processor, data_root=args.data_root, split="train", ce_enabled=True,
        seed=data_seed, deterministic_sampling=True, max_length=2068,
    ), processor.tokenizer)
    collator = RawQuestionCollator(processor)
    native_dataset = RawQuestionDataset(PathVQADataset(
        processor=native_processor, data_root=args.data_root, split="train", ce_enabled=True,
        seed=data_seed, deterministic_sampling=True, max_length=2048,
    ), native_processor.tokenizer)
    native_collator = RawQuestionCollator(native_processor)
    rng = _rng_state()
    real_batch_preflight(model, dataset, collator, native_dataset, native_collator,
                         processor, args.output_dir)
    generation_roundtrip_preflight(model, processor, args.data_root, args.cache_dir, args.output_dir)
    _restore_rng(rng)
    if torch.cuda.is_available():
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
        callbacks=[V3AuditCallback(args.output_dir, processor)],
    )
    trainer.model_accepts_loss_kwargs = False
    if (trainer.args.gradient_accumulation_steps != 16
            or trainer.accelerator.gradient_accumulation_steps != 1
            or trainer.model_accepts_loss_kwargs is not False):
        raise RuntimeError("V3 runtime differs from audited V1 normalized accumulation")
    print("[V3_LOSS_ACCUMULATION] trainer_model_accepts_loss_kwargs=False "
          "trainer_steps=16 accelerate_steps=1", flush=True)
    result = trainer.train()
    checkpoint = args.output_dir / "checkpoints" / "epoch_3"
    if not (checkpoint / "visual_selection_postvisual.pt").is_file():
        raise RuntimeError("V3 epoch3 checkpoint was not saved")
    report = {"experiment": args.experiment_name, "method": "visual_selection_postvisual_p20_v3",
              "dataset": "PathVQA", "model_seed": seed, "data_seed": data_seed, "epochs": 3,
              "trainable_parameters": counts, "total_trainable_parameters": sum(counts.values()),
              "train_metrics": result.metrics,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              "git_commit": commit, "runtime_versions": versions,
              "trainer_model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
              "accelerator_gradient_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
              "loss_accumulation_protocol": "equal_microbatch_mean_trainer_normalized",
              "optimizer": {"type": "AdamW", "weight_decay": 0.0, "betas": [0.9, 0.999],
                            "eps": 1e-8, "scheduler": "linear", "warmup_ratio": 0.03,
                            "max_grad_norm": 1.0, "per_device_batch_size": 2,
                            "gradient_accumulation_steps": 16}}
    with (args.output_dir / "train_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[V3_TRAIN_DONE] checkpoint={checkpoint} runtime={result.metrics.get('train_runtime')} "
          f"peak_gpu_bytes={report['peak_gpu_memory_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
