"""User-executed V10 training: explicit budget, default five; no evaluation here."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments

from diagnostics.v10_protocol import EXPERIMENT, METHOD, GROUP_LRS, EXPECTED_TRAINABLE
from pathvqa.train_visual_selection_prefix import RawQuestionDataset, RawQuestionCollator
from slake.data_pipeline import SLAKEDataset, SLAKEDataCollator
from slake.visual_selection_v10 import VisualSelectionV10Model
from slake.visual_selection_v10_interface import VisualSelectionV10Interface
from slake.slake_official_eval import build_prompt


def grad_norm(parameters):
    gradients = [p.grad for p in parameters]
    if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("V10 missing/nonfinite parameter gradient")
    return math.sqrt(sum(float(g.float().square().sum()) for g in gradients))


class V10Trainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        self.model._audit_parameters()
        groups = self.model.trainable_parameter_groups()
        rates = getattr(self.model, "group_learning_rates", GROUP_LRS)
        if set(groups) != set(rates):
            raise RuntimeError("V10 optimizer rate/group mismatch")
        self.optimizer = torch.optim.AdamW([
            {"params": groups[name], "lr": rates[name], "weight_decay": 0., "group_name": name}
            for name in groups], betas=(.9,.999), eps=1e-8)
        print("[V10_OPTIMIZER] " + json.dumps(rates), flush=True)
        return self.optimizer


class V10Callback(TrainerCallback):
    def __init__(self, output, processor, save_epochs=(3,4,5)):
        self.output, self.processor = output, processor
        self.save_epochs = set(save_epochs)
        self.saved = set()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        norms = {name: grad_norm(values) for name,values in model.trainable_parameter_groups().items()}
        if state.is_world_process_zero and int(state.global_step)%20 == 0:
            row = {"step": int(state.global_step), "epoch": float(state.epoch or 0),
                   "gradient_measurement": "on_pre_optimizer_step_after_Trainer_clipping",
                   "group_gradient_norms": norms,
                   **{key:float(value.float()) for key,value in model.debug_context.items()}}
            with (self.output/"v10_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row)+"\n")
            print("[V10_DIAGNOSTICS] " + json.dumps(row), flush=True)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(float(state.epoch or 0)))
        if state.is_world_process_zero and epoch in self.save_epochs and epoch not in self.saved:
            directory = self.output/"checkpoints"/f"epoch_{epoch}"
            kwargs["model"].save_v10(directory)
            self.processor.save_pretrained(directory)
            self.saved.add(epoch)
            print(f"[V10_EPOCH_CHECKPOINT] epoch={epoch} path={directory}", flush=True)
        return control


def real_batch_preflight(model, dataset, collator, processor, source, output, dataset_name="slake"):
    """Train-batch gradients and save/reload/cache audit; no optimizer step."""
    batch = collator([dataset[i] for i in range(2)])
    device = next(model.base_model.parameters()).device
    moved = {key:value.to(device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
             for key,value in batch.items()}
    model.train()
    counts = {"vision": 0, "language": 0}
    def count_vision(_module, _args, _output):
        counts["vision"] += 1
    def count_language(_module, _args, _output):
        counts["language"] += 1
    handles = [model.base_model.model.visual.register_forward_hook(count_vision),
               model.base_model.model.language_model.register_forward_hook(count_language)]
    before = model.diagnostic_prefill_calls
    prediction = model(**moved)
    if counts != {"vision":1,"language":1}:
        raise RuntimeError(f"V10 repeated complete vision/LLM forward: {counts}")
    if prediction.loss is None or not bool(torch.isfinite(prediction.loss)):
        raise RuntimeError("V10 real-batch loss invalid")
    prediction.loss.backward()
    groups = {name:grad_norm(values) for name,values in model.trainable_parameter_groups().items()}
    if any(value <= 0 for value in groups.values()):
        raise RuntimeError(f"V10 inactive real-batch group: {groups}")
    required = ["layer_gate.weight", "text_projection.weight", "meta_net.0.weight", "meta_net.2.weight"]
    required += [f"{kind}.{i}.weight" for kind in ("key_heads", "query_heads") for i in range(3)]
    if any(model.first_backward_gradients.get(name, 0) <= 0 for name in required):
        raise RuntimeError("V10 conditional-path gradients are not active")
    if model.diagnostic_prefill_calls-before != 1 or not model.last_injection_audit["native_embeddings_unchanged"]:
        raise RuntimeError("V10 did not preserve single-prefill/native embeddings")
    # Actual labels and original chat order are preserved by inherited expansion.
    expanded, _, _ = model._expand(moved)
    if not torch.equal(expanded["labels"][:,20:], moved["labels"]) or not bool((expanded["labels"][:,:20] == -100).all()):
        raise RuntimeError("V10 prefix supervision changed original labels")
    audit = {"loss": float(prediction.loss.detach()), "group_gradients_preclip": groups,
             "parameter_gradients_preclip": model.first_backward_gradients,
             "forward": model.first_batch_diagnostics, "injection": model.last_injection_audit,
             "parameter_counts": model._audit_parameters(), "labels_preserved_prefix_ignored": True,
             "initialization": model.initialization_audit}
    model.zero_grad(set_to_none=True)
    del prediction, batch, moved, expanded
    row = source.data[0]
    interface = VisualSelectionV10Interface.__new__(VisualSelectionV10Interface)
    interface.processor = processor
    image_source = source.load_image(row) if dataset_name == "pathvqa" else Image.open(row["image_path"])
    with image_source as image:
        if dataset_name == "pathvqa":
            from pathvqa.pathvqa_official_eval import build_prompt as pathvqa_prompt
            prompt = pathvqa_prompt(row["question"], None)
        elif dataset_name == "rsvqa_lr":
            from RSVQA.prompts import build_prompt as rsvqa_prompt
            prompt = rsvqa_prompt(row)
        else:
            prompt = build_prompt({"_slake_question": row["question"], "_slake_language": row["language"]}, None)
        inputs = interface.prepare_inputs(image.convert("RGB"), prompt, question=row["question"])
    moved = {key:value.to(device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
             for key,value in inputs.items()}
    old_cache = model.base_model.config.use_cache
    model.base_model.config.use_cache = True
    model.eval()
    try:
        def greedy():
            count = model.diagnostic_prefill_calls
            vision_count = counts["vision"]
            with torch.inference_mode():
                result = model.generate(**moved, max_new_tokens=8, do_sample=False, use_cache=True)
            if model.diagnostic_prefill_calls-count != 1:
                raise RuntimeError("V10 repeated prefix injection during KV-cache decoding")
            if counts["vision"]-vision_count != 1:
                raise RuntimeError("V10 repeated complete vision encoder during cached generation")
            return result.detach().cpu()
        first = greedy()
        directory = output/"preflight_roundtrip"
        model.save_v10(directory)
        with torch.no_grad():
            model.p20[0,0].add_(1.)
        model.load_v10(directory)
        second = greedy()
        if not torch.equal(first, second):
            raise RuntimeError("V10 save/reload changed greedy prediction")
    finally:
        for handle in handles:
            handle.remove()
        model.base_model.config.use_cache = old_cache
        model.train()
    audit.update(roundtrip_equal=True, cached_generation_single_prefill=True,
                 probe_split="train", question_id=row["question_id"], optimizer_steps=0)
    (output/"v10_real_batch_preflight.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("[V10_REAL_BATCH_PREFLIGHT] " + json.dumps(audit), flush=True)


def main(*, model_class=VisualSelectionV10Model, experiment_override=None,
         method=METHOD, expected_trainable=EXPECTED_TRAINABLE, group_lrs=GROUP_LRS,
         default_epochs=5, default_save_epochs=(3,4,5)):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-count", type=int, required=True)
    parser.add_argument("--dataset", choices=("slake", "pathvqa", "rsvqa_lr"), default="slake")
    parser.add_argument("--model-seed", type=int, default=44)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--save-epochs", type=int, nargs="+", default=list(default_save_epochs))
    args = parser.parse_args()
    if args.epochs < 1 or any(epoch < 1 or epoch > args.epochs for epoch in args.save_epochs):
        parser.error("Saved epochs must be within the training budget")
    seed, data_seed = args.model_seed, 42
    experiment = (f"pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed{seed}"
                  if args.dataset == "pathvqa" else EXPERIMENT)
    if experiment_override:
        experiment = experiment_override
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "accelerate": importlib.metadata.version("accelerate")}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print("[V10_RUNTIME] " + json.dumps({"experiment":experiment,"versions":versions,"commit":commit}), flush=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(str(args.model_path), torch_dtype=torch.bfloat16,
                                                    device_map="auto", trust_remote_code=True)
    base.config.use_cache = False
    model = model_class(base, init_seed=seed)
    model.group_learning_rates = dict(group_lrs)
    if args.dataset == "pathvqa":
        from pathvqa.data_pipeline import PathVQADataset, PathVQADataCollator
        source = PathVQADataset(processor=processor, data_root=args.data_root, split="train",
                               ce_enabled=True, seed=data_seed, deterministic_sampling=True, max_length=2048)
        base_collator = PathVQADataCollator(processor)
    elif args.dataset == "rsvqa_lr":
        from RSVQA.data_pipeline import RSVQALRDataset, RSVQADataCollator
        source = RSVQALRDataset(processor, args.data_root, split="train", ce_enabled=True,
                               seed=data_seed, deterministic_sampling=True, max_length=2048)
        base_collator = RSVQADataCollator(processor)
    else:
        source = SLAKEDataset(processor, str(args.data_root/"imgs"), questions_path=str(args.data_root/"train.json"),
                             languages=None, base_types=None, splits=("train",), ce_enabled=True,
                             seed=data_seed, deterministic_sampling=True, max_length=2048)
        base_collator = SLAKEDataCollator(processor)
    if len(source) != args.expected_train_count:
        raise RuntimeError("V10 effective train sample count differs from CPU manifest audit")
    dataset = RawQuestionDataset(source, processor.tokenizer)
    collator = RawQuestionCollator(processor, base_collator)
    cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state_all()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    real_batch_preflight(model, dataset, collator, processor, source, args.output_dir, args.dataset)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.cuda.reset_peak_memory_stats()
    callback = V10Callback(args.output_dir, processor, args.save_epochs)
    trainer = V10Trainer(model=model, args=TrainingArguments(
        output_dir=str(args.output_dir/"trainer"), num_train_epochs=args.epochs, per_device_train_batch_size=2,
        gradient_accumulation_steps=16, learning_rate=1e-4, weight_decay=0., warmup_ratio=.03,
        lr_scheduler_type="linear", max_grad_norm=1., logging_steps=20, save_strategy="no",
        bf16=True, gradient_checkpointing=False, dataloader_num_workers=2, remove_unused_columns=False,
        report_to="none", seed=seed, data_seed=data_seed), train_dataset=dataset,
        data_collator=collator, processing_class=processor, callbacks=[callback])
    trainer.model_accepts_loss_kwargs = False
    if trainer.accelerator.gradient_accumulation_steps != 1 or trainer.args.gradient_accumulation_steps != 16:
        raise RuntimeError("V10 accumulation differs from audited equal-microbatch-mean path")
    print("[V10_ACCUMULATION] trainer.model_accepts_loss_kwargs=False trainer=16 accelerate=1", flush=True)
    result = trainer.train()
    trainer.save_state()
    if callback.saved != set(args.save_epochs):
        raise RuntimeError(f"V10 required saved epochs missing: {callback.saved}")
    counts = model._audit_parameters()
    if sum(counts.values()) != expected_trainable:
        raise RuntimeError("V10 final parameter count changed")
    report = {"experiment": experiment, "method": method, "dataset": {"pathvqa":"PathVQA","slake":"SLAKE","rsvqa_lr":"RSVQA-LR"}[args.dataset], "model_seed":seed,
              "data_seed":data_seed,"epochs":args.epochs,"saved_epochs":sorted(callback.saved),"train_split":"train",
              "languages":"en" if args.dataset == "rsvqa_lr" else "all","train_samples":len(source),
              "train_manifest":getattr(source,"manifest",None) if args.dataset == "rsvqa_lr" else
                               None if args.dataset == "pathvqa" else str(args.data_root/"train.json"),
              "base_model":str(args.model_path),"max_length":2048,"workers":2,"bf16":True,
              "trainable_parameters":counts,"total_trainable_parameters":sum(counts.values()),
              "git_commit":commit,"runtime_versions":versions,"train_metrics":result.metrics,
              "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"initialization":model.initialization_audit,
              "trainer_model_accepts_loss_kwargs":False,"accelerator_gradient_accumulation_steps":1,
              "loss_accumulation_protocol":"equal_microbatch_mean_trainer_normalized",
              "optimizer":{"type":"AdamW","betas":[.9,.999],"eps":1e-8,"weight_decay":0.,
                           "per_device_batch_size":2,"gradient_accumulation_steps":16,"warmup_ratio":.03,
                           "scheduler":"linear","max_grad_norm":1.,"group_learning_rates":group_lrs}}
    (args.output_dir/"train_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("[V10_TRAIN_DONE] " + str(args.output_dir/"checkpoints"/f"epoch_{args.epochs}"),flush=True)


if __name__ == "__main__":
    main()
