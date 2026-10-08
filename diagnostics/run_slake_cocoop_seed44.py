"""CPU artifact checks/statistics; GPU child commands ONLY via user launch.

Independent fixed epoch3 bilingual SLAKE Test. No retries, shutdown or overwrite.
Ledger fragments are returned to Windows, never written to server source files.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Load the pure scoring file without slake/__init__.py importing torch/data_pipeline.
_spec = importlib.util.spec_from_file_location("slake_cpu_metric", ROOT / "slake/slake_vqa_metric.py")
_metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metric)
evaluate_slake_predictions, record_id = _metric.evaluate_slake_predictions, _metric.record_id
HISTORY = "pathvqa/outputs/cocoop/pathvqa_cocoop_style_p20_h160_seed44_20260909"
BASELINE = "slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928"
EXPERIMENT = "slake_cocoop_style_p20_h160_norm_fixed_3ep_seed44"
PROTOCOL = {
    "prompt_length": 20, "bottleneck_dim": 160, "trainable_parameters": 873120,
    "seed": 44, "data_seed": 42, "epochs": 3, "batch_size": 2,
    "gradient_accumulation": 16, "max_length": 2048, "dataloader_workers": 2,
    "prompt_learning_rate": .3, "meta_net_learning_rate": 3e-4,
    "bf16": True, "optimizer": {"type": "AdamW", "betas": [.9, .999],
        "eps": 1e-8, "weight_decay": 0, "warmup_ratio": .03,
        "scheduler": "linear", "max_grad_norm": 1},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def historical_audit(directory):
    """Recorded run evidence, NOT today's defaults/runtime or an inferred commit."""
    report = read(directory / "train_report.json")
    config = read(directory / "checkpoints/epoch_3/cocoop_prompt_config.json")
    log = (directory / "train.log").read_text(encoding="utf-8", errors="replace")
    require(report["experiment"] == "pathvqa_cocoop_style_p20_h160_seed44", "Wrong historical experiment")
    require(report["dataset"] == "PathVQA", "Wrong historical dataset")
    evidence = {}
    for key, expected in PROTOCOL.items():
        if key in report:
            require(report[key] == expected, f"Historical override conflicts at {key}: {report[key]!r}; stop for explicit reconciliation")
            evidence[key] = {"value": report[key], "source": "train_report.json"}
        else:
            evidence[key] = {"value": "unknown", "source": "not recorded; new run explicitly uses requested protocol"}
    require(report["train_metrics"]["epoch"] == 3, "Historical run did not finish epoch3")
    for key, expected in {"init_seed": 44, "hidden_size": 2560, "prompt_length": 20,
                          "bottleneck_dim": 160, "question_access": False,
                          "visual_source": "post_merger_llm_visual_token_mean",
                          "prompt_placement": "before_full_chat"}.items():
        require(config[key] == expected, f"Historical architecture mismatch: {key}")
    logged = [line for line in log.splitlines() if "[COCOOP_STYLE_OPTIMIZER]" in line or
              "[PATHVQA_COCOOP_STYLE_CONFIG]" in line or "[COCOOP_RUNTIME]" in line]
    # Check explicitly logged numeric overrides rather than treating missing fields as attested defaults.
    aliases = {"prompt_lr": "prompt_learning_rate", "meta_net_lr": "meta_net_learning_rate",
               "batch_size": "batch_size", "gradient_accumulation": "gradient_accumulation",
               "max_length": "max_length", "dataloader_workers": "dataloader_workers",
               "epochs": "epochs", "seed": "seed", "data_seed": "data_seed"}
    for line in logged:
        for source, key in aliases.items():
            match = re.search(r"(?:^|\s)" + source + r"=([\d.eE+-]+)(?:\s|$)", line)
            if match:
                value = float(match[1])
                require(value == PROTOCOL[key], f"Historical log override at {source}: {value}; stop")
                evidence[key] = {"value": value, "source": "train.log"}
        for key in ("weight_decay", "warmup_ratio"):
            match = re.search(r"(?:^|\s)" + key + r"=([\d.eE+-]+)(?:\s|$)", line)
            if match:
                require(float(match[1]) == PROTOCOL["optimizer"][key], f"Historical optimizer override: {key}")
                evidence["optimizer." + key] = {"value": float(match[1]), "source": "train.log"}
        match = re.search(r"(?:^|\s)scheduler=(\w+)", line)
        if match:
            require(match[1] == "linear", "Historical scheduler override")
            evidence["optimizer.scheduler"] = {"value": match[1], "source": "train.log"}
    return {"run": str(directory), "report": report, "checkpoint_config": config,
            "logged_configuration": logged, "configuration_evidence": evidence,
            "training_commit": report.get("git_commit", "unknown"),
            "runtime_versions": report.get("runtime_versions", "unknown"),
            "historical_model_accepts_loss_kwargs": report.get("trainer_model_accepts_loss_kwargs", "unknown"),
            "normalization_note": "Historical loss normalization is unverified unless explicitly recorded; new run uses False. Full training-protocol equivalence is not established.",
            "report_sha256": hashlib.sha256((directory / "train_report.json").read_bytes()).hexdigest()}


def references(path):
    rows = read(path)
    require(isinstance(rows, list) and rows, "Expected official nonempty SLAKE list")
    require(len({str(record_id(x)) for x in rows}) == len(rows), "Duplicate SLAKE IDs")
    require({x["q_lang"] for x in rows} == {"en", "zh"}, "Expected full bilingual split")
    return rows


def effective_train_records(rows):
    """Match SLAKEDataset's existing train-split/empty-answer filtering, without torch."""
    kept, excluded = [], []
    for row in rows:
        split = next((str(row.get(k) or "").strip().lower() for k in ("split", "subset", "set")
                      if str(row.get(k) or "").strip()), "")
        answer = row.get("answer", row.get("answers"))
        if isinstance(answer, list):
            answer = next((v for v in answer if str(v).strip()), "")
        reason = "non_train_split" if split and split != "train" else (
            "empty_answer" if not str(answer or "").strip() else None)
        if reason:
            excluded.append({"question_id": record_id(row), "reason": reason})
        else:
            kept.append(row)
    return kept, excluded


def training_audit(run, effective_count, data_root):
    report = read(run / "train_report.json")
    for key, expected in PROTOCOL.items():
        require(report[key] == expected, f"Actual training report differs at {key}")
    for key, expected in {"experiment": EXPERIMENT, "dataset": "SLAKE", "languages": "all",
            "train_split": "train", "method": "cocoop_style_conditional_prompt_tuning",
            "question_access": False, "visual_source": "post_merger_llm_visual_token_mean",
            "prompt_placement": "before_full_chat"}.items():
        require(report[key] == expected, f"Training source identity mismatch: {key}")
    require(Path(report["train_manifest"]).resolve() == (data_root / "train.json").resolve(), "Training manifest differs")
    require(report["train_metrics"]["epoch"] == 3, "Training epoch3 incomplete")
    require(report["trainer_model_accepts_loss_kwargs"] is False and
            report["accelerator_gradient_accumulation_steps"] == 1, "Normalization mismatch")
    require(report["backbone_frozen"] and report["trainable_parameter_groups"] ==
            {"soft_prompt": 51200, "meta_net": 821920}, "Unexpected trainable modules")
    require(report["train_samples"] == effective_count, "Effective training sample count mismatch")
    require(report["saved_epochs"] == [1,2,3], "Saved epoch list mismatch")
    checkpoint = run / "checkpoints/epoch_3"
    config = read(checkpoint / "cocoop_prompt_config.json")
    for key, expected in {"init_seed": 44, "hidden_size": 2560, "prompt_length": 20,
            "bottleneck_dim": 160, "question_access": False,
            "prompt_placement": "before_full_chat"}.items():
        require(config[key] == expected, f"Epoch3 checkpoint mismatch: {key}")
    weights = checkpoint / "cocoop_prompt.pt"
    require(weights.is_file() and weights.stat().st_size > 0, "Fixed epoch3 weights missing")
    return report, checkpoint


def scored(eval_dir, refs):
    summary = read(eval_dir / "slake_summary.json")
    require(summary["language"] == "all" and summary["expected_split"] == "test" and
            not summary["partial_evaluation"] and not summary["base_types"], "Not full bilingual Test")
    for key, expected in {"max_new_tokens": 32, "temperature": 0, "answer_mode": "raw",
                          "instruction": "language-aware-short-answer"}.items():
        require(summary[key] == expected, f"Generation protocol mismatch at {key}")
    predictions = read(eval_dir / "slake_predictions.json")
    fresh, comparisons = evaluate_slake_predictions(refs, predictions)
    require(not fresh["missing_predictions"] and not fresh["extra_predictions"], "Incomplete predictions")
    saved = read(eval_dir / "slake_comparisons.json")
    saved_by_id = {str(x["question_id"]): x for x in saved}
    require(len(saved_by_id) == len(saved) == len(refs), "Saved comparison IDs incomplete or duplicated")
    for row in comparisons:
        old = saved_by_id[str(row["question_id"])]
        for key in ("question", "ground_truth_answers", "normalized_ground_truth_answers",
                    "predicted_answer", "correct", "answer_type", "question_type", "language"):
            require(old[key] == row[key], f"Saved reference/scoring identity differs: {row['question_id']}/{key}")
    for key in ("total", "correct", "overall_accuracy", "per_answer_type_accuracy",
                "per_question_type_accuracy", "per_language_accuracy"):
        require(summary[key] == fresh[key], f"Saved score differs from recomputation: {key}")
    return summary, comparisons


def baseline_audit(run, refs, model):
    report = read(run / "train_report.json")
    for key, expected in {"experiment": "slake_v1_norm_fixed_5ep_seed44", "dataset": "SLAKE",
            "method": "visual_selection_prefix_p20_v1", "model_seed": 44, "data_seed": 42,
            "epochs": 5, "visual_prompt_mode": "split18", "train_split": "train",
            "total_trainable_parameters": 1864963, "trainer_model_accepts_loss_kwargs": False,
            "languages": "all"}.items():
        require(report[key] == expected, f"SLAKE V1 baseline identity mismatch: {key}")
    require((run / "checkpoints/epoch_5/visual_selection_prefix.pt").is_file(), "V1 epoch5 checkpoint missing")
    summary, _ = scored(run / "eval_test/epoch_5", refs)
    require(summary["backend"] == "visual-selection-prefix" and summary["overall_accuracy"] == 75.55,
            "Wrong V1 Test reference")
    require(Path(summary["checkpoint"]).resolve() == (run / "checkpoints/epoch_5").resolve(), "V1 checkpoint binding differs")
    require(Path(summary["base_model"]).resolve() == model.resolve(), "Frozen backbone path differs")
    return {"run": str(run), "report": report, "summary": summary}


def paired(baseline, variant, refs, iterations=10000):
    a_summary, a = scored(baseline, refs)
    b_summary, b = scored(variant, refs)
    ai, bi = [{str(x["question_id"]): x for x in rows} for rows in (a, b)]
    ref_map = {str(record_id(x)): x for x in refs}
    rows = [(ai[k], bi[k], str(ref_map[k]["img_name"])) for k in sorted(ai)]
    groups = {}
    names = [("overall", None, None)] + [(v, k, v) for k, vals in (
        ("answer_type", ("OPEN", "CLOSED")), ("question_type", ("kvqa", "vqa")),
        ("language", ("en", "zh"))) for v in vals]
    for name, field, value in names:
        subset = rows if field is None else [row for row in rows if row[0][field] == value]
        require(bool(subset), f"Empty required subgroup: {name}")
        clusters = {}
        for x, y, image in subset:
            item = clusters.setdefault(image, [0, 0])
            item[0] += y["correct"] - x["correct"]
            item[1] += 1
        values = [clusters[k] for k in sorted(clusters)]
        rng, draws = random.Random(42), []
        for _ in range(iterations):
            sampled = [values[rng.randrange(len(values))] for _ in values]
            draws.append(100 * sum(x[0] for x in sampled) / sum(x[1] for x in sampled))
        draws.sort()
        def quantile(p):
            index = (len(draws) - 1) * p
            i = int(index)
            return draws[i] + (draws[min(i+1, len(draws)-1)] - draws[i]) * (index-i)
        old = 100 * sum(x["correct"] for x, _, _ in subset) / len(subset)
        new = 100 * sum(y["correct"] for _, y, _ in subset) / len(subset)
        groups[name] = {"count": len(subset), "image_clusters": len(values),
            "baseline_accuracy": old, "variant_accuracy": new, "delta": new-old,
            "variant_only_correct": sum(not x["correct"] and y["correct"] for x,y,_ in subset),
            "baseline_only_correct": sum(x["correct"] and not y["correct"] for x,y,_ in subset),
            "ci95": [quantile(.025), quantile(.975)], "exploratory": name != "overall"}
    return {"baseline": str(baseline), "variant": str(variant), "groups": groups,
            "bootstrap": {"unit": "official img_name", "iterations": iterations, "seed": 42},
            "budget_note": "CoCoOp three epochs versus V1 five epochs; not equal training exposure",
            "summaries": {"baseline": a_summary, "variant": b_summary}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/dataset/slake"))
    parser.add_argument("--history-run", type=Path, default=ROOT / HISTORY)
    parser.add_argument("--baseline-run", type=Path, default=ROOT / BASELINE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "slake/outputs/cocoop")
    parser.add_argument("--precheck-only", action="store_true", help="CPU files only; never launches model")
    parser.add_argument("--evaluate-run", type=Path,
                        help="Evaluate this exact completed run's epoch3 only; NEVER retrain or overwrite it")
    args = parser.parse_args()
    output = args.output_root / (EXPERIMENT + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    training_run = args.evaluate_run.resolve() if args.evaluate_run else output
    state = {"experiment": EXPERIMENT, "output": str(output), "stage": "precheck",
             "status": "started", "protocol": PROTOCOL, "shutdown": False,
             "precheck_only": args.precheck_only, "stage_exits": []}
    state.update(evaluation_only=args.evaluate_run is not None, training_source=str(training_run))
    def save():
        write(output / "stage_status.json", state)
    def command(stage, tokens):
        state.update(stage=stage, status="running", command=list(map(str, tokens)))
        save()
        with (output / (stage + ".log")).open("w", encoding="utf-8") as log:
            with subprocess.Popen(list(map(str, tokens)), cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
        state["exit_code"] = code
        state["stage_exits"].append({"stage": stage, "exit_code": code})
        save()
        require(code == 0, f"{stage} failed, exit={code}; no retry")
    save()
    print("[SLAKE_COCOOP_OUTPUT] " + str(output), flush=True)
    try:
        historical = historical_audit(args.history_run)
        refs = references(args.data_root / "test.json")
        train_refs = references(args.data_root / "train.json")
        effective, excluded = effective_train_records(train_refs)
        for row in train_refs + refs:
            require((args.data_root / "imgs" / row["img_name"]).is_file(), f"Missing image: {row['img_name']}")
        baseline = baseline_audit(args.baseline_run, refs, args.model_path)
        require(len(refs) == 2094, "Official full bilingual Test count differs from verified reference")
        write(output / "historical_protocol_audit.json", historical)
        write(output / "baseline_binding.json", baseline)
        write(output / "train_sample_audit.json", {"raw_count": len(train_refs),
              "effective_count": len(effective), "excluded": excluded,
              "policy": "existing_SLAKEDataset_train_split_and_nonempty_answer"})
        state.update(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                     test_count=len(refs), train_count=len(effective), status="precheck_passed")
        if args.evaluate_run:
            report, checkpoint = training_audit(training_run, len(effective), args.data_root)
            write(output / "training_source_binding.json", {"run": str(training_run),
                  "report": report, "checkpoint": str(checkpoint)})
        save()
        if args.precheck_only:
            state["exit_code"] = 0
            save()
            return 0
        if not args.evaluate_run:
            command("train", [sys.executable, "-m", "pathvqa.train_cocoop", "--dataset", "slake",
            "--model-path", args.model_path, "--data-root", args.data_root,
            "--output-dir", output, "--experiment-name", EXPERIMENT,
            "--prompt-length", 20, "--bottleneck-dim", 160, "--expected-trainable-parameters", 873120,
            "--prompt-learning-rate", .3, "--meta-net-learning-rate", 3e-4,
            "--epochs", 3, "--seed", 44, "--data-seed", 42, "--batch-size", 2,
            "--gradient-accumulation", 16, "--max-length", 2048, "--dataloader-workers", 2,
            "--correct-loss-accumulation"])
        report, checkpoint = training_audit(training_run, len(effective), args.data_root)
        eval_dir = output / "eval_test/epoch_3"
        command("eval_test_epoch_3", [sys.executable, "-m", "slake.slake_official_eval",
            "--backend", "cocoop-style", "--base-model", args.model_path, "--checkpoint", checkpoint,
            "--questions", args.data_root / "test.json", "--image-root", args.data_root / "imgs",
            "--language", "all", "--expected-split", "test", "--max-new-tokens", 32,
            "--temperature", 0, "--answer-mode", "raw", "--output-dir", eval_dir])
        comparison = paired(args.baseline_run / "eval_test/epoch_5", eval_dir, refs)
        write(output / "paired_vs_v1_seed44.json", comparison)
        summary = comparison["summaries"]["variant"]
        fragment = (f"### {EXPERIMENT}\n\nSLAKE seed44/data42, fixed epoch3 full bilingual Test; "
            f"Overall {summary['overall_accuracy']}; OPEN/CLOSED {summary['per_answer_type_accuracy']}; "
            f"KVQA/VQA {summary['per_question_type_accuracy']}; EN/ZH {summary['per_language_accuracy']}. "
            f"Parameters 873120; evaluation commit {state['git_commit']}; training commit "
            f"{report['git_commit']}; source {training_run}; output {output}. "
            f"Train runtime {report['train_metrics'].get('train_runtime')}; peak allocated GPU bytes "
            f"{report['peak_gpu_memory_bytes']}; versions {report['runtime_versions']}. "
            f"Paired vs V1 {comparison['groups']}; CoCoOp 3 epochs vs V1 5 epochs. "
            "Historical PathVQA runtime/normalization remains unverified.\n")
        (output / "ledger_fragment.md").write_text(fragment, encoding="utf-8")
        state.update(stage="completed", status="completed", exit_code=0)
        save()
        print("[SLAKE_COCOOP_DONE] " + json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        state.update(status="failed", error=str(exc), exit_code=1)
        save()
        (output / "ledger_fragment.md").write_text(
            f"{EXPERIMENT}: FAILED at {state['stage']}; {exc}; output {output}; "
            "no retry, no shutdown.\n", encoding="utf-8")
        print("[SLAKE_COCOOP_FAILED] " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
