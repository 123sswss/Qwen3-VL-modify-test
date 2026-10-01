"""CPU orchestration; child GPU commands are launched only by the user entrypoint.

No model imports. Strict artifact binding, fixed Test epochs, fail-stop execution.
Ledger fragments stay with outputs: the Windows checkout remains source of truth.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import zipfile
from datetime import datetime


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_archive(path):
    # torch.save's existing zip format: CPU-only CRC/completeness check, no torch import.
    with zipfile.ZipFile(path) as archive:
        require(any(n.endswith('/data.pkl') for n in archive.namelist()), f"Missing tensor manifest: {path}")
        require(archive.testzip() is None, f"Checkpoint CRC failure: {path}")


def v1_rates_evidence(run, report):
    """Read newer reports or recover older fields from actual logs/recorded commit.

    Never substitute current HEAD's defaults for a historical training protocol.
    """
    if 'group_learning_rates' in report['optimizer']:
        return report['optimizer']['group_learning_rates'], 'train_report.optimizer.group_learning_rates'
    log = run/'train.log'
    if log.is_file():
        matches = re.findall(r'\[V1_OPTIMIZER\] rates=(\{[^\r\n]*?\})\s+warmup_ratio=',
                             log.read_text(encoding='utf-8', errors='replace'))
        if matches:
            rates = [json.loads(x) for x in matches]
            require(all(x == rates[0] for x in rates), f'Conflicting optimizer logs: {run}')
            return rates[0], str(log) + ':[V1_OPTIMIZER]'
    commit = report.get('git_commit', '')
    require(bool(re.fullmatch(r'[0-9a-f]{40}', commit)),
            f'Missing historical optimizer evidence (log or full training commit): {run}')
    source = subprocess.run(['git', 'show', commit+':pathvqa/train_visual_selection_prefix.py'],
                            capture_output=True, text=True, encoding='utf-8', check=False)
    require(source.returncode == 0, f'Historical training source unavailable: {commit}')
    tree = ast.parse(source.stdout)
    values = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == 'V1Trainer':
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == 'create_optimizer':
                    for node in ast.walk(fn):
                        if isinstance(node, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == 'rates' for t in node.targets):
                            try:
                                values.append(ast.literal_eval(node.value))
                            except (ValueError, TypeError):
                                pass
    require(len(values) == 1 and isinstance(values[0], dict),
            f'Cannot recover literal historical LR groups: {commit}; provide original optimizer log')
    return values[0], f'git:{commit}:pathvqa/train_visual_selection_prefix.py:V1Trainer.create_optimizer'


def audit_v1(run, seed, dataset="PathVQA", mode="split18", params=1864963):
    report = read(run / "train_report.json")
    config = read(run / "checkpoints/epoch_5/visual_selection_prefix_config.json")
    require(report["dataset"] == dataset and report["model_seed"] == seed and
            report["data_seed"] == 42 and report["epochs"] == 5 and
            5 in report["saved_epochs"], f"Wrong dataset/seed/epoch: {run}")
    # Five-epoch reports predating Visual20 do not contain visual_prompt_mode.
    # Missing fields can be resolved only by an explicit original method identity.
    inferred_mode = report.get('visual_prompt_mode')
    if inferred_mode is None:
        require(mode == 'split18' and report.get('method') == 'visual_selection_prefix_p20_v1'
                and config.get('method') == 'visual_selection_prefix_p20_v1'
                and not config.get('ablation_mode'), f'Missing mode without original V1 evidence: {run}')
        inferred_mode = 'split18'
    require(inferred_mode == mode and report["total_trainable_parameters"] == params and
            report["trainer_model_accepts_loss_kwargs"] is False and
            report["accelerator_gradient_accumulation_steps"] == 1,
            f"Wrong architecture/normalization: {run}")
    require(config["init_seed"] == seed and config["visual_tokens"] == [8, 10]
            and config["prefix_tokens"] == 20 and config["layers"] == [5, 11, 17],
            f"Wrong checkpoint layout: {run}")
    if mode == "split18":
        require(config["method"] == "visual_selection_prefix_p20_v1" and
                not config.get("ablation_mode"), f"Not original V1: {run}")
    else:
        require(config.get("ablation_mode") == "c_static", f"Not C-static: {run}")
    require(sum(config["trainable_parameters"].values()) == params,
            f"Checkpoint parameter count: {run}")
    opt = report["optimizer"]
    require(opt["per_device_batch_size"] == (2 if dataset == "PathVQA" else 4)
            and opt["gradient_accumulation_steps"] == (16 if dataset == "PathVQA" else 8)
            and opt["warmup_ratio"] == .03 and opt["scheduler"] == "linear"
            and opt["max_grad_norm"] == 1,
            f"Training protocol mismatch: {run}")
    rates, rates_source = v1_rates_evidence(run, report)
    require(rates.get("p20") == .3 and rates.get("visual_s8") == 3e-5
            and rates.get("visual_av10") == 1e-4, f"Visual18/P20 LR mismatch: {run}")
    if 'visual_av10_learning_rate' in report:
        require(report['visual_av10_learning_rate'] == rates['visual_av10'],
                f'Conflicting Av10 report/log evidence: {run}')
    weights = run / "checkpoints/epoch_5/visual_selection_prefix.pt"
    require(weights.is_file() and weights.stat().st_size > 0, f"Missing weights: {weights}")
    audit_archive(weights)
    return {"run": str(run.resolve()), "checkpoint": str(weights.parent.resolve()),
            "weights_sha256": fingerprint(weights), "report": report,
            "report_sha256": fingerprint(run / "train_report.json"),
            "compatibility_evidence": {'visual_prompt_mode': inferred_mode,
                'mode_source': 'report.visual_prompt_mode' if 'visual_prompt_mode' in report
                    else 'original method in report and checkpoint config',
                'group_learning_rates': rates, 'rates_source': rates_source}}


def bind_pathvqa(root):
    parent = root / "pathvqa/outputs/visual_selection_prefix"
    # Confirmed by read-only SSH inspection on 2026-10-01, not directory recency.
    exact_runs = {44: 'pathvqa_v1_norm_fixed_5ep_seed44_20260926_2',
                  45: 'pathvqa_v1_norm_fixed_5ep_seed45_20260926',
                  46: 'pathvqa_v1_norm_fixed_5ep_seed46_20260926'}
    bindings = []
    for seed, score in [(44, 59.3386), (45, 58.5077), (46, 58.7314)]:
        name = f"pathvqa_v1_norm_fixed_5ep_seed{seed}"
        candidates = [parent / exact_runs[seed]]
        valid, rejected = [], []
        for run in candidates:
            try:
                require(read(run / "train_report.json")["experiment"] == name, "experiment mismatch")
                binding = audit_v1(run, seed)
                summary = read(run / "eval_validation/epoch_5/pathvqa_summary.json")
                require(summary["split"] == "validation" and summary["count"] == 6259
                        and abs(summary["overall_accuracy"] - score) < .00011,
                        "Recorded Validation identity mismatch")
                valid.append(binding)
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
                rejected.append({"run": str(run), "reason": str(exc)})
        require(len(valid) == 1, f"seed{seed}: require exactly one verified artifact; "
                f"valid={[x['run'] for x in valid]}, rejected={rejected}")
        bindings.append({"seed": seed, **valid[0], "rejected_candidates": rejected})
    return bindings


def audit_eval(directory, dataset, checkpoint, backend):
    prefix = "pathvqa" if dataset == "PathVQA" else "rsvqa"
    summary = read(directory / f"{prefix}_summary.json")
    predictions = read(directory / f"{prefix}_predictions.json")
    comparisons = read(directory / f"{prefix}_comparisons.json")
    expected = 6719 if dataset == "PathVQA" else 10004
    require(summary["split"] == "test" and summary["count"] == expected and
            not summary.get("partial_evaluation", False) and summary["backend"] == backend
            and Path(summary["checkpoint"]).resolve() == Path(checkpoint).resolve(),
            f"Test identity mismatch: {directory}")
    require(summary["max_new_tokens"] == (32 if dataset == "PathVQA" else 16),
            f"Generation length mismatch: {directory}")
    if dataset == "PathVQA":
        require(summary["temperature"] == 0 and summary["answer_mode"] == "raw"
                and summary["instruction"] == "short-answer"
                and summary.get('v0_intervention', 'normal') == 'normal'
                and summary.get('dynamic_prompt_intervention', {}).get('mode', 'normal') == 'normal'
                and len({str(x['image_id']) for x in comparisons}) == 858,
                f"PathVQA Test protocol mismatch: {directory}")
    require(len(predictions) == len(comparisons) == expected and
            len({str(x['question_id']) for x in predictions}) == expected and
            {str(x['question_id']) for x in predictions} ==
            {str(x['question_id']) for x in comparisons}, f"Incomplete IDs: {directory}")
    if dataset != "PathVQA":
        require(summary["metric"] == "official_range_normalized_exact_match" and
                summary["prompt_policy"] == "question_type_specific_short_answer_v1" and
                len({str(x['image_id']) for x in comparisons}) == 100,
                "RSVQA scoring/cluster mismatch")
        from RSVQA.metric import evaluate_rsvqa_predictions
        references = [{"question_id": x['question_id'], "image_id": x['image_id'],
                       "question_type": x['question_type'], "question": x['question'],
                       "answer": x['ground_truth_answer']} for x in comparisons]
        recomputed, recomparisons = evaluate_rsvqa_predictions(references, predictions, bootstrap_iterations=0)
        require(all(x['correct'] == y['correct'] for x,y in zip(comparisons,recomparisons))
                and recomputed['overall_accuracy'] == summary['overall_accuracy']
                and recomputed['average_accuracy'] == summary['average_accuracy'],
                f"RSVQA stored scoring is inconsistent: {directory}")
    else:
        require(abs(100 * sum(x['correct'] for x in comparisons)/expected -
                    summary['overall_accuracy']) < .00011, "PathVQA stored score mismatch")
    return summary


def audit_cocoop(run, experiment):
    r = read(run / "train_report.json")
    c = read(run / "checkpoints/epoch_5/cocoop_prompt_config.json")
    require(r['experiment'] == experiment and r['dataset'] == 'RSVQA-LR'
            and r['method'] == 'cocoop_style_conditional_prompt_tuning'
            and r['trainable_parameters'] == 873120 and r['epochs'] == 5 and r['seed'] == 44
            and r['data_seed'] == 42 and r['trainer_model_accepts_loss_kwargs'] is False
            and r['accelerator_gradient_accumulation_steps'] == 1
            and r['batch_size'] == 4 and r['gradient_accumulation'] == 8
            and r['prompt_learning_rate'] == .3 and r['meta_net_learning_rate'] == 3e-4
            and r['question_access'] is False and r['prompt_length'] == 20
            and r['bottleneck_dim'] == 160, f"CoCoOp training protocol mismatch: {run}")
    require(c['init_seed'] == 44 and c['bottleneck_dim'] == 160 and c['hidden_size'] == 2560
            and c['prompt_length'] == 20 and c['question_access'] is False
            and c['visual_source'] == 'post_merger_llm_visual_token_mean'
            and c['prompt_placement'] == 'before_full_chat', "CoCoOp checkpoint mismatch")
    require(r['answer_supervision'] == 'raw_release_answer_evaluated_with_official_count_ranges'
            and r['optimizer'] == {'type': 'AdamW', 'weight_decay': 0.0, 'betas': [.9,.999],
                                   'eps': 1e-8, 'warmup_ratio': .03, 'scheduler': 'linear',
                                   'max_grad_norm': 1.0}, 'CoCoOp optimizer/supervision mismatch')
    require((run/'checkpoints/epoch_5/cocoop_prompt.pt').stat().st_size > 0, "Missing CoCoOp weights")
    audit_archive(run/'checkpoints/epoch_5/cocoop_prompt.pt')
    return r


def reuse_test(candidates, dataset, checkpoint, backend, expected_hash=None):
    valid = []
    for directory in candidates:
        if not (directory / ("pathvqa_summary.json" if dataset == "PathVQA" else "rsvqa_summary.json")).exists():
            continue
        summary = audit_eval(directory, dataset, checkpoint, backend)
        if expected_hash:
            if (directory / "suite_eval_identity.json").exists():
                provenance = read(directory / "suite_eval_identity.json")
                require(provenance['weights_sha256'] == expected_hash, "Reused checkpoint digest changed")
            else:
                # Unknown historical weight provenance is not sufficient for skipping.
                print(f"[SUITE_NO_REUSE] missing checkpoint digest provenance: {directory}", flush=True)
                continue
        valid.append((directory, summary))
    require(len(valid) <= 1, f"Multiple matching completed results; require explicit resolution: {valid}")
    return valid[0] if valid else None


def paired_rsvqa(baseline_dir, variant_dir):
    a, b = [read(p / "rsvqa_comparisons.json") for p in (baseline_dir, variant_dir)]
    ai, bi = [{str(x['question_id']): x for x in rows} for rows in (a, b)]
    require(len(ai) == len(bi) == 10004 and set(ai) == set(bi), "RSVQA paired IDs differ")
    rows = []
    for key in ai:
        x, y = ai[key], bi[key]
        for field in ("image_id", "question_type", "question", "ground_truth_answer",
                      "normalized_ground_truth_answer"):
            require(x[field] == y[field], f"RSVQA paired metadata differs: {key}/{field}")
        rows.append((x, y))
    groups = {}
    for name in ["overall", "rural_urban", "presence", "count", "comp"]:
        subset = rows if name == "overall" else [(x, y) for x, y in rows if x['question_type'] == name]
        clusters = {}
        for x, y in subset:
            item = clusters.setdefault(str(x['image_id']), [0, 0])
            item[0] += int(y['correct']) - int(x['correct'])
            item[1] += 1
        values = [clusters[key] for key in sorted(clusters)]
        rng, draws = random.Random(42), []
        for _ in range(10000):
            sampled = [values[rng.randrange(len(values))] for _ in values]
            draws.append(100 * sum(x[0] for x in sampled) / sum(x[1] for x in sampled))
        draws.sort()
        def quantile(p):
            v = (len(draws) - 1) * p
            i = int(v)
            return draws[i] + (draws[min(i + 1, len(draws)-1)] - draws[i]) * (v-i)
        old = 100 * sum(x['correct'] for x, _ in subset) / len(subset)
        new = 100 * sum(y['correct'] for _, y in subset) / len(subset)
        groups[name] = {"count": len(subset), "image_clusters": len(values),
                        "baseline_accuracy": old, "variant_accuracy": new, "delta": new-old,
                        "variant_only_correct": sum(not x['correct'] and y['correct'] for x,y in subset),
                        "baseline_only_correct": sum(x['correct'] and not y['correct'] for x,y in subset),
                        "ci95": [quantile(.025), quantile(.975)], "exploratory": name != "overall"}
    types = ['rural_urban', 'presence', 'count', 'comp']
    clusters = {}
    for x, y in rows:
        value = clusters.setdefault(str(x['image_id']), [[0, 0] for _ in types])
        k = types.index(x['question_type'])
        value[k][0] += int(y['correct']) - int(x['correct'])
        value[k][1] += 1
    values = [clusters[key] for key in sorted(clusters)]
    rng, macro_draws = random.Random(42), []
    for _ in range(10000):
        totals = [[0, 0] for _ in types]
        for _ in values:
            drawn = values[rng.randrange(len(values))]
            for k in range(4):
                totals[k][0] += drawn[k][0]
                totals[k][1] += drawn[k][1]
        if all(x[1] for x in totals):
            macro_draws.append(sum(100 * x[0]/x[1] for x in totals)/4)
    macro_draws.sort()
    def macro_q(p):
        v = (len(macro_draws)-1)*p
        i = int(v)
        return macro_draws[i] + (macro_draws[min(i+1,len(macro_draws)-1)]-macro_draws[i])*(v-i)
    return {"baseline": str(baseline_dir), "variant": str(variant_dir), "groups": groups,
            "bootstrap": {"unit": "image_id", "iterations": 10000, "seed": 42},
            "AA_ci95": [macro_q(.025), macro_q(.975)],
            "AA_valid_bootstrap_draws": len(macro_draws),
            "AA_delta": read(variant_dir/'rsvqa_summary.json')['average_accuracy'] -
                        read(baseline_dir/'rsvqa_summary.json')['average_accuracy']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--precheck-only", action="store_true", help="CPU file identity audit only")
    parser.add_argument("--import-results", type=Path, help="CPU-only import suite fragments into Windows ledgers")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.import_results:
        require(os.name == "nt", "Ledger import is restricted to the Windows source checkout")
        marker = "\n\n## Five-task suite " + args.import_results.name + "\n\n"
        for target, fragment in [("EXPERIMENT_RESULTS.md", "EXPERIMENT_RESULTS_append.md"),
                                 ("result.md", "result_append.md"), ("plan.md", "plan_append.md")]:
            path = root / target
            original = path.read_text(encoding='utf-8')
            require(marker not in original, f"Already imported {target}; preserve history without duplicates")
            text = (args.import_results / fragment).read_text(encoding='utf-8')
            with path.open('a', encoding='utf-8') as handle:
                handle.write(marker + text)
        return 0
    try:
        bindings = bind_pathvqa(root)  # All identities verified before the first GPU child.
        v1 = root / "RSVQA/outputs/visual_selection_prefix/rsvqa_v1_norm_fixed_5ep_b4a8_seed44_20260929"
        lora = root / "RSVQA/outputs/lora/rsvqa_lr_lora_full_model_attention_r8_b4a8_seed44_20260929"
        audit_v1(v1, 44, "RSVQA-LR")
        old_lora = read(lora/'train_report.json')
        require(old_lora['dataset'] == 'rsvqa_lr' and old_lora['seed'] == 44
                and old_lora['data_seed'] == 42 and old_lora['epochs'] == 3
                and old_lora['rank'] == 8 and old_lora['alpha'] == 16
                and old_lora['parameter_counts']['trainable'] == 7077888
                and old_lora['per_device_train_batch_size'] == 4
                and old_lora['gradient_accumulation_steps'] == 8
                and old_lora['trainer_model_accepts_loss_kwargs'] is False,
                'Existing RSVQA LoRA training identity mismatch')
        for run, epoch, backend in [(v1, 5, "visual-selection-prefix"), (lora, 3, "lora")]:
            audit_eval(run / f"eval_test/epoch_{epoch}", "RSVQA-LR", run / f"checkpoints/epoch_{epoch}", backend)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        if not args.precheck_only:
            failure = root/'outputs/five_task_test_suite'/datetime.now().strftime('precheck_failed_%Y%m%d_%H%M%S_%f')
            failure.mkdir(parents=True)
            row = {'task': 'prechecks', 'status': 'failed', 'exit_code': 1, 'error': str(exc),
                   'gpu_children_started': False}
            write(failure/'stage_status.json', [row])
            for name in ('EXPERIMENT_RESULTS', 'result', 'plan'):
                (failure/(name+'_append.md')).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'[SUITE_FAILED] output={failure}', file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    if args.precheck_only:
        print(json.dumps(bindings, ensure_ascii=False, indent=2))
        return 0
    parent = root / "outputs/five_task_test_suite"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / datetime.now().strftime("suite_%Y%m%d_%H%M%S_%f")
    output.mkdir()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    write(output / "bindings.json", bindings)
    model = os.environ.get("MMRL_MODEL_PATH", "/root/autodl-tmp/model")
    path_data = os.environ.get("PATHVQA_DATA_ROOT", "/root/autodl-tmp/dataset/pathVQA")
    rs_data = os.environ.get("RSVQA_DATA_ROOT", "/root/autodl-tmp/dataset/RSVQA/6344334")
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')}
    write(output/'suite_config.json', {'git_commit': commit, 'runtime_versions': versions,
          'base_model': model, 'pathvqa_data': path_data, 'rsvqa_data': rs_data,
          'order': ['pathvqa_seed44_test','pathvqa_seed45_test','pathvqa_seed46_test',
                    'rsvqa_static_train5_test','rsvqa_cocoop_train5_test'],
          'auto_shutdown': False, 'retry': False})
    states = []

    def record(task, status, **info):
        states.append({"task": task, "status": status, "time": datetime.now().isoformat(),
                       "git_commit": commit,
                       "exit_code": 0 if status == 'completed' else None, **info})
        write(output / "stage_status.json", states)
        text = "\n\n".join(json.dumps(x, ensure_ascii=False, indent=2) for x in states)
        (output / "EXPERIMENT_RESULTS_append.md").write_text(text, encoding="utf-8")
        (output / "result_append.md").write_text("\n".join(
            f"- {x['task']}: {x['status']} {json.dumps(x.get('summary', {}), ensure_ascii=False)}" for x in states), encoding="utf-8")
        (output / "plan_append.md").write_text("\n".join(
            f"- {x['task']}: {x['status']}" for x in states), encoding="utf-8")

    def command(task, module, options, directory):
        cmd = [sys.executable, "-m", module, *map(str, options)]
        write(directory / (task + "_command.json"), {"command": cmd, "git_commit": commit})
        print("[SUITE_COMMAND]", task, cmd, flush=True)
        with (directory / (task + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            error = RuntimeError(f"{task} exited {code}; see {directory}")
            error.exit_code = code
            raise error

    current = "prechecks"
    try:
        scores = []
        for binding in bindings:
            seed = binding["seed"]
            current = f"pathvqa_v1_5ep_seed{seed}_epoch5_test"
            directory = output / current
            directory.mkdir()
            checkpoint = Path(binding["checkpoint"])
            require(fingerprint(checkpoint/'visual_selection_prefix.pt') == binding['weights_sha256'],
                    f"Checkpoint changed since binding: {checkpoint}")
            record(current, "started", checkpoint=str(checkpoint))
            candidates = [Path(binding['run']) / 'eval_test/epoch_5']
            candidates += [p for p in parent.glob('suite_*/' + current) if p != directory]
            reused = reuse_test(candidates, "PathVQA", checkpoint, "visual-selection-prefix", binding['weights_sha256'])
            if reused:
                directory, summary = reused
                print(f"[SUITE_REUSE] {current}: {directory}", flush=True)
            else:
                command(current, "pathvqa.pathvqa_official_eval", ["--data-root", path_data,
                        "--split", "test", "--backend", "visual-selection-prefix", "--base-model", model,
                        "--checkpoint", checkpoint, "--output-dir", directory], directory)
                summary = audit_eval(directory, "PathVQA", checkpoint, "visual-selection-prefix")
                write(directory / 'suite_eval_identity.json', {'weights_sha256': binding['weights_sha256'],
                      'source_report_sha256': binding['report_sha256'], 'git_commit': commit})
            scores.append(summary)
            record(current, "completed", output=str(directory), summary=summary, reused=bool(reused),
                   source_training=binding["report"], weights_sha256=binding["weights_sha256"])
        keys = {"Overall": "overall_accuracy", "Yes/No": "yes_no_accuracy", "Free-form": "free_form_accuracy"}
        aggregate = {name: {"values": [s[key] for s in scores],
                            "mean": statistics.mean(s[key] for s in scores),
                            "sample_sd": statistics.stdev(s[key] for s in scores)} for name,key in keys.items()}
        aggregate["question_types"] = {q: {"values": [s['per_question_type_accuracy'][q] for s in scores],
            "mean": statistics.mean(s['per_question_type_accuracy'][q] for s in scores),
            "sample_sd": statistics.stdev(s['per_question_type_accuracy'][q] for s in scores)}
            for q in scores[0]['per_question_type_accuracy']}
        write(output / "pathvqa_three_seed_summary.json", aggregate)
        for mode, count, backend in [("static_visual18_p20", 69632, "visual-selection-prefix"),
                                     ("cocoop_style_p20_h160", 873120, "cocoop-style")]:
            current = f"rsvqa_{mode}_norm_fixed_5ep_b4a8_seed44"
            directory = output / current
            directory.mkdir()
            record(current, "started", output=str(directory))
            prior = []
            for old in parent.glob('suite_*/' + current):
                if old == directory or not (old/'eval_test/epoch_5/rsvqa_summary.json').exists():
                    continue
                if mode.startswith('static'):
                    audit_v1(old, 44, 'RSVQA-LR', 'c_static', count)
                else:
                    audit_cocoop(old, current)
                summary = audit_eval(old/'eval_test/epoch_5', 'RSVQA-LR', old/'checkpoints/epoch_5', backend)
                prior.append((old, summary))
            require(len(prior) <= 1, f"Ambiguous completed {current} artifacts")
            if prior:
                old, summary = prior[0]
                paired = paired_rsvqa(v1/'eval_test/epoch_5', old/'eval_test/epoch_5')
                write(directory/'paired_vs_v1.json', paired)
                record(current, 'completed', reused=True, output=str(old), summary=summary,
                       train_report=read(old/'train_report.json'), paired_vs_v1=paired)
                continue
            common = ["--dataset", "rsvqa_lr", "--model-path", model, "--data-root", rs_data,
                      "--output-dir", directory, "--experiment-name", current, "--epochs", 5,
                      "--batch-size", 4, "--gradient-accumulation", 8, "--correct-loss-accumulation"]
            if mode.startswith("static"):
                command("train", "pathvqa.train_visual_selection_prefix", common +
                        ["--model-seed", 44, "--visual-prompt-mode", "c_static", "--save-epochs", 3, 4, 5], directory)
                audit_v1(directory, 44, "RSVQA-LR", "c_static", count)
            else:
                command("train", "pathvqa.train_cocoop", common + ["--seed", 44, "--data-seed", 42,
                        "--prompt-learning-rate", .3, "--meta-net-learning-rate", .0003,
                        "--expected-trainable-parameters", count], directory)
                audit_cocoop(directory, current)
            record(current, 'training_completed', output=str(directory),
                   train_report=read(directory/'train_report.json'), exit_code=0)
            checkpoint = directory / "checkpoints/epoch_5"
            eval_dir = directory / "eval_test/epoch_5"
            eval_dir.mkdir(parents=True)
            command("test", "RSVQA.rsvqa_lr_official_eval", ["--data-root", rs_data, "--split", "test",
                    "--backend", backend, "--base-model", model, "--checkpoint", checkpoint,
                    "--output-dir", eval_dir, "--bootstrap-iterations", 10000, "--bootstrap-seed", 42], directory)
            summary = audit_eval(eval_dir, "RSVQA-LR", checkpoint, backend)
            record(current, 'evaluation_completed', output=str(directory), summary=summary, exit_code=0)
            paired = paired_rsvqa(v1 / "eval_test/epoch_5", eval_dir)
            write(directory / "paired_vs_v1.json", paired)
            record(current, "completed", output=str(directory), summary=summary,
                   train_report=read(directory/'train_report.json'), paired_vs_v1=paired)
        paired = paired_rsvqa(lora / "eval_test/epoch_3", v1 / "eval_test/epoch_5")
        paired["budget_note"] = "V1 five epochs versus LoRA three epochs, not equal exposure"
        write(output / "rsvqa_v1_vs_lora.json", paired)
        record("five_task_suite", "completed", paired_v1_vs_lora=paired)
    except Exception as exc:
        record(current, "failed", error=str(exc), exit_code=getattr(exc, 'exit_code', 1))
        print(f"[SUITE_FAILED] {exc}; output={output}", file=sys.stderr)
        return 1
    print(f"[SUITE_DONE] output={output} auto_shutdown=false", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
