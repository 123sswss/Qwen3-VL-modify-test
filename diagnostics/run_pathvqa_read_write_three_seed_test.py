"""Fixed epoch5 Test for the three recorded V1 read/write-LR runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime


RUNS = {
    44: "pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44_20261014_092451_34346",
    45: "pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed45_20261016_024534_10401",
    46: "pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed46_20261017_104312_1230",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def bind(root, seed, name):
    run = root / "pathvqa/outputs/visual_selection_prefix" / name
    report = read(run / "train_report.json")
    checkpoint = run / "checkpoints/epoch_5"
    config = read(checkpoint / "visual_selection_prefix_config.json")
    expected = {
        "experiment": f"pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed{seed}",
        "dataset": "PathVQA", "model_seed": seed, "data_seed": 42,
        "epochs": 5, "visual_prompt_mode": "split18",
        "read_write_learning_rate": 3e-4,
        "trainer_model_accepts_loss_kwargs": False,
        "total_trainable_parameters": 1864963,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{run}: {key}={report.get(key)!r}, expected {value!r}")
    rates = report["optimizer"]["group_learning_rates"]
    if rates["value_projection"] != 3e-4 or rates["prefix_output"] != 3e-4:
        raise ValueError(f"Wrong read/write optimizer groups: {run}")
    if (config["method"] != "visual_selection_prefix_p20_v1"
            or config["layers"] != [5, 11, 17] or config["init_seed"] != seed
            or report.get("ablation_mode")):
        raise ValueError(f"Wrong V1 checkpoint identity: {run}")
    if not (checkpoint / "visual_selection_prefix.pt").is_file():
        raise FileNotFoundError(checkpoint / "visual_selection_prefix.pt")
    return {"seed": seed, "run": str(run), "checkpoint": str(checkpoint),
            "training_commit": report.get("git_commit")}


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = root / "outputs" / (
        "pathvqa_v1_read_write_three_seed_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    state = {"status": "precheck", "output": str(output), "completed_seeds": [],
             "evaluation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    print(f"[THREE_SEED_TEST_OUTPUT] {output}", flush=True)
    try:
        bindings = [bind(root, seed, run) for seed, run in RUNS.items()]
        write(output / "binding.json", bindings)
        scores = []
        for binding in bindings:
            seed = binding["seed"]
            state.update(status="running", seed=seed)
            write(output / "status.json", state)
            eval_dir = output / f"seed{seed}/eval_test/epoch_5"
            command = [sys.executable, "-m", "pathvqa.pathvqa_official_eval",
                       "--backend", "visual-selection-prefix", "--base-model",
                       os.environ.get("MMRL_MODEL_PATH", "/root/autodl-tmp/model"),
                       "--checkpoint", binding["checkpoint"], "--data-root",
                       os.environ.get("PATHVQA_DATA_ROOT", "/root/autodl-tmp/dataset/pathVQA"),
                       "--split", "test", "--output-dir", str(eval_dir)]
            write(output / f"seed{seed}_command.json", command)
            with (output / f"seed{seed}.log").open("w", encoding="utf-8") as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
            if code:
                raise subprocess.CalledProcessError(code, command)
            summary = read(eval_dir / "pathvqa_summary.json")
            scores.append({"seed": seed, "summary": summary, "eval_dir": str(eval_dir)})
            state["completed_seeds"].append(seed)
            print(f"[TEST_DONE] seed={seed} Overall={summary['overall_accuracy']}", flush=True)
        fields = ["overall_accuracy", "yes_no_accuracy", "free_form_accuracy"]
        aggregates = {}
        for field in fields:
            values = [s["summary"][field] for s in scores]
            aggregates[field] = {"values": values, "mean": statistics.mean(values),
                                 "sample_std_ddof1": statistics.stdev(values)}
        for kind in sorted(set().union(*(s["summary"]["per_question_type_accuracy"] for s in scores))):
            values = [s["summary"]["per_question_type_accuracy"][kind] for s in scores]
            aggregates[f"question_type/{kind}"] = {"values": values, "mean": statistics.mean(values),
                                                   "sample_std_ddof1": statistics.stdev(values)}
        write(output / "three_seed_summary.json", {"runs": scores, "aggregate": aggregates})
        print(json.dumps(aggregates, ensure_ascii=False, indent=2), flush=True)
        state.update(status="completed", exit_code=0)
    except Exception as exc:
        state.update(status="failed", exit_code=getattr(exc, "returncode", 1), error=str(exc))
        raise
    finally:
        state["not_completed_seeds"] = [s for s in RUNS if s not in state["completed_seeds"]]
        write(output / "status.json", state)


if __name__ == "__main__":
    main()
