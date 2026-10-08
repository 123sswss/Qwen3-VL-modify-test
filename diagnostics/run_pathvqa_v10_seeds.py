"""User-launched PathVQA V10 44->45->46; fixed epoch5, fail-stop."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from diagnostics.run_five_task_test_suite import audit_v1
from diagnostics.v10_protocol import CONFIG_NAME, EXPECTED_GROUP_COUNTS, WEIGHTS_NAME, METHOD

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (44, 45, 46)
V1_RUNS = {
    44: "pathvqa_v1_norm_fixed_5ep_seed44_20260926_2",
    45: "pathvqa_v1_norm_fixed_5ep_seed45_20260926",
    46: "pathvqa_v1_norm_fixed_5ep_seed46_20260926",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def experiment(seed):
    return f"pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed{seed}"


def validation(directory, checkpoint, backend, model_path, data_root):
    summary = read(directory/"pathvqa_summary.json")
    rows = read(directory/"pathvqa_comparisons.json")
    predictions = read(directory/"pathvqa_predictions.json")
    expected = {"split":"validation", "count":6259, "backend":backend,
                "max_new_tokens":32, "temperature":0., "answer_mode":"raw",
                "instruction":"short-answer", "partial_evaluation":False}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Validation protocol differs: {directory}: {key}")
    for key, value in (("checkpoint",checkpoint), ("base_model",model_path), ("data_root",data_root)):
        if Path(summary[key]).resolve() != Path(value).resolve():
            raise ValueError(f"Validation binding differs: {directory}: {key}")
    ids = {str(row["question_id"]) for row in rows}
    if len(rows) != 6259 or len(predictions) != 6259 or len(ids) != 6259 or ids != {
            str(row["question_id"]) for row in predictions} or len({str(row["image_id"]) for row in rows}) != 832:
        raise ValueError(f"Incomplete Validation: {directory}")
    if abs(100*sum(row["correct"] for row in rows)/6259-summary["overall_accuracy"]) > .00011:
        raise ValueError(f"Stored Validation score differs: {directory}")
    return summary


def aggregate(summaries):
    values = {key:[row[key] for row in summaries] for key in
              ("overall_accuracy", "yes_no_accuracy", "free_form_accuracy")}
    for kind in sorted(set.intersection(*(set(row["per_question_type_accuracy"]) for row in summaries))):
        values[f"question_type:{kind}"] = [row["per_question_type_accuracy"][kind] for row in summaries]
    return {key:{"values":items, "mean":statistics.mean(items),
                 "sample_std":statistics.stdev(items), "ddof":1} for key,items in values.items()}


def shutdown_worker(output):
    """AutoDL shutdown may ignore delay arguments: wait before calling it."""
    path = output/"shutdown_status.json"
    state = read(path)
    time.sleep(max(0., state["scheduled_unix"]-time.time()))
    state.update(status="calling", called_at=datetime.now().isoformat())
    write(path,state)
    try:
        result = subprocess.run(["/usr/bin/shutdown"], capture_output=True, text=True)
        state.update(status="called" if result.returncode == 0 else "call_failed",
                     exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except Exception as exc:
        state.update(status="call_failed", error=str(exc))
    write(path,state)


def schedule_shutdown(output):
    state = {"status":"scheduled", "delay_seconds":600, "scheduled_unix":time.time()+600}
    write(output/"shutdown_status.json",state)
    try:
        with (output/"shutdown.log").open("a",encoding="utf-8") as log:
            worker = subprocess.Popen([sys.executable,"-m","diagnostics.run_pathvqa_v10_seeds",
                "--shutdown-worker",str(output)], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state["worker_pid"] = worker.pid
    except Exception as exc:
        state.update(status="schedule_failed",error=str(exc))
    write(output/"shutdown_status.json",state)
    print("[V10_SHUTDOWN] " + json.dumps(state),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",type=Path,default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root",type=Path,default=Path("/root/autodl-tmp/dataset/pathVQA"))
    parser.add_argument("--shutdown-after",action="store_true")
    parser.add_argument("--shutdown-worker",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shutdown_worker:
        shutdown_worker(args.shutdown_worker)
        return 0
    output = ROOT/"pathvqa/outputs/v10"/("suite_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True,exist_ok=False)
    state = {"status":"started", "exit_code":None, "completed_seeds":[],
             "not_run_seeds":list(SEEDS), "stage_exits":[], "shutdown_after":args.shutdown_after,
             "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    def save():
        write(output/"suite_status.json",state)
    def command(stage, tokens, directory):
        state.update(stage=stage)
        save()
        with (directory/(stage+".log")).open("w",encoding="utf-8") as log:
            process = subprocess.Popen(list(map(str,tokens)),cwd=ROOT,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        state["stage_exits"].append({"seed":state["current_seed"],"stage":stage,"exit_code":code})
        save()
        if code:
            raise subprocess.CalledProcessError(code,tokens)
    save()
    print("[PATHVQA_V10_SUITE_OUTPUT] " + str(output),flush=True)
    summaries, results = [], []
    try:
        for seed in SEEDS:
            state.update(current_seed=seed,stage="baseline_binding")
            save()
            baseline = ROOT/"pathvqa/outputs/visual_selection_prefix"/V1_RUNS[seed]
            binding = audit_v1(baseline,seed)
            validation(baseline/"eval_validation/epoch_5",baseline/"checkpoints/epoch_5",
                       "visual-selection-prefix",args.model_path,args.data_root)
            directory = output/experiment(seed)
            directory.mkdir()
            write(directory/"baseline_binding.json",binding)
            state["not_run_seeds"].remove(seed)
            state["current_output"] = str(directory)
            command("train",[sys.executable,"-m","slake.train_v10","--dataset","pathvqa",
                "--model-seed",seed,"--model-path",args.model_path,"--data-root",args.data_root,
                "--output-dir",directory,"--expected-train-count",19654],directory)
            report = read(directory/"train_report.json")
            for key,value in {"experiment":experiment(seed),"method":METHOD,"dataset":"PathVQA",
                    "model_seed":seed,"data_seed":42,"epochs":5,"saved_epochs":[3,4,5],
                    "trainable_parameters":EXPECTED_GROUP_COUNTS,"trainer_model_accepts_loss_kwargs":False}.items():
                if report[key] != value:
                    raise ValueError(f"V10 training report differs: {key}")
            checkpoint = directory/"checkpoints/epoch_5"
            config = read(checkpoint/CONFIG_NAME)
            if config["init_seed"] != seed or config["method"] != METHOD or not (checkpoint/WEIGHTS_NAME).is_file():
                raise ValueError("V10 checkpoint binding differs")
            evaluation = directory/"eval_validation/epoch_5"
            command("eval_validation_epoch5",[sys.executable,"-m","pathvqa.pathvqa_official_eval",
                "--backend","v10-weighted-map","--base-model",args.model_path,"--checkpoint",checkpoint,
                "--data-root",args.data_root,"--split","validation","--max-new-tokens",32,
                "--temperature",0,"--answer-mode","raw","--output-dir",evaluation],directory)
            summary = validation(evaluation,checkpoint,"v10-weighted-map",args.model_path,args.data_root)
            command("paired_vs_v1",[sys.executable,"-m","diagnostics.compare_pathvqa_v1_training_budget",
                "--baseline-eval",baseline/"eval_validation/epoch_5","--variant-eval",evaluation,
                "--experiment",experiment(seed),"--baseline",f"pathvqa_v1_norm_fixed_5ep_seed{seed}",
                "--all-question-types","--output",directory/"paired_vs_v1.json"],directory)
            result = {"seed":seed,"experiment":experiment(seed),"output":str(directory),
                      "summary":summary,"train_report":report,"paired":read(directory/"paired_vs_v1.json")}
            results.append(result)
            summaries.append(summary)
            write(output/"seed_results.json",results)
            with (output/"ledger_fragment.md").open("a",encoding="utf-8") as handle:
                handle.write(f"\n### {experiment(seed)}\nPathVQA44/45/46 individually, data42; fixed epoch5 Validation. "
                    f"commit={state['git_commit']}; output={directory}\n"+json.dumps(result,ensure_ascii=False)+"\n")
            state["completed_seeds"].append(seed)
            save()
        write(output/"three_seed_summary.json",{"seeds":list(SEEDS),"fixed_epoch":5,
              "dataset":"PathVQA","split":"validation","statistics":aggregate(summaries)})
        with (output/"ledger_fragment.md").open("a",encoding="utf-8") as handle:
            handle.write("\nThree-seed mean ± sample standard deviation (ddof=1):\n"+
                         json.dumps(aggregate(summaries),ensure_ascii=False)+"\n")
        state.update(status="completed",stage="completed",exit_code=0)
    except Exception as exc:
        code = exc.returncode if isinstance(exc,subprocess.CalledProcessError) else 1
        state.update(status="failed",exit_code=code,failed_seed=state.get("current_seed"),error=str(exc))
        state["not_run_seeds"] = [seed for seed in SEEDS if seed not in state["completed_seeds"]
                                  and seed != state.get("failed_seed")]
        with (output/"ledger_fragment.md").open("a",encoding="utf-8") as handle:
            handle.write("\nFAILED: "+json.dumps(state,ensure_ascii=False)+"\n")
    finally:
        save()
        write(output/"final_report.json",{"state":state,"results":results})
        print("[PATHVQA_V10_SUITE_DONE] "+json.dumps(state,ensure_ascii=False),flush=True)
        if args.shutdown_after:
            schedule_shutdown(output)
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
