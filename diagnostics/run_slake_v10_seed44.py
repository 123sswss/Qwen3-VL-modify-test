"""CPU binding/statistics and user-launched V10 train/Test; fail-stop, no shutdown."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from diagnostics.v10_protocol import (
    EXPERIMENT, METHOD, CONFIG_NAME, WEIGHTS_NAME, GROUP_LRS, EXPECTED_GROUP_COUNTS,
    EXPECTED_TRAINABLE, COCOOP_EVAL, COCOOP_TRAIN, V1_RUN,
)
from diagnostics.run_slake_cocoop_seed44 import (
    read, write, require, references, effective_train_records, scored,
    baseline_audit, training_audit as cocoop_training_audit, paired,
)

ROOT = Path(__file__).resolve().parents[1]


def bind_baselines(data_root, model_path):
    refs = references(data_root/"test.json")
    require(len(refs) == 2094, "Official bilingual Test count mismatch")
    raw = references(data_root/"train.json")
    effective, excluded = effective_train_records(raw)
    for row in raw + refs:
        require((data_root/"imgs"/row["img_name"]).is_file(), f"Missing image: {row['img_name']}")
    v1 = baseline_audit(ROOT/V1_RUN, refs, model_path)
    cocoop, _ = scored(ROOT/COCOOP_EVAL/"eval_test/epoch_3", refs)
    require(cocoop["backend"] == "cocoop-style" and cocoop["overall_accuracy"] == 77.03 and
            cocoop["correct"] == 1613, "Not the specified CoCoOp seed44 Test result")
    require(Path(cocoop["base_model"]).resolve() == model_path.resolve(), "CoCoOp backbone differs")
    report, checkpoint = cocoop_training_audit(ROOT/COCOOP_TRAIN, len(effective), data_root)
    require(Path(cocoop["checkpoint"]).resolve() == checkpoint.resolve(), "CoCoOp epoch3 binding differs")
    return refs, effective, {"v1":v1,"cocoop":{"evaluation":str(ROOT/COCOOP_EVAL),
             "training":str(ROOT/COCOOP_TRAIN),"report":report,"summary":cocoop},
             "train_raw_count":len(raw),"train_effective_count":len(effective),"train_excluded":excluded}


def verify_v10_report(run, effective_count, model_path):
    report = read(run/"train_report.json")
    for key, expected in {"experiment":EXPERIMENT,"method":METHOD,"dataset":"SLAKE","model_seed":44,
            "data_seed":42,"epochs":5,"saved_epochs":[3,4,5],"train_samples":effective_count,
            "train_split":"train","languages":"all","max_length":2048,"workers":2,"bf16":True,
            "trainable_parameters":EXPECTED_GROUP_COUNTS,"total_trainable_parameters":EXPECTED_TRAINABLE,
            "trainer_model_accepts_loss_kwargs":False,"accelerator_gradient_accumulation_steps":1}.items():
        require(report[key] == expected, f"V10 training report mismatch: {key}")
    require(Path(report["base_model"]).resolve() == model_path.resolve(), "V10 backbone differs")
    require(report["train_metrics"]["epoch"] == 5, "V10 incomplete five epochs")
    require(report["optimizer"] == {"type":"AdamW","betas":[.9,.999],"eps":1e-8,"weight_decay":0.,
             "per_device_batch_size":2,"gradient_accumulation_steps":16,"warmup_ratio":.03,
             "scheduler":"linear","max_grad_norm":1.,"group_learning_rates":GROUP_LRS}, "V10 optimizer changed")
    for epoch in (3,4,5):
        checkpoint = run/"checkpoints"/f"epoch_{epoch}"
        config = read(checkpoint/CONFIG_NAME)
        require(config["method"] == METHOD and config["init_seed"] == 44 and
                config["trainable_parameters"] == EXPECTED_GROUP_COUNTS, "V10 checkpoint identity differs")
        require((checkpoint/WEIGHTS_NAME).is_file(), f"V10 epoch{epoch} weights missing")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/dataset/slake"))
    parser.add_argument("--precheck-only", action="store_true", help="CPU JSON/file checks only; no model import")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT/"slake/outputs/v10"/("prechecks" if args.precheck_only else "runs")/(EXPERIMENT+"_"+stamp)
    output.mkdir(parents=True, exist_ok=False)
    state = {"experiment":EXPERIMENT,"status":"started","stage":"precheck","output":str(output),
             "precheck_only":args.precheck_only,"shutdown":False,"stage_exits":[]}
    def save():
        write(output/"stage_status.json",state)
    def command(stage, tokens, cpu_only=False):
        state.update(stage=stage,status="running",command=list(map(str,tokens)))
        save()
        with (output/(stage+".log")).open("w",encoding="utf-8") as log:
            environment = os.environ.copy()
            if cpu_only:
                environment["CUDA_VISIBLE_DEVICES"] = ""
            with subprocess.Popen(list(map(str,tokens)),cwd=ROOT,stdout=subprocess.PIPE,env=environment,
                    stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace") as proc:
                for line in proc.stdout:
                    print(line,end="",flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
        state["stage_exits"].append({"stage":stage,"exit_code":code})
        save()
        require(code == 0, f"{stage} failed exit={code}; no retry")
    save()
    print("[SLAKE_V10_OUTPUT] " + str(output),flush=True)
    try:
        refs, effective, binding = bind_baselines(args.data_root,args.model_path)
        write(output/"baseline_binding.json",binding)
        commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        state.update(status="precheck_passed",git_commit=commit,train_samples=len(effective),test_samples=len(refs))
        write(output/"requested_config.json",{"experiment":EXPERIMENT,"method":METHOD,
            "group_counts":EXPECTED_GROUP_COUNTS,"group_lrs":GROUP_LRS,"epochs":5,"fixed_epoch":5,
            "saved_epochs":[3,4,5],"model_seed":44,"data_seed":42,"batch":2,"accumulation":16,
            "model_accepts_loss_kwargs":False,"max_length":2048,"git_commit":commit})
        save()
        if args.precheck_only:
            state["exit_code"] = 0
            save()
            return 0
        command("cpu_tensor_checks",[sys.executable,"-m","unittest","test_visual_selection_v10","-v"],cpu_only=True)
        command("train",[sys.executable,"-m","slake.train_v10","--model-path",args.model_path,
            "--data-root",args.data_root,"--output-dir",output,"--expected-train-count",len(effective)])
        report = verify_v10_report(output,len(effective),args.model_path)
        evaluation = output/"eval_test/epoch_5"
        command("eval_test_epoch_5",[sys.executable,"-m","slake.slake_official_eval",
            "--backend","v10-weighted-map","--base-model",args.model_path,
            "--checkpoint",output/"checkpoints/epoch_5","--questions",args.data_root/"test.json",
            "--image-root",args.data_root/"imgs","--language","all","--expected-split","test",
            "--max-new-tokens",32,"--temperature",0,"--answer-mode","raw","--output-dir",evaluation])
        summary, _ = scored(evaluation,refs)
        require(summary["backend"] == "v10-weighted-map" and
                Path(summary["checkpoint"]).resolve() == (output/"checkpoints/epoch_5").resolve(),
                "V10 Test evaluation checkpoint/backend binding differs")
        state.update(stage="paired_statistics")
        save()
        comparisons = {}
        for name, reference in (("cocoop",ROOT/COCOOP_EVAL/"eval_test/epoch_3"),
                                ("v1",ROOT/V1_RUN/"eval_test/epoch_5")):
            comparison = paired(reference,evaluation,refs)
            comparison["budget_note"] = ("V10 five epochs with Visual18; CoCoOp three epochs without Visual18"
                if name == "cocoop" else "V10 and original V1 both five epochs with original Visual18")
            write(output/f"paired_vs_{name}_seed44.json",comparison)
            comparisons[name] = comparison["groups"]
        report_text = (f"### {EXPERIMENT}\n\nSLAKE44/data42; fixed epoch5 complete bilingual Test2094. "
            f"Scores {json.dumps(summary,ensure_ascii=False)}. Trainable1685923, groups{EXPECTED_GROUP_COUNTS}; "
            f"commit{commit},output{output};train runtime{report['train_metrics'].get('train_runtime')}, "
            f"peak allocated GPU bytes{report['peak_gpu_memory_bytes']},versions{report['runtime_versions']}. "
            f"Paired10000/seed42 img_name clusters:{json.dumps(comparisons,ensure_ascii=False)}. "
            "CoCoOp3epochs/noVisual18 versus V10five/Visual18; subgroup intervals exploratory. "
            "No additional seed, Validation, tuning, retry or shutdown.\n")
        (output/"ledger_fragment.md").write_text(report_text,encoding="utf-8")
        state.update(stage="completed",status="completed",exit_code=0)
        save()
        print("[SLAKE_V10_DONE] " + json.dumps({"summary":summary,"comparisons":comparisons},ensure_ascii=False),flush=True)
        return 0
    except Exception as exc:
        state.update(status="failed",exit_code=1,error=str(exc))
        save()
        (output/"ledger_fragment.md").write_text(f"{EXPERIMENT}: FAILED at {state['stage']}; {exc}; "
            f"output{output}; no retry or shutdown.\n",encoding="utf-8")
        print("[SLAKE_V10_FAILED] " + str(exc),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
