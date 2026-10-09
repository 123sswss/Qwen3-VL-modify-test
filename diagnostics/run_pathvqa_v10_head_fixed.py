"""One user-launched seed44 five-epoch Validation; no retry or shutdown."""
import argparse
from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys

from diagnostics.v10_protocol import METHOD as ORIGINAL_METHOD, CONFIG_NAME as ORIGINAL_CONFIG
from diagnostics.v10_head_fixed_protocol import EXPERIMENT, METHOD, GROUP_COUNTS, GROUP_LRS, TOTAL
from diagnostics.run_pathvqa_v10_seeds import ROOT, V1_RUNS, read, write, validation
from diagnostics.run_five_task_test_suite import audit_v1


def bind_original(run, model_path, data_root):
    report = read(run/"train_report.json")
    config = read(run/"checkpoints/epoch_5"/ORIGINAL_CONFIG)
    expected = {"experiment":"pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44",
                "method":ORIGINAL_METHOD,"dataset":"PathVQA","model_seed":44,
                "data_seed":42,"epochs":5,"total_trainable_parameters":1685923,
                "trainer_model_accepts_loss_kwargs":False}
    if any(report[key] != value for key,value in expected.items()) or config["init_seed"] != 44:
        raise ValueError(f"Not the original V10 seed44 five-epoch run: {run}")
    summary = validation(run/"eval_validation/epoch_5",run/"checkpoints/epoch_5",
                         "v10-weighted-map",model_path,data_root)
    if abs(summary["overall_accuracy"]-57.7089) > .00011:
        raise ValueError("Not the specified V10 57.7089 reference")
    return {"run":str(run),"report":report,"summary":summary}


def main(*, seven_epochs=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",type=Path,default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root",type=Path,default=Path("/root/autodl-tmp/dataset/pathVQA"))
    parser.add_argument("--original-v10-run",type=Path,help="Optional explicit original seed44 directory")
    args = parser.parse_args()
    budget = 7 if seven_epochs else 5
    experiment = EXPERIMENT.replace("_5ep_","_7ep_") if seven_epochs else EXPERIMENT
    group_lrs = GROUP_LRS
    saved_epochs = [5,6,7] if seven_epochs else [3,4,5]
    evaluation_epochs = [5,6,7] if seven_epochs else [5]
    output = ROOT/"pathvqa/outputs/v10_head_fixed"/(experiment+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True,exist_ok=False)
    state = {"experiment":experiment,"status":"started","stage":"baseline_binding","stage_exits":[],
             "output":str(output),"shutdown":False,
             "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    def save():
        write(output/"stage_status.json",state)
    def command(stage,tokens):
        state["stage"] = stage
        save()
        with (output/(stage+".log")).open("w",encoding="utf-8") as log:
            process = subprocess.Popen(list(map(str,tokens)),cwd=ROOT,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        state["stage_exits"].append({"stage":stage,"exit_code":code})
        save()
        if code:
            raise subprocess.CalledProcessError(code,tokens)
    save()
    print("[PATHVQA_V10_HEAD_FIXED_OUTPUT] "+str(output),flush=True)
    try:
        if seven_epochs:
            run = ROOT/"pathvqa/outputs/v10_head_fixed"/(EXPERIMENT+"_20261009_102158_643548")
            report = read(run/"train_report.json")
            expected = {"experiment":EXPERIMENT,"method":METHOD,"epochs":5,"model_seed":44,
                        "data_seed":42,"total_trainable_parameters":TOTAL,"trainer_model_accepts_loss_kwargs":False}
            if any(report[key] != value for key,value in expected.items()):
                raise ValueError("Not the completed five-epoch head-fixed reference")
            summary = validation(run/"eval_validation/epoch_5",run/"checkpoints/epoch_5",
                                 "v10-head-fixed",args.model_path,args.data_root)
            if abs(summary["overall_accuracy"]-58.6835) > .00011:
                raise ValueError("Not the specified 58.6835 reference")
            original = {"run":str(run),"report":report,"summary":summary}
        elif args.original_v10_run:
            original = bind_original(args.original_v10_run.resolve(),args.model_path,args.data_root)
        else:
            # Identity/score, never mtime/latest-directory selection.
            matches = []
            for run in (ROOT/"pathvqa/outputs/v10").glob(
                    "suite_*/pathvqa_v10_weighted_map_metanet_h160_norm_fixed_5ep_seed44"):
                if (run/"train_report.json").is_file() and (run/"eval_validation/epoch_5/pathvqa_summary.json").is_file():
                    matches.append(bind_original(run,args.model_path,args.data_root))
            if len(matches) != 1:
                raise ValueError(f"Require one verified V10 57.7089 run, found {len(matches)}; use --original-v10-run")
            original = matches[0]
        v1 = ROOT/"pathvqa/outputs/visual_selection_prefix"/V1_RUNS[44]
        binding = audit_v1(v1,44)
        v1_summary = validation(v1/"eval_validation/epoch_5",v1/"checkpoints/epoch_5",
                                "visual-selection-prefix",args.model_path,args.data_root)
        if abs(v1_summary["overall_accuracy"]-59.3386) > .00011:
            raise ValueError("Not the specified V1 59.3386 reference")
        write(output/"baseline_binding.json",{"head_fixed_5ep" if seven_epochs else "v10":original,"v1":binding})
        write(output/"requested_config.json",{"experiment":experiment,"method":METHOD,"model_seed":44,
            "data_seed":42,"epochs":budget,"fixed_epoch":budget,"saved_epochs":saved_epochs,
            "evaluation_epochs":evaluation_epochs,"batch":2,"accumulation":16,
            "group_counts":GROUP_COUNTS,"group_lrs":group_lrs,"total":TOTAL,
            "summary_norm":"LayerNorm(2560)","meta_init":"nn.Linear_default_both_layers",
            "git_commit":state["git_commit"],"gradient_preclip":"real_batch_preflight_and_Trainer_grad_norm",
            "gradient_postclip":"on_pre_optimizer_step_group_norms"})
        train_module = "pathvqa.train_v10_head_fixed_7ep" if seven_epochs else "pathvqa.train_v10_head_fixed"
        command("train",[sys.executable,"-m",train_module,"--dataset","pathvqa",
            "--model-seed",44,"--model-path",args.model_path,"--data-root",args.data_root,
            "--output-dir",output,"--expected-train-count",19654,
            "--epochs",budget,"--save-epochs",*saved_epochs])
        report = read(output/"train_report.json")
        if report["method"] != METHOD or report["total_trainable_parameters"] != TOTAL or report["saved_epochs"] != saved_epochs or report["epochs"] != budget:
            raise ValueError("Independent training budget/report differs")
        if report["optimizer"]["group_learning_rates"] != group_lrs:
            raise ValueError("Training optimizer groups differ from requested learning rates")
        if seven_epochs:
            command("plot_curves",[sys.executable,"-m","diagnostics.plot_v10_training_logs","--input-dir",output])
        epoch_results = {}
        for epoch in evaluation_epochs:
            evaluation = output/"eval_validation"/f"epoch_{epoch}"
            checkpoint = output/"checkpoints"/f"epoch_{epoch}"
            command(f"eval_validation_epoch{epoch}",[sys.executable,"-m","pathvqa.pathvqa_official_eval",
                "--backend","v10-head-fixed","--base-model",args.model_path,"--checkpoint",checkpoint,
                "--data-root",args.data_root,"--split","validation","--output-dir",evaluation])
            summary = validation(evaluation,checkpoint,"v10-head-fixed",args.model_path,args.data_root)
            comparisons = {}
            reference_name = "head_fixed_5ep" if seven_epochs else "v10"
            for name,run in ((reference_name,Path(original["run"])),("v1",v1)):
                paired_path = output/f"paired_epoch{epoch}_vs_{name}.json" if seven_epochs else output/f"paired_vs_{name}.json"
                command(f"paired_epoch{epoch}_vs_{name}",[sys.executable,"-m","diagnostics.compare_pathvqa_v1_training_budget",
                    "--baseline-eval",run/"eval_validation/epoch_5","--variant-eval",evaluation,
                    "--experiment",experiment,"--baseline",read(run/"train_report.json")["experiment"],
                    "--output",paired_path])
                comparisons[name] = read(paired_path)
            epoch_results[epoch] = {"summary":summary,"comparisons":comparisons,"primary":epoch==budget}
            write(output/"epoch_scores.json",epoch_results)
        result = {**epoch_results[budget],"primary_epoch":budget,"epoch_results":epoch_results,"train_report":report,
                  "controlled_change":"training budget 5->7 with warmup/linear scheduler to new endpoint" if seven_epochs else
                                      "LN + default Meta-Net initialization + Meta-Net LR3e-4 combined"}
        write(output/"final_report.json",result)
        (output/"ledger_fragment.md").write_text(f"### {experiment}\noutput={output}\n"+
            json.dumps(result,ensure_ascii=False,indent=2)+"\nExploratory subgroup CIs.\n",encoding="utf-8")
        state.update(status="completed",stage="completed",exit_code=0)
        print("[V10_HEAD_FIXED_DONE] "+json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        state.update(status="failed",exit_code=exc.returncode if isinstance(exc,subprocess.CalledProcessError) else 1,error=str(exc))
        (output/"ledger_fragment.md").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        print("[V10_HEAD_FIXED_FAILED] "+str(exc),file=sys.stderr,flush=True)
    finally:
        save()
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
