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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",type=Path,default=Path("/root/autodl-tmp/model"))
    parser.add_argument("--data-root",type=Path,default=Path("/root/autodl-tmp/dataset/pathVQA"))
    parser.add_argument("--original-v10-run",type=Path,help="Optional explicit original seed44 directory")
    args = parser.parse_args()
    output = ROOT/"pathvqa/outputs/v10_head_fixed"/(EXPERIMENT+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True,exist_ok=False)
    state = {"experiment":EXPERIMENT,"status":"started","stage":"baseline_binding","stage_exits":[],
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
        if args.original_v10_run:
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
        write(output/"baseline_binding.json",{"v10":original,"v1":binding})
        write(output/"requested_config.json",{"experiment":EXPERIMENT,"method":METHOD,"model_seed":44,
            "data_seed":42,"epochs":5,"fixed_epoch":5,"batch":2,"accumulation":16,
            "group_counts":GROUP_COUNTS,"group_lrs":GROUP_LRS,"total":TOTAL,
            "summary_norm":"LayerNorm(2560)","meta_init":"nn.Linear_default_both_layers",
            "git_commit":state["git_commit"],"gradient_preclip":"real_batch_preflight_and_Trainer_grad_norm",
            "gradient_postclip":"on_pre_optimizer_step_group_norms"})
        command("train",[sys.executable,"-m","pathvqa.train_v10_head_fixed","--dataset","pathvqa",
            "--model-seed",44,"--model-path",args.model_path,"--data-root",args.data_root,
            "--output-dir",output,"--expected-train-count",19654])
        report = read(output/"train_report.json")
        if report["method"] != METHOD or report["total_trainable_parameters"] != TOTAL or report["saved_epochs"] != [3,4,5]:
            raise ValueError("Independent five-epoch training report differs")
        evaluation = output/"eval_validation/epoch_5"
        command("eval_validation_epoch5",[sys.executable,"-m","pathvqa.pathvqa_official_eval",
            "--backend","v10-head-fixed","--base-model",args.model_path,"--checkpoint",output/"checkpoints/epoch_5",
            "--data-root",args.data_root,"--split","validation","--output-dir",evaluation])
        summary = validation(evaluation,output/"checkpoints/epoch_5","v10-head-fixed",args.model_path,args.data_root)
        comparisons = {}
        for name,run in (("v10",Path(original["run"])),("v1",v1)):
            command("paired_vs_"+name,[sys.executable,"-m","diagnostics.compare_pathvqa_v1_training_budget",
                "--baseline-eval",run/"eval_validation/epoch_5","--variant-eval",evaluation,
                "--experiment",EXPERIMENT,"--baseline",read(run/"train_report.json")["experiment"],
                "--output",output/f"paired_vs_{name}.json"])
            comparisons[name] = read(output/f"paired_vs_{name}.json")
        result = {"summary":summary,"comparisons":comparisons,"train_report":report,
                  "controlled_change":"LN + default Meta-Net initialization + Meta-Net LR3e-4 combined"}
        write(output/"final_report.json",result)
        (output/"ledger_fragment.md").write_text(f"### {EXPERIMENT}\noutput={output}\n"+
            json.dumps(result,ensure_ascii=False,indent=2)+"\nExploratory subgroup CIs; no single-factor attribution.\n",encoding="utf-8")
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
