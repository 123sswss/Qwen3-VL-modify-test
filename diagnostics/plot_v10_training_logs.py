"""CPU/Pillow plots over archived logs only. No model, torch, or CUDA imports."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = ["#2563eb","#ea580c","#16a34a","#9333ea","#0891b2","#db2777","#854d0e","#475569"]
FONT_PATH = "C:/Windows/Fonts/arial.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH if Path(FONT_PATH).exists() else "DejaVuSans.ttf",size)
    except OSError:
        return ImageFont.load_default()


def panel(draw, box, title, series, *, total_steps, warmup, log=False, limits=None):
    left,top,width,height = box
    draw.text((left,top),title,font=font(23),fill="#111827")
    x0,y0,x1,y1 = left+83,top+60,left+width-30,top+height-115
    finite = [v for _,xs,ys in series for v in ys if math.isfinite(v) and (not log or v>0)]
    low,high = limits if limits else (min(finite),max(finite))
    if log:
        low,high = math.log10(max(low,1e-12)),math.log10(high)
    margin = (high-low)*.07 if high>low else .1
    low,high = low-margin,high+margin
    def xx(x): return x0+(x1-x0)*x/total_steps
    def yy(y):
        value = math.log10(max(y,1e-12)) if log else y
        return y1-(y1-y0)*(value-low)/(high-low)
    draw.rectangle((x0,y0,xx(warmup),y1),fill="#fff7ed")
    for tick in np.linspace(low,high,5):
        y = y1-(y1-y0)*(tick-low)/(high-low)
        draw.line((x0,y,x1,y),fill="#e5e7eb",width=1)
        value = 10**tick if log else tick
        draw.text((left+4,y-9),f"{value:.2g}",font=font(17),fill="#4b5563")
    for tick in np.linspace(0,total_steps,6).round().astype(int):
        x = xx(tick)
        draw.line((x,y0,x,y1),fill="#e5e7eb",width=1)
        draw.text((x-21,y1+8),str(tick),font=font(17),fill="#4b5563")
    draw.rectangle((x0,y0,x1,y1),outline="#9ca3af",width=1)
    for index,(label,xs,ys) in enumerate(series):
        points = [(xx(x),yy(y)) for x,y in zip(xs,ys) if math.isfinite(y) and (not log or y>0)]
        if len(points)>1:
            draw.line(points,fill=COLORS[index%len(COLORS)],width=3)
        lx,ly = x0+(index%3)*((x1-x0)/3),y1+38+(index//3)*22
        draw.line((lx,ly+8,lx+20,ly+8),fill=COLORS[index%len(COLORS)],width=3)
        draw.text((lx+25,ly),label,font=font(15),fill="#374151")


def figure(path, title, panels, note, total_steps, warmup):
    image = Image.new("RGB",(1800,1400),"white")
    draw = ImageDraw.Draw(image)
    draw.text((45,22),title,font=font(30),fill="#111827")
    for i,(name,series,kwargs) in enumerate(panels):
        panel(draw,(35+(i%2)*880,85+(i//2)*605,855,575),name,series,
              total_steps=total_steps,warmup=warmup,**kwargs)
    draw.text((45,1320),note,font=font(17),fill="#475569")
    image.save(path)


def describe(items):
    values = np.array(items,dtype=float)
    return {"count":len(items),"first":float(values[0]),"last":float(values[-1]),
            "mean":float(values.mean()),"median":float(np.median(values)),
            "p10":float(np.quantile(values,.1)),"p90":float(np.quantile(values,.9)),
            "min":float(values.min()),"max":float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True)
    args = parser.parse_args()
    root = args.input_dir
    report = json.loads((root/"train_report.json").read_text(encoding="utf-8"))
    state_path = root/"trainer_state.json"
    if not state_path.exists():
        state_path = root/"trainer/trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    diagnostics = [json.loads(line) for line in (root/"v10_diagnostics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    logs = [row for row in state["log_history"] if "loss" in row]
    steps = [row["step"] for row in logs]
    dsteps = [row["step"]+1 for row in diagnostics]
    epochs = [row["epoch"] for row in logs]
    losses = [float(row["loss"]) for row in logs]
    gradients = [float(row["grad_norm"]) for row in logs]
    def ds(key): return [float(row[key]) for row in diagnostics]
    rates = report["optimizer"]["group_learning_rates"]
    optimizer_lines = re.findall(r"\[V10_OPTIMIZER\] (\{[^\n]+\})",(root/"train.log").read_text(encoding="utf-8",errors="replace"))
    observed_rates = json.loads(optimizer_lines[0])
    if observed_rates != rates or next(iter(observed_rates)) != "p20":
        raise ValueError("Archived optimizer groups disagree with report or first-group P20 LR assumption")
    # Recorded Trainer LR is the FIRST optimizer group (p20), not Meta-Net.
    factors = [float(row["learning_rate"])/rates["p20"] for row in logs]
    actual_lrs = {key:[factor*base for factor in factors] for key,base in rates.items()}
    warmup = math.ceil(state["max_steps"]*report["optimizer"]["warmup_ratio"])
    total_steps = state["max_steps"]
    total_epochs = int(report["epochs"])
    expected_factors = [(step-1)/warmup if step-1<warmup else
                        max(0.,(state["max_steps"]-(step-1))/(state["max_steps"]-warmup)) for step in steps]
    scheduler_error = max(abs(a-b) for a,b in zip(factors,expected_factors))
    if scheduler_error>1e-8:
        raise ValueError("Recorded LR disagrees with saved linear scheduler; cannot reconstruct group rates")
    smooth = [statistics.median(losses[max(0,i-2):i+3]) for i in range(len(losses))]
    groups = list(rates)
    post = {name:[row["group_gradient_norms"][name] for row in diagnostics] for name in groups}
    post_total = [math.sqrt(sum(row["group_gradient_norms"][name]**2 for name in groups)) for row in diagnostics]
    figures = []
    def emit(name,title,panels,note):
        file = root/name
        figure(file,title,panels,note,total_steps,warmup)
        figures.append(name)
    emit("01_loss_lr.png","1. Training loss and group learning rates",[
        ("Loss (20-update averages)",[("raw",steps,losses),("5-point median",steps,smooth)],{}),
        ("Loss after warmup (zoom)",[("raw",steps,[v if s>warmup else float('nan') for s,v in zip(steps,losses)]),
            ("5-point median",steps,[v if s>warmup else float('nan') for s,v in zip(steps,smooth)])],{}),
        ("Per-group LR (log scale; reconstructed)",[(name,steps,actual_lrs[name]) for name in groups],{"log":True}),
        ("Shared LR factor; P20 logged / base 0.3",[("scheduler factor",steps,factors)],{})],
        f"LRs reconstructed from archived group bases + recorded common scheduler factor. Orange: {warmup}-step warmup.")
    emit("02_gradients.png","2. Gradients: preclip and postclip are separate",[
        ("Preclip total norm (Trainer log)",[("preclip total",steps,gradients),("clip threshold 1",[0,total_steps],[1,1])],{"log":True}),
        ("Postclip group norms (callback)",[(name,dsteps,post[name]) for name in groups],{"log":True}),
        ("Global clipping factor at logged updates",[("min(1, 1 / norm)",steps,[min(1.,1/g) for g in gradients])],{}),
        ("Postclip total from group-norm squares",[("postclip total",dsteps,post_total)],{})],
        "Trainer samples every 20 updates, NOT every update. Callback raw step k refers to update k+1; no pointwise alignment.")
    emit("03_scales.png","3. Summary LayerNorm and conditional offset scales",[
        ("Summary RMS before / after LN",[("pre-LN",dsteps,ds("summary_pre_ln_rms")),
                                          ("post-LN",dsteps,ds("summary_post_ln_rms"))],{}),
        ("Offset and P20 RMS",[(key,dsteps,ds(key)) for key in ("offset_rms","p20_rms")],{}),
        ("Offset / P20 RMS",[("offset_to_p20_rms",dsteps,ds("offset_to_p20_rms"))],{}),
        ("Offset / P20 after warmup (zoom)",[("offset / P20",dsteps,
            [v if step>warmup else float('nan') for step,v in zip(dsteps,ds("offset_to_p20_rms"))])],{})],
        "Values are batch summaries on changing training samples; RMS alone does not show representation direction or information.")
    emit("04_maps_summary.png","4. Effective merged map, layer gates and Value summary",[
        ("Effective merged-grid map entropy",[("fused_map_entropy_norm",dsteps,ds("fused_map_entropy_norm"))],{}),
        ("Fine-patch entropy (secondary only)",[(f"layer{layer}",dsteps,ds(f"map{layer}_entropy_norm")) for layer in (5,11,17)],{}),
        ("Problem-conditioned layer weights",[(f"layer{layer}",dsteps,ds(f"layer{layer}_weight")) for layer in (5,11,17)],{}),
        ("Effective Value summary / native Value RMS",[("summary RMS",dsteps,ds("summary_rms")),
            ("native Value RMS",dsteps,ds("native_value_rms"))],{})],
        "Merged per-layer maps / summary vectors were NOT archived: no fixed-sample map changes, cosine, or localization claim.")
    metrics = ("summary_pre_ln_rms","summary_post_ln_rms","offset_rms","p20_rms","offset_to_p20_rms",
               "fused_map_entropy_norm","summary_rms","native_value_rms",
               "layer5_weight","layer11_weight","layer17_weight")
    epoch_loss = {}
    for epoch in range(1,total_epochs+1):
        vals = [v for e,v in zip(epochs,losses) if epoch-1<e<=epoch]
        epoch_loss[str(epoch)] = describe(vals)
    after = [g for s,g in zip(steps,gradients) if s>warmup]
    final_first = [v for e,v in zip(epochs,losses) if total_epochs-1<e<=total_epochs-.5]
    final_second = [v for e,v in zip(epochs,losses) if total_epochs-.5<e<=total_epochs]
    analysis = {"experiment":report["experiment"],"commit":report["git_commit"],
        "global_step":state["global_step"],"trainable_parameters":report["total_trainable_parameters"],
        "train_runtime_seconds":report["train_metrics"]["train_runtime"],
        "peak_gpu_memory_bytes_recorded":report["peak_gpu_memory_bytes"],
        "sources_sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                           (("train_report.json",root/"train_report.json"),("trainer_state.json",state_path),
                            ("v10_diagnostics.jsonl",root/"v10_diagnostics.jsonl"),("train.log",root/"train.log"))},
        "warmup_steps":warmup,"lr_reconstruction_error":scheduler_error,"base_group_lrs":rates,
        "optimizer_log_report_match":True,
        "lr_note":"reconstructed actual group rates; recorded LR is p20; observed factor corresponds to step-1",
        "epoch_loss":epoch_loss,"final_epoch_first_half_loss_mean":statistics.mean(final_first),
        "final_epoch_second_half_loss_mean":statistics.mean(final_second),
        "post_warmup_sampled_preclip_gradient":describe(after),
        "post_warmup_sampled_clip_fraction":sum(g>1 for g in after)/len(after),
        "postclip_total":describe(post_total),"postclip_groups":{key:describe(vals) for key,vals in post.items()},
        "diagnostics":{key:describe(ds(key)) for key in metrics},
        "missing":["per-layer merged-grid map entropy","saved summary vector directions / fixed-sample change",
                   "direct per-step per-group LR logs","per-step complete clipping history"],
        "timing_note":"diagnostic callback raw step k is update k+1; changing minibatches; not paired fixed-sample probes"}
    write_text = json.dumps(analysis,ensure_ascii=False,indent=2)
    (root/"curve_analysis.json").write_text(write_text,encoding="utf-8")
    with (root/"reconstructed_group_lrs.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["optimizer_step","epoch","recorded_p20_lr","scheduler_factor",*groups])
        for i,step in enumerate(steps):
            writer.writerow([step,epochs[i],logs[i]["learning_rate"],factors[i],*[actual_lrs[key][i] for key in groups]])
    (root/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>V10 CPU curve audit</title>"+
        "<h1>Archived V10 head-fixed training logs</h1><p>No model execution. LR reconstructed; gradients distinguished.</p>"+
        "".join(f"<img src='{name}' style='max-width:100%;display:block;margin:24px 0'>" for name in figures),encoding="utf-8")
    print(write_text)


if __name__ == "__main__":
    main()
