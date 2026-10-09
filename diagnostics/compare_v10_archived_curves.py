"""CPU-only comparison of two explicitly supplied archived run directories."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from plot_v10_training_logs import figure, describe


def load(root):
    report = json.loads((root/'train_report.json').read_text(encoding='utf-8'))
    state = json.loads((root/'trainer_state.json').read_text(encoding='utf-8'))
    diag = [json.loads(s) for s in (root/'v10_diagnostics.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    logs = [r for r in state['log_history'] if 'loss' in r]
    rates = report['optimizer']['group_learning_rates']
    text = (root/'train.log').read_text(encoding='utf-8', errors='replace')
    archived_rates = json.loads(re.findall(r'\[V10_OPTIMIZER\] (\{[^\n]+\})', text)[0])
    assert archived_rates == rates and next(iter(archived_rates)) == 'p20'
    total, warmup = state['max_steps'], math.ceil(state['max_steps'] * report['optimizer']['warmup_ratio'])
    error = max(abs(float(r['learning_rate']) / rates['p20'] -
                    ((r['step']-1)/warmup if r['step']-1 < warmup else
                     max(0, (total-r['step']+1)/(total-warmup)))) for r in logs)
    assert error < 1e-8
    summaries = {int(p.stem.replace('eval_epoch','')): json.loads(p.read_text(encoding='utf-8'))
                 for p in root.glob('eval_epoch*.json')}
    return dict(report=report, state=state, diag=diag, logs=logs, summaries=summaries,
                total=total, warmup=warmup, lr_error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    runs = {name: load(root/name) for name in ('five','seven')}
    five, seven = runs.values()
    differences = {k: [five['report'].get(k),seven['report'].get(k)] for k in five['report']
                   if five['report'].get(k) != seven['report'].get(k)}
    summary_protocol_keys = ('count','backend','base_model','v0_intervention','question_source_policy',
                            'data_root','split','max_new_tokens','temperature','answer_mode','instruction','partial_evaluation')
    protocol = {k: [five['summaries'][5].get(k),seven['summaries'][5].get(k)] for k in summary_protocol_keys}
    assert all(a == b for a,b in protocol.values())
    metrics = ['summary_pre_ln_rms','summary_post_ln_rms','offset_rms','p20_rms','offset_to_p20_rms',
               'fused_map_entropy_norm','summary_rms','native_value_rms','layer5_weight','layer11_weight','layer17_weight']
    def series(key, source='diag', group=None, cutoff=None, percent=False):
        result = []
        for name, run in runs.items():
            rows = run[source]
            xs = [r['step']+(source == 'diag') for r in rows]
            if group:
                ys = [r['group_gradient_norms'][group] for r in rows]
            elif key.startswith('lr:'):
                ys = [float(r['learning_rate'])/run['report']['optimizer']['group_learning_rates']['p20'] *
                      run['report']['optimizer']['group_learning_rates'][key[3:]] for r in rows]
            else:
                ys = [float(r[key]) for r in rows]
            pairs = [(x,y) for x,y in zip(xs,ys) if cutoff is None or x <= cutoff]
            result.append((name, [100*x/run['total'] if percent else x for x,y in pairs], [y for x,y in pairs]))
        return result
    figures = []
    def emit(file, title, panels, cutoff=None, percent=False):
        figure(root/file,title,panels,
               'Blue=five epochs; orange=seven. X=optimizer update'+(' (% of each budget)' if percent else '')+
               '. Diag raw k -> update k+1; changing minibatches.',
               100 if percent else cutoff or seven['total'],0)
        figures.append(file)
    emit('01_loss_lr.png','Loss and reconstructed group LR (same optimizer steps)',[
        ('Training loss: 20-update averages',series('loss','logs'),{}),
        ('P20 LR (reconstructed)',series('lr:p20','logs'),{}),
        ('Meta-Net LR (reconstructed)',series('lr:meta_net','logs'),{}),
        ('Other branch LR 1e-4 (reconstructed)',series('lr:maps','logs'),{})])
    emit('02_first_five.png','Focus: shared first five epochs / 3075 updates',[
        ('Loss after step 140',[(n,x,[y if s>140 else float('nan') for s,y in zip(x,ys)])
                              for n,x,ys in series('loss','logs',cutoff=five['total'])],{}),
        ('Effective fused-grid entropy',series('fused_map_entropy_norm',cutoff=five['total']),{}),
        ('Offset / P20 RMS',[(n,x,[y if s>140 else float('nan') for s,y in zip(x,ys)])
                             for n,x,ys in series('offset_to_p20_rms',cutoff=five['total'])],{}),
        ('Preclip total gradient (Trainer)',series('grad_norm','logs',cutoff=five['total']),{'log':True})],five['total'])
    emit('03_gradients.png','Separate gradient conventions',[
        ('Preclip total (sampled every 20)',series('grad_norm','logs'),{'log':True}),
        ('Postclip Meta-Net',series('',group='meta_net'),{'log':True}),
        ('Postclip maps',series('',group='maps'),{'log':True}),
        ('Postclip P20',series('',group='p20'),{'log':True})])
    groups = list(five['report']['optimizer']['group_learning_rates'])
    for chunk in range(2):
        emit(f'03_groups_{chunk+1}.png','All groups: postclip norms only',[
            (g,series('',group=g),{'log':True}) for g in groups[chunk*4:chunk*4+4]])
    emit('01_other_lrs.png','Remaining groups: reconstructed LR, not independent logs',[
        (g,series('lr:'+g,'logs'),{}) for g in ('visual_s8','visual_av10','question_context','summary_norm')])
    emit('04_scales.png','P20 / offset / summary scales',[
        (key,series(key),{}) for key in ('p20_rms','offset_rms','summary_pre_ln_rms','summary_post_ln_rms')])
    emit('05_maps.png','Effective map and three layer weights',[
        (key,series(key),{}) for key in ('fused_map_entropy_norm','layer5_weight','layer11_weight','layer17_weight')])
    emit('06_progress.png','Normalized budget progress (not the same samples / epochs)',[
        (key,series(key,'logs' if key=='loss' else 'diag',percent=True),{})
        for key in ('loss','offset_to_p20_rms','summary_rms','fused_map_entropy_norm')],percent=True)
    # Multiply epoch by 615 to share the training-step x axis with sparse saved Validation.
    scores = {}
    eval_series = {}
    for key in ('overall_accuracy','yes_no_accuracy','free_form_accuracy','where'):
        eval_series[key] = []
        for name,run in runs.items():
            points = sorted(run['summaries'].items())
            vals = [s['per_question_type_accuracy'][key] if key=='where' else s[key] for e,s in points]
            eval_series[key].append((name,[e*run['total']/run['report']['epochs'] for e,s in points],vals))
    emit('07_validation.png','Archived Validation: sparse saved epochs only',[
        (key,eval_series[key],{}) for key in eval_series])
    earliest = {}
    for key in metrics:
        pairs = [(a,b) for a,b in zip(five['diag'],seven['diag']) if a['step']==b['step']]
        found = next(((a,b) for a,b in pairs if a[key] != b[key]),None)
        earliest[key] = None if found is None else dict(update=found[0]['step']+1,
                            five=found[0][key],seven=found[1][key],difference=found[1][key]-found[0][key])
    for name,run in runs.items():
        epoch_stats = {}
        per_epoch = run['total']/run['report']['epochs']
        for epoch in range(1,run['report']['epochs']+1):
            logs = [r for r in run['logs'] if (epoch-1)*per_epoch < r['step'] <= epoch*per_epoch]
            diag = [r for r in run['diag'] if (epoch-1)*per_epoch < r['step']+1 <= epoch*per_epoch]
            grads = [float(r['grad_norm']) for r in logs]
            epoch_stats[epoch] = {'loss':describe([float(r['loss']) for r in logs]),
                'preclip_gradient':describe(grads),'sampled_clipped_count':sum(g>1 for g in grads),
                **{key:describe([r[key] for r in diag]) for key in metrics},
                'postclip_group_gradients':{g:describe([r['group_gradient_norms'][g] for r in diag])
                                            for g in run['report']['optimizer']['group_learning_rates']}}
        scores[name] = dict(experiment=run['report']['experiment'],commit=run['report']['git_commit'],
                           max_steps=run['total'],warmup=run['warmup'],epoch_stats=epoch_stats,
                           validation=run['summaries'],lr_reconstruction_error=run['lr_error'],
                           sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in (root/name).iterdir() if p.is_file()})
    initial_gradient_relative = {g:seven['diag'][0]['group_gradient_norms'][g]/five['diag'][0]['group_gradient_norms'][g]-1
                                for g in groups}
    result = dict(runs=scores,report_differences=differences,validation_protocol=protocol,
                  initial_forward_statistics_identical=all(five['diag'][0][k]==seven['diag'][0][k] for k in metrics),
                  initial_gradient_relative_differences=initial_gradient_relative,
                  initial_diagnostic_identical=five['diag'][0]==seven['diag'][0],earliest_logged_divergence=earliest,
                  missing=['per-step complete clipping history','direct per-group LR time series',
                           'fixed-sample map/summary vectors and representation changes','batch sample IDs / shuffle trace',
                           'per-layer merged-grid entropy'],
                  conventions={'lr':'reconstructed using recorded P20 LR / base LR; scheduler step-1 verified',
                               'trainer_grad':'preclip total','callback_grad':'postclip groups; raw step k is update k+1',
                               'loss':'20-update averages; epoch summaries average these windows, not exact epoch CE'})
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'index.html').write_text("<!doctype html><meta charset='utf-8'><h1>V10 5/7 epoch CPU comparison</h1>"+
        ''.join(f"<img src='{p}' style='max-width:100%;display:block'>" for p in figures),encoding='utf-8')
    compact = {n:{e:{k:round(v[k]['mean'],6) for k in ('loss','fused_map_entropy_norm','offset_to_p20_rms',
                    'summary_pre_ln_rms','offset_rms','p20_rms')} for e,v in r['epoch_stats'].items()} for n,r in scores.items()}
    print(json.dumps({'differences':list(differences),'initial_identical':result['initial_diagnostic_identical'],
                      'epochs':compact},indent=2))


if __name__ == '__main__':
    main()
