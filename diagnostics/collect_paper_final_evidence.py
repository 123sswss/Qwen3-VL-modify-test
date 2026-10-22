"""CPU-only collection of existing PathVQA final-LR evidence; no model imports."""
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL = {
    44: 'pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44_20261014_092451_34346',
    45: 'pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed45_20261016_024534_10401',
    46: 'pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed46_20261017_104312_1230',
}
BASE = ROOT / 'pathvqa/outputs/visual_selection_prefix'
TEST = ROOT / 'outputs/pathvqa_v1_read_write_three_seed_test_20261017_164339_485798'
AB = {
    'image_only': ['c_image_only', '20261019_203001_1166', '20261021_104337_1478', '20261021_135520_1478'],
    'qmap': ['c_qmap', '20261019_171742_1166', '20261021_170747_1478', '20261021_202052_1478'],
    'no_visual': ['c_no_visual', '20261019_234027_1166', '20261020_160928_1339', '20261020_213525_1339'],
}
COUNTS = dict(full=1864963, image_only=1470339, qmap=1815811, no_visual=1846531)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def csv_write(p, rows):
    if not rows:
        return
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

spec = importlib.util.spec_from_file_location('paper_cpu_metric', ROOT/'pathvqa/pathvqa_vqa_metric.py')
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)

def groups(rows):
    result = {'overall': rows,
              'yes_no': [r for r in rows if r['answer_type']=='yes/no'],
              'free_form': [r for r in rows if r['answer_type']!='yes/no']}
    result.update({'type:'+t: [r for r in rows if r['question_type']==t]
                   for t in sorted({r['question_type'] for r in rows})})
    return {k:v for k,v in result.items() if v}

def stats(rows):
    return {name: dict(questions=len(r), image_clusters=len({x['image_id'] for x in r}),
                      correct=sum(bool(x['correct']) for x in r),
                      accuracy=100*sum(bool(x['correct']) for x in r)/len(r))
            for name, r in groups(rows).items()}

def paired(full, other):
    oi = {r['question_id']: r for r in other}
    assert len(oi)==len(other)==len(full)
    assert set(oi)=={r['question_id'] for r in full}
    for r in full:
        assert all(r[k]==oi[r['question_id']][k] for k in ('image_id','question','reference','answer_type','question_type'))
    output = {}
    for name, rows in groups(full).items():
        images = sorted({r['image_id'] for r in rows})
        ix = {k:i for i,k in enumerate(images)}
        n = np.zeros(len(images), dtype=np.int64)
        d = n.copy()
        fonly = oonly = both = neither = 0
        for r in rows:
            a, b = bool(r['correct']), bool(oi[r['question_id']]['correct'])
            j=ix[r['image_id']]; n[j]+=1; d[j]+=int(a)-int(b)
            fonly+=a and not b; oonly+=b and not a; both+=a and b; neither+=not a and not b
        rng=np.random.default_rng(42)
        samples=[]
        for start in range(0,10000,250):
            draw=rng.integers(0,len(images),size=(min(250,10000-start),len(images)))
            samples.extend((100*d[draw].sum(1)/n[draw].sum(1)).tolist())
        output[name]=dict(questions=len(rows), image_clusters=len(images),
            full_accuracy=100*(both+fonly)/len(rows), other_accuracy=100*(both+oonly)/len(rows),
            delta_full_minus_other=100*(fonly-oonly)/len(rows), full_only_correct=fonly,
            other_only_correct=oonly, both_correct=both, both_wrong=neither,
            ci95=np.quantile(samples,[.025,.975]).tolist(), exploratory=name!='overall')
    return dict(groups=output, direction='full QDPT minus comparator',
                bootstrap=dict(iterations=10000,seed=42,unit='image_id',rng='NumPy PCG64',method='paired percentile; subgroup images sampled separately; question-weighted accuracy'))

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    sources=[]; artifacts={}; warnings=[]
    def load(name, run, evaluation, seed, split, params=None):
        report=read(run/'train_report.json'); summary=read(evaluation/'pathvqa_summary.json')
        rows=read(evaluation/'pathvqa_comparisons.json'); preds=read(evaluation/'pathvqa_predictions.json')
        count, images=(6259,832) if split=='validation' else (6719,858)
        assert summary['split']==split and summary['count']==count and not summary['partial_evaluation']
        assert len(rows)==len(preds)==count and len({r['image_id'] for r in rows})==images
        assert len({r['question_id'] for r in rows})==count
        assert {r['question_id'] for r in rows}=={f'pathvqa:{split}:{i}' for i in range(count)}
        assert report.get('model_seed',report.get('seed'))==seed and report['data_seed']==42 and report['epochs']==5
        assert report['trainer_model_accepts_loss_kwargs'] is False
        assert (ROOT/summary['checkpoint']).resolve()==(run/'checkpoints/epoch_5').resolve()
        assert summary['answer_mode']=='raw' and summary['instruction']=='short-answer' and summary['max_new_tokens']==32 and summary['temperature']==0
        pi={p['question_id']:p for p in preds}
        for r in rows:
            assert r['prediction']==metric.normalize_pathvqa_answer(pi[r['question_id']]['answer'])
            assert bool(r['correct'])==(r['prediction']==metric.normalize_pathvqa_answer(r['reference']))
            assert r['answer_type']==metric.answer_type(r['reference'])
            assert r['question_type']==metric.question_type(r['question'],r['reference'])
        measured=stats(rows)
        assert abs(measured['overall']['accuracy']-summary['overall_accuracy'])<.000051
        actual=report.get('total_trainable_parameters', report.get('parameter_counts',{}).get('trainable'))
        if params is not None:
            assert actual==params,(name,actual,params)
        files=[run/'train_report.json']+[evaluation/f'pathvqa_{x}.json' for x in ('summary','predictions','comparisons')]
        init_file=run/'v1_shared_initialization_audit.json'
        if init_file.exists():
            files.append(init_file)
        record=dict(name=name,seed=seed,split=split,run=str(run),evaluation=str(evaluation),checkpoint=summary['checkpoint'],
            training_commit=report.get('git_commit'),parameters=actual,metrics=measured,
            optimizer=report.get('optimizer'),summary=summary,report=report,
            initialization_audit=read(init_file) if init_file.exists() else None,
            files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files])
        sources.append(record); artifacts[name]=(rows,record)
        print('BOUND',name,measured['overall']['accuracy'],flush=True)
        return rows

    full={}; validations={}; test={}
    for seed in (44,45,46):
        run=BASE/FULL[seed]
        full[seed]=load(f'full_validation_{seed}',run,run/'eval_validation/epoch_5',seed,'validation',COUNTS['full'])
        rates=artifacts[f'full_validation_{seed}'][1]['report']['optimizer']['group_learning_rates']
        assert rates['value_projection']==rates['prefix_output']==3e-4
        test[seed]=load(f'full_test_{seed}',run,TEST/f'seed{seed}/eval_test/epoch_5',seed,'test',COUNTS['full'])
    validations['full']=full
    for family, (mode,*suffixes) in AB.items():
        validations[family]={}
        for seed,suffix in zip((44,45,46),suffixes):
            run=BASE/'ablations'/f'pathvqa_v1_read_write_lr3e4_{mode}_norm_fixed_5ep_seed{seed}_{suffix}'
            rr=load(f'{family}_{seed}',run,run/'eval_validation/epoch_5',seed,'validation',COUNTS[family])
            rates=artifacts[f'{family}_{seed}'][1]['report']['optimizer']['group_learning_rates']
            assert rates['value_projection']==rates['prefix_output']==3e-4
            validations[family][seed]=rr
            write(out/'paired'/f'validation_full_minus_{family}_seed{seed}.json',paired(full[seed],rr))
    extra={
        'c_static': (BASE/'ablations/pathvqa_v1_c_static_norm_fixed_5ep_seed44_20260930',69632),
        'A': (ROOT/'pathvqa/outputs/visual_selection_prefix_direct/pathvqa_v1_direct_summary_norm_fixed_5ep_seed44_20260929',879364),
        'B': (BASE/'ablations/pathvqa_v1_read_write_lr3e4_evidence_token_norm_fixed_5ep_seed44_20261020_024446_1166',1864963),
    }
    for name,(run,params) in extra.items():
        rr=load(name,run,run/'eval_validation/epoch_5',44,'validation',params)
        write(out/'paired'/f'validation_full_minus_{name}_seed44.json',paired(full[44],rr))
    r2run=ROOT/'pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261001_1'
    rr=load('lora_r2_test_44',r2run,r2run/'test_epoch5_20261010_095918_1103',44,'test',1769472)
    r2report=artifacts['lora_r2_test_44'][1]['report']
    assert r2report['rank']==2 and r2report['alpha']==4 and r2report['use_rslora'] is False
    write(out/'paired/test_full_minus_lora_r2_seed44.json',paired(test[44],rr))

    def summarize(families,filename):
        result={}; flat=[]
        for family,byseed in families.items():
            per={str(s):stats(rows) for s,rows in byseed.items()}
            means={}
            for g in per['44']:
                values=[per[str(s)][g]['accuracy'] for s in (44,45,46)]
                means[g]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),ddof=1)
                for s in (44,45,46):
                    m=per[str(s)][g]
                    baseline=stats((test if filename.startswith('test') else full)[s])[g]['accuracy']
                    flat.append(dict(configuration=family,seed=s,group=g,**m,full_minus_configuration=baseline-m['accuracy']))
            result[family]=dict(per_seed=per,three_seed=means,parameters=COUNTS[family])
        write(out/(filename+'.json'),result);csv_write(out/(filename+'.csv'),flat)
        return result
    validation=summarize(validations,'validation_three_seed_summary')
    testing=summarize({'full':test},'test_three_seed_summary')
    (out/'paired').mkdir(exist_ok=True)
    paired_rows=[]
    for p in sorted((out/'paired').glob('*.json')):
        for g,m in read(p)['groups'].items():
            paired_rows.append(dict(comparison=p.stem,group=g,**{k:v for k,v in m.items() if k!='ci95'},ci_lower=m['ci95'][0],ci_upper=m['ci95'][1]))
    csv_write(out/'paired/all_groups.csv',paired_rows)
    single=[]
    for name in extra:
        record=artifacts[name][1]; report=record['report']
        rates=report.get('optimizer',{}).get('group_learning_rates',{})
        single.append(dict(name=name,parameters=record['parameters'],report=report,
            deleted_rate_groups=[k for k in ('value_projection','prefix_output') if k not in rates],
            available_group_rates=rates))
    write(out/'single_seed_reuse_audit.json',single)
    write(out/'source_manifest.json',dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=sources,notes=['Predictions and checkpoints remain at their source paths.','Intervals are image-sampling uncertainty conditional on each fitted model; seed SD reported separately.','Retrospective protocol verification; no test-based tuning in this task.']))

    branch=ROOT/'outputs/v1_three_branch_audit_20261015'
    meta=read(branch/'manifest_meta.json'); completed=read(branch/'lr3e4/completed.json')
    assert meta['manifest_sha256']==completed['manifest_sha256']=='5cc1c48c508266aed345ad98cdeb8f88a39ba74a261a16f85e9c6e8806e9d861'
    assert hashlib.sha256((branch/'manifest.json').read_bytes()).hexdigest()==meta['manifest_sha256']
    assert meta['samples']==1000 and meta['images']==122
    grouped=read(branch/'lr3e4/grouped_statistics.json')
    write(out/'branch_current_lr.json',dict(manifest=meta,completed=completed,groups=grouped,
        binding=read(branch/'binding.json'),warning='Fixed original-order first1000; norm shares are descriptive, not causal contributions.'))
    branch_rows=[]
    for group,g in grouped.items():
        if group not in ('all','question_type:what','question_type:where'):
            continue
        for layer,fields in g['layers'].items():
            for field,values in fields.items():
                if isinstance(values,dict):
                    branch_rows.append(dict(group=group,layer=layer,metric=field,mean=values.get('mean'),median=values.get('median'),p10=values.get('p10'),p90=values.get('p90')))
    csv_write(out/'branch_current_lr.csv',branch_rows)
    cases=out/'cases';cases.mkdir()
    old=ROOT/'outputs/paper_cases_20261014'
    selected=read(old/'selected_cases.json')
    selected=[g for g in selected if g['name'] in ('main_heart','appendix_burn')]
    expected=['pathvqa:test:'+str(i) for i in (513,524,435,5911,5918,5915)]
    assert [r['question_id'] for g in selected for r in g['rows']]==expected
    index={r['question_id']:r for r in test[44]}
    image_rows=[]
    from PIL import Image
    for g in selected:
        (cases/'images').mkdir(exist_ok=True)
        src=old/'images'/f"{g['image_id']}.png"
        shutil.copy2(src,cases/'images'/src.name)
        with Image.open(src) as image:
            rgb=image.convert('RGB'); digest=hashlib.sha256()
            digest.update(b'RGB');digest.update(f'{rgb.width}x{rgb.height}'.encode());digest.update(rgb.tobytes())
            assert digest.hexdigest()==g['image_id']
            image_rows.append(dict(image_id=g['image_id'],size=list(rgb.size),source=str(src)))
        for r in g['rows']:
            actual=index[r['question_id']]
            assert all(r[k]==actual[k] for k in ('question','reference','image_id'))
            r['answers']['QDPT']=dict(prediction=actual['prediction'],correct=bool(actual['correct']))
    write(cases/'selected_cases.json',selected)
    write(cases/'images.json',image_rows)
    shutil.copy2(old/'selection_policy.json',cases/'selection_policy.json')
    historical={}
    for name,file in [('CoCoOp','cocoop'),('LoRA-r8','lora_r8')]:
        source=old/'sources'/f'{file}.json'; summary=read(old/'sources'/f'{file}_summary.json')
        index_old={r['question_id']:r for r in read(source)}
        for g in selected:
            for r in g['rows']:
                archived=index_old[r['question_id']]
                assert all(r[k]==archived[k] for k in ('question','reference','image_id'))
                assert r['answers'][name]['prediction']==archived['prediction']
                assert r['answers'][name]['correct']==archived['correct']
        historical[name]=dict(summary=summary,source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write(cases/'answer_provenance.json',dict(QDPT=artifacts['full_test_44'][1]['evaluation'],
        historical_answers=historical,selection='same fixed six IDs as manuscript; selected before new maps; qualitative purposive cases'))
    text=['# Final-LR PathVQA evidence','',
          'All paired differences use full QDPT minus comparator; 10,000 image-cluster percentile draws, seed42 (NumPy PCG64).',
          'Within each subgroup, resample its images with all associated questions. Subgroups are exploratory.',
          'Three-seed sample SD and conditional image-bootstrap CI describe distinct variation sources.','',
          '## Validation','|Configuration|Mean|Sample SD|Full minus configuration|','|---|---:|---:|---:|']
    for family,v in validation.items():
        m=v['three_seed']['overall'];f=validation['full']['three_seed']['overall']['mean']
        text.append(f"|{family}|{m['mean']:.4f}|{m['sample_sd']:.4f}|{f-m['mean']:+.4f}|")
    m=testing['full']['three_seed']['overall']
    text+=['',f"Test QDPT: {m['mean']:.4f} +/- {m['sample_sd']:.4f} (3 seeds, sample SD).",'',
           'Detailed subgroup summaries: validation_three_seed_summary.csv and test_three_seed_summary.csv.',
           'Paired counts and intervals: paired/all_groups.csv and per-comparison JSON.',
           'C-static/A reuse is audited in single_seed_reuse_audit.json: actual training reports retained; source implementations remove the adjusted read/write groups.',
           'Current-LR branch data are archived exports, not new forward results; source binding and first1000 manifest retained.',
           'Case answers use archived Test predictions. Case maps are a separate explicitly authorized six-question prefill export.',
           'No training, checkpoint update, full evaluation or additional seed is performed.']
    (out/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('CPU_COLLECTION_DONE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.output.resolve())
