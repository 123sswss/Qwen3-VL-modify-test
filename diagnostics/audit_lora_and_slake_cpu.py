"""Read archived JSON/logs/safetensors headers only; no torch/model imports."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import shutil
import struct


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def header_audit(checkpoint):
    p=checkpoint/'adapter_model.safetensors'
    if not p.exists(): return {'missing':'safetensors header; no pickle or model load'}
    with p.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        h=json.loads(f.read(size))
    modules=sorted(k.split('.lora_A.')[0] for k in h if '.lora_A.' in k)
    visual=[k for k in modules if '.visual.blocks.' in k]
    language=[k for k in modules if '.language_model.layers.' in k]
    return {'source':str(p),'header_sha256':hashlib.sha256(json.dumps(h,sort_keys=True).encode()).hexdigest(),
            'modules':modules,'count':len(modules),'visual_count':len(visual),'language_count':len(language),
            'visual_layers':sorted({int(re.search(r'\.blocks\.(\d+)\.',k)[1]) for k in visual}),
            'language_layers':sorted({int(re.search(r'\.layers\.(\d+)\.',k)[1]) for k in language}),
            'visual_roles':sorted({k.rsplit('.',1)[-1] for k in visual}),
            'language_roles':sorted({k.rsplit('.',1)[-1] for k in language}),
            'adapter_tensor_parameters':sum(math.prod(v['shape']) for k,v in h.items() if k!='__metadata__')}


def config_audit(root,out):
    specs=[('PathVQA','r2','pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261001_1',5),
           ('PathVQA','r8','pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r8_seed44_20260830',3),
           ('SLAKE','r2','slake/outputs/lora/slake_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261010_101607_1852',5),
           ('RSVQA-LR','r2','RSVQA/outputs/lora/rsvqa_lr_lora_full_model_attention_r2_norm_fixed_5ep_b4a8_seed44_20261010_150020_9726',5),
           ('RSVQA-LR','r8','RSVQA/outputs/lora/rsvqa_lr_lora_full_model_attention_r8_b4a8_seed44_20260929',3)]
    result=[]
    for dataset,method,rel,epoch in specs:
        run=root/rel; report=read(run/'train_report.json'); cp=run/'checkpoints'/f'epoch_{epoch}'
        config=read(cp/'adapter_config.json')
        hashes={}
        for name,p in [('train_report.json',run/'train_report.json'),('adapter_config.json',cp/'adapter_config.json')]:
            hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            write(out/'sources'/f'{dataset}_{method}'/name,read(p))
        summaries=[]
        for pattern in ('**/pathvqa_summary.json','**/slake_summary.json','**/rsvqa_summary.json'):
            for p in run.glob(pattern):
                s=read(p)
                summaries.append({'path':str(p),**{k:s.get(k) for k in ('checkpoint','backend','split','expected_split','count','total','overall_accuracy')}})
        result.append({'dataset':dataset,'method':method,'run':str(run),'checkpoint':str(cp),
                       'report':report,'adapter_config':config,'header_audit':header_audit(cp),
                       'evaluation_summaries':summaries,'source_hashes':hashes})
    # Dataset-specific V1 controls, not V10 or latest-directory guesses.
    v1={}
    for ds,rel in [('PathVQA','pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2'),
                   ('SLAKE','slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928'),
                   ('RSVQA-LR','RSVQA/outputs/visual_selection_prefix/rsvqa_v1_norm_fixed_5ep_b4a8_seed44_20260929')]:
        p=root/rel/'train_report.json'; v1[ds]={'path':str(p),'report':read(p)}
    failed=[]
    for run in sorted((root/'pathvqa/outputs/lora').glob('*r2*seed44*')):
        if (run/'train_report.json').exists(): continue
        log=run/'train.log'
        tail=''
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-18000));tail=f.read().decode('utf-8',errors='replace')
        failed.append({'run':str(run),'has_train_log':log.exists(),'log_size':log.stat().st_size if log.exists() else 0,
                       'tail_evidence':'\n'.join(s for s in tail.splitlines() if any(x in s for x in ('OutOfMemory','out of memory','Traceback','Error:','train_runtime')))[-5000:]})
    sources={}
    for rel in ('pathvqa/train_visual_lora.py','slake/train_visual_peft.py','run_lora_r2_dataset_seed44.sh',
                'pathvqa/run_lora_full_model_attention_r2_norm_fixed_5ep_seed44.sh','loraTest/trainLora.py'):
        p=root/rel
        text=p.read_text(encoding='utf-8',errors='replace')
        sources[rel]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'path':str(p),
            'snippets':[{ 'line':i,'text':line} for i,line in enumerate(text.splitlines(),1)
                        if any(s in line for s in ('seed','rank','alpha','dropout','target_modules','batch_size',
                          'gradient_accumulation','warmup','learning_rate','num_train_epochs','max_grad_norm',
                          'model_accepts_loss_kwargs','optim','save_epochs','work_dir'))]}
    legacy=[]
    for rank in (8,16,32):
        cp=root/f'loraTest/runs/lora/rank{rank}/final'
        if cp.exists():legacy.append({'candidate_not_bound_to_paper':True,'checkpoint':str(cp),
                      'adapter_config':read(cp/'adapter_config.json'),'header_audit':header_audit(cp)})
    logfile=root/'loraTest/logs-fullattention/eval_lora.log'
    if logfile.exists():sources[str(logfile)]={'text':logfile.read_text(encoding='utf-8',errors='replace')[-5000:]}
    payload={'runs':result,'v1_controls':v1,'incomplete_or_failed_r2_seed44':failed,'source_code':sources,
             'legacy_electrical_candidates_not_paper_bound':legacy,
             'pending':['SLAKE-r8 actual 81.95 run unavailable on this instance',
                        'electrical-r2 seed/epoch/checkpoint unavailable; user says possibly another instance',
                        'electrical-r8 70.96 run unavailable; current legacy r8 log gives different score',
                        'historical missing versions/normalization/search provenance not inferred from current defaults']}
    write(out/'lora_configuration_audit.json',payload)
    return payload


def stratified_sample(pool,n,rng):
    groups=collections.defaultdict(list)
    for row in sorted(pool,key=lambda x:x['question_id']):
        groups[(row['question_type'],row['answer_type'],row['language'])].append(row)
    allocation={k:0 for k in sorted(groups)}
    n=min(n,len(pool))
    if n>=len(groups): allocation={k:1 for k in sorted(groups)}
    # Repeated largest deficit allocation approximates proportional quotas and caps rare strata.
    while sum(allocation.values())<n:
        candidates=[k for k in allocation if allocation[k]<len(groups[k])]
        k=max(candidates,key=lambda k:(n*len(groups[k])/len(pool)-allocation[k],k))
        allocation[k]+=1
    selected=[]
    for k in sorted(groups):
        rows=groups[k][:];rng.shuffle(rows);selected+=rows[:allocation[k]]
    return selected,{'|'.join(k):{'population':len(groups[k]),'sample':allocation[k]} for k in allocation}


def slake_audit(root,data,out):
    spec=importlib.util.spec_from_file_location('archived_slake_metric',root/'slake/slake_vqa_metric.py')
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    refs=read(data/'test.json'); assert isinstance(refs,list) and len(refs)==2094
    runs={'v1':root/'slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928',
          'r2':root/'slake/outputs/lora/slake_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261010_101607_1852'}
    comparisons={};binding={}
    for name,run in runs.items():
        ev=run/'eval_test/epoch_5';preds=read(ev/'slake_predictions.json');summary=read(ev/'slake_summary.json')
        details={str(x['question_id']):x for x in read(ev/'slake_details.json')}
        archived={str(x['question_id']):x for x in read(ev/'slake_comparisons.json')}
        assert len(preds)==2094 and len(details)==2094
        s,c=metric.evaluate_slake_predictions(refs,preds)
        assert s['overall_accuracy']==summary['overall_accuracy']
        assert not s['missing_predictions'] and not s['extra_predictions']
        assert str(summary['checkpoint']).endswith(str(run.relative_to(root))+'/checkpoints/epoch_5')
        for row,ref in zip(c,refs):
            q=str(row['question_id']); assert row['correct']==archived[q]['correct']
            assert row['ground_truth_answers']==archived[q]['ground_truth_answers']
            assert row['question']==details[q]['question'] and ref['img_name']==details[q]['image']
        comparisons[name]={str(x['question_id']):x for x in c}
        binding[name]={'run':str(run),'summary':summary,'recomputed_summary':s,
          'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [ev/'slake_predictions.json',ev/'slake_summary.json',ev/'slake_comparisons.json',ev/'slake_details.json']}}
    train_counts=collections.Counter(metric.normalize_vqa_answer(x.get('answer','')) for x in read(data/'train.json')
                                     if str(x.get('answer','')).strip())
    rows=[]
    for ref in refs:
        q=str(metric.record_id(ref));a=comparisons['v1'][q];b=comparisons['r2'][q]
        assert (a['question'],a['ground_truth_answers'],a['answer_type'],a['language'],a['question_type']) == (
                b['question'],b['ground_truth_answers'],b['answer_type'],b['language'],b['question_type'])
        category=('both_correct' if a['correct'] and b['correct'] else 'v1_only' if a['correct'] else
                  'r2_only' if b['correct'] else 'both_wrong')
        rows.append({'question_id':q,'image_id':ref['img_id'],'image':ref['img_name'],'question':ref['question'],
                     'reference':ref['answer'],'question_type':ref['base_type'],'answer_type':ref['answer_type'],
                     'language':ref['q_lang'],'content_type':ref.get('content_type'),'modality':ref.get('modality'),
                     'v1_prediction':a['predicted_answer'],'r2_prediction':b['predicted_answer'],
                     'v1_correct':a['correct'],'r2_correct':b['correct'],'category':category,
                     'train_normalized_answer_frequency':train_counts[metric.normalize_vqa_answer(ref['answer'])]})
    stats={}
    for group,key,value in [('overall',None,None)]+[(v,k,v) for k,vs in
          [('question_type',('kvqa','vqa')),('answer_type',('OPEN','CLOSED')),('language',('en','zh'))] for v in vs]:
        selected=[x for x in rows if key is None or x[key]==value]
        stats[group]={'count':len(selected),'images':len({x['image_id'] for x in selected}),
            'v1_accuracy':100*sum(x['v1_correct'] for x in selected)/len(selected),
            'r2_accuracy':100*sum(x['r2_correct'] for x in selected)/len(selected),
            'outcomes':dict(collections.Counter(x['category'] for x in selected))}
    rng=random.Random(42);pools={k:[x for x in rows if x['category']==k] for k in ('r2_only','v1_only')}
    quotas={'r2_only':min(40,len(pools['r2_only'])),'v1_only':min(20,len(pools['v1_only']))}
    for name in ('r2_only','v1_only'):
        quotas[name]+=min(60-sum(quotas.values()),len(pools[name])-quotas[name])
    sample=[];allocation={}
    for name in ('r2_only','v1_only'):
        part,allocation[name]=stratified_sample(pools[name],quotas[name],rng);sample+=part
    if len(sample)<60:
        part,allocation['both_wrong']=stratified_sample([x for x in rows if x['category']=='both_wrong'],60-len(sample),rng);sample+=part
    images=out/'images';images.mkdir(exist_ok=True)
    for i,row in enumerate(sample,1):
        row['review_index']=i
        source=data/'imgs'/row['image'];target=images/(str(row['image_id'])+source.suffix)
        assert source.is_file()
        if not target.exists():shutil.copyfile(source,target)
        row['exported_image']=str(target.relative_to(out));row['server_source_image']=str(source)
    write(out/'slake_binding_and_statistics.json',{'binding':binding,'statistics':stats,
        'identity_audit':'2094 unique IDs, same image/question/reference and archived scoring, exact epoch5 checkpoints',
        'metric_source':str(root/'slake/slake_vqa_metric.py'),
        'metric_sha256':hashlib.sha256((root/'slake/slake_vqa_metric.py').read_bytes()).hexdigest()})
    write(out/'slake_all_paired_rows.json',rows)
    write(out/'slake_sample_60.json',{'seed':42,'sampling_rule':'40 r2-only/20 V1-only; each nonempty type x answer x language stratum gets >=1 if budget permits; remaining proportional largest deficit capped by population; deficit transfers to other exclusive group then both-wrong; sorted IDs before seeded shuffle',
                                   'quotas':quotas,'strata':allocation,'rows':sample})
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    config_audit(a.root,a.output)
    stats=slake_audit(a.root,a.data,a.output)
    print(json.dumps({'output':str(a.output),'statistics':stats},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
