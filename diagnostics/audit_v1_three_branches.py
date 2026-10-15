"""CPU binding / user-launched prefill export for three fixed PathVQA checkpoints."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SPECS={
    'v1':('pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2',[5,11,17],'visual_selection_prefix_p20_v1'),
    'lr3e4':(None,[5,11,17],'visual_selection_prefix_p20_v1'),
    'v20':('pathvqa/outputs/v20/pathvqa_v20_deeper_three_l7_15_23_norm_fixed_5ep_seed44_20261013_170205_10331',[7,15,23],'visual_selection_prefix_v20_deeper_three'),
}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def require(ok,message):
    if not ok:raise ValueError(message)

def scorer():
    spec=importlib.util.spec_from_file_location('cpu_pathvqa_metric',ROOT/'pathvqa/pathvqa_vqa_metric.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def bind_one(name,run):
    _,layers,method=SPECS[name]
    report=read(run/'train_report.json');checkpoint=run/'checkpoints/epoch_5'
    config=read(checkpoint/'visual_selection_prefix_config.json')
    evaluation=run/'eval_validation/epoch_5';summary=read(evaluation/'pathvqa_summary.json')
    expected_experiment={'v1':'pathvqa_v1_norm_fixed_5ep_seed44','lr3e4':'pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44',
        'v20':'pathvqa_v20_deeper_three_l7_15_23_norm_fixed_5ep_seed44'}[name]
    require(report['experiment']==expected_experiment,'Experiment identity: '+str(run))
    for k,v in {'epochs':5,'model_seed':44,'data_seed':42,'total_trainable_parameters':1864963,
                'trainer_model_accepts_loss_kwargs':False}.items():require(report[k]==v,f'{name} report {k}')
    require(report['train_metrics']['epoch']==5,f'{name} incomplete epoch5')
    require(config['method']==method and config['init_seed']==44 and config['layers']==layers,f'{name} architecture')
    require(config['visual_tokens']==[8,10] and config['prefix_tokens']==20,f'{name} visual/P20 layout')
    require(sum(config['trainable_parameters'].values())==1864963,f'{name} checkpoint parameter count')
    optimizer=report['optimizer']
    for k,v in {'per_device_batch_size':2,'gradient_accumulation_steps':16,'warmup_ratio':.03,
                'scheduler':'linear','max_grad_norm':1.0}.items():require(optimizer[k]==v,f'{name} optimizer {k}')
    matches=re.findall(r'\[V1_OPTIMIZER\] rates=(\{[^\r\n]*?\})', (run/'train.log').read_text(encoding='utf-8',errors='replace'))
    require(bool(matches),f'{name} logged optimizer groups absent')
    rates=json.loads(matches[-1])
    expected={'p20':.3,'question_context':1e-4,'maps':1e-4,'layer_condition':1e-4,
              'prefix_output':3e-4 if name=='lr3e4' else 1e-4,'visual_s8':3e-5,'visual_av10':1e-4}
    if name=='lr3e4':expected['value_projection']=3e-4
    require(rates==expected,f'{name} actual logged LR groups differ: {rates}')
    if 'group_learning_rates' in optimizer:require(optimizer['group_learning_rates']==rates,f'{name} report/log LRs')
    if name=='lr3e4':require(report['read_write_learning_rate']==3e-4,'LR experiment flag')
    for k,v in {'count':6259,'split':'validation','partial_evaluation':False,'max_new_tokens':32,
                'temperature':0,'answer_mode':'raw','instruction':'short-answer'}.items():require(summary[k]==v,f'{name} evaluation {k}')
    ck=Path(summary['checkpoint']);ck=ck if ck.is_absolute() else ROOT/ck
    require(ck.resolve()==checkpoint.resolve(),f'{name} evaluated checkpoint differs')
    weights=checkpoint/'visual_selection_prefix.pt';require(weights.is_file(),f'{name} missing weights')
    rows=read(evaluation/'pathvqa_comparisons.json');predictions=read(evaluation/'pathvqa_predictions.json')
    index={r['question_id']:r for r in rows};pi={r['question_id']:r for r in predictions}
    require(len(index)==len(rows)==len(pi)==len(predictions)==6259,f'{name} predictions incomplete/duplicate')
    metric=scorer()
    for i in range(6259):
        qid=f'pathvqa:validation:{i}';r=index[qid]
        require(metric.normalize_pathvqa_answer(pi[qid]['answer'])==r['prediction'],f'{name} prediction {qid}')
        require(r['correct']==(r['prediction']==metric.normalize_pathvqa_answer(r['reference'])),f'{name} score {qid}')
    require(abs(sum(r['correct'] for r in rows)*100/6259-summary['overall_accuracy'])<.000051,f'{name} score summary')
    return {'run':str(run),'checkpoint':str(checkpoint),'layers':layers,'report':report,'config':config,
        'summary':summary,'logged_group_lrs':rates,'weights_sha256':sha(weights),
        'config_sha256':sha(checkpoint/'visual_selection_prefix_config.json'),
        'comparisons_sha256':sha(evaluation/'pathvqa_comparisons.json')}, index

def precheck(out,data):
    import pyarrow.parquet as pq
    from PIL import Image
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/'binding.json').exists(),'Binding already exists; choose a new independent output directory')
    bindings={};indices={};rejected=[]
    for name,(relative,_,_) in SPECS.items():
        candidates=[ROOT/relative] if relative else sorted((ROOT/'pathvqa/outputs/visual_selection_prefix').glob('pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44*'))
        valid=[]
        for run in candidates:
            try:valid.append(bind_one(name,run))
            except Exception as exc:rejected.append({'name':name,'run':str(run),'error':str(exc)})
        require(len(valid)==1,f'{name} requires one verified artifact; valid={len(valid)} rejected={rejected}')
        bindings[name],indices[name]=valid[0]
    samples=[];shards=sorted(data.glob('validation-*.parquet'))
    require(sum(pq.ParquetFile(p).metadata.num_rows for p in shards)==6259,'Official Validation shard count')
    metric=scorer()
    for path in shards:
        f=pq.ParquetFile(path)
        for ri in range(f.num_row_groups):
            table=f.read_row_group(ri,columns=['image','question','answer'])
            for raw in table.slice(0,min(len(table),1000-len(samples))).to_pylist():
                i=len(samples);qid=f'pathvqa:validation:{i}';entry=indices['v1'][qid]
                image=raw['image'];source=Image.open(io.BytesIO(image['bytes'])) if image['bytes'] is not None else Image.open(image['path'])
                with source:
                    rgb=source.convert('RGB');d=hashlib.sha256()
                    d.update(rgb.mode.encode('ascii'));d.update(f'{rgb.width}x{rgb.height}'.encode('ascii'));d.update(rgb.tobytes());image_id=d.hexdigest();rgb.close()
                require(image_id==entry['image_id'] and raw['question'].strip()==entry['question'],'Official original order/image mismatch: '+qid)
                require(metric.normalize_pathvqa_answer(raw['answer'])==entry['reference'],'Official reference mismatch: '+qid)
                for name,index in indices.items():
                    require(all(index[qid][k]==entry[k] for k in ['image_id','question','reference','answer_type','question_type']),name+' paired metadata '+qid)
                samples.append({'question_id':qid,'row_index':i,'image_id':image_id,'question':raw['question'].strip(),
                    'reference':entry['reference'],'answer_type':entry['answer_type'],'question_type':entry['question_type'],
                    'archived':{name:{'prediction':idx[qid]['prediction'],'correct':idx[qid]['correct']} for name,idx in indices.items()}})
            if len(samples)==1000:break
        if len(samples)==1000:break
    require(len(samples)==1000,'Expected first1000')
    write(out/'manifest.json',samples)
    meta={'split':'validation','rule':'official original indices 0..999; no correctness filtering','samples':1000,
        'images':len({r['image_id'] for r in samples}),'question_types':dict(Counter(r['question_type'] for r in samples)),
        'answer_types':dict(Counter(r['answer_type'] for r in samples)),'manifest_sha256':sha(out/'manifest.json')}
    write(out/'manifest_meta.json',meta)
    write(out/'binding.json',{'models':bindings,'rejected':rejected,'data_root':str(data.resolve()),'git_commit':commit(),'manifest':meta})
    print('[BRANCH_AUDIT_CPU_PRECHECK]',json.dumps(meta),flush=True)

def export(out,name,model_path):
    # Only this function imports Torch/model code; user launches this stage.
    import numpy as np
    import torch
    from pathvqa.data_pipeline import PathVQAParquetStore
    from pathvqa.pathvqa_official_eval import build_prompt,image_fingerprint
    from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface
    binding=read(out/'binding.json');info=binding['models'][name];manifest=read(out/'manifest.json')
    require(Path(info['summary']['base_model']).resolve()==model_path.resolve(),'Frozen backbone path differs from archived evaluation')
    require(sha(out/'manifest.json')==binding['manifest']['manifest_sha256'],'Manifest changed')
    checkpoint=Path(info['checkpoint'])
    require(sha(checkpoint/'visual_selection_prefix.pt')==info['weights_sha256'],'Checkpoint changed')
    require(sha(checkpoint/'visual_selection_prefix_config.json')==info['config_sha256'],'Checkpoint config changed')
    target=out/name;target.mkdir(exist_ok=False)
    (target/'tensors').mkdir()
    interface=VisualSelectionPrefixInterface(str(checkpoint),str(model_path));m=interface.model
    require(list(m.selection_layers)==info['layers'],'Loaded layer identity')
    store=PathVQAParquetStore(binding['data_root'],'validation')
    require([r['question_id'] for r in store.samples[:1000]]==[r['question_id'] for r in manifest],'Store order')
    def arr(t):return t.detach().float().cpu().numpy().copy()
    w=arr(m.prefix_output.weight);bias=arr(m.prefix_output.bias)
    np.savez_compressed(target/'head_parameters.npz',weight=w,bias=bias)
    with (target/'rows.jsonl').open('w',encoding='utf-8') as handle:
        for i,(record,meta) in enumerate(zip(store.samples,manifest)):
            require(record['question']==meta['question'],'Question text changed after precheck')
            image=store.load_image(record);require(image_fingerprint(image)==meta['image_id'],'GPU input image identity')
            inputs=interface.prepare_inputs(image,build_prompt(meta['question'],None),question=meta['question'])
            moved={k:v.to(interface.device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            require('labels' not in moved,'Prefill must exclude teacher-forcing answers')
            captured={};hooks=[]
            def capture(key):
                def hook(module,args,result):captured[key]=result.detach().clone()
                return hook
            for j in range(3):
                hooks.append(m.value_norms[j].register_forward_hook(capture(f'normalized{j}')))
                hooks.append(m.value_blocks[j].register_forward_hook(capture(f'projected{j}')))
            def head_hook(module,args,result):
                captured['uniform_raw']=m._features['value'].detach().float().mean(dim=0,keepdim=True)
            hooks.append(m.prefix_output.register_forward_hook(head_hook))
            m.diagnostic_probe={};before=m.diagnostic_prefill_calls
            try:
                with torch.inference_mode():m(**moved,logits_to_keep=1)
                require(m.diagnostic_prefill_calls==before+1,'Exactly one prefill required')
            finally:
                for hook in hooks:hook.remove()
                image.close()
            p=m.diagnostic_probe;beta=arr(p['layer_weights'])[0];grid=p['grid'][0].tolist()
            t,h,v=grid;require(t==1 and h%2==0 and v%2==0,'Expected one aligned 2D image')
            maps=np.stack([arr(p[f'map{l}'][0]) for l in info['layers']])
            require(maps.shape==(3,h*v//4) and np.isfinite(maps).all() and (maps>=0).all() and np.allclose(maps.sum(1),1,atol=1e-4),'Mapped probability geometry')
            normalized=np.stack([arr(captured[f'normalized{j}'])[0] for j in range(3)])
            projected=np.stack([arr(captured[f'projected{j}'])[0] for j in range(3)])
            weighted=arr(p['condition']).reshape(3,64)
            require(np.allclose(weighted,projected*beta[:,None],atol=1e-6,rtol=1e-5),'Gate/block alignment')
            u_norm=[];u_projected=[]
            with torch.inference_mode():
                for j in range(3):
                    z=m.value_norms[j](captured['uniform_raw']);u_norm.append(arr(z)[0]);u_projected.append(arr(m.value_blocks[j](z))[0])
            u_projected=np.stack(u_projected);u_weighted=u_projected*beta[:,None]
            components=np.stack([w[:,j*64:(j+1)*64]@np.maximum(weighted[j],0) for j in range(3)])
            uniform_components=np.stack([w[:,j*64:(j+1)*64]@np.maximum(u_weighted[j],0) for j in range(3)])
            shift=arr(p['shift'])[0];reconstructed=components.sum(0)+bias
            absolute=float(np.max(np.abs(reconstructed-shift)));relative=float(np.linalg.norm(reconstructed-shift)/max(np.linalg.norm(shift),1e-12))
            require(np.allclose(reconstructed,shift,atol=1e-5,rtol=1e-4),f'Head reconstruction failed {i}: {absolute} / {relative}')
            tensors={'beta':beta,'maps':maps.reshape(3,h//2,v//2),'raw_summaries':np.stack([arr(p[f'summary{l}'])[0] for l in info['layers']]),
                'normalized_summaries':normalized,'projected_summaries':projected,'weighted_blocks':weighted,
                'components':components,'uniform_raw_summary':arr(captured['uniform_raw'])[0],
                'uniform_normalized':np.stack(u_norm),'uniform_projected':u_projected,'uniform_weighted_blocks':u_weighted,
                'uniform_components':uniform_components,'shift':shift,'output_bias':bias}
            require(all(np.isfinite(x).all() for x in tensors.values()),'Nonfinite export')
            np.savez_compressed(target/'tensors'/f'{i:04d}.npz',**tensors)
            row={**meta,'model':name,'layers':info['layers'],'grid_thw':grid,'tensor_file':f'tensors/{i:04d}.npz',
                'prediction':meta['archived'][name]['prediction'],'correct':meta['archived'][name]['correct'],
                'reconstruction_max_abs':absolute,'reconstruction_relative_l2':relative}
            handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush()
            m.diagnostic_probe=None;captured.clear()
            if (i+1)%25==0:print(f'[BRANCH_EXPORT] {name} {i+1}/1000',flush=True)
    require(sha(checkpoint/'visual_selection_prefix.pt')==info['weights_sha256'],'Checkpoint file changed during read-only export')
    write(target/'completed.json',{'samples':1000,'model':name,'layers':info['layers'],'manifest_sha256':sha(out/'manifest.json'),
        'git_commit':commit(),'torch':torch.__version__,'numpy':np.__version__,'head_weight_dtype':str(m.prefix_output.weight.dtype),
        'reconstruction_tolerance':{'atol':1e-5,'rtol':1e-4},'checkpoint_unchanged':True,
        'mode':'eval/inference_mode, no labels, single prefill per question; no generated answers/optimizer/backward',
        'uniform_control':'same-forward native Value mean -> original LN/projection/gate/output slices; representation diagnostic'})
    print('[BRANCH_EXPORT_DONE]',name,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['precheck','export']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/dataset/pathVQA'))
    p.add_argument('--model',choices=list(SPECS));p.add_argument('--base-model',type=Path,default=Path('/root/autodl-tmp/model'))
    a=p.parse_args()
    if a.stage=='precheck':precheck(a.output,a.data_root)
    else:
        require(a.model is not None,'--model required for GPU export');export(a.output,a.model,a.base_model)

if __name__=='__main__':main()
