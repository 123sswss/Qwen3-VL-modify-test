"""CPU-only analysis of frozen first-1000 branch exports (NumPy/Pillow)."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def cosine(a,b):
    d=norm(a)*norm(b)
    return float(np.clip(np.dot(np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64))/d,-1,1)) if d>1e-12 else None
def relative(a,b):
    d=(norm(a)+norm(b))/2
    return norm(a-b)/d if d>1e-12 else None
def describe(values):
    finite=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not finite:return {'count':0,'missing':len(values)}
    return {'count':len(finite),'missing':len(values)-len(finite),'mean':float(np.mean(finite)),
        'median':float(np.median(finite)),'p10':float(np.quantile(finite,.1)),
        'p90':float(np.quantile(finite,.9)),'min':min(finite),'max':max(finite)}
def js(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a/a.sum();b=b/b.sum()
    m=(a+b)/2
    return max(0.,float(.5*(np.sum(a*np.log(np.maximum(a,1e-30)/np.maximum(m,1e-30)))+
                     np.sum(b*np.log(np.maximum(b,1e-30)/np.maximum(m,1e-30))))))

def analyze(out):
    manifest=read(out/'manifest.json');meta=read(out/'manifest_meta.json')
    digest=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
    assert digest==meta['manifest_sha256'] and len(manifest)==1000
    all_rows={};summaries={};pairs_all={};headers=[];strata_all={}
    for name in ('v1','lr3e4','v20'):
        target=out/name;completed=read(target/'completed.json')
        assert completed['samples']==1000 and completed['manifest_sha256']==digest
        rows=[json.loads(line) for line in (target/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==1000 and [r['question_id'] for r in rows]==[r['question_id'] for r in manifest]
        layers=completed['layers'];derived=[];by_image=defaultdict(list)
        for row,source in zip(rows,manifest):
            assert all(row[k]==source[k] for k in ('question_id','image_id','question','reference','question_type','answer_type'))
            with np.load(target/row['tensor_file'],allow_pickle=False) as a:
                beta=a['beta'];maps=a['maps'].reshape(3,-1);components=a['components'];uni=a['uniform_components'];bias=a['output_bias']
                assert np.allclose(components.sum(0)+bias,a['shift'],atol=1e-5,rtol=1e-4)
                cn=np.linalg.norm(components.astype(np.float64),axis=1);total=float(cn.sum());bias_norm=norm(bias)
                combined=components.astype(np.float64).sum(0);weighted=a['weighted_blocks']
                item={k:row[k] for k in ('question_id','image_id','question_type','answer_type','correct','prediction')}
                item.update({'model':name,'reconstruction_max_abs':row['reconstruction_max_abs'],
                    'reconstruction_relative_l2':row['reconstruction_relative_l2'],'bias_norm':bias_norm,
                    'shift_norm':norm(a['shift']),'bias_norm_share':bias_norm/(total+bias_norm) if total+bias_norm>1e-12 else None,
                    'cancellation_without_bias':1-norm(combined)/total if total>1e-12 else None,
                    'cancellation_with_bias':1-norm(a['shift'])/(total+bias_norm) if total+bias_norm>1e-12 else None,
                    'uniform_shift_difference_norm':norm(components.sum(0)-uni.sum(0)),
                    'uniform_shift_relative_difference':relative(a['shift'],uni.sum(0)+bias),
                    'zero_component_count':int((cn<=1e-12).sum()),'layers':{}})
                for j,l in enumerate(layers):
                    p=maps[j];m=len(p)
                    item['layers'][str(l)]={'gate':float(beta[j]),'gate_below_001':bool(beta[j]<.01),'gate_below_005':bool(beta[j]<.05),
                        'largest_gate':bool(j==int(np.argmax(beta))),
                        'map_entropy':float(-np.sum(p*np.log(np.maximum(p,1e-30)))/math.log(m)) if m>1 else 0.,
                        'map_tv_uniform':float(np.abs(p-1/m).sum()/2),'map_peak_multiplier':float(p.max()*m),
                        'raw_summary_norm':norm(a['raw_summaries'][j]),'normalized_summary_norm':norm(a['normalized_summaries'][j]),
                        'projected_summary_norm':norm(a['projected_summaries'][j]),'weighted_block_norm':norm(weighted[j]),
                        'component_norm':float(cn[j]),'component_norm_share':float(cn[j]/total) if total>1e-12 else None,
                        'uniform_component_norm':norm(uni[j]),'component_uniform_difference_norm':norm(components[j]-uni[j]),
                        'component_uniform_relative_difference':relative(components[j],uni[j]),
                        'component_shift_cosine':cosine(components[j],a['shift'])}
                for j,k in itertools.combinations(range(3),2):
                    item[f'component_cosine_{layers[j]}_{layers[k]}']=cosine(components[j],components[k])
                derived.append(item)
                by_image[row['image_id']].append({'row':row,'maps':maps.copy(),'beta':beta.copy()})
        groups={'all':derived,'correct':[r for r in derived if r['correct']],'incorrect':[r for r in derived if not r['correct']]}
        for t in sorted({r['question_type'] for r in derived}):
            groups['question_type:'+t]=[r for r in derived if r['question_type']==t]
            for correct in (True,False):
                groups[f'question_type:{t}/'+('correct' if correct else 'incorrect')]=[r for r in derived if r['question_type']==t and r['correct']==correct]
        for t in ('yes/no','free-form'):groups['answer_type:'+t]=[r for r in derived if r['answer_type']==t]
        summary={}
        for group,subset in groups.items():
            if not subset:continue
            s={'n':len(subset),'images':len({r['image_id'] for r in subset}),'layers':{}}
            for key in [k for k in derived[0] if (isinstance(derived[0][k],(int,float)) or k.startswith('component_cosine_')) and k!='correct']:
                s[key]=describe([r[key] for r in subset])
            for j,l in enumerate(layers):
                s['layers'][str(l)]={k:describe([r['layers'][str(l)][k] for r in subset]) for k in derived[0]['layers'][str(l)]}
            summary[group]=s
        question_pairs=[];duplicates=0
        for image,items in by_image.items():
            for a,b in itertools.combinations(items,2):
                if ' '.join(a['row']['question'].lower().split())==' '.join(b['row']['question'].lower().split()):
                    duplicates+=1;continue
                assert a['maps'].shape==b['maps'].shape
                question_pairs.append({'image_id':image,'question_a':a['row']['question_id'],'question_b':b['row']['question_id'],
                    'question_type_a':a['row']['question_type'],'question_type_b':b['row']['question_type'],
                    'both_correct':bool(a['row']['correct'] and b['row']['correct']),
                    'correctness_pair':('both_correct' if a['row']['correct'] and b['row']['correct'] else
                        'both_incorrect' if not a['row']['correct'] and not b['row']['correct'] else 'mixed'),
                    'map_js_nats':{str(l):js(a['maps'][j],b['maps'][j]) for j,l in enumerate(layers)},
                    'gate_abs_change':{str(l):float(abs(a['beta'][j]-b['beta'][j])) for j,l in enumerate(layers)},
                    'gate_l1_change':float(np.abs(a['beta']-b['beta']).sum())})
        pair_summary={'pairs':len(question_pairs),'images_with_distinct_questions':len({p['image_id'] for p in question_pairs}),
            'duplicate_text_pairs_excluded':duplicates,'gate_l1_change':describe([p['gate_l1_change'] for p in question_pairs]),
            'layers':{str(l):{'map_js_nats':describe([p['map_js_nats'][str(l)] for p in question_pairs]),
                             'gate_abs_change':describe([p['gate_abs_change'][str(l)] for p in question_pairs])} for l in layers}}
        pair_groups=defaultdict(list)
        for pair in question_pairs:
            pair_groups[pair['correctness_pair']].append(pair)
            pair_groups['type_pair:'+','.join(sorted([pair['question_type_a'],pair['question_type_b']]))].append(pair)
        pair_summary['groups']={g:{'pairs':len(ps),'images':len({p['image_id'] for p in ps}),
            'gate_l1_change':describe([p['gate_l1_change'] for p in ps]),
            'layers':{str(l):{'js':describe([p['map_js_nats'][str(l)] for p in ps]),
                             'gate_change':describe([p['gate_abs_change'][str(l)] for p in ps])} for l in layers}}
            for g,ps in pair_groups.items()}
        image_gates=defaultdict(list)
        for row in derived:image_gates[row['image_id']].append(row)
        summary['image_coverage']={str(l):{'images':len(image_gates),
            'images_with_any_gate_below_001':sum(any(r['layers'][str(l)]['gate']<.01 for r in rs) for rs in image_gates.values()),
            'images_with_all_gates_below_001':sum(all(r['layers'][str(l)]['gate']<.01 for r in rs) for rs in image_gates.values()),
            'images_with_any_gate_below_005':sum(any(r['layers'][str(l)]['gate']<.05 for r in rs) for rs in image_gates.values()),
            'images_with_all_gates_below_005':sum(all(r['layers'][str(l)]['gate']<.05 for r in rs) for rs in image_gates.values())}
            for l in layers}
        strata={}
        for l in layers:
            strata[str(l)]={}
            for label,lo,hi in [('below_001',0,.01),('001_to_005',.01,.05),('005_to_05',.05,.5),('at_least_05',.5,1.000001)]:
                subset=[r for r in derived if lo<=r['layers'][str(l)]['gate']<hi]
                strata[str(l)][label]={'n':len(subset),'images':len({r['image_id'] for r in subset}),
                    'types':{t:sum(r['question_type']==t for r in subset) for t in sorted({r['question_type'] for r in subset})},
                    'component_norm_share':describe([r['layers'][str(l)]['component_norm_share'] for r in subset]),
                    'component_norm':describe([r['layers'][str(l)]['component_norm'] for r in subset]),
                    'map_entropy':describe([r['layers'][str(l)]['map_entropy'] for r in subset]),
                    'uniform_relative_difference':describe([r['layers'][str(l)]['component_uniform_relative_difference'] for r in subset])}
        strata_all[name]=strata
        write(target/'derived_rows.json',derived);write(target/'grouped_statistics.json',summary)
        write(target/'same_image_question_pairs.json',question_pairs);write(target/'same_image_summary.json',pair_summary)
        write(target/'gate_strata.json',strata)
        summaries[name]=summary;pairs_all[name]=pair_summary;all_rows[name]=derived
        header=np.load(target/'head_parameters.npz',allow_pickle=False)
        headers.append({'model':name,'bias_norm':norm(header['bias']),'layer_matrix_frobenius':{str(l):norm(header['weight'][:,j*64:(j+1)*64]) for j,l in enumerate(layers)}})
    write(out/'comparison.json',{'manifest':meta,'models':summaries,'same_image_question_pairs':pairs_all,'output_head_norms':headers,'gate_strata':strata_all,
        'definitions':{'component':'W_l ReLU(beta_l projected_LN_summary_l), output bias separate',
        'component_norm_share':'norm(b_l) / sum_l norm(b_l); representation measure, no causal interpretation',
        'cancellation':'1 - norm(sum b_l) / sum norm(b_l), bias excluded unless explicitly named',
        'uniform_relative_difference':'norm(actual-uniform) / (0.5*(norm(actual)+norm(uniform))); near-zero denominator -> missing',
        'map_entropy':'entropy of actual merged probability / log(M)',
        'same_image_pairs':'all distinct normalized question texts in the fixed first1000; pair-weighted, not independent examples'}})
    with (out/'quantiles.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','group','n','layer','metric','count','missing','mean','median','p10','p90','min','max'])
        for name,groups in summaries.items():
            for group,s in groups.items():
                if group=='image_coverage':continue
                for l,metrics in s['layers'].items():
                    for metric,q in metrics.items():writer.writerow([name,group,s['n'],l,metric,*[q.get(k) for k in ['count','missing','mean','median','p10','p90','min','max']]])
    plot(out,summaries)
    lines=['# V1 three-branch usage audit','',f"Manifest: {meta['samples']} original-order Validation questions, {meta['images']} images; SHA256 {digest}.",
        '', 'Fixed first-1000 diagnostic subset; all answers/correctness reused from archived Validation predictions.',
        'Norm shares are representation descriptors. Selection gates, projections and vector cancellation are reported separately.',
        '', '| Model/layer | gate median | gate<.01 | gate<.05 | largest gate | component-share median | map entropy median | uniform TV median |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,summary in summaries.items():
        for l,s in summary['all']['layers'].items():
            lines.append(f"|{name}/{l}|{s['gate']['median']:.6f}|{s['gate_below_001']['mean']:.4f}|{s['gate_below_005']['mean']:.4f}|{s['largest_gate']['mean']:.4f}|{s['component_norm_share']['median']:.6f}|{s['map_entropy']['median']:.6f}|{s['map_tv_uniform']['median']:.6f}|")
    lines+=['','Detailed mean/median/P10/P90, type/correctness groups and same-image pairs: comparison.json and quantiles.csv.',
        'Uniform pooling reuses the same-forward native Value, original LN/projection/gate/output weights; counterfactual vectors are diagnostic representations.',
        'Each checkpoint retains its own archived outcomes. Cross-model associations and low gates alone do not establish causal branch importance.',
        'The first1000 are not a random or full-Validation estimate; same-image pair statistics weight question pairs.',
        'User-generated GPU exports are required; this analyzer uses NumPy/Pillow only.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[THREE_BRANCH_CPU_ANALYSIS_DONE]',out)

def plot(out,models):
    from PIL import Image,ImageDraw,ImageFont
    path=next(p for p in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
    font=ImageFont.truetype(str(path),20);small=ImageFont.truetype(str(path),15)
    sets=[('branch_usage',[('gate','Gate'),('component_norm_share','Component norm share'),('weighted_block_norm','Weighted 64-D norm')]),
          ('map_characteristics',[('map_entropy','Merged-map entropy'),('map_tv_uniform','TV to uniform'),('map_peak_multiplier','Peak / uniform probability')]),
          ('uniform_readout',[('component_uniform_relative_difference','Actual/uniform relative difference'),('component_norm','Component norm'),('normalized_summary_norm','LN summary norm')])]
    colors=[(51,102,170),(220,125,50),(60,150,100)]
    for filename,metrics in sets:
        image=Image.new('RGB',(1500,940),'white');draw=ImageDraw.Draw(image)
        draw.text((20,10),'Median with P10-P90; original Validation first1000; norm shares are representation metrics',font=font,fill='black')
        maxima={key:max(s['all']['layers'][l][key]['p90'] for s in models.values() for l in s['all']['layers'])*1.1 or 1 for key,_ in metrics}
        for col,(name,summary) in enumerate(models.items()):
            for row,(key,title) in enumerate(metrics):
                x=col*500+65;y=row*285+75
                draw.text((x-35,y-30),f'{name}: {title}',font=small,fill='black')
                series=list(summary['all']['layers'].items());maximum=maxima[key]
                if key in ('gate','component_norm_share','map_entropy','map_tv_uniform'):maximum=1
                draw.line((x,y,x,y+180,x+370,y+180),fill='gray',width=2)
                for tick in (0,.5,1):
                    yy=y+180-180*tick;draw.text((x-50,yy-8),f'{maximum*tick:.2g}',font=small,fill='black')
                for j,(layer,q) in enumerate(series):
                    q=q[key];xx=x+65+j*115
                    coords=[y+180-180*q[k]/maximum for k in ('p10','median','p90')]
                    draw.line((xx,coords[0],xx,coords[2]),fill=colors[j],width=3)
                    draw.ellipse((xx-6,coords[1]-6,xx+6,coords[1]+6),fill=colors[j])
                    draw.text((xx-20,y+190),f'L{layer}',font=small,fill='black')
        image.save(out/(filename+'.png'),dpi=(200,200))
    types=['yes/no','what','where','how','other','when','why']
    image=Image.new('RGB',(1450,650),'white');draw=ImageDraw.Draw(image)
    draw.text((20,15),'Middle branch by question type: mean gate / mean component norm share',font=font,fill='black')
    for col,(name,summary) in enumerate(models.items()):
        middle=list(summary['all']['layers'])[1];x=25+col*475
        draw.text((x,65),f'{name}: layer {middle}',font=font,fill='black')
        for row,t in enumerate(types):
            s=summary.get('question_type:'+t)
            if not s:continue
            y=110+row*67;draw.text((x,y),f'{t} (n={s["n"]})',font=small,fill='black')
            for k,key in enumerate(['gate','component_norm_share']):
                v=s['layers'][middle][key]['mean'];xx=x+150+k*135
                draw.rectangle((xx,y-5,xx+125,y+35),fill=(int(245-170*v),int(245-100*v),int(255-30*v)))
                draw.text((xx+20,y),f'{v:.3f}',font=font,fill='black')
    draw.text((25,610),'Same 1000 questions; rare types have small n. Colors use a shared [0,1] scale.',font=small,fill='black')
    image.save(out/'middle_by_question_type.png',dpi=(200,200))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    analyze(p.parse_args().output)
