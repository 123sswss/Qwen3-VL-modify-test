"""Paper cases: CPU image export/render; explicitly authorized small V1 GPU probe."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_EVAL = ROOT / 'outputs/five_task_test_suite/suite_20261001_134036_879815/pathvqa_v1_5ep_seed44_epoch5_test'
CHECKPOINT = ROOT / 'pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2/checkpoints/epoch_5'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def candidates(out):
    from collections import defaultdict
    sources={name:read(out/'sources'/file) for name,file in [('QDPT','v1.json'),('CoCoOp','cocoop.json'),('LoRA-r8','lora_r8.json')]}
    indexed={name:{r['question_id']:r for r in rows} for name,rows in sources.items()}
    assert all(len(x)==6719 for x in indexed.values())
    groups=defaultdict(list)
    for row in sources['QDPT']:
        row=dict(row);row['answers']={}
        for name,index in indexed.items():
            other=index[row['question_id']]
            assert all(row[k]==other[k] for k in ['image_id','question','reference','answer_type','question_type'])
            row['answers'][name]={'prediction':other['prediction'],'correct':other['correct']}
        groups[row['image_id']].append(row)
    # Shortlist formed from question/answer targets and archived answers BEFORE maps.
    prefixes=['04fd316d12d9','8d2431314d89','346132943f46','c5896ef10fa3','de88d322b17c','3ee3168716aa','62a2abb9041d','860fd4e92bbf']
    result=[]
    for prefix in prefixes:
        image=next(im for im in groups if im.startswith(prefix))
        result.append({'image_id':image,'rows':groups[image]})
    write(out/'candidates.json',result)
    write(out/'selection_policy.json',{'shortlist_before_maps':True,'selection_type':'purposive qualitative selection, not random or representative frequency estimation',
        'criteria':['same image with distinct question/answer targets','include spatial and entity/attribute questions','inspect image clarity before maps','include an error group'],
        'candidate_image_prefixes':prefixes,'models':'original V1 five-epoch seed44 Test; original CoCoOp/r8 three-epoch seed44 Test',
        'all_questions_in_shortlisted_images_retained':True,'warning':'answer differences may reflect wording/exact-match and training-budget differences'})
    print('Shortlist:',len(result),'images',sum(len(c['rows']) for c in result),'questions')

def export_images(out,data):
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    candidates=read(out/'candidates.json')
    wanted={int(x['question_id'].split(':')[-1]):x for c in candidates for x in c['rows']}
    offset=0
    for path in sorted(data.glob('test-*.parquet')):
        f=pq.ParquetFile(path)
        for k in range(f.num_row_groups):
            n=f.metadata.row_group(k).num_rows
            selected=[i for i in wanted if offset<=i<offset+n]
            if selected:
                t=f.read_row_group(k,columns=['image','question','answer'])
                for i in selected:
                    row=wanted[i]; actual=t.slice(i-offset,1).to_pylist()[0]
                    assert actual['question'].strip()==row['question'] and actual['answer'].strip()==row['reference']
                    im=actual['image']; raw=im['bytes'] if im['bytes'] is not None else Path(im['path']).read_bytes()
                    decoded=Image.open(io.BytesIO(raw)).convert('RGB')
                    digest=hashlib.sha256()
                    digest.update(decoded.mode.encode('ascii'))
                    digest.update(f'{decoded.width}x{decoded.height}'.encode('ascii'))
                    digest.update(decoded.tobytes())
                    assert digest.hexdigest()==row['image_id']
                    target=out/'images'/f"{row['image_id']}.png"
                    target.parent.mkdir(exist_ok=True)
                    decoded.save(target)
                    decoded.close()
            offset+=n
    assert offset==6719
    print('CPU image export:',len(candidates),'groups')

def select(out):
    candidates=read(out/'candidates.json')
    index={r['question_id']:r for c in candidates for r in c['rows']}
    groups=[('main_heart',[513,524,435],
             'Clear four-chamber gross image; entity/right-side question, inverse location question, and an atrium error. Spatial interpretation needs expert confirmation.'),
            ('main_testis',[2212,2313,2269],
             'Clear paired gross regions; disease/knowledge question, normal-region location, and abscess-entity error. Location baseline differences are exact-match wording; gonococcal causation cannot be established from this image.'),
            ('appendix_burn',[5911,5918,5915],
             'Clear hand/skin injury; burn attribute, tissue location, and open-image answer granularity error. Thermal burns versus thermal burn is an exact-match issue.')]
    selected=[]
    for name,ids,reason in groups:
        rows=[index[f'pathvqa:test:{i}'] for i in ids]
        assert len({r['image_id'] for r in rows})==1
        selected.append({'name':name,'image_id':rows[0]['image_id'],'rationale_before_maps':reason,'rows':rows})
    write(out/'selected_cases.json',selected)
    print('Selection frozen before maps:',[r['question_id'] for c in selected for r in c['rows']])

def maps(out,data):
    import torch
    from pathvqa.data_pipeline import PathVQAParquetStore
    from pathvqa.pathvqa_official_eval import build_prompt
    from slake.visual_selection_prefix_interface import VisualSelectionPrefixInterface
    cfg=read(CHECKPOINT/'visual_selection_prefix_config.json')
    report=read(CHECKPOINT.parent.parent/'train_report.json')
    summary=read(V1_EVAL/'pathvqa_summary.json')
    assert report['experiment']=='pathvqa_v1_norm_fixed_5ep_seed44' and report['epochs']==5
    assert report['model_seed']==44 and report['trainer_model_accepts_loss_kwargs'] is False
    assert cfg['method']=='visual_selection_prefix_p20_v1' and cfg['init_seed']==44
    assert abs(summary['overall_accuracy']-59.8006)<1e-4
    assert Path(summary['checkpoint']).resolve()==CHECKPOINT.resolve()
    store=PathVQAParquetStore(data,'test')
    records={r['question_id']:r for r in store.samples}
    interface=VisualSelectionPrefixInterface(str(CHECKPOINT),'/root/autodl-tmp/model')
    m=interface.model
    captured=[]
    for group in read(out/'selected_cases.json'):
        for row in group['rows']:
            r=records[row['question_id']]
            image=store.load_image(r)
            inputs=interface.prepare_inputs(image,build_prompt(r['question'],None),question=r['question'])
            moved={k:v.to(interface.device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            m.diagnostic_probe={}
            before=m.diagnostic_prefill_calls
            with torch.inference_mode(): m(**moved)
            assert m.diagnostic_prefill_calls==before+1
            p=m.diagnostic_probe
            t,h,w=p['grid'][0].tolist()
            assert t==1 and h%2==0 and w%2==0
            item={'question_id':r['question_id'],'image_id':row['image_id'],'grid_thw':[t,h,w],
                  'layer_weights':p['layer_weights'][0].tolist(),'maps':{},'statistics':{}}
            for l in (5,11,17):
                v=p[f'map{l}'][0].float()
                assert v.numel()==h*w//4 and torch.isfinite(v).all() and abs(float(v.sum())-1)<1e-4
                item['maps'][str(l)]=v.reshape(h//2,w//2).tolist()
                item['statistics'][str(l)]={'min':float(v.min()),'max':float(v.max()),
                    'entropy_normalized':float(-(v*v.clamp_min(1e-12).log()).sum()/math.log(v.numel()))}
            captured.append(item)
            print('CAPTURE',r['question_id'],item['statistics'],flush=True)
            image.close()
            m.diagnostic_probe=None
    write(out/'raw_maps.json',captured)
    write(out/'map_provenance.json',{'checkpoint':str(CHECKPOINT),'source_test':str(V1_EVAL),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'torch':torch.__version__,'layers_0based':[5,11,17],'selected_questions':len(captured),
        'protocol':'eval/inference_mode, single prefill per question, no answer supervision, no parameter update',
        'geometry':'native merger row-major [H/2,W/2]; sum each consecutive 2x2 merge group; prompt tokens excluded'})

def control(out,data):
    original=read(out/'selected_cases.json')[0]
    repeat=out/'determinism_control'
    repeat.mkdir(exist_ok=True)
    write(repeat/'selected_cases.json',[dict(original,rows=original['rows'][:1])])
    maps(repeat,data)
    a=read(out/'raw_maps.json')[0];b=read(repeat/'raw_maps.json')[0]
    errors={str(l):max(abs(x-y) for ar,br in zip(a['maps'][str(l)],b['maps'][str(l)]) for x,y in zip(ar,br)) for l in (5,11,17)}
    errors['layer_weights']=max(abs(x-y) for x,y in zip(a['layer_weights'],b['layer_weights']))
    assert a['question_id']==b['question_id'] and a['grid_thw']==b['grid_thw']
    write(out/'determinism_check.json',{'question_id':a['question_id'],'maximum_absolute_differences':errors,
        'tolerance':1e-6,'passed':all(x<=1e-6 for x in errors.values()),'mode':'independent reload, same image and question; one additional prefill'})
    assert all(x<=1e-6 for x in errors.values())

def render(out):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    fontpath=next(p for p in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
    font=ImageFont.truetype(str(fontpath),20); small=ImageFont.truetype(str(fontpath),17)
    cases=read(out/'selected_cases.json'); probes={p['question_id']:p for p in read(out/'raw_maps.json')}
    scales=[np.log2(np.maximum(np.array(v)*np.array(v).size,1e-12)) for p in probes.values() for v in p['maps'].values()]
    lower=min(float(a.min()) for a in scales)
    upper=max(float(a.max()) for a in scales)
    assert lower<0<upper
    raw_max=max(float(np.array(v).max()) for p in probes.values() for v in p['maps'].values())
    write(out/'visualization_config.json',{'quantity':'log2(probability / uniform probability)',
        'global_color_range':[lower,upper],'center_uniform':0,'overlay_alpha':.45,
        'normalization':'two-slope linear color interpolation, common minimum/zero/maximum across all panels',
        'raw_grid_png_global_probability_max':raw_max,
        'colors':'blue=below uniform, white=uniform, red=above uniform',
        'interpolation':'bilinear overlay; raw probability grids retained in JSON',
        'note':'identical color range/opacity across every selected question and layer'})
    def wrapped(draw,text,x,y,width,f=small):
        line=''
        for word in text.split():
            new=(line+' '+word).strip()
            if draw.textlength(new,font=f)>width and line:
                draw.text((x,y),line,font=f,fill='black');y+=24;line=word
            else:line=new
        draw.text((x,y),line,font=f,fill='black');return y+27
    for gi,c in enumerate(cases,1):
        canvas=Image.new('RGB',(1860,92+310*len(c['rows'])+100),'white');d=ImageDraw.Draw(canvas)
        d.text((20,12),f"Case {gi}: question-conditioned visual selection (ViT indices 5 / 11 / 17)",font=font,fill='black')
        for x,title in [(20,'Question / reference'),(410,'Original'),(650,'Layer 5'),(890,'Layer 11'),(1130,'Layer 17'),(1390,'Saved Test answers')]:d.text((x,53),title,font=font,fill='black')
        for ri,r in enumerate(c['rows']):
            y=92+ri*310;p=probes[r['question_id']]
            yy=wrapped(d,'Q: '+r['question'],20,y,370);wrapped(d,'Reference: '+r['reference'],20,yy+8,370)
            d.text((20,y+275),r['question_id'],font=small,fill='gray')
            orig=Image.open(out/'images'/f"{r['image_id']}.png").convert('RGB')
            im=orig.copy();im.thumbnail((225,245));canvas.paste(im,(410,y))
            for x,l in [(650,5),(890,11),(1130,17)]:
                v=np.array(p['maps'][str(l)],dtype=np.float32);log=np.log2(np.maximum(v*v.size,1e-12))
                a=np.where(log>=0,log/upper,log/abs(lower))
                rgb=np.ones((*a.shape,3),dtype=np.float32)
                rgb[...,0]=np.where(a>=0,1,1+a);rgb[...,1]=1-np.abs(a);rgb[...,2]=np.where(a<=0,1,1-a)
                heat=Image.fromarray((np.clip(rgb,0,1)*255).astype('uint8')).resize(orig.size,Image.Resampling.BILINEAR)
                overlay=Image.blend(orig,heat,.45);overlay.thumbnail((225,245));canvas.paste(overlay,(x,y))
                d.text((x,y+253),f"Layer weight: {p['layer_weights'][(5,11,17).index(l)]:.3f}",font=small,fill='black')
                Image.fromarray((v/raw_max*65535).astype('uint16')).save(out/f"{r['question_id'].replace(':','_')}_layer{l}_grid.png")
            yy=y
            for name in ['QDPT','CoCoOp','LoRA-r8']:
                answer=r['answers'][name]
                yy=wrapped(d,f"{name}: {answer['prediction']} [{'correct' if answer['correct'] else 'incorrect'}]",1390,yy,445)+10
        y=canvas.height-85
        d.text((20,y),f"Common color scale: log2(p / uniform), range [{lower:.2f}, {upper:.2f}], centered at 0; opacity 0.45",font=small,fill='black')
        for k in range(300):
            a=2*k/299-1;color=(255,int(255*(1-abs(a))),int(255*(1-a))) if a>=0 else (int(255*(1+a)),int(255*(1-abs(a))),255)
            d.line((20+k,y+32,20+k,y+48),fill=color)
        canvas.save(out/f'case_{gi}.png',dpi=(300,300));canvas.save(out/f'case_{gi}.pdf',resolution=180)
        orig.close()
    comparisons=[]
    import itertools
    for c in cases:
        for r,s in itertools.combinations(c['rows'],2):
            a,b=probes[r['question_id']],probes[s['question_id']]
            row={'case':c['name'],'question_a':r['question_id'],'question_b':s['question_id'],'map_js_nats':{},'map_total_variation':{}}
            for l in (5,11,17):
                p=np.array(a['maps'][str(l)]).flatten();q=np.array(b['maps'][str(l)]).flatten();m=(p+q)/2
                row['map_js_nats'][str(l)]=float(.5*(np.sum(p*np.log(p/m))+np.sum(q*np.log(q/m))))
                row['map_total_variation'][str(l)]=float(np.abs(p-q).sum()/2)
            row['layer_weight_l1']=float(np.abs(np.array(a['layer_weights'])-b['layer_weights']).sum())
            comparisons.append(row)
    write(out/'same_image_question_statistics.json',comparisons)
    import csv
    with (out/'answers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['case','question_id','question','reference','model','prediction','official_correct'])
        for c in cases:
            for r in c['rows']:
                for name,a in r['answers'].items():
                    writer.writerow([c['name'],r['question_id'],r['question'],r['reference'],name,a['prediction'],a['correct']])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PathVQA paper cases</title>'
        '<style>body{font:16px sans-serif;margin:24px}img{width:100%;max-width:1860px}a{margin-right:20px}</style>'
        '<h1>PathVQA: question-conditioned visual selection</h1><p>Original V1 seed44, epoch5 Test. '
        'Three layers shown together, common color scale and opacity. Answers are archived predictions.</p>'
        '<p><a href="README.md">Analysis and caveats</a><a href="raw_maps.json">Raw maps</a>'
        '<a href="same_image_question_statistics.json">Same-image statistics</a><a href="answers.csv">Answer table</a></p>'
        + ''.join(f'<h2>Case {i+1}: {c["name"]}</h2><img src="case_{i+1}.png"><p><a href="case_{i+1}.pdf">PDF</a></p>'
                  for i,c in enumerate(cases)),encoding='utf-8')
    print('Rendered cases:',len(cases),'common scale',lower,upper)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['candidates','images','select','maps','control','render']);p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/dataset/pathVQA'))
    a=p.parse_args()
    if a.stage=='candidates':candidates(a.output)
    elif a.stage=='images':export_images(a.output,a.data)
    elif a.stage=='select':select(a.output)
    elif a.stage=='maps':maps(a.output,a.data)
    elif a.stage=='control':control(a.output,a.data)
    else:render(a.output)
