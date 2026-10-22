"""CPU tables, print-size case panels and report for the collected evidence."""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main(out):
    cases=out/'cases'; groups=read(cases/'selected_cases.json')
    maps={r['question_id']:r for r in read(cases/'raw_maps.json')}
    provenance=read(cases/'map_provenance.json')
    assert provenance['selected_questions']==6 and 'read_write_lr3e4' in provenance['checkpoint']
    cfg=read(cases/'visualization_config.json'); lo,hi=cfg['global_color_range']
    fontpath=next(p for p in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
    def font(size): return ImageFont.truetype(str(fontpath),size)
    def wrap(draw,text,x,y,width,size=31,color='black'):
        f=font(size);line=''
        for word in text.split():
            proposal=(line+' '+word).strip()
            if draw.textlength(proposal,font=f)>width and line:
                draw.text((x,y),line,font=f,fill=color);y+=size+7;line=word
            else: line=proposal
        draw.text((x,y),line,font=f,fill=color)
        return y+size+7
    answers=[]
    printdir=cases/'grids';printdir.mkdir(exist_ok=True)
    for gi,case in enumerate(groups,1):
        page=Image.new('RGB',(1800,1600),'white');draw=ImageDraw.Draw(page)
        original=Image.open(cases/'images'/f"{case['image_id']}.png").convert('RGB')
        thumb=original.copy();thumb.thumbnail((195,150));page.paste(thumb,(25,10))
        draw.text((245,15),'Original image',font=font(34),fill='black')
        draw.text((245,64),'Question-conditioned visual selection',font=font(36),fill='black')
        # Numeric labels are below the colorbar, with a common shared scale.
        zero=int(1200+550*(-lo)/(hi-lo))
        for x in range(550):
            z=lo+(hi-lo)*x/549;v=z/abs(lo) if z<0 else z/hi
            rgb=(round(255*(1+min(v,0))),round(255*(1-abs(v))),round(255*(1-max(v,0))))
            draw.line((1200+x,100,1200+x,122),fill=rgb)
        for x,label,anchor in [(1200,f'{lo:.2f}','lt'),(zero,'0','mt'),(1749,f'{hi:.2f}','rt')]:
            draw.text((x,130),label,font=font(25),fill='black',anchor=anchor)
        draw.text((25,178),'Question and saved Test answers',font=font(33),fill='black')
        for j,l in enumerate((5,11,17)):
            draw.text((700+j*355,178),f'Layer {l}',font=font(36),fill='black')
        for i,row in enumerate(case['rows']):
            y=245+i*415;draw.line((25,y-15,1775,y-15),fill='#cccccc',width=2)
            yy=wrap(draw,'Q: '+row['question'],25,y,625,size=33)
            yy=wrap(draw,'Ref: '+row['reference'],25,yy+8,625,size=31)
            for method in ('QDPT','CoCoOp','LoRA-r8'):
                answer=row['answers'][method]
                label='CoCoOp-style' if method=='CoCoOp' else method
                yy=wrap(draw,label+': '+answer['prediction'],25,yy+7,625,size=30,
                        color='#17693c' if answer['correct'] else '#a02b2b')
                answers.append(dict(case=case['name'],question_id=row['question_id'],question=row['question'],reference=row['reference'],method=label,answer=answer['prediction'],official_correct=answer['correct']))
            if yy>y+370:
                raise ValueError(f'Answer text overflows at {row["question_id"]}: {yy-y}')
            draw.text((25,y+375),row['question_id'],font=font(24),fill='#555555')
            p=maps[row['question_id']]
            assert p['image_id']==row['image_id']
            for j,l in enumerate((5,11,17)):
                a=np.asarray(p['maps'][str(l)],dtype=np.float64)
                assert np.isfinite(a).all() and abs(a.sum()-1)<1e-5
                z=np.log2(np.maximum(a,np.finfo(float).tiny)*a.size)
                v=np.where(z<0,z/abs(lo),z/hi).clip(-1,1)
                colors=np.ones((*v.shape,3));colors[...,0]=np.where(v<0,1+v,1)
                colors[...,1]=1-abs(v);colors[...,2]=np.where(v>0,1-v,1)
                heat=Image.fromarray(np.rint(colors*255).astype('uint8')).resize(original.size,Image.Resampling.BILINEAR)
                overlay=Image.blend(original,heat,.60);overlay.thumbnail((305,315))
                page.paste(overlay,(690+j*355+(305-overlay.width)//2,y))
                draw.text((695+j*355,y+325),f'Weight: {p["layer_weights"][j]:.3f}',font=font(32),fill='black')
                write(printdir/f'{row["question_id"].replace(":","_")}_layer{l}.json',
                      dict(question_id=row['question_id'],image_id=row['image_id'],layer=l,
                           grid_thw=p['grid_thw'],merger_shape=list(a.shape),probability=a.tolist(),
                           layer_weight=p['layer_weights'][j],checkpoint=provenance['checkpoint']))
        draw.text((25,1513),'Shared scale: log2(p / uniform); blue < 0, white = 0, red > 0; opacity 0.60.',font=font(29),fill='black')
        draw.text((25,1555),'Green / red: official correct / incorrect. QDPT: final read/write LR, seed44, epoch5.',font=font(27),fill='black')
        page.save(cases/f'case_{gi}.png',dpi=(300,300))
        c=canvas.Canvas(str(cases/f'case_{gi}.pdf'),pagesize=(540,480))
        c.drawImage(ImageReader(page),0,0,540,480);c.save()
        original.close()
    with (cases/'answers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(answers[0]));w.writeheader();w.writerows(answers)
    write(cases/'answers.json',answers)
    cfg['manuscript_overlay_alpha']=.60;cfg['print_width_points']=540
    cfg['print_body_font_points']=[9.0,9.3,9.9]
    write(cases/'visualization_config.json',cfg)
    js=read(cases/'same_image_question_statistics.json')
    caption=['# Captions and interpretation','',
        '**Heart (main text).** Question-conditioned visual selection for three fixed PathVQA Test questions sharing a four-chamber gross heart image. Columns show the effective merger-grid selection probabilities from ViT code indices 5, 11 and 17. The shared scale is log2(p/uniform), with identical opacity across panels. Layer weights are the question-conditioned gates. Answers are archived official Test predictions: final-LR QDPT seed44 epoch5, CoCoOp-style epoch3, and LoRA-r8 epoch3. The fixed cases were selected before the current map export.',
        '', '**Burn (appendix).** The same visualization and scoring protocol applied to three preselected questions about one burn image. All six original questions are retained. Lexical and answer-granularity differences are shown under official exact-match scoring. Spatial relevance of highlighted regions requires domain-expert assessment.',
        '', '## Observed case behavior']
    for g in groups:
        rows=[x for x in js if x['case']==g['name']]
        caption.append(f"- {g['name']}: same-image question-pair JS (nats) ranges "+'; '.join(
            f"layer{l}: {min(r['map_js_nats'][str(l)] for r in rows):.6g} to {max(r['map_js_nats'][str(l)] for r in rows):.6g}" for l in (5,11,17))+'.')
    caption+=['','The heart maps vary modestly across the selected questions; the burn maps show larger question-dependent changes. These examples describe branch behavior and archived outcomes. They do not establish localization accuracy or a causal explanation of a correct answer. Gates and map contrast are descriptive. The cases are purposive qualitative examples, with no frequency or representativeness claim.']
    (cases/'captions.md').write_text('\n'.join(caption)+'\n',encoding='utf-8')

    v=read(out/'validation_three_seed_summary.json'); t=read(out/'test_three_seed_summary.json')
    for filename,data in [('validation_three_seed_means.csv',v),('test_three_seed_means.csv',t)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['configuration','group','mean','sample_sd','ddof'])
            w.writeheader()
            for name,record in data.items():
                for group,values in record['three_seed'].items():
                    w.writerow(dict(configuration=name,group=group,**values))
    lines=['# PathVQA final-LR evidence report','',
           '## Three-seed Validation (accuracy %, sample SD)','|Configuration|Overall|Yes/No|Free-form|what|where|','|---|---:|---:|---:|---:|---:|']
    for name,record in v.items():
        cells=[]
        for key in ('overall','yes_no','free_form','type:what','type:where'):
            m=record['three_seed'][key];cells.append(f"{m['mean']:.4f} ± {m['sample_sd']:.4f}")
        lines.append('|'+name+'|'+'|'.join(cells)+'|')
    lines+=['','## Seed44 paired comparisons: full QDPT minus comparator',
            '|Split / comparator|Full|Comparator|Difference pp|95% image-paired CI|Exclusive full / comparator|',
            '|---|---:|---:|---:|---|---:|']
    for p in sorted((out/'paired').glob('*seed44.json')):
        m=read(p)['groups']['overall'];a,b=m['ci95']
        lines.append(f"|{p.stem}|{m['full_accuracy']:.4f}|{m['other_accuracy']:.4f}|{m['delta_full_minus_other']:+.4f}|[{a:+.4f}, {b:+.4f}]|{m['full_only_correct']}/{m['other_only_correct']}|")
    lines+=['','## Test three-seed breakdown', '|Metric|Mean|Sample SD|','|---|---:|---:|']
    for g,m in t['full']['three_seed'].items():
        lines.append(f"|{g}|{m['mean']:.4f}|{m['sample_sd']:.4f}|")
    b=read(out/'branch_current_lr.json');lines+=['','## Existing branch audit: current LR only',
        'Original-order first1000 Validation questions, 122 images. Gates and norm shares are representation descriptors.',
        '|Group/layer|Gate mean|Gate median|Component norm-share mean|Map normalized entropy mean|','|---|---:|---:|---:|---:|']
    for group in ('all','question_type:what','question_type:where'):
        for layer,m in b['groups'][group]['layers'].items():
            lines.append(f"|{group}/{layer}|{m['gate']['mean']:.6f}|{m['gate']['median']:.6f}|{m['component_norm_share']['mean']:.6f}|{m['map_entropy']['mean']:.6f}|")
    plot=Image.new('RGB',(1500,650),'white');draw=ImageDraw.Draw(plot)
    draw.text((25,12),'Current-LR branch descriptors: first1000 Validation / 122 images',font=font(30),fill='black')
    colors=['#4477aa','#ee9944','#55aa77']
    for k,metric_name in enumerate(('gate','component_norm_share','map_entropy')):
        x=45+k*495
        draw.text((x,65),metric_name,font=font(25),fill='black')
        draw.line((x,520,x+410,520),fill='black',width=2)
        for j,layer in enumerate(('5','11','17')):
            value=b['groups']['all']['layers'][layer][metric_name]['mean']
            left=x+j*135
            draw.rectangle((left,520-round(value*360),left+90,520),fill=colors[j])
            draw.text((left,530),f'L{layer}',font=font(25),fill='black')
            draw.text((left,520-round(value*360)-34),f'{value:.3f}',font=font(25),fill='black')
    draw.text((25,602),'Means; norm shares are representation measures. Gates and entropy alone do not establish causal effects.',font=font(24),fill='black')
    plot.save(out/'branch_current_lr.png')
    lines+=['','## Reuse and limitations',
        '- C-static and A delete both adjusted read/write modules. Their retained optimizer groups match the full parent base rates; saved reports include fixed five epochs, data42, batch2/accum16, corrected normalization and same-seed initialization audits. A retains its required alpha calibration. These seed44 comparisons are reusable at the new reference; legacy confidence intervals stay separate.',
        '- Image-only changes both question access and trainable capacity. Its three-seed mean gap is a full-configuration comparison.',
        '- C-qmap has a small overall mean gap; question-conditioned maps have no clear overall advantage in these three runs. Keep the type-specific tradeoff visible.',
        '- Visual18 has a positive three-seed average contribution with seed-dependent magnitude.',
        '- Image-bootstrap CIs condition on each checkpoint and describe image sampling; seed SD describes three training initializations. No seed-level significance test or equivalence/noninferiority claim is made.',
        '- New intervals use 10,000 image-cluster percentile draws with NumPy PCG64 seed42. Legacy intervals may use Python Random and have small Monte Carlo differences. All subgroup analyses are exploratory.',
        '- Scores are recomputed from integer correct counts; last-digit differences from previously rounded user summaries are retained (e.g. full minus qmap mean rounds to 0.1385 pp).',
        '- Raw checkpoint files, complete source predictions and branch NPZ remain at the bound server paths. This package contains only compact reports, source metadata, selected originals and case maps.',
        '- Six authorized prefill calls completed without training or checkpoint writes; answers were reused from complete Test evaluation. GPU exception ends with this handoff.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    header=(out/'README.md').read_text(encoding='utf-8').split('\n## Completion')[0]
    (out/'README.md').write_text(header+'\n## Completion\nAll requested CPU statistics and six-case GPU maps are complete. See report.md, cases/captions.md and cases/case_1.pdf (heart), case_2.pdf (burn). No missing evaluation or case answer. Print panels use 0.60 opacity and the same shared probability-ratio scale; grid JSON stores unmodified probabilities.\n\nFinal print panels are authored on Windows with its existing ReportLab runtime. The server lacks ReportLab; no server dependency was installed. The CPU collection and GPU map export had already succeeded before that optional server rendering attempt.\n',encoding='utf-8')
    write(out/'completion.json',dict(status='completed',case_prefills=6,new_generation_calls=0,training=False,
        case_maps=18,full_evaluations_run=0,gpu_exception='expired after this task',
        output=str(out),figure_pdf_count=2))
    print('FINALIZED',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    main(p.parse_args().output.resolve())
