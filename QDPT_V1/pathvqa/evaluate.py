import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from .data import encode, load_split


def normalize(text):
    text = unicodedata.normalize('NFKC', text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return re.sub(r'\s+([,.;:!?])', r'\1', text).rstrip(' .')


def score(rows):
    def accuracy(group):
        return round(100 * sum(r['correct'] for r in group) / len(group), 4)
    def by(key):
        return {name: accuracy([r for r in rows if r[key] == name]) for name in sorted({r[key] for r in rows})}
    clusters = defaultdict(list)
    for row in rows:
        clusters[row['image_id']].append(row['correct'])
    images = sorted(clusters)
    counts = [(sum(clusters[i]), len(clusters[i])) for i in images]
    rng = random.Random(42)
    samples = []
    for _ in range(2000):
        draws = [counts[rng.randrange(len(images))] for _ in images]
        samples.append(100 * sum(c for c, n in draws) / sum(n for c, n in draws))
    answers, questions = by('answer_type'), by('question_type')
    free = [v for k, v in questions.items() if k != 'yes/no']
    return dict(count=len(rows), overall_accuracy=accuracy(rows),
                yes_no_accuracy=answers['yes/no'], free_form_accuracy=answers['free-form'],
                per_answer_type_accuracy=answers, per_question_type_accuracy=questions,
                free_form_question_type_macro_accuracy=round(sum(free) / len(free), 4),
                clustered_bootstrap=dict(point_estimate=accuracy(rows), confidence_level=.95,
                    lower=round(float(np.quantile(samples, .025)), 4),
                    upper=round(float(np.quantile(samples, .975)), 4),
                    iterations=2000, seed=42, image_clusters=len(images)))


@torch.inference_mode()
def evaluate(model, processor, data, split, output, checkpoint):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.base_model.config.use_cache = True
    device = model.p20.device
    predictions, comparisons = [], []
    with (output / 'pathvqa_progress.jsonl').open('w', encoding='utf-8') as log:
        for i, row in enumerate(tqdm(load_split(data, split))):
            question, reference = row['question'].strip(), row['answer'].strip()
            image = row['image'].convert('RGB')
            digest = hashlib.sha256()
            digest.update(b'RGB')
            digest.update(f'{image.width}x{image.height}'.encode('ascii'))
            digest.update(image.tobytes())
            inputs = {k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                      for k, v in encode(processor, image, question).items()}
            length = inputs['input_ids'].shape[1] + 20
            # [1,t] -> [1,t+20+g]
            ids = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
            # [1,t+20+g] -> [1,g]
            answer = processor.batch_decode(ids[:, length:], skip_special_tokens=True)[0].strip()
            item = dict(question_id=f'pathvqa:{split}:{i}', image_id=digest.hexdigest(),
                        question=question, answer=answer)
            predictions.append(item)
            ref, pred = normalize(reference), normalize(answer)
            answer_type = 'yes/no' if ref in ('yes', 'no') else 'free-form'
            word = re.search(r'[a-z]+', normalize(question)).group()
            kind = word if word in ('what', 'where', 'when', 'why', 'who', 'which', 'how') else 'other'
            comparisons.append(dict(question_id=item['question_id'], image_id=item['image_id'],
                question=question, reference=ref, prediction=pred, answer_type=answer_type,
                question_type='yes/no' if answer_type == 'yes/no' else kind, correct=ref == pred))
            log.write(json.dumps(item, ensure_ascii=False) + '\n')
            log.flush()
            image.close()
    summary = score(comparisons)
    summary.update(split=split, max_new_tokens=32, temperature=0., answer_mode='raw',
                   instruction='short-answer', partial_evaluation=False,
                   checkpoint=str(checkpoint.resolve()), data_root=str(data.resolve()))
    for name, value in [('predictions', predictions), ('comparisons', comparisons), ('summary', summary)]:
        (output / f'pathvqa_{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
