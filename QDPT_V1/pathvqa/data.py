import random
from pathlib import Path

import torch
from datasets import Image, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def load_split(root, split):
    root = Path(root)
    files = [str(p) for p in sorted(root.glob(f'{split}-*.parquet'))]
    return load_dataset('parquet', data_files=files, split='train',
                        cache_dir=str(root / '.hf_cache')).cast_column('image', Image())


def encode(processor, image, question, answer=None):
    text = question if answer is not None else question + '\nAnswer with only a short answer.'
    messages = [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=text)])]
    if answer is not None:
        messages.append(dict(role='assistant', content=[dict(type='text', text=answer)]))
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=answer is None)
    inputs = dict(processor(images=image, text=chat, padding=False, truncation=False,
                            max_length=2048, return_tensors='pt'))
    ids = processor.tokenizer.encode(question, add_special_tokens=False)
    inputs['question_source_ids'] = torch.tensor([ids], dtype=torch.long)
    inputs['question_source_mask'] = torch.ones((1, len(ids)), dtype=torch.bool)
    if answer is not None:
        tokens = inputs['input_ids'][0].tolist()
        header = processor.tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        prefix = processor.tokenizer.encode('<|im_start|>assistant', add_special_tokens=False)
        start = next(i for i in range(len(tokens)) if tokens[i:i + len(header)] == header) + len(prefix)
        labels = inputs['input_ids'].clone()
        labels[:, :start] = -100
        inputs['labels'] = labels
    return inputs


class TrainSet(Dataset):
    def __init__(self, root, processor):
        self.rows = load_split(root, 'train')
        self.order = list(range(len(self.rows)))
        random.Random(42).shuffle(self.order)
        self.processor = processor

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        row = self.rows[self.order[index]]
        image = row['image'].convert('RGB')
        inputs = encode(self.processor, image, row['question'].strip(), row['answer'].strip())
        image.close()
        return inputs


class Collator:
    def __init__(self, processor):
        self.pad = processor.tokenizer.pad_token_id

    def __call__(self, rows):
        padding = dict(input_ids=self.pad, attention_mask=0, labels=-100,
                       question_source_ids=0, question_source_mask=False)
        batch = {key: pad_sequence([r[key][0] for r in rows], batch_first=True, padding_value=value)
                 for key, value in padding.items()}
        batch['pixel_values'] = torch.cat([r['pixel_values'] for r in rows])
        batch['image_grid_thw'] = torch.cat([r['image_grid_thw'] for r in rows])
        return batch
