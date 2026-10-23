import argparse
import json
import subprocess
from pathlib import Path

import torch
import transformers
import accelerate
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed

from model import QDPT
from pathvqa.data import Collator, TrainSet
from pathvqa.evaluate import evaluate


class SaveEpochs(TrainerCallback):
    def __init__(self, output):
        self.output = output

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if epoch == 5:
            kwargs['model'].save(self.output / 'checkpoints' / f'epoch_{epoch}')


def train(model, processor, data, output):
    model.base_model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.optimizer_groups(), betas=(.9, .999), eps=1e-8)
    trainer = Trainer(model=model, processing_class=processor,
        args=TrainingArguments(output_dir=str(output / 'trainer'), num_train_epochs=5,
            per_device_train_batch_size=2, gradient_accumulation_steps=16,
            learning_rate=1e-4, weight_decay=0., warmup_ratio=.03,
            lr_scheduler_type='linear', max_grad_norm=1., logging_steps=20,
            save_strategy='no', bf16=True, gradient_checkpointing=False,
            dataloader_num_workers=2, remove_unused_columns=False, report_to='none',
            seed=44, data_seed=42),
        train_dataset=TrainSet(data, processor), data_collator=Collator(processor),
        optimizers=(optimizer, None), callbacks=[SaveEpochs(output)])
    trainer.model_accepts_loss_kwargs = False  # 累积让Trainer自己除
    report = dict(model_seed=44, data_seed=42, epochs=5, saved_epochs=[5],
        per_device_batch_size=2, gradient_accumulation_steps=16, effective_batch_size=32,
        max_length=2048, warmup_ratio=.03, scheduler='linear', max_grad_norm=1.,
        optimizer='AdamW', betas=[.9, .999], eps=1e-8, weight_decay=0.,
        group_learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
        trainer_model_accepts_loss_kwargs=trainer.model_accepts_loss_kwargs,
        accelerator_gradient_accumulation_steps=trainer.accelerator.gradient_accumulation_steps,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__, accelerate=accelerate.__version__),
        git_commit=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'], text=True).strip())
    print(json.dumps(report, indent=2))
    report['train_metrics'] = trainer.train().metrics
    trainer.save_state()
    (output / 'train_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('action', choices=['run', 'train', 'eval'])
    cli.add_argument('--model', default='/root/autodl-tmp/model')
    cli.add_argument('--data', type=Path, default=Path('/root/autodl-tmp/dataset/pathVQA'))
    cli.add_argument('--output', type=Path, default=Path('outputs/v1_seed44'))
    cli.add_argument('--checkpoint', type=Path)
    cli.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(44)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     device_map='auto', trust_remote_code=True)
    model = QDPT(base)
    checkpoint = args.output / 'checkpoints/epoch_5'
    if args.action in ('run', 'train'):
        train(model, processor, args.data, args.output)
    else:
        checkpoint = args.checkpoint or checkpoint
        model.load(checkpoint)
    if args.action in ('run', 'eval'):
        evaluate(model, processor, args.data, args.split, args.output / f'eval_{args.split}/epoch_5', checkpoint)


if __name__ == '__main__':
    main()
