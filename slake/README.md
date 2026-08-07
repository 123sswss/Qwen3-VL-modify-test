# SLAKE data pipeline

`SLAKEDataset` scans a directory containing `xmlabN` folders. Each object in
each `question.json` is treated as one training sample, while all questions in
the same folder reuse that folder's `source.jpg`.

```text
SLAKE_ROOT/
  xmlab0/
    source.jpg
    question.json
  xmlab1/
    source.jpg
    question.json
```

Basic usage:

```python
from slake import SLAKEDataCollator, SLAKEDataset

dataset = SLAKEDataset(
    processor=processor,
    image_root="/path/to/imgs",
    languages=("en",),       # None keeps English and Chinese
    base_types=("vqa",),     # None also keeps knowledge-VQA records
    ce_enabled=True,
    seed=42,
)
collator = SLAKEDataCollator(processor)
```

The dataset item and collated batch use the same keys as
`train/data_pipeline.py`: token tensors, visual tensors, labels, expert label,
task metadata, and multimodal sample metadata.

## MMRL training

`train_mmrl.py` is a thin entry point over the existing two-stage trainer. It
uses an official SLAKE train manifest and injects datasets without copying the
optimizers, Stage 3 losses, callbacks, or model construction code.

Stage 1 trains the existing focal-BCE expert/general gate objective with:

- SLAKE questions as positive expert samples;
- the main training pipeline's LLaVA/general manifests as negative samples;
- one multimodal and one text-only prompt view per raw sample;
- prompt-only chat templates, so neither side exposes assistant answers.

Stage 3 then trains only on SLAKE answer supervision. It keeps the Stage 1
gate poolers, task classifier, and both visual backbones frozen. The dynamic
Rep generator pools the question and the natural layer-16 visual states into
128 memory tokens per modality, then uses one residual Cross-Attention to
produce 40 image-specific Rep tokens for each of natural layers 17 through 24.
The old residual Router/Adapter path has been removed. The visual output is:

```text
H_out = H_base + G * (H_rep - H_base)
```

The default Stage 3 hyperparameters are:

```text
MMRL=6e-5
visual memory queries=128
text memory queries=128
Cross-Attention heads=8
relation=0.050
scheduler=constant_with_warmup
```

Full training:

```bash
python slake/train_mmrl.py \
  --questions /root/autodl-tmp/dataset/SLAKE/slake_train.json \
  --image-root /root/autodl-tmp/dataset/SLAKE/imgs \
  --model-path /root/autodl-tmp/model \
  --output-dir ./slake_outputs/mmrl_train
```

Score the resulting checkpoint with the existing official evaluator:

```bash
python slake/slake_official_eval.py \
  --backend mmrl \
  --base-model /root/autodl-tmp/model \
  --checkpoint ./slake_outputs/mmrl_train/final \
  --questions /root/autodl-tmp/dataset/SLAKE/slake_test.json \
  --image-root /root/autodl-tmp/dataset/SLAKE/imgs \
  --output-dir ./slake_outputs/mmrl_test \
  --language en \
  --overwrite
```

FROST-VL training writes only `mmrl_manifest.json` and
`mmrl_delta.safetensors` for model weights. A matching Base model must be
provided during evaluation; full-model legacy checkpoints are not supported.

Use `--mmrl-lr`, `--relation-weight`, `--rp-space-length`,
`--memory-query-count`, `--memory-attention-dim`,
`--projector-hidden-dim`, and `--cross-attention-heads` to change one
training variable without editing source.

The default Stage 1 general data paths match `train/train.py`. Override them
by repeating `--stage1-general-json` and `--stage1-general-image-root`.

## 200-sample GPU smoke test

The smoke test uses exactly 200 deterministic SLAKE train samples. Stage 1
adds balanced general-domain negatives, then Stage 3 consumes those 200 SLAKE
samples. It does not save the large final checkpoint:

```bash
bash slake/run_200_smoke.sh \
  --questions /root/autodl-tmp/dataset/SLAKE/slake_train.json
```

When exactly one of `slake_train.json`, `train.json`, or
`train_questions.json` exists under `/root/autodl-tmp/dataset/SLAKE`,
`--questions` may be omitted.

Before training, the smoke test checks:

- the official manifest and image resolution;
- chat-template assistant boundaries and decoded supervision;
- no overlap between supervised answer tokens and MMRL pooling context;
- collator tensor shapes and image packing;
- exclusion of visual placeholder tokens from text pooling;
- actual updates to both Stage 1 pooling modules;
- one finite full MMRL forward loss.

SLAKE visual placeholders are never truncated. Samples are tokenized in full,
then right-padded to the longest sequence in each batch. The default 2048-token
limit is only a safety ceiling for a complete sample.

After training, it performs two real autoregressive generations and records
the output, gate value, token count, and timing in
`slake_train_report.json`.
