# QDPT

Minimal implementation of the final **Question-Directed Prompt Tuning (QDPT)**
model used with Qwen3-VL-4B-Instruct on PathVQA.

This directory contains only the final Dense-D768 Sandwich configuration. It
does not include historical MMRL variants, ablations, interventions, profiling,
or multi-dataset launch code.

## Method

The Qwen3-VL backbone is frozen. QDPT learns 7,805,184 parameters:

```text
Question tokens --attention pooling--> Q10
Layer17 visual tokens ------------K/V--+--> Cross-Attention --> Z10

Visual Block17 input: [S8; A_v10; Visual]
LLM input:            [P20; Visual; A_t10 + g(Z10); Question]
```

The visual Prompt tokens pass through visual Block17 exactly once and are then
removed. Z10 is written only to the language-side Prompt; there is no dynamic
visual write-back.

## Fixed configuration

| Item | Value |
|---|---:|
| Backbone | Qwen3-VL-4B-Instruct |
| Precision | BF16 |
| Text / visual width | 2560 / 1024 |
| Text Prompts | P20 + A_t10 |
| Visual Prompts | S8 + A_v10 |
| Visual read/injection layer | zero-index 17 |
| Q/Z tokens | 10 |
| Workspace | D768, 16 heads |
| Epochs | 3 |
| Microbatch / accumulation | 2 / 16 |
| Scheduler | 3% warmup, linear decay |

The learning rates are 0.3 for P20+A_t10, 3e-5 for S8, and 1e-4 for A_v10
and the remaining QDPT modules. AdamW uses betas 0.9/0.999, epsilon 1e-8,
zero weight decay, and maximum gradient norm 1.0.

## Installation

```bash
python -m pip install -r requirements.txt
```

The model can be supplied as a local directory or as
`Qwen/Qwen3-VL-4B-Instruct` from Hugging Face.

## PathVQA data

Place the official Hugging Face Parquet shards in one directory:

```text
pathvqa/
├── train-00000-of-00001-....parquet
├── validation-00000-of-00001-....parquet
└── test-00000-of-00001-....parquet
```

The loader checks the official split sizes: 19,654 train, 6,259 validation,
and 6,719 test questions.

## Training

```bash
python train.py \
  --model-path /path/to/Qwen3-VL-4B-Instruct \
  --data-root /path/to/pathvqa \
  --output-dir outputs/seed44 \
  --seed 44
```

The adapter is saved to:

```text
outputs/seed44/checkpoints/epoch_3/
├── dynamic_prompt_config.json
└── dynamic_prompt.pt
```

The frozen Qwen3-VL weights are not copied into the checkpoint.

## Single-image inference

```bash
python infer.py \
  --model-path /path/to/Qwen3-VL-4B-Instruct \
  --checkpoint outputs/seed44/checkpoints/epoch_3 \
  single \
  --image example.jpg \
  --question "What organ is shown?"
```

## PathVQA validation or test

```bash
python infer.py \
  --model-path /path/to/Qwen3-VL-4B-Instruct \
  --checkpoint outputs/seed44/checkpoints/epoch_3 \
  dataset \
  --data-root /path/to/pathvqa \
  --split validation \
  --output-dir outputs/seed44/eval_validation
```

This writes `pathvqa_predictions.json` and `pathvqa_summary.json`. Evaluation
uses normalized exact match and reports Overall, Yes/No, and Free-form accuracy.
An existing prediction file can be evaluated separately:

```bash
python evaluate.py \
  --data-root /path/to/pathvqa \
  --split test \
  --predictions pathvqa_predictions.json \
  --output pathvqa_summary.json
```

## Reference results

The frozen Dense-D768 Sandwich configuration obtained the following PathVQA
scores over seeds 44/45/46:

| Split | Overall | Yes/No | Free-form |
|---|---:|---:|---:|
| Validation | 59.1522 +/- 1.7528 | 91.6480 +/- 1.0300 | 26.7496 +/- 2.5107 |
| Test | 58.8827 +/- 1.8141 | 91.3246 +/- 1.1491 | 26.3926 +/- 2.4909 |

Values are mean +/- sample standard deviation. Prompt optimization is sensitive
to initialization, so multi-seed reporting is recommended.

