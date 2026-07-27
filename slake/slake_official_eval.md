# Isolated SLAKE evaluation

All files in this evaluation path are isolated under `slake/`. Existing
training, test, and PEFT scripts are imported read-only and are not modified.

## Expected data

Use the official SLAKE split manifest and image root:

```text
/root/autodl-tmp/dataset/SLAKE/
  slake_test.json
  imgs/
    xmlab0/
      source.jpg
    ...
```

The question JSON may be a list directly or contain a list under `questions`,
`annotations`, `data`, or `records`. Records must provide:

- `question_id`, `qid`, or `id`
- `question`
- `answer`
- `img_name`, `image_name`, or `image`

The evaluator also preserves `q_lang`, `answer_type`, `question_type`, and
`base_type` when available.

## Audit without loading a model

```bash
python slake/slake_official_eval.py \
  --questions /root/autodl-tmp/dataset/SLAKE/slake_test.json \
  --image-root /root/autodl-tmp/dataset/SLAKE/imgs \
  --base-model /root/autodl-tmp/model \
  --output-dir ./slake_outputs/audit \
  --language en \
  --audit-only
```

The default filter is English plus `split=test` when those metadata fields are
present. Missing split metadata is accepted because many preprocessed official
English manifests omit it.

## Base Qwen3-VL

```bash
python slake/slake_official_eval.py \
  --backend base \
  --base-model /root/autodl-tmp/model \
  --questions /root/autodl-tmp/dataset/SLAKE/slake_test.json \
  --image-root /root/autodl-tmp/dataset/SLAKE/imgs \
  --output-dir ./slake_outputs/base_en \
  --language en \
  --overwrite
```

## MMRL checkpoint

```bash
python slake/slake_official_eval.py \
  --backend mmrl \
  --base-model /root/autodl-tmp/model \
  --checkpoint /path/to/mmrl/final \
  --questions /root/autodl-tmp/dataset/SLAKE/slake_test.json \
  --image-root /root/autodl-tmp/dataset/SLAKE/imgs \
  --output-dir ./slake_outputs/mmrl_en \
  --language en \
  --overwrite
```

## LoRA checkpoint

```bash
python slake/slake_official_eval.py \
  --backend lora \
  --base-model /root/autodl-tmp/model \
  --checkpoint /path/to/lora/final \
  --questions /root/autodl-tmp/dataset/SLAKE/slake_test.json \
  --image-root /root/autodl-tmp/dataset/SLAKE/imgs \
  --output-dir ./slake_outputs/lora_en \
  --language en \
  --overwrite
```

Other supported backends are `lora-vision`, `dora`, `dora-vision`, `ia3`, and
`adapter`.

## Smoke test and resume

Use `--limit 5` for a real-model smoke test. A full run writes every completed
sample to `slake_progress.jsonl`. Resume an interrupted run with `--resume`.

```bash
python slake/slake_eval_smoke_test.py
```

## Outputs and metric

Each run writes:

- `slake_predictions.json`: standard `question_id`/`answer` result list
- `slake_summary.json`: overall and grouped accuracy
- `slake_comparisons.json`: normalized GT/prediction comparison
- `slake_details.json`: raw model output and per-sample status
- `slake_progress.jsonl`: crash-safe incremental progress

The primary metric is normalized exact match compatible with the VQA evaluator
used by common SLAKE implementations. It reports overall accuracy and accuracy
grouped by answer type, question type, and language. The publication-comparable
default is `--answer-mode raw`; no fuzzy matching or LLM judge is used.
