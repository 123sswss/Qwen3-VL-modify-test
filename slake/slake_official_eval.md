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
  --timing-warmup-runs 3 \
  --overwrite
```

The same command evaluates an existing checkpoint and records performance. The
three warmup requests are excluded. `slake_summary.json` reports TTFT, weighted
time per output token (TPOT), decode tokens/s, model generation throughput, and
end-to-end request latency. Per-request timing is preserved in
`slake_details.json` and `slake_progress.jsonl`. TTFT and TPOT are measured
separately; the expensive visual prefill is never averaged into TPOT.

For the 66.95 checkpoint created by the matrix runner, the command is:

```bash
cd /root/autodl-tmp/Qwen3-VL-modify-test
python slake/slake_official_eval.py \
  --backend mmrl \
  --base-model /root/autodl-tmp/model \
  --checkpoint /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/mmrl/slake_mmrl_constraint_matrix_r0500_w0008_d058_seed44_20260731/final \
  --questions /root/autodl-tmp/dataset/slake/test.json \
  --image-root /root/autodl-tmp/dataset/slake/imgs \
  --output-dir /root/autodl-tmp/Qwen3-VL-modify-test/slake/outputs/eval_r0500_w0008_timing \
  --language all \
  --max-new-tokens 32 \
  --timing-warmup-runs 3 \
  --overwrite
```

If the checkpoint folder has an automatically appended suffix, replace the
checkpoint path with the actual directory containing
`final/mmrl_manifest.json` and `final/mmrl_delta.safetensors`.

The compact format deliberately contains no Base-model weights. The evaluator
rebuilds frozen weights and Rep blocks from `--base-model`, then strictly loads
the saved FROST-VL delta.

## Evaluate all existing comparison checkpoints

The evaluation-only runner loads the already trained Base, LoRA, and DoRA
checkpoints. It never calls a training script:

```bash
cd /root/autodl-tmp/Qwen3-VL-modify-test
bash slake/eval_existing_checkpoints.sh
```

Use a two-sample server smoke test before the full run:

```bash
EVAL_LIMIT=2 bash slake/eval_existing_checkpoints.sh
```

Checkpoint, dataset, and output roots can be overridden without editing code:

```bash
CHECKPOINT_ROOT=/path/to/visual_peft_comparison \
RESULT_ROOT=/path/to/timing_results \
bash slake/eval_existing_checkpoints.sh
```

The runner continues after individual failures and writes
`comparison_summary.json`, `comparison_summary.csv`, and
`comparison_summary.md` under `timing_results/`.

## Fair vision-only LoRA checkpoints

The primary matched-scope baseline is LoRA on the last eight vision-encoder
layers. Two rank-32 checkpoints are produced by the existing training script:

- `last8_full_linear_rank32`: attention and MLP linear layers in vision blocks
  16 through 23
- `last8_attention_rank32`: attention `qkv` and `proj` layers in vision blocks
  16 through 23

Evaluate the full-linear variant with:

```bash
python slake/slake_official_eval.py \
  --backend lora-vision-last8 \
  --base-model /root/autodl-tmp/model \
  --checkpoint ./loraTest/runs/lora_vision_last8/last8_full_linear_rank32/final \
  --questions /root/autodl-tmp/dataset/slake/test.json \
  --image-root /root/autodl-tmp/dataset/slake/imgs \
  --output-dir ./slake_outputs/lora_vision_last8_full_linear_r32_en \
  --language en \
  --overwrite
```

Use the same backend with
`./loraTest/runs/lora_vision_last8/last8_attention_rank32/final` for the
attention-only variant. The existing `lora-vision` backend evaluates LoRA over
all vision-attention layers and is a useful larger-scope comparison.

## Full-VLM LoRA checkpoint

Full-VLM LoRA also adapts language-model attention layers. It is supported for
completeness, but it is not the primary parameter-scope-matched comparison for
a method that trains only the final eight vision layers.

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
