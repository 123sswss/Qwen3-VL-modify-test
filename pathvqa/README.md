# PathVQA training and evaluation

This directory is the PathVQA counterpart of `slake/`. It reads the official
Hugging Face Parquet release directly and keeps PathVQA-specific data handling,
training, metrics, and inference outputs isolated from SLAKE.

Expected data layout:

```text
/root/autodl-tmp/dataset/pathVQA/
  train-00000-of-00007-*.parquet
  ...
  validation-00000-of-00003-*.parquet
  ...
  test-00000-of-00003-*.parquet
  ...
```

Install the Parquet reader once in the same Python environment used for
training:

```bash
python -m pip install -r pathvqa/requirements.txt
```

Run the complete seed44 MMRL training and test evaluation:

```bash
RUN_TARGET=pathvqa bash run_experiment.sh
```

The default run uses the same controlled SLAKE method configuration:

- Stage 1 specialist/general Gate training with PathVQA positives and the
  existing LLaVA/general negatives;
- Stage 3 `open_full_ce` MMRL training for three epochs;
- 128-slot Attention Pooling, shared Cross-Attention, same-initialized layer
  MLPs, 40 Rep tokens, and Relation weight 0.05;
- seed44 for initialization and seed42 for deterministic data order.

The Parquet loader verifies all split shards and official row counts before
loading the model. It uses an Arrow cache under
`/root/autodl-tmp/dataset/pathVQA/.hf_cache` by default.

Evaluation reports normalized exact-match Overall, Yes/No, and Free-form
accuracy, question-prefix breakdowns, free-form macro accuracy, generation
timing, and a 95% bootstrap interval clustered by decoded image content. The
image hash reconstructs same-image QA groups even though the Hugging Face
conversion removed original image identifiers.

Standalone evaluation of an existing MMRL checkpoint:

```bash
python pathvqa/pathvqa_official_eval.py \
  --backend mmrl \
  --base-model /root/autodl-tmp/model \
  --checkpoint /path/to/checkpoint/final \
  --data-root /root/autodl-tmp/dataset/pathVQA \
  --output-dir /path/to/eval \
  --overwrite
```

No PathVQA smoke-test target is provided. The first scheduled run is the full
seed44 experiment.

## Visual Attention LoRA-r128

The first LoRA baseline is intentionally limited to one configuration: LoRA
is attached to the qkv/proj attention linears in all 24 visual layers, while
the language model and visual MLPs remain frozen. Rank is fixed at 128 and
alpha at 256.

```bash
RUN_TARGET=pathvqa_lora_visual_attn_r128 bash run_experiment.sh
```

The run saves each of three adapter epochs, evaluates all three on the
validation split, selects the highest validation Overall score with the
earliest epoch as the tie-breaker, and evaluates only that checkpoint on test.
The selected checkpoint and scores are written to `selected_result.tsv`.

Run the LoRA experiment and then the frozen Base-model test inference in one
serial job:

```bash
RUN_TARGET=pathvqa_lora_visual_attn_r128_then_base bash run_experiment.sh
```

The Base result is stored separately under `pathvqa/outputs/base/`. To run or
repeat only the Base inference, use `RUN_TARGET=pathvqa_base`.
