#!/usr/bin/env bash
# Ordinary single-run entry: no automatic shutdown.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
PY="${V1_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATASET="${1:?usage: bash run_v1_read_write_lr3e4_single.sh DATASET SEED}"
SEED="${2:?model seed required}"
case "$DATASET" in
  rsvqa_lr)
    NAME=rsvqa; ROOT=RSVQA; BATCH=4; ACCUM=8
    DATA="${RSVQA_DATA_ROOT:-/root/autodl-tmp/dataset/RSVQA/6344334}" ;;
  pathvqa)
    NAME=pathvqa; ROOT=pathvqa; BATCH=2; ACCUM=16
    DATA="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}" ;;
  slake)
    NAME=slake; ROOT=slake; BATCH=2; ACCUM=16
    DATA="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
EXPERIMENT="${NAME}_v1_read_write_lr3e4_norm_fixed_5ep_seed${SEED}"
OUT="$ROOT/outputs/visual_selection_prefix/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
STAGE=train
trap 'code=$?; printf "stage=%s\nexit_code=%s\n" "$STAGE" "$code" > "$OUT/exit_status.txt"' EXIT
printf 'experiment=%s\ncommit=%s\nmodel_seed=%s\ndata_seed=42\nepochs=5\nbatch_size=%s\ngradient_accumulation=%s\nvalue_projection_lr=0.0003\nprefix_output_lr=0.0003\nexpected_trainable=1864963\n' \
  "$EXPERIMENT" "$(git rev-parse HEAD)" "$SEED" "$BATCH" "$ACCUM" > "$OUT/config.txt"
echo "[V1_READ_WRITE_OUTPUT] $OUT"
"$PY" -m pathvqa.train_visual_selection_prefix --dataset "$DATASET" \
  --model-path "$MODEL" --data-root "$DATA" --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
  --visual-prompt-mode split18 --read-write-learning-rate 3e-4 \
  --model-seed "$SEED" --epochs 5 --save-epochs 3 4 5 \
  --batch-size "$BATCH" --gradient-accumulation "$ACCUM" --correct-loss-accumulation \
  2>&1 | tee "$OUT/train.log"
STAGE=eval_epoch5
case "$DATASET" in
  pathvqa)
    "$PY" -m pathvqa.pathvqa_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
      --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split validation \
      --output-dir "$OUT/eval_validation/epoch_5" 2>&1 | tee "$OUT/eval_validation.log" ;;
  slake)
    "$PY" -m slake.slake_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
      --checkpoint "$OUT/checkpoints/epoch_5" --questions "$DATA/test.json" --image-root "$DATA/imgs" \
      --language all --expected-split test --max-new-tokens 32 --temperature 0 --answer-mode raw \
      --output-dir "$OUT/eval_test/epoch_5" 2>&1 | tee "$OUT/eval_test.log" ;;
  rsvqa_lr)
    "$PY" -m RSVQA.rsvqa_lr_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
      --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split test \
      --output-dir "$OUT/eval_test/epoch_5" 2>&1 | tee "$OUT/eval_test.log" ;;
esac
STAGE=completed
echo "[V1_READ_WRITE_DONE] $OUT"
