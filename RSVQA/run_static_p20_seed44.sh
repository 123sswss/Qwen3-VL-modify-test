#!/usr/bin/env bash
# Corrected pure P20 baseline; save status before Bash shutdown on exit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${RSVQA_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA="${RSVQA_DATA_ROOT:-/root/autodl-tmp/dataset/RSVQA/6344334}"
EXPERIMENT=rsvqa_lr_static_p20_norm_fixed_5ep_b4a8_seed44
OUT="RSVQA/outputs/prompt_tuning/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
STAGE=train
finish() {
  local code=$?
  trap - EXIT
  set +e
  printf 'stage=%s\nexit_code=%s\nauto_shutdown=true\n' "$STAGE" "$code" > "$OUT/exit_status.txt"
  printf 'calling /usr/bin/shutdown %s\n' "$(date -Is)" > "$OUT/shutdown.log"
  sync
  /usr/bin/shutdown >> "$OUT/shutdown.log" 2>&1
  printf 'shutdown_exit_code=%s\n' "$?" >> "$OUT/shutdown.log"
  sync
  exit "$code"
}
trap finish EXIT
printf 'experiment=%s\ncommit=%s\nvisual_prompt_length=0\nexpected_trainable=51200\n' \
  "$EXPERIMENT" "$(git rev-parse HEAD)" > "$OUT/config.txt"
echo "[RSVQA_STATIC_P20_OUTPUT] $OUT"
"$PY" -m RSVQA.train_static_p20 --model-path "$MODEL" --data-root "$DATA" \
  --output-dir "$OUT" --experiment-name "$EXPERIMENT" --prompt-length 20 --visual-prompt-length 0 \
  --expected-trainable-parameters 51200 --seed 44 --data-seed 42 --epochs 5 --save-epochs 3 4 5 \
  --batch-size 4 --gradient-accumulation 8 --learning-rate 0.3 --max-length 2048 --dataloader-workers 2 \
  --correct-loss-accumulation 2>&1 | tee "$OUT/train.log"
STAGE=test_epoch5
"$PY" -m RSVQA.rsvqa_lr_official_eval --backend prompt-tuning --base-model "$MODEL" \
  --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split test --max-new-tokens 32 \
  --output-dir "$OUT/eval_test/epoch_5" 2>&1 | tee "$OUT/eval_test.log"
STAGE=completed
echo "[RSVQA_STATIC_P20_DONE] $OUT"
