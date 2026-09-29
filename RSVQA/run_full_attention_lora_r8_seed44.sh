#!/bin/bash
set -euo pipefail

ROOT_DIR="${MMRL_ROOT_DIR:-/root/autodl-tmp/Qwen3-VL-modify-test}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA_ROOT="${RSVQA_DATA_ROOT:-/root/autodl-tmp/dataset/RSVQA/6344334}"
OUTPUT_ROOT="${RSVQA_LORA_OUTPUT_ROOT:-$ROOT_DIR/RSVQA/outputs/lora}"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"
EXPERIMENT="rsvqa_lr_lora_full_model_attention_r8_seed44"

available_output_dir() {
  local candidate="$OUTPUT_ROOT/${EXPERIMENT}_${RUN_DATE}"
  local index=1
  while [ -e "$candidate" ]; do
    candidate="$OUTPUT_ROOT/${EXPERIMENT}_${RUN_DATE}_${index}"
    index=$((index + 1))
  done
  printf '%s\n' "$candidate"
}

OUTPUT_DIR="$(available_output_dir)"
TRAIN_LOG="$OUTPUT_DIR/train.log"
EVAL_DIR="$OUTPUT_DIR/eval_test/epoch_3"
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR"

printf '[RSVQA_LORA_CONFIG] experiment=%s dataset=RSVQA-LR train_split=train eval_split=test model_seed=44 data_seed=42 rank=8 alpha=16 dropout=0.05 target=full_model_attention expected_trainable=7077888 microbatch=2 accumulation=16 effective_batch=32 epochs=3 warmup=0.03 scheduler=linear max_grad_norm=1 learning_rate=1e-4 model_accepts_loss_kwargs=false auto_shutdown=false git_commit=%s output=%s\n' \
  "$EXPERIMENT" "$(git -C "$ROOT_DIR" rev-parse HEAD)" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/config.log"

cd "$ROOT_DIR"
python -m pathvqa.train_visual_lora \
  --dataset rsvqa_lr \
  --data-root "$DATA_ROOT" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT" \
  --target-scope full_model \
  --last-n-vision-layers 24 \
  --rank 8 \
  --expected-trainable-parameters 7077888 \
  --epochs 3 \
  --seed 44 \
  --data-seed 42 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --gradient-accumulation 16 \
  --dataloader-workers 2 \
  --max-length 2048 \
  --correct-loss-accumulation \
  2>&1 | tee "$TRAIN_LOG"

python -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p,encoding="utf-8")); assert d["dataset"]=="rsvqa_lr"; assert d["target_scope"]=="full_model_attention"; assert d["rank"]==8 and d["alpha"]==16; assert d["epochs"]==3; assert d["per_device_train_batch_size"]==2 and d["gradient_accumulation_steps"]==16 and d["effective_batch_size"]==32; assert d["parameter_counts"]["trainable"]==7077888; assert d["trainer_model_accepts_loss_kwargs"] is False; print("[RSVQA_LORA_TRAIN_REPORT_AUDIT] pass=True trainable=7077888 effective_batch=32 normalized=True")' \
  "$OUTPUT_DIR/train_report.json" | tee "$OUTPUT_DIR/train_report_audit.log"

CHECKPOINT="$OUTPUT_DIR/checkpoints/epoch_3"
if [ ! -d "$CHECKPOINT" ]; then
  echo "[ERR] Missing fixed epoch3 checkpoint: $CHECKPOINT" >&2
  exit 1
fi

python -m RSVQA.rsvqa_lr_official_eval \
  --data-root "$DATA_ROOT" \
  --split test \
  --backend lora \
  --base-model "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EVAL_DIR" \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 42 \
  --overwrite \
  2>&1 | tee "$OUTPUT_DIR/eval_test_epoch3.log"

echo "[RSVQA_LORA_DONE] output=$OUTPUT_DIR checkpoint=$CHECKPOINT evaluation=$EVAL_DIR auto_shutdown=false"
