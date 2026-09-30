#!/bin/bash
set -euo pipefail

# Independent experiment: never called by the active C-ablation queue.
ROOT_DIR="${MMRL_ROOT_DIR:-/root/autodl-tmp/Qwen3-VL-modify-test}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA_ROOT="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
CACHE_ROOT="${PATHVQA_CACHE_ROOT:-$DATA_ROOT/.hf_cache}"
OUTPUT_ROOT="${PATHVQA_LORA_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/lora}"
V1_ROOT="${PATHVQA_V1_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/visual_selection_prefix}"
BASELINE_RUN="${PATHVQA_V1_5EP_BASELINE_RUN:-$V1_ROOT/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2}"
BASELINE_EVAL="$BASELINE_RUN/eval_validation/epoch_5"
EXPERIMENT="pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"
OUTPUT_DIR="$OUTPUT_ROOT/${EXPERIMENT}_${RUN_DATE}"
INDEX=1
while [ -e "$OUTPUT_DIR" ]; do
  OUTPUT_DIR="$OUTPUT_ROOT/${EXPERIMENT}_${RUN_DATE}_${INDEX}"
  INDEX=$((INDEX + 1))
done
mkdir -p "$OUTPUT_DIR"
STAGE="prechecks"

finish() {
  local code=$?
  trap - EXIT
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$STAGE" "$code" >> "$OUTPUT_DIR/stage_status.tsv"
  printf '{"experiment":"%s","final_stage":"%s","exit_code":%s,"auto_shutdown":false}\n' \
    "$EXPERIMENT" "$STAGE" "$code" > "$OUTPUT_DIR/run_exit_status.json"
  echo "[LORA_R2_EXIT] stage=$STAGE exit_code=$code output=$OUTPUT_DIR auto_shutdown=false"
  exit "$code"
}
trap finish EXIT

cd "$ROOT_DIR"
if [ ! -f "$BASELINE_EVAL/pathvqa_summary.json" ] || [ ! -f "$BASELINE_EVAL/pathvqa_comparisons.json" ]; then
  echo "[ERR] Original five-epoch V1 baseline artifacts missing: $BASELINE_EVAL" >&2
  exit 1
fi
python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); assert d["split"]=="validation" and d["count"]==6259; assert abs(d["overall_accuracy"]-59.3386)<1e-4; print("[LORA_R2_BASELINE] fixed_V1_epoch5=59.3386")' \
  "$BASELINE_EVAL/pathvqa_summary.json" | tee "$OUTPUT_DIR/baseline_precheck.log"

echo "[LORA_R2_CONFIG] experiment=$EXPERIMENT git_commit=$(git rev-parse HEAD) dataset=PathVQA model_seed=44 data_seed=42 rank=2 alpha=4 alpha_over_r=2 dropout=0.05 rslora=false target=visual24_qkv_proj+language36_qkvo target_modules=192 expected_trainable=1769472 budget=approximately_matched_to_V1_1864963 difference_percent=5.12 batch=1 accumulation=32 effective_batch=32 epochs=5 save_epochs=3,4,5 primary_epoch=5 learning_rate=1e-4 warmup=0.03 scheduler=linear clip=1 model_accepts_loss_kwargs=false auto_shutdown=false output=$OUTPUT_DIR" | tee "$OUTPUT_DIR/config.log"

STAGE="training"
python -m pathvqa.train_visual_lora \
  --dataset pathvqa --model-path "$MODEL_PATH" --data-root "$DATA_ROOT" \
  --cache-dir "$CACHE_ROOT" --output-dir "$OUTPUT_DIR" --experiment-name "$EXPERIMENT" \
  --target-scope full_model --last-n-vision-layers 24 --rank 2 \
  --expected-trainable-parameters 1769472 \
  --seed 44 --data-seed 42 --epochs 5 --save-epochs 3 4 5 \
  --batch-size 1 --gradient-accumulation 32 --learning-rate 1e-4 \
  --max-length 2048 --dataloader-workers 2 --correct-loss-accumulation \
  2>&1 | tee "$OUTPUT_DIR/train.log"

STAGE="checkpoint_audit"
python - "$OUTPUT_DIR" <<'PY' | tee "$OUTPUT_DIR/train_report_audit.log"
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
d=json.loads((root/"train_report.json").read_text(encoding="utf-8"))
assert d["dataset"]=="pathvqa" and d["target_scope"]=="full_model_attention"
assert d["seed"]==44 and d["data_seed"]==42 and d["epochs"]==5
assert d["saved_epochs"]==[3,4,5] and d["rank"]==2 and d["alpha"]==4
assert d["dropout"]==0.05 and d["use_rslora"] is False and d["use_dora"] is False
assert d["parameter_counts"]["trainable"]==1769472 and len(set(d["target_modules"]))==192
assert d["selected_vision_layers_0based"]==list(range(24))
assert d["selected_language_layers_0based"]==list(range(36))
assert d["per_device_train_batch_size"]==1 and d["gradient_accumulation_steps"]==32
assert d["effective_batch_size"]==32 and d["learning_rate"]==1e-4
assert d["trainer_model_accepts_loss_kwargs"] is False and d["accelerator_gradient_accumulation_steps"]==1
for epoch in (3,4,5):
    c=json.loads((root/f"checkpoints/epoch_{epoch}/adapter_config.json").read_text(encoding="utf-8"))
    assert c["r"]==2 and c["lora_alpha"]==4 and c["use_rslora"] is False
print("[LORA_R2_TRAIN_REPORT_AUDIT] pass=True trainable=1769472 target_modules=192 saved_epochs=3,4,5")
print("Training seconds:",d["train_metrics"].get("train_runtime"))
print("Peak GPU bytes:",d["peak_gpu_memory_bytes"])
PY

STAGE="validation_epoch5"
EVAL_DIR="$OUTPUT_DIR/eval_validation/epoch_5"
python -m pathvqa.pathvqa_official_eval \
  --backend lora --base-model "$MODEL_PATH" \
  --checkpoint "$OUTPUT_DIR/checkpoints/epoch_5" \
  --data-root "$DATA_ROOT" --cache-dir "$CACHE_ROOT" --split validation \
  --output-dir "$EVAL_DIR" \
  2>&1 | tee "$OUTPUT_DIR/eval_validation_epoch_5.log"

STAGE="paired_vs_v1"
python -m diagnostics.compare_pathvqa_v1_training_budget \
  --baseline-eval "$BASELINE_EVAL" --variant-eval "$EVAL_DIR" \
  --experiment "$EXPERIMENT" --baseline pathvqa_v1_norm_fixed_5ep_seed44 \
  --all-question-types --output "$OUTPUT_DIR/paired_vs_v1_5ep.json" \
  2>&1 | tee "$OUTPUT_DIR/paired_vs_v1_5ep.log"

STAGE="completed"
echo "[LORA_R2_DONE] output=$OUTPUT_DIR fixed_epoch=5 split=validation budget=approximately_matched auto_shutdown=false"
