#!/usr/bin/env bash
# Shared PathVQA r2 protocol; only dataset and official Test scorer differ.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
DATASET="${1:?Choose slake or rsvqa_lr}"
case "$DATASET" in
  slake) DATA="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"; BASE=slake ;;
  rsvqa_lr) DATA="${RSVQA_DATA_ROOT:-/root/autodl-tmp/dataset/RSVQA/6344334}"; BASE=RSVQA ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
PY="${LORA_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
EXPERIMENT="${DATASET}_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44"
OUT="$BASE/outputs/lora/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
STAGE=train
trap 'code=$?; printf "stage=%s\nexit_code=%s\nauto_shutdown=false\n" "$STAGE" "$code" > "$OUT/exit_status.txt"' EXIT
printf 'experiment=%s\ncommit=%s\n' "$EXPERIMENT" "$(git rev-parse HEAD)" > "$OUT/config.txt"
echo "[LORA_R2_OUTPUT] $OUT"
"$PY" -m pathvqa.train_visual_lora \
  --dataset "$DATASET" --model-path "$MODEL" --data-root "$DATA" \
  --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
  --target-scope full_model --last-n-vision-layers 24 --rank 2 \
  --expected-trainable-parameters 1769472 \
  --seed 44 --data-seed 42 --epochs 5 --save-epochs 3 4 5 \
  --batch-size 1 --gradient-accumulation 32 --learning-rate 1e-4 \
  --max-length 2048 --dataloader-workers 2 --correct-loss-accumulation \
  2>&1 | tee "$OUT/train.log"
STAGE=report_check
"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads((p/'train_report.json').read_text(encoding='utf-8'))
assert d['rank']==2 and d['alpha']==4 and d['epochs']==5 and d['saved_epochs']==[3,4,5]
assert d['seed']==44 and d['data_seed']==42 and d['parameter_counts']['trainable']==1769472
assert d['per_device_train_batch_size']==1 and d['gradient_accumulation_steps']==32
assert d['trainer_model_accepts_loss_kwargs'] is False
PY
STAGE=test_epoch5
if [[ "$DATASET" == slake ]]; then
  "$PY" -m slake.slake_official_eval --backend lora --base-model "$MODEL" \
    --checkpoint "$OUT/checkpoints/epoch_5" --questions "$DATA/test.json" --image-root "$DATA/imgs" \
    --language all --expected-split test --max-new-tokens 32 --temperature 0 --answer-mode raw \
    --output-dir "$OUT/eval_test/epoch_5" 2>&1 | tee "$OUT/eval_test.log"
else
  "$PY" -m RSVQA.rsvqa_lr_official_eval --backend lora --base-model "$MODEL" \
    --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split test --max-new-tokens 32 \
    --output-dir "$OUT/eval_test/epoch_5" 2>&1 | tee "$OUT/eval_test.log"
fi
STAGE=completed
echo "[LORA_R2_DONE] $OUT fixed_epoch=5 split=test auto_shutdown=false"
