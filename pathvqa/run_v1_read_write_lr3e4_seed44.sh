#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${V1_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
BASELINE=pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2
EXPERIMENT=pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44
OUT="pathvqa/outputs/visual_selection_prefix/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
STAGE=baseline_check
trap 'code=$?; printf "stage=%s\nexit_code=%s\n" "$STAGE" "$code" > "$OUT/exit_status.txt"' EXIT
"$PY" - "$BASELINE" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);d=json.loads((r/'train_report.json').read_text(encoding='utf-8'))
s=json.loads((r/'eval_validation/epoch_5/pathvqa_summary.json').read_text(encoding='utf-8'))
assert d['experiment']=='pathvqa_v1_norm_fixed_5ep_seed44' and d['epochs']==5 and d['model_seed']==44
assert d['trainer_model_accepts_loss_kwargs'] is False and abs(s['overall_accuracy']-59.3386)<.0001
PY
printf 'experiment=%s\ncommit=%s\nvalue_projection_lr=0.0003\nprefix_output_lr=0.0003\nselection_indices=5,11,17\nexpected_trainable=1864963\n' \
    "$EXPERIMENT" "$(git rev-parse HEAD)" > "$OUT/config.txt"
echo "[V1_READ_WRITE_OUTPUT] $OUT"
STAGE=train
"$PY" -m pathvqa.train_visual_selection_prefix --dataset pathvqa \
    --model-path "$MODEL" --data-root "$DATA" --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
    --visual-prompt-mode split18 --read-write-learning-rate 3e-4 \
    --model-seed 44 --epochs 5 --save-epochs 3 4 5 \
    --batch-size 2 --gradient-accumulation 16 --correct-loss-accumulation 2>&1 | tee "$OUT/train.log"
STAGE=validation_epoch5
"$PY" -m pathvqa.pathvqa_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
    --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split validation \
    --output-dir "$OUT/eval_validation/epoch_5" 2>&1 | tee "$OUT/eval_validation.log"
STAGE=paired_vs_v1
"$PY" -m diagnostics.compare_pathvqa_v1_training_budget \
    --baseline-eval "$BASELINE/eval_validation/epoch_5" --variant-eval "$OUT/eval_validation/epoch_5" \
    --experiment "$EXPERIMENT" --baseline pathvqa_v1_norm_fixed_5ep_seed44 \
    --output "$OUT/paired_vs_v1.json" 2>&1 | tee "$OUT/paired_vs_v1.log"
STAGE=completed
echo "[V1_READ_WRITE_DONE] $OUT test=false shutdown=false"
