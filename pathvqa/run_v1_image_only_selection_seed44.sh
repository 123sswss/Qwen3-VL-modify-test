#!/usr/bin/env bash
# Single user-started Image-only selection ablation. No retries/Test/shutdown.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${V1_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
V1="pathvqa/outputs/visual_selection_prefix/pathvqa_v1_norm_fixed_5ep_seed44_20260926_2"
QMAP="pathvqa/outputs/visual_selection_prefix/ablations/pathvqa_v1_c_qmap_norm_fixed_5ep_seed44_20260930"
EXPERIMENT=pathvqa_v1_image_only_selection_norm_fixed_5ep_seed44
OUT="pathvqa/outputs/visual_selection_prefix/ablations/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
STAGE=baseline_check
trap 'code=$?; printf "stage=%s\nexit_code=%s\n" "$STAGE" "$code" > "$OUT/exit_status.txt"' EXIT
"$PY" - "$V1" "$QMAP" <<'PY' | tee "$OUT/baseline_check.log"
import json,sys
from pathlib import Path
for path,name in zip(sys.argv[1:],('pathvqa_v1_norm_fixed_5ep_seed44','pathvqa_v1_c_qmap_norm_fixed_5ep_seed44')):
    root=Path(path)
    r=json.loads((root/'train_report.json').read_text(encoding='utf-8'))
    s=json.loads((root/'eval_validation/epoch_5/pathvqa_summary.json').read_text(encoding='utf-8'))
    assert r['experiment']==name and r['model_seed']==44 and r['data_seed']==42 and r['epochs']==5
    assert r['trainer_model_accepts_loss_kwargs'] is False
    assert s['split']=='validation' and s['count']==6259
    assert (root/'eval_validation/epoch_5/pathvqa_predictions.json').is_file()
    print(name,s['overall_accuracy'])
PY
printf 'experiment=%s\ncommit=%s\nexpected_trainable=1470339\nv1=%s\nc_qmap=%s\n' \
    "$EXPERIMENT" "$(git rev-parse HEAD)" "$V1" "$QMAP" > "$OUT/config.txt"
echo "[V1_IMAGE_ONLY_OUTPUT] $OUT"
STAGE=train
"$PY" -m pathvqa.train_visual_selection_prefix --dataset pathvqa \
    --model-path "$MODEL" --data-root "$DATA" --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
    --visual-prompt-mode c_image_only --model-seed 44 --epochs 5 --save-epochs 3 4 5 \
    --batch-size 2 --gradient-accumulation 16 --correct-loss-accumulation \
    2>&1 | tee "$OUT/train.log"
STAGE=validation_epoch5
"$PY" -m pathvqa.pathvqa_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
    --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split validation \
    --output-dir "$OUT/eval_validation/epoch_5" 2>&1 | tee "$OUT/eval_validation.log"
for REFERENCE in v1 c_qmap; do
    BASELINE="$V1"
    [[ "$REFERENCE" == c_qmap ]] && BASELINE="$QMAP"
    STAGE="paired_vs_$REFERENCE"
    "$PY" -m diagnostics.compare_pathvqa_v1_training_budget \
        --baseline-eval "$BASELINE/eval_validation/epoch_5" --variant-eval "$OUT/eval_validation/epoch_5" \
        --experiment "$EXPERIMENT" --baseline "$REFERENCE" --output "$OUT/paired_vs_$REFERENCE.json" \
        2>&1 | tee "$OUT/paired_vs_$REFERENCE.log"
done
STAGE=completed
echo "[V1_IMAGE_ONLY_DONE] $OUT"
