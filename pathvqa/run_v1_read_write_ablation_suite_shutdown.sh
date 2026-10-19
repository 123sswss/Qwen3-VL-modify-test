#!/usr/bin/env bash
# User-started GPU suite: qmap -> image-only -> noVisual -> evidence-token B.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${V1_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
BASELINE=pathvqa/outputs/visual_selection_prefix/pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44_20261014_092451_34346
SUITE="outputs/pathvqa_v1_read_write_ablation_suite_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$SUITE"
STAGE=baseline_check
finish() {
  CODE=$?
  trap - EXIT
  set +e
  printf 'stage=%s\nexit_code=%s\nfinished_at=%s\n' "$STAGE" "$CODE" "$(date -Is)" > "$SUITE/exit_status.txt"
  echo "[SUITE_FINISHED] stage=$STAGE code=$CODE; shutdown in 600 seconds" | tee -a "$SUITE/suite.log"
  sync
  sleep 600
  printf 'called_at=%s\n' "$(date -Is)" > "$SUITE/shutdown_status.txt"
  /usr/bin/shutdown >> "$SUITE/shutdown.log" 2>&1
  printf 'shutdown_exit_code=%s\n' "$?" >> "$SUITE/shutdown_status.txt"
  sync
  exit "$CODE"
}
trap finish EXIT
printf 'commit=%s\nbaseline=%s\norder=c_qmap,c_image_only,c_no_visual,evidence_token\nmodel_seed=44\ndata_seed=42\nepochs=5\nbatch=2\naccumulation=16\nread_write_lr=0.0003\n' \
  "$(git rev-parse HEAD)" "$BASELINE" > "$SUITE/config.txt"
echo "[ABLATION_SUITE_OUTPUT] $SUITE"
"$PY" - "$BASELINE" <<'PY' | tee "$SUITE/baseline_check.log"
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
d=json.loads((r/'train_report.json').read_text(encoding='utf-8'))
s=json.loads((r/'eval_validation/epoch_5/pathvqa_summary.json').read_text(encoding='utf-8'))
assert d['experiment']=='pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44'
assert d['model_seed']==44 and d['data_seed']==42 and d['epochs']==5
assert d['trainer_model_accepts_loss_kwargs'] is False
rates=d['optimizer']['group_learning_rates']
assert rates['value_projection']==3e-4 and rates['prefix_output']==3e-4
assert s['split']=='validation' and s['count']==6259
assert (r/'eval_validation/epoch_5/pathvqa_predictions.json').is_file()
print('Baseline:',r,'Overall:',s['overall_accuracy'])
PY
for MODE in c_qmap c_image_only c_no_visual evidence_token; do
  EXPERIMENT="pathvqa_v1_read_write_lr3e4_${MODE}_norm_fixed_5ep_seed44"
  OUT="pathvqa/outputs/visual_selection_prefix/ablations/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$OUT"
  printf '%s\t%s\n' "$MODE" "$OUT" >> "$SUITE/runs.tsv"
  printf 'commit=%s\nexperiment=%s\nmode=%s\nread_write_lr=0.0003\n' \
    "$(git rev-parse HEAD)" "$EXPERIMENT" "$MODE" > "$OUT/config.txt"
  STAGE="$MODE:train"
  echo "[TASK_START] $MODE output=$OUT" | tee -a "$SUITE/suite.log"
  "$PY" -m pathvqa.train_visual_selection_prefix --dataset pathvqa \
    --model-path "$MODEL" --data-root "$DATA" --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
    --visual-prompt-mode "$MODE" --read-write-learning-rate 3e-4 \
    --model-seed 44 --epochs 5 --save-epochs 3 4 5 \
    --batch-size 2 --gradient-accumulation 16 --correct-loss-accumulation \
    2>&1 | tee "$OUT/train.log"
  STAGE="$MODE:validation_epoch5"
  "$PY" -m pathvqa.pathvqa_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
    --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split validation \
    --output-dir "$OUT/eval_validation/epoch_5" 2>&1 | tee "$OUT/eval_validation.log"
  STAGE="$MODE:paired_vs_final_v1"
  "$PY" -m diagnostics.compare_pathvqa_v1_training_budget \
    --baseline-eval "$BASELINE/eval_validation/epoch_5" --variant-eval "$OUT/eval_validation/epoch_5" \
    --experiment "$EXPERIMENT" --baseline pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed44 \
    --output "$OUT/paired_vs_final_v1.json" 2>&1 | tee "$OUT/paired_vs_final_v1.log"
  printf 'stage=completed\nexit_code=0\n' > "$OUT/exit_status.txt"
  echo "[TASK_DONE] $MODE output=$OUT" | tee -a "$SUITE/suite.log"
done
STAGE=completed
