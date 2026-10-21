#!/usr/bin/env bash
# Image-only 45/46, then C-qmap 45/46; fail-stop, no automatic shutdown.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${V1_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
SUITE="outputs/pathvqa_v1_read_write_qmap_image_only_seeds45_46_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$SUITE"
STAGE=baseline_check
trap 'code=$?; printf "stage=%s\nexit_code=%s\n" "$STAGE" "$code" > "$SUITE/exit_status.txt"' EXIT
printf 'commit=%s\norder=c_image_only:45,c_image_only:46,c_qmap:45,c_qmap:46\ndata_seed=42\nepochs=5\nread_write_lr=0.0003\nbatch=2\naccumulation=16\nshutdown=false\n' \
  "$(git rev-parse HEAD)" > "$SUITE/config.txt"
echo "[QMAP_IMAGE_ONLY_SUITE_OUTPUT] $SUITE"
"$PY" - <<'PY' | tee "$SUITE/baseline_check.log"
from pathlib import Path
from diagnostics.run_pathvqa_read_write_three_seed_test import RUNS, bind
for seed in (45, 46):
    binding = bind(Path.cwd(), seed, RUNS[seed])
    comparison = Path(binding["run"]) / "eval_validation/epoch_5/pathvqa_comparisons.json"
    if not comparison.is_file():
        raise FileNotFoundError(comparison)
    print(binding)
PY
for MODE in c_image_only c_qmap; do
  case "$MODE" in
    c_image_only) EXPECTED=1470339 ;;
    c_qmap) EXPECTED=1815811 ;;
  esac
  for SEED in 45 46; do
    case "$SEED" in
      45) BASELINE=pathvqa/outputs/visual_selection_prefix/pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed45_20261016_024534_10401 ;;
      46) BASELINE=pathvqa/outputs/visual_selection_prefix/pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed46_20261017_104312_1230 ;;
    esac
    EXPERIMENT="pathvqa_v1_read_write_lr3e4_${MODE}_norm_fixed_5ep_seed${SEED}"
    OUT="pathvqa/outputs/visual_selection_prefix/ablations/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
    mkdir -p "$OUT"
    printf '%s\t%s\t%s\n' "$MODE" "$SEED" "$OUT" >> "$SUITE/runs.tsv"
    printf 'experiment=%s\ncommit=%s\nbaseline=%s\nexpected_trainable=%s\n' \
      "$EXPERIMENT" "$(git rev-parse HEAD)" "$BASELINE" "$EXPECTED" > "$OUT/config.txt"
    STAGE="${MODE}:seed${SEED}:train"
    "$PY" -m pathvqa.train_visual_selection_prefix --dataset pathvqa \
      --model-path "$MODEL" --data-root "$DATA" --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
      --visual-prompt-mode "$MODE" --read-write-learning-rate 3e-4 \
      --model-seed "$SEED" --epochs 5 --save-epochs 3 4 5 \
      --batch-size 2 --gradient-accumulation 16 --correct-loss-accumulation \
      2>&1 | tee "$OUT/train.log"
    STAGE="${MODE}:seed${SEED}:validation_epoch5"
    "$PY" -m pathvqa.pathvqa_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
      --checkpoint "$OUT/checkpoints/epoch_5" --data-root "$DATA" --split validation \
      --output-dir "$OUT/eval_validation/epoch_5" 2>&1 | tee "$OUT/eval_validation.log"
    STAGE="${MODE}:seed${SEED}:paired_vs_final_v1"
    "$PY" -m diagnostics.compare_pathvqa_v1_training_budget \
      --baseline-eval "$BASELINE/eval_validation/epoch_5" --variant-eval "$OUT/eval_validation/epoch_5" \
      --experiment "$EXPERIMENT" --baseline "pathvqa_v1_read_write_lr3e4_norm_fixed_5ep_seed${SEED}" \
      --output "$OUT/paired_vs_final_v1.json" 2>&1 | tee "$OUT/paired_vs_final_v1.log"
    printf 'stage=completed\nexit_code=0\n' > "$OUT/exit_status.txt"
    echo "[ABLATION_DONE] mode=$MODE seed=$SEED output=$OUT"
  done
done
STAGE=completed
