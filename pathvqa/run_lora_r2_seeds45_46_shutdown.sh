#!/usr/bin/env bash
# User-started seed45 -> seed46, fixed epoch5 Test; shutdown from Bash on exit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${LORA_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
CACHE="${PATHVQA_CACHE_ROOT:-$DATA/.hf_cache}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SUITE="pathvqa/outputs/lora/seed45_46_suite_$STAMP"
mkdir -p "$SUITE"
STAGE=starting
SEED=45
declare -A STATUS=([45]=not_run [46]=not_run)
finish() {
    local code=$?
    trap - EXIT
    set +e
    if (( code != 0 )); then STATUS[$SEED]=failed; fi
    printf 'stage=%s\nseed=%s\nexit_code=%s\nseed45=%s\nseed46=%s\n' \
      "$STAGE" "$SEED" "$code" "${STATUS[45]}" "${STATUS[46]}" > "$SUITE/final_status.txt"
    printf 'calling /usr/bin/shutdown %s\n' "$(date -Is)" > "$SUITE/shutdown.log"
    sync
    /usr/bin/shutdown >> "$SUITE/shutdown.log" 2>&1
    printf 'shutdown_exit_code=%s\n' "$?" >> "$SUITE/shutdown.log"
    sync
    exit "$code"
}
trap finish EXIT
git rev-parse HEAD > "$SUITE/execution_commit.txt"
step() {
    STAGE="$1"
    local log="$2"
    shift 2
    local code=0
    "$@" 2>&1 | tee "$log" || code=$?
    printf 'seed=%s stage=%s exit_code=%s %s\n' "$SEED" "$STAGE" "$code" "$(date -Is)" >> "$SUITE/stages.log"
    if (( code != 0 )); then exit "$code"; fi
}
echo "[PATHVQA_LORA_R2_SUITE] $SUITE"
for SEED in 45 46; do
    STATUS[$SEED]=running
    EXPERIMENT="pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed$SEED"
    OUT="pathvqa/outputs/lora/${EXPERIMENT}_$STAMP"
    mkdir -p "$OUT"
    printf 'seed%s=%s\n' "$SEED" "$OUT" >> "$SUITE/run_paths.txt"
    step train "$OUT/train.log" "$PY" -m pathvqa.train_visual_lora \
      --dataset pathvqa --model-path "$MODEL" --data-root "$DATA" --cache-dir "$CACHE" \
      --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
      --target-scope full_model --last-n-vision-layers 24 --rank 2 --expected-trainable-parameters 1769472 \
      --seed "$SEED" --data-seed 42 --epochs 5 --save-epochs 3 4 5 \
      --batch-size 1 --gradient-accumulation 32 --learning-rate 1e-4 \
      --max-length 2048 --dataloader-workers 2 --correct-loss-accumulation
    step test_epoch5 "$OUT/eval_test.log" "$PY" -m pathvqa.pathvqa_official_eval \
      --backend lora --base-model "$MODEL" --checkpoint "$OUT/checkpoints/epoch_5" \
      --data-root "$DATA" --cache-dir "$CACHE" --split test --output-dir "$OUT/eval_test/epoch_5"
    STATUS[$SEED]=completed
done
STAGE=completed
echo "[PATHVQA_LORA_R2_DONE] $SUITE; shutting down"
