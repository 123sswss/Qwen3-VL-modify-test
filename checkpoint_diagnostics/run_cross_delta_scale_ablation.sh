#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-${MMRL_CHECKPOINT:-}}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/model}"
SLAKE_DATA_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -z "$TARGET" ]]; then
  echo "Usage: bash checkpoint_diagnostics/run_cross_delta_scale_ablation.sh /path/to/experiment-or-final" >&2
  exit 2
fi

if [[ -f "$TARGET/mmrl_manifest.json" ]]; then
  CHECKPOINT="$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
  EXPERIMENT_DIR="$(dirname "$CHECKPOINT")"
elif [[ -f "$TARGET/final/mmrl_manifest.json" ]]; then
  EXPERIMENT_DIR="$(cd "$TARGET" && pwd)"
  CHECKPOINT="$EXPERIMENT_DIR/final"
else
  echo "[ERR] no compact final checkpoint found under: $TARGET" >&2
  exit 2
fi

if [[ ! -f "$SLAKE_DATA_ROOT/test.json" ]]; then
  echo "[ERR] SLAKE test manifest not found: $SLAKE_DATA_ROOT/test.json" >&2
  exit 2
fi

EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoint_diagnostics/outputs/cross_delta_scale_${EXPERIMENT_NAME}_${RUN_STAMP}}"
mkdir -p "$OUTPUT_ROOT"

LIMIT_ARGS=()
if [[ -n "${SLAKE_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$SLAKE_LIMIT")
fi

run_scale() {
  local scale="$1"
  local tag="$2"
  local output_dir="$OUTPUT_ROOT/$tag"
  mkdir -p "$output_dir"
  echo "[CROSS_DELTA_SCALE] scale=$scale checkpoint=$CHECKPOINT output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python slake/slake_official_eval.py \
      --backend mmrl \
      --base-model "$MODEL_PATH" \
      --checkpoint "$CHECKPOINT" \
      --questions "$SLAKE_DATA_ROOT/test.json" \
      --image-root "$SLAKE_DATA_ROOT/imgs" \
      --output-dir "$output_dir" \
      --language all \
      --mmrl-cross-delta-scale "$scale" \
      --overwrite \
      "${LIMIT_ARGS[@]}" \
      2>&1 | tee "$output_dir/eval.log"
  )
}

failures=0
run_scale 0.0 scale_0 || failures=$((failures + 1))
run_scale 0.5 scale_0p5 || failures=$((failures + 1))

BASELINE_SUMMARY="$EXPERIMENT_DIR/eval/slake_summary.json"
if [[ -z "${SLAKE_LIMIT:-}" && -f "$BASELINE_SUMMARY" ]]; then
  mkdir -p "$OUTPUT_ROOT/scale_1"
  cp "$BASELINE_SUMMARY" "$OUTPUT_ROOT/scale_1/slake_summary.json"
  echo "[CROSS_DELTA_SCALE] scale=1.0 reused=$BASELINE_SUMMARY"
else
  run_scale 1.0 scale_1 || failures=$((failures + 1))
fi

python "$ROOT_DIR/checkpoint_diagnostics/summarize_cross_delta_scale.py" "$OUTPUT_ROOT"
summary_status=$?
if [[ $summary_status -ne 0 ]]; then
  failures=$((failures + 1))
fi

echo "[CROSS_DELTA_SCALE_DONE] output=$OUTPUT_ROOT failures=$failures"
if [[ $failures -ne 0 ]]; then
  exit 1
fi
