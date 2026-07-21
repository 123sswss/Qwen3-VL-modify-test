#!/usr/bin/env bash
set -euo pipefail

SCALE="${1:-${MMRL_SCALE:-0.7}}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/visual_router_relation_retest_seed47_20260721/final}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCALE_TAG="${SCALE//./p}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoint_diagnostics/outputs/mmrl_scale_${SCALE_TAG}_${TIMESTAMP}}"

mkdir -p "$OUTPUT_DIR"

echo "[MMRL_SCALE] checkpoint=$CHECKPOINT"
echo "[MMRL_SCALE] scale=$SCALE"
echo "[MMRL_SCALE] output=$OUTPUT_DIR"

python checkpoint_diagnostics/evaluate_mmrl_scale.py \
  --checkpoint "$CHECKPOINT" \
  --base-model "$BASE_MODEL" \
  --scale "$SCALE" \
  --output-dir "$OUTPUT_DIR"

echo "[MMRL_SCALE_COMPLETE] output=$OUTPUT_DIR"
