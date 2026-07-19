#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/visual_router_layer_fixed_v4_diversity_recover_20260719_4/final}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoint_diagnostics/outputs/low_expert_sweep_${TIMESTAMP}}"

mkdir -p "$OUTPUT_ROOT"

for EXPERT_INDEX in 3 2 1 0; do
  echo "[SWEEP] evaluating fixed expert ${EXPERT_INDEX}"
  python checkpoint_diagnostics/evaluate_fixed_route.py \
    --checkpoint "$CHECKPOINT" \
    --base-model "$BASE_MODEL" \
    --expert-index "$EXPERT_INDEX" \
    --stop-on-regex-fail \
    --output-dir "$OUTPUT_ROOT/expert_${EXPERT_INDEX}"
done

echo "[SWEEP_COMPLETE] $OUTPUT_ROOT"
