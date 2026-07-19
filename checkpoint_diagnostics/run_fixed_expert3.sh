#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/visual_router_layer_fixed_v4_diversity_recover_20260719_5/final}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model}"

python checkpoint_diagnostics/evaluate_fixed_route.py \
  --checkpoint "$CHECKPOINT" \
  --base-model "$BASE_MODEL" \
  --expert-index 3
