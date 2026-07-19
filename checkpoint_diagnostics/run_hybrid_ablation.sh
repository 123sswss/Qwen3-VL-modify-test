#!/usr/bin/env bash
set -euo pipefail

HIGH_CHECKPOINT="${HIGH_CHECKPOINT:-/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/visual_router_layer_fixed_v4_diversity_recover_20260719_5/final}"
LOW_CHECKPOINT="${LOW_CHECKPOINT:-/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/visual_router_layer_fixed_v4_diversity_recover_20260719_4/final}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model}"
PERMUTATION="${PERMUTATION:-1,3,0,2}"

python checkpoint_diagnostics/evaluate_hybrid.py \
  --high-checkpoint "$HIGH_CHECKPOINT" \
  --low-checkpoint "$LOW_CHECKPOINT" \
  --base-model "$BASE_MODEL" \
  --permutation "$PERMUTATION" \
  --mode high_adapters_low_router

python checkpoint_diagnostics/evaluate_hybrid.py \
  --high-checkpoint "$HIGH_CHECKPOINT" \
  --low-checkpoint "$LOW_CHECKPOINT" \
  --base-model "$BASE_MODEL" \
  --permutation "$PERMUTATION" \
  --mode low_adapters_high_router
