#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
python checkpoint_diagnostics/diagnose_checkpoint.py \
  --checkpoint "/root/autodl-tmp/Qwen3-VL-modify-test/experiment_outputs/output/visual_router_layer_fixed_v4_diversity_recover_20260719_5/final" \
  --base-model "/root/autodl-tmp/model" \
  "$@"

