#!/usr/bin/env bash
# Untrained backbone only; full bilingual official Test, no shutdown.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
DATA="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
OUT="slake/outputs/base/slake_base_qwen3vl_test_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$OUT/exit_status.txt"' EXIT
printf 'experiment=slake_base_qwen3vl_test\ncommit=%s\ntraining=false\ncheckpoint=none\n' \
  "$(git rev-parse HEAD)" > "$OUT/config.txt"
"${SLAKE_PYTHON:-/root/miniconda3/bin/python}" -m slake.slake_official_eval \
  --backend base --base-model "${MMRL_MODEL_PATH:-/root/autodl-tmp/model}" \
  --questions "$DATA/test.json" --image-root "$DATA/imgs" --language all --expected-split test \
  --max-new-tokens 32 --temperature 0 --answer-mode raw --output-dir "$OUT" \
  2>&1 | tee "$OUT/eval.log"
echo "[SLAKE_BASE_TEST_DONE] output=$OUT"
