#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-${MMRL_CHECKPOINT:-}}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/model}"
SLAKE_DATA_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: bash checkpoint_diagnostics/run_memory_collapse_ablation.sh /path/to/checkpoint/final" >&2
  exit 2
fi

CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
if [[ ! -f "$CHECKPOINT/mmrl_manifest.json" ]]; then
  echo "[ERR] compact checkpoint manifest not found: $CHECKPOINT/mmrl_manifest.json" >&2
  exit 2
fi
if [[ ! -f "$SLAKE_DATA_ROOT/test.json" ]]; then
  echo "[ERR] SLAKE test manifest not found: $SLAKE_DATA_ROOT/test.json" >&2
  exit 2
fi

CHECKPOINT_PARENT="$(dirname "$CHECKPOINT")"
CHECKPOINT_NAME="$(basename "$CHECKPOINT_PARENT")"
if [[ "$CHECKPOINT_NAME" == "stage3" ]]; then
  CHECKPOINT_NAME="$(basename "$(dirname "$CHECKPOINT_PARENT")")"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoint_diagnostics/outputs/memory_collapse_${CHECKPOINT_NAME}_${RUN_STAMP}}"
mkdir -p "$OUTPUT_ROOT"

LIMIT_ARGS=()
if [[ -n "${SLAKE_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$SLAKE_LIMIT")
fi

modes=(none text visual both)
failures=0
for mode in "${modes[@]}"; do
  output_dir="$OUTPUT_ROOT/$mode"
  mkdir -p "$output_dir"
  echo "[MEMORY_COLLAPSE] mode=$mode checkpoint=$CHECKPOINT output=$output_dir"
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
      --mmrl-memory-collapse "$mode" \
      --overwrite \
      "${LIMIT_ARGS[@]}" \
      2>&1 | tee "$output_dir/eval.log"
  )
  status=$?
  if [[ $status -ne 0 ]]; then
    failures=$((failures + 1))
    echo "[ERR] mode=$mode failed with exit_code=$status; continuing" >&2
  fi
done

python "$ROOT_DIR/checkpoint_diagnostics/summarize_memory_collapse.py" "$OUTPUT_ROOT"
summary_status=$?
if [[ $summary_status -ne 0 ]]; then
  failures=$((failures + 1))
fi

echo "[MEMORY_COLLAPSE_DONE] output=$OUTPUT_ROOT failures=$failures"
if [[ $failures -ne 0 ]]; then
  exit 1
fi
