#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_44="${1:-}"
TARGET_45="${2:-}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/model}"
SLAKE_DATA_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -z "$TARGET_44" || -z "$TARGET_45" ]]; then
  echo "Usage: bash checkpoint_diagnostics/run_query_generator_swap_ablation.sh CHECKPOINT_44 CHECKPOINT_45" >&2
  exit 2
fi

resolve_checkpoint() {
  local target="$1"
  if [[ -f "$target/mmrl_manifest.json" ]]; then
    (cd "$(dirname "$target")" && printf '%s/%s\n' "$PWD" "$(basename "$target")")
  elif [[ -f "$target/final/mmrl_manifest.json" ]]; then
    (cd "$target" && printf '%s/final\n' "$PWD")
  else
    echo "[ERR] no compact checkpoint found under: $target" >&2
    return 2
  fi
}

CHECKPOINT_44="$(resolve_checkpoint "$TARGET_44")" || exit $?
CHECKPOINT_45="$(resolve_checkpoint "$TARGET_45")" || exit $?
EXPERIMENT_44="$(dirname "$CHECKPOINT_44")"
EXPERIMENT_45="$(dirname "$CHECKPOINT_45")"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoint_diagnostics/outputs/query_generator_swap_$RUN_STAMP}"
mkdir -p "$OUTPUT_ROOT"

LIMIT_ARGS=()
if [[ -n "${SLAKE_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$SLAKE_LIMIT")
fi

run_eval() {
  local recipient="$1"
  local donor="$2"
  local tag="$3"
  local output_dir="$OUTPUT_ROOT/$tag"
  mkdir -p "$output_dir"
  if [[ -f "$output_dir/slake_summary.json" ]]; then
    echo "[QUERY_GENERATOR_SWAP] tag=$tag reused=$output_dir/slake_summary.json"
    return 0
  fi
  local progress_args=(--overwrite)
  if [[ -f "$output_dir/slake_progress.jsonl" ]]; then
    progress_args=(--resume)
    echo "[QUERY_GENERATOR_SWAP] tag=$tag resume=$output_dir/slake_progress.jsonl"
  fi
  local donor_args=()
  if [[ -n "$donor" ]]; then
    donor_args=(
      --mmrl-component-donor "$donor"
      --mmrl-component-swap query_generator
    )
  fi
  echo "[QUERY_GENERATOR_SWAP] tag=$tag recipient=$recipient donor=${donor:-none}"
  (
    cd "$ROOT_DIR" || exit 1
    python slake/slake_official_eval.py \
      --backend mmrl \
      --base-model "$MODEL_PATH" \
      --checkpoint "$recipient" \
      --questions "$SLAKE_DATA_ROOT/test.json" \
      --image-root "$SLAKE_DATA_ROOT/imgs" \
      --output-dir "$output_dir" \
      --language all \
      "${donor_args[@]}" \
      "${progress_args[@]}" \
      "${LIMIT_ARGS[@]}" \
      2>&1 | tee "$output_dir/eval.log"
  )
}

failures=0
if [[ -z "${SLAKE_LIMIT:-}" \
      && -f "$EXPERIMENT_44/eval/slake_summary.json" \
      && -f "$EXPERIMENT_45/eval/slake_summary.json" ]]; then
  mkdir -p "$OUTPUT_ROOT/baseline_44" "$OUTPUT_ROOT/baseline_45"
  cp "$EXPERIMENT_44/eval/slake_summary.json" "$OUTPUT_ROOT/baseline_44/slake_summary.json"
  cp "$EXPERIMENT_45/eval/slake_summary.json" "$OUTPUT_ROOT/baseline_45/slake_summary.json"
  echo "[QUERY_GENERATOR_SWAP] reused baselines from recipient experiments"
else
  run_eval "$CHECKPOINT_44" "" baseline_44 || failures=$((failures + 1))
  run_eval "$CHECKPOINT_45" "" baseline_45 || failures=$((failures + 1))
fi

run_eval "$CHECKPOINT_45" "$CHECKPOINT_44" query44_rest45 \
  || failures=$((failures + 1))
run_eval "$CHECKPOINT_44" "$CHECKPOINT_45" query45_rest44 \
  || failures=$((failures + 1))

python "$ROOT_DIR/checkpoint_diagnostics/summarize_query_generator_swap.py" "$OUTPUT_ROOT"
summary_status=$?
if [[ $summary_status -ne 0 ]]; then
  failures=$((failures + 1))
fi

echo "[QUERY_GENERATOR_SWAP_DONE] output=$OUTPUT_ROOT failures=$failures"
if [[ $failures -ne 0 ]]; then
  exit 1
fi
