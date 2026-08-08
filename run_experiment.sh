#!/bin/bash
set -uo pipefail

ROOT_DIR="${MMRL_ROOT_DIR:-/root/autodl-tmp/Qwen3-VL-modify-test}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
OUTPUT_ROOT="${MMRL_OUTPUT_ROOT:-$ROOT_DIR/experiment_outputs/output}"
SLAKE_DATA_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
SLAKE_OUTPUT_ROOT="${SLAKE_OUTPUT_ROOT:-$ROOT_DIR/slake/outputs/mmrl}"
RUN_TARGET="${1:-${MMRL_RUN_TARGET:-all}}"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"
SEED="${MMRL_FIXED_SEED:-44}"
SHUTDOWN_ON_EXIT="${MMRL_SHUTDOWN_ON_EXIT:-1}"

mkdir -p "$OUTPUT_ROOT" "$SLAKE_OUTPUT_ROOT"

cancel_shutdown_on_interrupt() {
  SHUTDOWN_ON_EXIT=0
  trap - EXIT
  echo "[INT] 检测到 Ctrl+C，已取消自动关机。"
  exit 130
}

shutdown_on_exit() {
  local exit_code=$?
  if [ "$SHUTDOWN_ON_EXIT" != "1" ]; then
    return "$exit_code"
  fi
  echo "[EXIT] 脚本退出，exit_code=$exit_code"
  echo "[EXIT] 600 秒后自动关机；按 Ctrl+C 可取消。"
  sleep 600
  /usr/bin/shutdown
}

trap shutdown_on_exit EXIT
trap cancel_shutdown_on_interrupt INT

available_output_dir() {
  local root="$1"
  local base="$2"
  local candidate="$root/$base"
  local index=1
  while [ -d "$candidate" ]; do
    candidate="$root/${base}_${index}"
    index=$((index + 1))
  done
  printf '%s\n' "$candidate"
}

run_train_dataset() {
  local experiment_name="${MMRL_EXPERIMENT_NAME:-dynamic_rep_cross_attention_v1}"
  local output_dir
  output_dir="$(available_output_dir "$OUTPUT_ROOT" "${experiment_name}_seed${SEED}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[TRAIN] experiment=$experiment_name seed=$SEED output=$output_dir"
  (
    cd "$ROOT_DIR/train" || exit 1
    MMRL_EXPERIMENT="$experiment_name" \
    MMRL_MODEL_PATH="$MODEL_PATH" \
    MMRL_OUTPUT_DIR="$output_dir" \
    MMRL_SEED="$SEED" \
    MMRL_DATA_SAMPLING_SEED=42 \
    MMRL_DATA_ORDER_SEED=42 \
    MMRL_DETERMINISTIC_SAMPLING=1 \
    MMRL_EVAL_EACH_EPOCH=0 \
    MMRL_LIVE_FINAL_EVAL=1 \
    python train.py 2>&1 | tee "$output_dir/train.log"
  )
}

run_slake() {
  local experiment_name="${SLAKE_EXPERIMENT_NAME:-slake_mmrl_dynamic_rep_cross_attention}"
  local epochs="${SLAKE_STAGE3_EPOCHS:-3}"
  local relation_weight="${MMRL_RELATION_LOSS_WEIGHT:-0.05}"
  local layer_lora_rank="${MMRL_LAYER_LORA_RANK:-0}"
  local query_architecture="${MMRL_QUERY_ARCHITECTURE:-layer_mlp_post_cross}"
  local rep_update_mode="${MMRL_REP_UPDATE_MODE:-replace}"
  local output_dir
  output_dir="$(available_output_dir "$SLAKE_OUTPUT_ROOT" "${experiment_name}_seed${SEED}_${RUN_DATE}")"
  mkdir -p "$output_dir/eval"
  echo "[SLAKE] experiment=$experiment_name seed=$SEED epochs=$epochs relation=$relation_weight query_architecture=$query_architecture rep_update_mode=$rep_update_mode layer_lora_rank=$layer_lora_rank output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python slake/train_mmrl.py \
      --data-root "$SLAKE_DATA_ROOT" \
      --model-path "$MODEL_PATH" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --language all \
      --seed "$SEED" \
      --data-seed 42 \
      --stage3-epochs "$epochs" \
      --stage3-epoch-lr-decay 0.5 \
      --batch-size "${SLAKE_STAGE1_BATCH_SIZE:-4}" \
      --gradient-accumulation "${SLAKE_STAGE1_GRAD_ACCUM:-8}" \
      --dataloader-workers "${SLAKE_STAGE1_WORKERS:-8}" \
      --stage3-batch-size "${SLAKE_STAGE3_BATCH_SIZE:-2}" \
      --stage3-gradient-accumulation "${SLAKE_STAGE3_GRAD_ACCUM:-16}" \
      --stage3-dataloader-workers "${SLAKE_STAGE3_WORKERS:-4}" \
      --rp-space-length "${MMRL_RP_SPACE_LENGTH:-40}" \
      --memory-query-count "${MMRL_MEMORY_QUERY_COUNT:-128}" \
      --memory-attention-dim "${MMRL_MEMORY_ATTENTION_DIM:-128}" \
      --projector-hidden-dim "${MMRL_PROJECTOR_HIDDEN_DIM:-1024}" \
      --cross-attention-heads "${MMRL_CROSS_ATTENTION_HEADS:-8}" \
      --query-architecture "$query_architecture" \
      --rep-update-mode "$rep_update_mode" \
      --layer-lora-rank "$layer_lora_rank" \
      --mmrl-lr 6e-5 \
      --relation-weight "$relation_weight" \
      --scheduler constant_with_warmup \
      --warmup-ratio 0.10 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  (
    cd "$ROOT_DIR" || exit 1
    python slake/slake_official_eval.py \
      --backend mmrl \
      --base-model "$MODEL_PATH" \
      --checkpoint "$output_dir/final" \
      --questions "$SLAKE_DATA_ROOT/test.json" \
      --image-root "$SLAKE_DATA_ROOT/imgs" \
      --output-dir "$output_dir/eval" \
      --language all \
      --overwrite \
      2>&1 | tee "$output_dir/eval.log"
  )
}

run_slake_force_g_one() {
  local checkpoint="${MMRL_CHECKPOINT:-}"
  if [ -z "$checkpoint" ]; then
    echo "[ERR] slake_force_g_one requires MMRL_CHECKPOINT=/path/to/final" >&2
    return 2
  fi
  local checkpoint_name
  checkpoint_name="$(basename "$(dirname "$checkpoint")")"
  local output_dir
  output_dir="$(available_output_dir "$SLAKE_OUTPUT_ROOT" "${checkpoint_name}_force_g_one_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[SLAKE FORCE G=1] checkpoint=$checkpoint output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python slake/slake_official_eval.py \
      --backend mmrl \
      --base-model "$MODEL_PATH" \
      --checkpoint "$checkpoint" \
      --questions "$SLAKE_DATA_ROOT/test.json" \
      --image-root "$SLAKE_DATA_ROOT/imgs" \
      --output-dir "$output_dir" \
      --language all \
      --force-g-one \
      --overwrite \
      2>&1 | tee "$output_dir/eval.log"
  )
}

failures=0
case "$RUN_TARGET" in
  train)
    run_train_dataset || failures=$((failures + 1))
    ;;
  slake)
    run_slake || failures=$((failures + 1))
    ;;
  slake_shared_direct)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_dynamic_rep_shared_direct" \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
      run_slake || failures=$((failures + 1))
    ;;
  slake_lowrank_matrix)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_rank16_relation0050" \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_LAYER_LORA_RANK=16 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_rank16_relation0000" \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_LAYER_LORA_RANK=16 \
    MMRL_RELATION_LOSS_WEIGHT=0.0 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_rank64_relation0050" \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_LAYER_LORA_RANK=64 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_structure_matrix)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_layer_linear_post_cross_replace_r0050" \
    MMRL_QUERY_ARCHITECTURE=layer_linear_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_lowdim_cross_layer_linear_replace_r0050" \
    MMRL_QUERY_ARCHITECTURE=lowdim_cross_layer_linear \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_layer_mlp_persistent_delta_r0050" \
    MMRL_QUERY_ARCHITECTURE=layer_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=persistent_delta \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_force_g_one)
    run_slake_force_g_one || failures=$((failures + 1))
    ;;
  train_shared_direct)
    MMRL_EXPERIMENT_NAME="dynamic_rep_shared_direct_v1" \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
      run_train_dataset || failures=$((failures + 1))
    ;;
  all)
    run_train_dataset || failures=$((failures + 1))
    run_slake || failures=$((failures + 1))
    ;;
  *)
    echo "[ERR] 未知目标: $RUN_TARGET，可选 train、slake、slake_shared_direct、slake_lowrank_matrix、slake_structure_matrix、slake_force_g_one、train_shared_direct、all。" >&2
    exit 2
    ;;
esac

if [ "$failures" -ne 0 ]; then
  echo "[ERR] 已执行全部计划，失败实验数=$failures。" >&2
  exit 1
fi
echo "[DONE] 已完成实验目标: $RUN_TARGET"
