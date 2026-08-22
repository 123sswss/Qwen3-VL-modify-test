#!/bin/bash
set -uo pipefail

ROOT_DIR="${MMRL_ROOT_DIR:-/root/autodl-tmp/Qwen3-VL-modify-test}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
OUTPUT_ROOT="${MMRL_OUTPUT_ROOT:-$ROOT_DIR/experiment_outputs/output}"
SLAKE_DATA_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
SLAKE_OUTPUT_ROOT="${SLAKE_OUTPUT_ROOT:-$ROOT_DIR/slake/outputs/mmrl}"
ENV_RUN_TARGET="${RUN_TARGET:-}"
RUN_TARGET="${1:-${ENV_RUN_TARGET:-${MMRL_RUN_TARGET:-all}}}"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"
SEED="${MMRL_FIXED_SEED:-44}"
SHUTDOWN_ON_EXIT="${MMRL_SHUTDOWN_ON_EXIT:-1}"

mkdir -p "$OUTPUT_ROOT" "$SLAKE_OUTPUT_ROOT"
echo "[RUN_TARGET] selected=$RUN_TARGET positional=${1:-<unset>} env=${ENV_RUN_TARGET:-<unset>} mmrl_env=${MMRL_RUN_TARGET:-<unset>}"

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

run_slake_checkpoint_eval() {
  local checkpoint="$1"
  local eval_output_dir="$2"
  local eval_log="$3"
  if [ ! -f "$checkpoint/mmrl_delta.safetensors" ]; then
    echo "[ERR] checkpoint 缺少 mmrl_delta.safetensors: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python slake/slake_official_eval.py \
      --backend mmrl \
      --base-model "$MODEL_PATH" \
      --checkpoint "$checkpoint" \
      --questions "$SLAKE_DATA_ROOT/test.json" \
      --image-root "$SLAKE_DATA_ROOT/imgs" \
      --output-dir "$eval_output_dir" \
      --language all \
      --overwrite \
      2>&1 | tee "$eval_log"
  )
}

run_slake() {
  local experiment_name="${SLAKE_EXPERIMENT_NAME:-slake_mmrl_layer_mlp_relation0050}"
  local run_seed="${SLAKE_RUN_SEED:-$SEED}"
  local architecture="${MMRL_QUERY_ARCHITECTURE:-layer_mlp_post_cross}"
  local epochs="${SLAKE_STAGE3_EPOCHS:-3}"
  local same_init="${MMRL_SAME_INIT_LAYER_PROJECTORS:-0}"
  local ablate_visual_gate="${MMRL_ABLATE_VISUAL_GATE:-0}"
  local use_alpha_prob_train_gate="${MMRL_USE_ALPHA_PROB_TRAIN_GATE:-0}"
  local use_alpha_mean_train_gate="${MMRL_USE_ALPHA_MEAN_TRAIN_GATE:-0}"
  local enable_deepstack="${MMRL_ENABLE_DEEPSTACK_MMRL_RESIDUAL:-0}"
  local cross_relation_weight="${MMRL_CROSS_RELATION_LOSS_WEIGHT:-0.0}"
  local extra_args=()
  if [ "$same_init" = "1" ]; then
    extra_args+=(--same-init-layer-projectors)
  elif [ "$same_init" != "0" ]; then
    echo "[ERR] MMRL_SAME_INIT_LAYER_PROJECTORS must be 0 or 1" >&2
    return 2
  fi
  if [ "$ablate_visual_gate" = "1" ]; then
    extra_args+=(--ablate-visual-gate)
  elif [ "$ablate_visual_gate" != "0" ]; then
    echo "[ERR] MMRL_ABLATE_VISUAL_GATE must be 0 or 1" >&2
    return 2
  fi
  if [ "$use_alpha_prob_train_gate" = "1" ]; then
    extra_args+=(--use-alpha-prob-train-gate)
  elif [ "$use_alpha_prob_train_gate" != "0" ]; then
    echo "[ERR] MMRL_USE_ALPHA_PROB_TRAIN_GATE must be 0 or 1" >&2
    return 2
  fi
  if [ "$use_alpha_mean_train_gate" = "1" ]; then
    extra_args+=(--use-alpha-mean-train-gate)
  elif [ "$use_alpha_mean_train_gate" != "0" ]; then
    echo "[ERR] MMRL_USE_ALPHA_MEAN_TRAIN_GATE must be 0 or 1" >&2
    return 2
  fi
  if [ "$use_alpha_prob_train_gate" = "1" ] && [ "$use_alpha_mean_train_gate" = "1" ]; then
    echo "[ERR] Alpha probability and Alpha batch-mean gates are mutually exclusive" >&2
    return 2
  fi
  if [ "$enable_deepstack" = "1" ]; then
    extra_args+=(--enable-deepstack-mmrl-residual)
  elif [ "$enable_deepstack" != "0" ]; then
    echo "[ERR] MMRL_ENABLE_DEEPSTACK_MMRL_RESIDUAL must be 0 or 1" >&2
    return 2
  fi
  local output_dir
  output_dir="$(available_output_dir "$SLAKE_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir/eval"
  local train_gate_mode="hard_concrete"
  if [ "$use_alpha_prob_train_gate" = "1" ]; then
    train_gate_mode="alpha_probability"
  elif [ "$use_alpha_mean_train_gate" = "1" ]; then
    train_gate_mode="alpha_batch_mean"
  fi
  echo "[SLAKE] experiment=$experiment_name seed=$run_seed architecture=$architecture same_init=$same_init ablate_visual_gate=$ablate_visual_gate train_gate=$train_gate_mode deepstack=$enable_deepstack:scale1.0 relation=${MMRL_RELATION_LOSS_WEIGHT:-0.05} cross_relation=$cross_relation_weight output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python slake/train_mmrl.py \
      --data-root "$SLAKE_DATA_ROOT" \
      --model-path "$MODEL_PATH" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --language all \
      --seed "$run_seed" \
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
      --query-architecture "$architecture" \
      "${extra_args[@]}" \
      --mmrl-lr "${SLAKE_MMRL_LR:-6e-5}" \
      --relation-weight "${MMRL_RELATION_LOSS_WEIGHT:-0.05}" \
      --cross-relation-weight "$cross_relation_weight" \
      --relation-max-tokens "${MMRL_RELATION_MAX_TOKENS:-64}" \
      --scheduler constant_with_warmup \
      --warmup-ratio 0.10 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  run_slake_checkpoint_eval \
    "$output_dir/final" \
    "$output_dir/eval" \
    "$output_dir/eval.log"
}

run_repro_seeds() {
  local architecture="$1"
  local experiment_name="$2"
  local seed
  for seed in 44 45 46; do
    SLAKE_RUN_SEED="$seed" \
    SLAKE_EXPERIMENT_NAME="$experiment_name" \
    MMRL_QUERY_ARCHITECTURE="$architecture" \
      run_slake || return 1
  done
}

run_final_serial3() {
  run_final_seed \
    "slake_mmrl_layer_mlp_same_init_relation0050" 1 0 0.0 || return 1
  run_final_seed \
    "slake_mmrl_layer_mlp_deepstack_relation0050" 0 1 0.0 || return 1
  run_final_seed \
    "slake_mmrl_layer_mlp_cross_relation0010_relation0050" 0 0 0.01 || return 1
}

run_final_seed() {
  local experiment_name="$1"
  local same_init="$2"
  local enable_deepstack="$3"
  local cross_relation_weight="$4"
  local run_seed="${5:-44}"
  local ablate_visual_gate="${6:-0}"
  local use_alpha_prob_train_gate="${7:-0}"
  local use_alpha_mean_train_gate="${8:-0}"
  SLAKE_EXPERIMENT_NAME="$experiment_name" \
  SLAKE_RUN_SEED="$run_seed" \
  SLAKE_STAGE3_EPOCHS=3 \
  SLAKE_MMRL_LR=6e-5 \
  MMRL_QUERY_ARCHITECTURE="layer_mlp_post_cross" \
  MMRL_RP_SPACE_LENGTH=40 \
  MMRL_MEMORY_QUERY_COUNT=128 \
  MMRL_MEMORY_ATTENTION_DIM=128 \
  MMRL_PROJECTOR_HIDDEN_DIM=1024 \
  MMRL_CROSS_ATTENTION_HEADS=8 \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
  MMRL_RELATION_MAX_TOKENS=64 \
  MMRL_SAME_INIT_LAYER_PROJECTORS="$same_init" \
  MMRL_ABLATE_VISUAL_GATE="$ablate_visual_gate" \
  MMRL_USE_ALPHA_PROB_TRAIN_GATE="$use_alpha_prob_train_gate" \
  MMRL_USE_ALPHA_MEAN_TRAIN_GATE="$use_alpha_mean_train_gate" \
  MMRL_ENABLE_DEEPSTACK_MMRL_RESIDUAL="$enable_deepstack" \
  MMRL_CROSS_RELATION_LOSS_WEIGHT="$cross_relation_weight" \
    run_slake
}

run_same_init_repro_seeds4() {
  local seed
  for seed in 44 45 46 47; do
    run_final_seed \
      "slake_mmrl_layer_mlp_same_init_relation0050_repro4" \
      1 0 0.0 "$seed" || return 1
  done
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
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_relation0050" \
    MMRL_QUERY_ARCHITECTURE="shared_direct_post_cross" \
      run_slake || failures=$((failures + 1))
    ;;
  slake_current_control)
    run_final_seed \
      "slake_mmrl_layer_mlp_current_control_relation0050" 0 0 0.0 \
      || failures=$((failures + 1))
    ;;
  slake_same_init)
    run_final_seed \
      "slake_mmrl_layer_mlp_same_init_relation0050" 1 0 0.0 \
      || failures=$((failures + 1))
    ;;
  slake_deepstack)
    run_final_seed \
      "slake_mmrl_layer_mlp_deepstack_relation0050" 0 1 0.0 \
      || failures=$((failures + 1))
    ;;
  slake_cross_relation)
    run_final_seed \
      "slake_mmrl_layer_mlp_cross_relation0010_relation0050" 0 0 0.01 \
      || failures=$((failures + 1))
    ;;
  slake_same_init_repro_seeds4)
    run_same_init_repro_seeds4 || failures=$((failures + 1))
    ;;
  slake_same_init_no_gate)
    run_final_seed \
      "slake_mmrl_layer_mlp_same_init_no_gate_relation0050" \
      1 0 0.0 44 1 || failures=$((failures + 1))
    ;;
  slake_same_init_alpha_prob_gate)
    run_final_seed \
      "slake_mmrl_layer_mlp_same_init_alpha_prob_gate_relation0050" \
      1 0 0.0 44 0 1 || failures=$((failures + 1))
    ;;
  slake_same_init_alpha_mean_gate)
    run_final_seed \
      "slake_mmrl_layer_mlp_same_init_alpha_mean_gate_relation0050" \
      1 0 0.0 44 0 0 1 || failures=$((failures + 1))
    ;;
  slake_final_serial3)
    run_final_serial3 || failures=$((failures + 1))
    ;;
  slake_layer_mlp_repro_seeds3)
    run_repro_seeds \
      "layer_mlp_post_cross" \
      "slake_mmrl_layer_mlp_full_ca_relation0050_repro3" \
      || failures=$((failures + 1))
    ;;
  slake_shared_direct_repro_seeds3)
    run_repro_seeds \
      "shared_direct_post_cross" \
      "slake_mmrl_shared_direct_relation0050_repro3" \
      || failures=$((failures + 1))
    ;;
  train_shared_direct)
    MMRL_EXPERIMENT_NAME="dynamic_rep_shared_direct_v1" \
      run_train_dataset || failures=$((failures + 1))
    ;;
  all)
    run_train_dataset || failures=$((failures + 1))
    run_slake || failures=$((failures + 1))
    ;;
  *)
    echo "[ERR] 未知目标: $RUN_TARGET，可选 train、slake、slake_shared_direct、slake_current_control、slake_same_init、slake_deepstack、slake_cross_relation、slake_same_init_repro_seeds4、slake_same_init_no_gate、slake_same_init_alpha_prob_gate、slake_same_init_alpha_mean_gate、slake_final_serial3、slake_layer_mlp_repro_seeds3、slake_shared_direct_repro_seeds3、train_shared_direct、all。" >&2
    exit 2
    ;;
esac

if [ "$failures" -ne 0 ]; then
  echo "[ERR] 已执行全部计划，失败实验数=$failures。" >&2
  exit 1
fi
echo "[DONE] 已完成实验目标: $RUN_TARGET"
