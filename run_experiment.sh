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

run_slake_checkpoint_eval() {
  local checkpoint="$1"
  local eval_output_dir="$2"
  local eval_log="$3"
  if [ ! -f "$checkpoint/mmrl_delta.safetensors" ]; then
    echo "[ERR] checkpoint 缺少 mmrl_delta.safetensors: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  echo "[SLAKE EVAL] checkpoint=$checkpoint output=$eval_output_dir"
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
  local experiment_name="${SLAKE_EXPERIMENT_NAME:-slake_mmrl_dynamic_rep_cross_attention}"
  local run_seed="${SLAKE_RUN_SEED:-$SEED}"
  local epochs="${SLAKE_STAGE3_EPOCHS:-3}"
  local mmrl_lr="${SLAKE_MMRL_LR:-6e-5}"
  local relation_weight="${MMRL_RELATION_LOSS_WEIGHT:-0.05}"
  local relation_mode="${MMRL_RELATION_MODE:-linear}"
  local relation_threshold="${MMRL_RELATION_THRESHOLD:-0.03}"
  local relation_max_tokens="${MMRL_RELATION_MAX_TOKENS:-64}"
  local memory_pooling_mode="${MMRL_MEMORY_POOLING_MODE:-independent}"
  local memory_slot_diversity_weight="${MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT:-0.0}"
  local memory_slot_cosine_max="${MMRL_MEMORY_SLOT_COSINE_MAX:-0.995}"
  local cross_attention_routing_mode="${MMRL_CROSS_ATTENTION_ROUTING_MODE:-dynamic_qk}"
  local static_modality_visual_prior="${MMRL_STATIC_MODALITY_VISUAL_PRIOR:-0.44}"
  local factorized_visual_residual_scale="${MMRL_FACTORIZED_VISUAL_RESIDUAL_SCALE:-1.0}"
  local factorized_text_residual_scale="${MMRL_FACTORIZED_TEXT_RESIDUAL_SCALE:-1.0}"
  local layer_lora_rank="${MMRL_LAYER_LORA_RANK:-0}"
  local ca_layer_lora_target="${MMRL_CA_LAYER_LORA_TARGET:-none}"
  local ca_layer_lora_rank="${MMRL_CA_LAYER_LORA_RANK:-0}"
  local ca_layer_lora_alpha="${MMRL_CA_LAYER_LORA_ALPHA:-1.0}"
  local query_architecture="${MMRL_QUERY_ARCHITECTURE:-layer_mlp_post_cross}"
  local rep_update_mode="${MMRL_REP_UPDATE_MODE:-replace}"
  local independent_layer_rep="${MMRL_INDEPENDENT_LAYER_REP:-0}"
  local independent_layer_rep_args=()
  if [ "$independent_layer_rep" = "1" ]; then
    independent_layer_rep_args=(--independent-layer-rep)
  elif [ "$independent_layer_rep" != "0" ]; then
    echo "[ERR] MMRL_INDEPENDENT_LAYER_REP must be 0 or 1" >&2
    return 2
  fi
  local output_dir
  output_dir="$(available_output_dir "$SLAKE_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir/eval"
  echo "[SLAKE] experiment=$experiment_name seed=$run_seed epochs=$epochs mmrl_lr=$mmrl_lr relation=$relation_mode:weight$relation_weight:threshold$relation_threshold:max_tokens$relation_max_tokens memory_pooling=$memory_pooling_mode memory_slot_weight=$memory_slot_diversity_weight memory_slot_cosine_max=$memory_slot_cosine_max cross_attention_routing=$cross_attention_routing_mode static_visual_prior=$static_modality_visual_prior factorized_residual_scales=visual$factorized_visual_residual_scale:text$factorized_text_residual_scale query_architecture=$query_architecture rep_update_mode=$rep_update_mode independent_layer_rep=$independent_layer_rep layer_lora_rank=$layer_lora_rank ca_layer_lora=$ca_layer_lora_target:r$ca_layer_lora_rank:alpha$ca_layer_lora_alpha output=$output_dir"
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
      --memory-pooling-mode "$memory_pooling_mode" \
      --memory-slot-diversity-weight "$memory_slot_diversity_weight" \
      --memory-slot-cosine-max "$memory_slot_cosine_max" \
      --projector-hidden-dim "${MMRL_PROJECTOR_HIDDEN_DIM:-1024}" \
      --cross-attention-heads "${MMRL_CROSS_ATTENTION_HEADS:-8}" \
      --cross-attention-routing-mode "$cross_attention_routing_mode" \
      --static-modality-visual-prior "$static_modality_visual_prior" \
      --factorized-visual-residual-scale "$factorized_visual_residual_scale" \
      --factorized-text-residual-scale "$factorized_text_residual_scale" \
      --query-architecture "$query_architecture" \
      --rep-update-mode "$rep_update_mode" \
      "${independent_layer_rep_args[@]}" \
      --layer-lora-rank "$layer_lora_rank" \
      --ca-layer-lora-target "$ca_layer_lora_target" \
      --ca-layer-lora-rank "$ca_layer_lora_rank" \
      --ca-layer-lora-alpha "$ca_layer_lora_alpha" \
      --mmrl-lr "$mmrl_lr" \
      --relation-weight "$relation_weight" \
      --relation-mode "$relation_mode" \
      --relation-threshold "$relation_threshold" \
      --relation-max-tokens "$relation_max_tokens" \
      --scheduler constant_with_warmup \
      --warmup-ratio 0.10 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  if [ "${SLAKE_EVAL_EPOCH_CHECKPOINTS:-0}" = "1" ]; then
    local epoch_id
    local checkpoint
    local epoch_eval_dir
    for ((epoch_id = 1; epoch_id < epochs; epoch_id++)); do
      checkpoint="$output_dir/checkpoints/stage3_epoch_${epoch_id}"
      epoch_eval_dir="$output_dir/eval_stage3_epoch_${epoch_id}"
      run_slake_checkpoint_eval \
        "$checkpoint" \
        "$epoch_eval_dir" \
        "$output_dir/eval_stage3_epoch_${epoch_id}.log" || return 1
    done
  fi

  run_slake_checkpoint_eval \
    "$output_dir/final" \
    "$output_dir/eval" \
    "$output_dir/eval.log" || return 1

  if [ "${SLAKE_RUN_MEMORY_COLLAPSE_BOTH:-0}" = "1" ]; then
    mkdir -p "$output_dir/eval_memory_collapse_both"
    (
      cd "$ROOT_DIR" || exit 1
      python slake/slake_official_eval.py \
        --backend mmrl \
        --base-model "$MODEL_PATH" \
        --checkpoint "$output_dir/final" \
        --questions "$SLAKE_DATA_ROOT/test.json" \
        --image-root "$SLAKE_DATA_ROOT/imgs" \
        --output-dir "$output_dir/eval_memory_collapse_both" \
        --language all \
        --mmrl-memory-collapse both \
        --overwrite \
        2>&1 | tee "$output_dir/eval_memory_collapse_both.log"
    ) || return 1
  fi
}

run_existing_shared_direct_repro_evals() {
  local eval_failures=0
  local repro_seed
  local candidate
  local output_dir
  local epoch_id
  local checkpoint
  for repro_seed in 44 45 46; do
    output_dir=""
    for candidate in \
      "$SLAKE_OUTPUT_ROOT"/slake_mmrl_shared_direct_relation0050_repro3_seed${repro_seed}_*
    do
      if [ ! -d "$candidate" ]; then
        continue
      fi
      if [ -z "$output_dir" ] || [ "$candidate" -nt "$output_dir" ]; then
        output_dir="$candidate"
      fi
    done
    if [ -z "$output_dir" ]; then
      echo "[ERR] 未找到 seed $repro_seed 的已训练复现实验目录" >&2
      eval_failures=$((eval_failures + 1))
      continue
    fi
    echo "[SLAKE EXISTING REPRO EVAL] seed=$repro_seed experiment=$output_dir"
    for epoch_id in 1 2; do
      checkpoint="$output_dir/checkpoints/stage3_epoch_${epoch_id}"
      run_slake_checkpoint_eval \
        "$checkpoint" \
        "$output_dir/eval_stage3_epoch_${epoch_id}" \
        "$output_dir/eval_stage3_epoch_${epoch_id}.log" \
        || eval_failures=$((eval_failures + 1))
    done
    run_slake_checkpoint_eval \
      "$output_dir/final" \
      "$output_dir/eval" \
      "$output_dir/eval.log" \
      || eval_failures=$((eval_failures + 1))
  done
  return "$eval_failures"
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
  slake_full_geometry_budget4)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_full_geometry_repro" \
    SLAKE_RUN_SEED=45 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_QUERY_ARCHITECTURE=layer_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_full_geometry_lr8e5" \
    SLAKE_RUN_SEED=45 \
    SLAKE_MMRL_LR=8e-5 \
    MMRL_QUERY_ARCHITECTURE=layer_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_full_geometry_rep80" \
    SLAKE_RUN_SEED=45 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=80 \
    MMRL_QUERY_ARCHITECTURE=layer_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_full_geometry_memory256" \
    SLAKE_RUN_SEED=45 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_MEMORY_QUERY_COUNT=256 \
    MMRL_QUERY_ARCHITECTURE=layer_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_memory_pooling_serial3)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_pooling8_independent" \
    SLAKE_RUN_SEED=44 \
    MMRL_MEMORY_QUERY_COUNT=8 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_MEMORY_SLOT_COSINE_MAX=0.995 \
    SLAKE_RUN_MEMORY_COLLAPSE_BOTH=1 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_pooling8_cosine995" \
    SLAKE_RUN_SEED=44 \
    MMRL_MEMORY_QUERY_COUNT=8 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.03 \
    MMRL_MEMORY_SLOT_COSINE_MAX=0.995 \
    SLAKE_RUN_MEMORY_COLLAPSE_BOTH=1 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_pooling8_competitive" \
    SLAKE_RUN_SEED=44 \
    MMRL_MEMORY_QUERY_COUNT=8 \
    MMRL_MEMORY_POOLING_MODE=competitive \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_MEMORY_SLOT_COSINE_MAX=0.995 \
    SLAKE_RUN_MEMORY_COLLAPSE_BOTH=1 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_memory_pooling_competitive128)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_pooling128_competitive" \
    SLAKE_RUN_SEED=44 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_POOLING_MODE=competitive \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_MEMORY_SLOT_COSINE_MAX=0.995 \
    SLAKE_RUN_MEMORY_COLLAPSE_BOTH=1 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_independent_layer_rep)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_independent_layer_rep_full_ca" \
    SLAKE_RUN_SEED=44 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_INDEPENDENT_LAYER_REP=1 \
    MMRL_QUERY_ARCHITECTURE=layer_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
    SLAKE_RUN_MEMORY_COLLAPSE_BOTH=1 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_shared_layer_delta)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_layer_delta_full_ca" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_delta_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
    SLAKE_RUN_MEMORY_COLLAPSE_BOTH=1 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_ca_ablation_serial3)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_mlp512_full_ca" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=512 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=none \
    MMRL_CA_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_ALPHA=1.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_q_lora_r4" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=query \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_o_lora_r4" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=output \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_ca_corrected_serial3)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_layer_mlp_hidden64_full_ca" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=64 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=layer_mlp_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=none \
    MMRL_CA_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_ALPHA=1.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_q_lora_r4_fanincorrected" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=query \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_o_lora_r4_fanincorrected" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=output \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_ca_corrected_lora2)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_q_lora_r4_fanincorrected" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=query \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_o_lora_r4_fanincorrected" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=output \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_ca_ce_only_serial3)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ce_only" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=none \
    MMRL_CA_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_ALPHA=1.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.0 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_q_lora_r4_ce_only" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=query \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.0 \
      run_slake || failures=$((failures + 1))
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_ca_o_lora_r4_ce_only" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=output \
    MMRL_CA_LAYER_LORA_RANK=4 \
    MMRL_CA_LAYER_LORA_ALPHA=4.0 \
    MMRL_RELATION_LOSS_WEIGHT=0.0 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_shared_direct_relation_sweep)
    for relation_spec in \
      "0.005:relation0005" \
      "0.015:relation0015" \
      "0.05:relation0050" \
      "0.15:relation0150" \
      "0.5:relation0500"
    do
      relation_weight="${relation_spec%%:*}"
      relation_tag="${relation_spec##*:}"
      SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_${relation_tag}" \
      SLAKE_RUN_SEED=44 \
      SLAKE_STAGE3_EPOCHS=3 \
      SLAKE_MMRL_LR=6e-5 \
      MMRL_RP_SPACE_LENGTH=40 \
      MMRL_MEMORY_QUERY_COUNT=128 \
      MMRL_MEMORY_ATTENTION_DIM=128 \
      MMRL_MEMORY_POOLING_MODE=independent \
      MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
      MMRL_PROJECTOR_HIDDEN_DIM=1024 \
      MMRL_CROSS_ATTENTION_HEADS=8 \
      MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
      MMRL_REP_UPDATE_MODE=replace \
      MMRL_LAYER_LORA_RANK=0 \
      MMRL_CA_LAYER_LORA_TARGET=none \
      MMRL_CA_LAYER_LORA_RANK=0 \
      MMRL_CA_LAYER_LORA_ALPHA=1.0 \
      MMRL_RELATION_LOSS_WEIGHT="$relation_weight" \
        run_slake || failures=$((failures + 1))
    done
    ;;
  slake_shared_direct_relation_trust_region)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_relation_trust_tau0030_w0050" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=none \
    MMRL_CA_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_ALPHA=1.0 \
    MMRL_RELATION_MODE=trust_region \
    MMRL_RELATION_THRESHOLD=0.03 \
    MMRL_RELATION_MAX_TOKENS=64 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_shared_direct_static_modality_relation0050)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_static_modality_relation0050" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_CROSS_ATTENTION_ROUTING_MODE=static_modality \
    MMRL_STATIC_MODALITY_VISUAL_PRIOR=0.44 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=none \
    MMRL_CA_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_ALPHA=1.0 \
    MMRL_RELATION_MODE=linear \
    MMRL_RELATION_MAX_TOKENS=64 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_factorized_ca_serial2)
    for factorized_spec in \
      "1.0:v100" \
      "2.0:v200"
    do
      visual_scale="${factorized_spec%%:*}"
      visual_tag="${factorized_spec##*:}"
      SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_factorized_ca_${visual_tag}_t100_relation0050" \
      SLAKE_RUN_SEED=44 \
      SLAKE_STAGE3_EPOCHS=3 \
      SLAKE_MMRL_LR=6e-5 \
      MMRL_RP_SPACE_LENGTH=40 \
      MMRL_MEMORY_QUERY_COUNT=128 \
      MMRL_MEMORY_ATTENTION_DIM=128 \
      MMRL_MEMORY_POOLING_MODE=independent \
      MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
      MMRL_PROJECTOR_HIDDEN_DIM=1024 \
      MMRL_CROSS_ATTENTION_HEADS=8 \
      MMRL_CROSS_ATTENTION_ROUTING_MODE=factorized_modality \
      MMRL_FACTORIZED_VISUAL_RESIDUAL_SCALE="$visual_scale" \
      MMRL_FACTORIZED_TEXT_RESIDUAL_SCALE=1.0 \
      MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
      MMRL_REP_UPDATE_MODE=replace \
      MMRL_LAYER_LORA_RANK=0 \
      MMRL_CA_LAYER_LORA_TARGET=none \
      MMRL_CA_LAYER_LORA_RANK=0 \
      MMRL_CA_LAYER_LORA_ALPHA=1.0 \
      MMRL_RELATION_MODE=linear \
      MMRL_RELATION_MAX_TOKENS=64 \
      MMRL_RELATION_LOSS_WEIGHT=0.05 \
        run_slake || failures=$((failures + 1))
    done
    ;;
  slake_factorized_ca_mean_only)
    SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_factorized_ca_v000_t000_relation0050" \
    SLAKE_RUN_SEED=44 \
    SLAKE_STAGE3_EPOCHS=3 \
    SLAKE_MMRL_LR=6e-5 \
    MMRL_RP_SPACE_LENGTH=40 \
    MMRL_MEMORY_QUERY_COUNT=128 \
    MMRL_MEMORY_ATTENTION_DIM=128 \
    MMRL_MEMORY_POOLING_MODE=independent \
    MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
    MMRL_PROJECTOR_HIDDEN_DIM=1024 \
    MMRL_CROSS_ATTENTION_HEADS=8 \
    MMRL_CROSS_ATTENTION_ROUTING_MODE=factorized_modality \
    MMRL_FACTORIZED_VISUAL_RESIDUAL_SCALE=0.0 \
    MMRL_FACTORIZED_TEXT_RESIDUAL_SCALE=0.0 \
    MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
    MMRL_REP_UPDATE_MODE=replace \
    MMRL_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_TARGET=none \
    MMRL_CA_LAYER_LORA_RANK=0 \
    MMRL_CA_LAYER_LORA_ALPHA=1.0 \
    MMRL_RELATION_MODE=linear \
    MMRL_RELATION_MAX_TOKENS=64 \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || failures=$((failures + 1))
    ;;
  slake_shared_direct_repro_seeds3)
    for repro_seed in 44 45 46
    do
      SLAKE_EXPERIMENT_NAME="slake_mmrl_shared_direct_relation0050_repro3" \
      SLAKE_RUN_SEED="$repro_seed" \
      SLAKE_STAGE3_EPOCHS=3 \
      SLAKE_MMRL_LR=6e-5 \
      SLAKE_STAGE1_BATCH_SIZE=4 \
      SLAKE_STAGE1_GRAD_ACCUM=8 \
      SLAKE_STAGE3_BATCH_SIZE=2 \
      SLAKE_STAGE3_GRAD_ACCUM=16 \
      SLAKE_EVAL_EPOCH_CHECKPOINTS=1 \
      MMRL_RP_SPACE_LENGTH=40 \
      MMRL_MEMORY_QUERY_COUNT=128 \
      MMRL_MEMORY_ATTENTION_DIM=128 \
      MMRL_MEMORY_POOLING_MODE=independent \
      MMRL_MEMORY_SLOT_DIVERSITY_WEIGHT=0.0 \
      MMRL_PROJECTOR_HIDDEN_DIM=1024 \
      MMRL_CROSS_ATTENTION_HEADS=8 \
      MMRL_CROSS_ATTENTION_ROUTING_MODE=dynamic_qk \
      MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
      MMRL_REP_UPDATE_MODE=replace \
      MMRL_INDEPENDENT_LAYER_REP=0 \
      MMRL_LAYER_LORA_RANK=0 \
      MMRL_CA_LAYER_LORA_TARGET=none \
      MMRL_CA_LAYER_LORA_RANK=0 \
      MMRL_CA_LAYER_LORA_ALPHA=1.0 \
      MMRL_RELATION_MODE=linear \
      MMRL_RELATION_MAX_TOKENS=64 \
      MMRL_RELATION_LOSS_WEIGHT=0.05 \
        run_slake || failures=$((failures + 1))
    done
    ;;
  slake_eval_shared_direct_repro_seeds3)
    run_existing_shared_direct_repro_evals \
      || failures=$((failures + 1))
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
    echo "[ERR] 未知目标: $RUN_TARGET，可选 train、slake、slake_shared_direct、slake_lowrank_matrix、slake_structure_matrix、slake_full_geometry_budget4、slake_memory_pooling_serial3、slake_memory_pooling_competitive128、slake_independent_layer_rep、slake_shared_layer_delta、slake_ca_ablation_serial3、slake_ca_corrected_serial3、slake_ca_corrected_lora2、slake_ca_ce_only_serial3、slake_shared_direct_relation_sweep、slake_shared_direct_relation_trust_region、slake_shared_direct_static_modality_relation0050、slake_factorized_ca_serial2、slake_factorized_ca_mean_only、slake_shared_direct_repro_seeds3、slake_eval_shared_direct_repro_seeds3、slake_force_g_one、train_shared_direct、all。" >&2
    exit 2
    ;;
esac

if [ "$failures" -ne 0 ]; then
  echo "[ERR] 已执行全部计划，失败实验数=$failures。" >&2
  exit 1
fi
echo "[DONE] 已完成实验目标: $RUN_TARGET"
