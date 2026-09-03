#!/bin/bash
set -uo pipefail

ROOT_DIR="${MMRL_ROOT_DIR:-/root/autodl-tmp/Qwen3-VL-modify-test}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
OUTPUT_ROOT="${MMRL_OUTPUT_ROOT:-$ROOT_DIR/experiment_outputs/output}"
SLAKE_DATA_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
SLAKE_OUTPUT_ROOT="${SLAKE_OUTPUT_ROOT:-$ROOT_DIR/slake/outputs/mmrl}"
SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT="${SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT:-$ROOT_DIR/slake/outputs/dynamic_prompt}"
PATHVQA_DATA_ROOT="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
PATHVQA_CACHE_ROOT="${PATHVQA_CACHE_ROOT:-$PATHVQA_DATA_ROOT/.hf_cache}"
PATHVQA_OUTPUT_ROOT="${PATHVQA_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/mmrl}"
PATHVQA_LORA_OUTPUT_ROOT="${PATHVQA_LORA_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/lora}"
PATHVQA_BASE_OUTPUT_ROOT="${PATHVQA_BASE_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/base}"
PATHVQA_PROMPT_OUTPUT_ROOT="${PATHVQA_PROMPT_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/prompt_tuning}"
PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT="${PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/dynamic_prompt}"
ELECTRICAL_DATA_ROOT="${ELECTRICAL_DATA_ROOT:-/root/autodl-tmp/dataset}"
ELECTRICAL_QDPT_OUTPUT_ROOT="${ELECTRICAL_QDPT_OUTPUT_ROOT:-$ROOT_DIR/electrical/outputs/qdpt}"
ENV_RUN_TARGET="${RUN_TARGET:-}"
RUN_TARGET="${1:-${ENV_RUN_TARGET:-${MMRL_RUN_TARGET:-all}}}"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"
SEED="${MMRL_FIXED_SEED:-44}"
SHUTDOWN_ON_EXIT="${MMRL_SHUTDOWN_ON_EXIT:-1}"

mkdir -p "$OUTPUT_ROOT" "$SLAKE_OUTPUT_ROOT" "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "$PATHVQA_OUTPUT_ROOT" "$PATHVQA_LORA_OUTPUT_ROOT" "$PATHVQA_BASE_OUTPUT_ROOT" "$PATHVQA_PROMPT_OUTPUT_ROOT" "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT" "$ELECTRICAL_QDPT_OUTPUT_ROOT"
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
  local experiment_name="${SLAKE_EXPERIMENT_NAME:-slake_mmrl_layer_mlp_same_init_full_ce_uniform_relation0050}"
  local run_seed="${SLAKE_RUN_SEED:-$SEED}"
  local epochs="${SLAKE_STAGE3_EPOCHS:-3}"
  local same_init="${MMRL_SAME_INIT_LAYER_PROJECTORS:-1}"
  local dynamic_cross_attention="${MMRL_USE_DYNAMIC_CROSS_ATTENTION:-1}"
  local memory_pooling_mode="${MMRL_MEMORY_POOLING_MODE:-multi_query}"
  local fusion_mode="${MMRL_FUSION_MODE:-cross_attention}"
  local extra_args=()
  if [ "$same_init" = "1" ]; then
    extra_args+=(--same-init-layer-projectors)
  elif [ "$same_init" = "0" ]; then
    extra_args+=(--no-same-init-layer-projectors)
  else
    echo "[ERR] MMRL_SAME_INIT_LAYER_PROJECTORS must be 0 or 1" >&2
    return 2
  fi
  if [ "$dynamic_cross_attention" = "0" ]; then
    extra_args+=(--disable-dynamic-cross-attention)
  elif [ "$dynamic_cross_attention" != "1" ]; then
    echo "[ERR] MMRL_USE_DYNAMIC_CROSS_ATTENTION must be 0 or 1" >&2
    return 2
  fi
  if [ "$memory_pooling_mode" != "multi_query" ] \
    && [ "$memory_pooling_mode" != "mean" ] \
    && [ "$memory_pooling_mode" != "text_guided" ]; then
    echo "[ERR] MMRL_MEMORY_POOLING_MODE must be multi_query, mean, or text_guided" >&2
    return 2
  fi
  if [ "$fusion_mode" != "cross_attention" ] && [ "$fusion_mode" != "concat_mlp" ]; then
    echo "[ERR] MMRL_FUSION_MODE must be cross_attention or concat_mlp" >&2
    return 2
  fi
  if [ "$fusion_mode" = "concat_mlp" ] && [ "$memory_pooling_mode" != "mean" ]; then
    echo "[ERR] concat_mlp fusion requires mean memory pooling" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$SLAKE_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir/eval"
  echo "[SLAKE] experiment=$experiment_name seed=$run_seed same_init=$same_init dynamic_cross_attention=$dynamic_cross_attention memory_pooling_mode=$memory_pooling_mode fusion_mode=$fusion_mode train_gate=open_full_ce relation=${MMRL_RELATION_LOSS_WEIGHT:-0.05} output=$output_dir"
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
      --memory-pooling-mode "$memory_pooling_mode" \
      --fusion-mode "$fusion_mode" \
      "${extra_args[@]}" \
      --mmrl-lr "${SLAKE_MMRL_LR:-6e-5}" \
      --relation-weight "${MMRL_RELATION_LOSS_WEIGHT:-0.05}" \
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

run_pathvqa_checkpoint_eval() {
  local checkpoint="$1"
  local eval_output_dir="$2"
  local eval_log="$3"
  local split="${4:-test}"
  if [ ! -f "$checkpoint/mmrl_delta.safetensors" ]; then
    echo "[ERR] checkpoint 缺少 mmrl_delta.safetensors: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python pathvqa/pathvqa_official_eval.py \
      --backend "${PATHVQA_MMRL_EVAL_BACKEND:-mmrl}" \
      --base-model "$MODEL_PATH" \
      --checkpoint "$checkpoint" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --split "$split" \
      --output-dir "$eval_output_dir" \
      --overwrite \
      2>&1 | tee "$eval_log"
  )
}

run_pathvqa() {
  local experiment_name="${PATHVQA_EXPERIMENT_NAME:-pathvqa_mmrl_layer_mlp_same_init_full_ce_uniform_relation0050}"
  local run_seed="${PATHVQA_RUN_SEED:-$SEED}"
  local epochs="${PATHVQA_STAGE3_EPOCHS:-3}"
  local same_init="${MMRL_SAME_INIT_LAYER_PROJECTORS:-1}"
  local dynamic_cross_attention="${MMRL_USE_DYNAMIC_CROSS_ATTENTION:-1}"
  local memory_pooling_mode="${MMRL_MEMORY_POOLING_MODE:-multi_query}"
  local fusion_mode="${MMRL_FUSION_MODE:-cross_attention}"
  local query_architecture="${MMRL_QUERY_ARCHITECTURE:-layer_mlp_post_cross}"
  local extra_args=()

  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] PathVQA 需要 datasets 和 pyarrow。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi
  if [ "$same_init" = "1" ]; then
    extra_args+=(--same-init-layer-projectors)
  elif [ "$same_init" = "0" ]; then
    extra_args+=(--no-same-init-layer-projectors)
  else
    echo "[ERR] MMRL_SAME_INIT_LAYER_PROJECTORS must be 0 or 1" >&2
    return 2
  fi
  if [ "$dynamic_cross_attention" = "0" ]; then
    extra_args+=(--disable-dynamic-cross-attention)
  elif [ "$dynamic_cross_attention" != "1" ]; then
    echo "[ERR] MMRL_USE_DYNAMIC_CROSS_ATTENTION must be 0 or 1" >&2
    return 2
  fi
  if [ "$memory_pooling_mode" != "multi_query" ] \
    && [ "$memory_pooling_mode" != "mean" ] \
    && [ "$memory_pooling_mode" != "text_guided" ]; then
    echo "[ERR] MMRL_MEMORY_POOLING_MODE must be multi_query, mean, or text_guided" >&2
    return 2
  fi
  if [ "$fusion_mode" != "cross_attention" ] && [ "$fusion_mode" != "concat_mlp" ]; then
    echo "[ERR] MMRL_FUSION_MODE must be cross_attention or concat_mlp" >&2
    return 2
  fi
  if [ "$fusion_mode" = "concat_mlp" ] && [ "$memory_pooling_mode" != "mean" ]; then
    echo "[ERR] concat_mlp fusion requires mean memory pooling" >&2
    return 2
  fi
  if [ "$query_architecture" != "layer_mlp_post_cross" ] \
    && [ "$query_architecture" != "shared_direct_post_cross" ]; then
    echo "[ERR] MMRL_QUERY_ARCHITECTURE must be layer_mlp_post_cross or shared_direct_post_cross" >&2
    return 2
  fi
  if [ -n "${PATHVQA_EXPECTED_MMRL_PARAMETERS:-}" ]; then
    extra_args+=(--expected-mmrl-parameters "$PATHVQA_EXPECTED_MMRL_PARAMETERS")
  fi
  if [ -n "${PATHVQA_SOFT_PROMPT_LENGTH:-}" ]; then
    extra_args+=(
      --soft-prompt-length "$PATHVQA_SOFT_PROMPT_LENGTH"
      --soft-prompt-init-seed "${PATHVQA_SOFT_PROMPT_INIT_SEED:-$run_seed}"
      --prompt-lr "${PATHVQA_PROMPT_LR:-0.3}"
      --prompt-warmup-ratio "${PATHVQA_PROMPT_WARMUP_RATIO:-0.03}"
    )
  fi
  if [ -n "${PATHVQA_EXPECTED_TOTAL_TRAINABLE_PARAMETERS:-}" ]; then
    extra_args+=(
      --expected-total-trainable-parameters \
      "$PATHVQA_EXPECTED_TOTAL_TRAINABLE_PARAMETERS"
    )
  fi
  if [ -n "${PATHVQA_STAGE1_CHECKPOINT_IN:-}" ]; then
    extra_args+=(--stage1-checkpoint-in "$PATHVQA_STAGE1_CHECKPOINT_IN")
  fi
  if [ -n "${PATHVQA_STAGE1_CHECKPOINT_OUT:-}" ]; then
    extra_args+=(--stage1-checkpoint-out "$PATHVQA_STAGE1_CHECKPOINT_OUT")
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir/eval"
  echo "[PATHVQA] experiment=$experiment_name seed=$run_seed query_architecture=$query_architecture same_init=$same_init dynamic_cross_attention=$dynamic_cross_attention memory_pooling_mode=$memory_pooling_mode fusion_mode=$fusion_mode train_gate=open_full_ce relation=${MMRL_RELATION_LOSS_WEIGHT:-0.05} output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m pathvqa.train_mmrl \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --model-path "$MODEL_PATH" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --seed "$run_seed" \
      --data-seed 42 \
      --stage3-epochs "$epochs" \
      --stage3-epoch-lr-decay 0.5 \
      --batch-size "${PATHVQA_STAGE1_BATCH_SIZE:-4}" \
      --gradient-accumulation "${PATHVQA_STAGE1_GRAD_ACCUM:-8}" \
      --dataloader-workers "${PATHVQA_STAGE1_WORKERS:-8}" \
      --stage3-batch-size "${PATHVQA_STAGE3_BATCH_SIZE:-2}" \
      --stage3-gradient-accumulation "${PATHVQA_STAGE3_GRAD_ACCUM:-16}" \
      --stage3-dataloader-workers "${PATHVQA_STAGE3_WORKERS:-4}" \
      --rp-space-length "${MMRL_RP_SPACE_LENGTH:-40}" \
      --memory-query-count "${MMRL_MEMORY_QUERY_COUNT:-128}" \
      --memory-attention-dim "${MMRL_MEMORY_ATTENTION_DIM:-128}" \
      --projector-hidden-dim "${MMRL_PROJECTOR_HIDDEN_DIM:-1024}" \
      --cross-attention-heads "${MMRL_CROSS_ATTENTION_HEADS:-8}" \
      --query-architecture "$query_architecture" \
      --memory-pooling-mode "$memory_pooling_mode" \
      --fusion-mode "$fusion_mode" \
      "${extra_args[@]}" \
      --mmrl-lr "${PATHVQA_MMRL_LR:-6e-5}" \
      --relation-weight "${MMRL_RELATION_LOSS_WEIGHT:-0.05}" \
      --relation-max-tokens "${MMRL_RELATION_MAX_TOKENS:-64}" \
      --scheduler constant_with_warmup \
      --warmup-ratio 0.10 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  if [ "${PATHVQA_SELECT_BEST_EPOCH:-0}" = "1" ]; then
    local epoch_id
    for ((epoch_id = 1; epoch_id <= epochs; epoch_id++)); do
      local checkpoint="$output_dir/checkpoints/stage3_epoch_${epoch_id}"
      echo "[PATHVQA_MMRL_VALIDATION] epoch=$epoch_id checkpoint=$checkpoint"
      run_pathvqa_checkpoint_eval \
        "$checkpoint" \
        "$output_dir/eval_validation/epoch_${epoch_id}" \
        "$output_dir/eval_validation_epoch_${epoch_id}.log" \
        validation || return 1
    done

    local selection
    selection="$(python "$ROOT_DIR/pathvqa/select_best_epoch.py" --root "$output_dir" --epochs "$epochs")" || return 1
    local best_epoch
    local best_validation_score
    IFS=$'\t' read -r best_epoch best_validation_score <<< "$selection"
    local best_checkpoint="$output_dir/checkpoints/stage3_epoch_${best_epoch}"
    echo "[PATHVQA_MMRL_TEST] best_epoch=$best_epoch validation=$best_validation_score"
    run_pathvqa_checkpoint_eval \
      "$best_checkpoint" \
      "$output_dir/eval_test/epoch_${best_epoch}" \
      "$output_dir/eval_test_epoch_${best_epoch}.log" \
      test || return 1
    local test_score
    test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_${best_epoch}/pathvqa_summary.json")" || return 1
    printf 'experiment\tseed\tbest_epoch\tvalidation_accuracy\ttest_accuracy\tcheckpoint\n' \
      > "$output_dir/selected_result.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$experiment_name" "$run_seed" "$best_epoch" "$best_validation_score" \
      "$test_score" "$best_checkpoint" \
      >> "$output_dir/selected_result.tsv"
    cat "$output_dir/selected_result.tsv"
  else
    run_pathvqa_checkpoint_eval \
      "$output_dir/final" \
      "$output_dir/eval" \
      "$output_dir/eval.log"
  fi
}

run_pathvqa_lora_eval() {
  local checkpoint="$1"
  local split="$2"
  local eval_output_dir="$3"
  local eval_log="$4"
  local backend="${5:-lora-vision}"
  if [ ! -f "$checkpoint/adapter_config.json" ]; then
    echo "[ERR] LoRA checkpoint 缺少 adapter_config.json: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python pathvqa/pathvqa_official_eval.py \
      --backend "$backend" \
      --base-model "$MODEL_PATH" \
      --checkpoint "$checkpoint" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --split "$split" \
      --output-dir "$eval_output_dir" \
      --overwrite \
      2>&1 | tee "$eval_log"
  )
}

run_pathvqa_base() {
  local experiment_name="pathvqa_base"
  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] PathVQA Base 推理需要 datasets 和 pyarrow。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_BASE_OUTPUT_ROOT" "${experiment_name}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_BASE] experiment=$experiment_name split=test output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python pathvqa/pathvqa_official_eval.py \
      --backend base \
      --base-model "$MODEL_PATH" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --split test \
      --output-dir "$output_dir" \
      --overwrite \
      2>&1 | tee "$output_dir/eval.log"
  )
}

run_pathvqa_prompt_eval() {
  local checkpoint="$1"
  local split="$2"
  local eval_output_dir="$3"
  local eval_log="$4"
  if [ ! -f "$checkpoint/prompt_config.json" ] \
    || [ ! -f "$checkpoint/soft_prompt.pt" ]; then
    echo "[ERR] Prompt checkpoint 不完整: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python pathvqa/pathvqa_official_eval.py \
      --backend prompt-tuning \
      --base-model "$MODEL_PATH" \
      --checkpoint "$checkpoint" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --split "$split" \
      --output-dir "$eval_output_dir" \
      --overwrite \
      2>&1 | tee "$eval_log"
  )
}

run_pathvqa_prompt_tuning_seed44() {
  local experiment_name="pathvqa_prompt_tuning_len20"
  local run_seed=44
  local epochs="${PATHVQA_PROMPT_EPOCHS:-3}"
  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] PathVQA Prompt Tuning 需要 datasets 和 pyarrow。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_PROMPT_TUNING] experiment=$experiment_name seed=$run_seed epochs=$epochs output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m pathvqa.train_prompt_tuning \
      --model-path "$MODEL_PATH" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length "${PATHVQA_PROMPT_LENGTH:-20}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --learning-rate "${PATHVQA_PROMPT_LR:-0.3}" \
      --batch-size "${PATHVQA_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${PATHVQA_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${PATHVQA_PROMPT_WORKERS:-2}" \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local epoch_id
  for ((epoch_id = 1; epoch_id <= epochs; epoch_id++)); do
    local checkpoint="$output_dir/checkpoints/epoch_${epoch_id}"
    echo "[PATHVQA_PROMPT_VALIDATION] epoch=$epoch_id checkpoint=$checkpoint"
    run_pathvqa_prompt_eval \
      "$checkpoint" \
      validation \
      "$output_dir/eval_validation/epoch_${epoch_id}" \
      "$output_dir/eval_validation_epoch_${epoch_id}.log" || return 1
  done

  local selection
  selection="$(python "$ROOT_DIR/pathvqa/select_best_epoch.py" --root "$output_dir" --epochs "$epochs")" || return 1
  local best_epoch
  local best_validation_score
  IFS=$'\t' read -r best_epoch best_validation_score <<< "$selection"
  local best_checkpoint="$output_dir/checkpoints/epoch_${best_epoch}"
  echo "[PATHVQA_PROMPT_TEST] best_epoch=$best_epoch validation=$best_validation_score"
  run_pathvqa_prompt_eval \
    "$best_checkpoint" \
    test \
    "$output_dir/eval_test/epoch_${best_epoch}" \
    "$output_dir/eval_test_epoch_${best_epoch}.log" || return 1

  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_${best_epoch}/pathvqa_summary.json")" || return 1
  printf 'experiment\tseed\tbest_epoch\tvalidation_accuracy\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" "$run_seed" "$best_epoch" "$best_validation_score" \
    "$test_score" "$best_checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_pathvqa_dynamic_prompt_eval() {
  local checkpoint="$1"
  local split="$2"
  local eval_output_dir="$3"
  local eval_log="$4"
  shift 4
  if [ ! -f "$checkpoint/dynamic_prompt_config.json" ] \
    || [ ! -f "$checkpoint/dynamic_prompt.pt" ]; then
    echo "[ERR] Dynamic Prompt checkpoint 不完整: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python pathvqa/pathvqa_official_eval.py \
      --backend dynamic-prompt \
      --base-model "$MODEL_PATH" \
      --checkpoint "$checkpoint" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --split "$split" \
      --output-dir "$eval_output_dir" \
      --overwrite \
      "$@" \
      2>&1 | tee "$eval_log"
  )
}

run_pathvqa_dynamic_prompt_epoch3_validation() {
  local output_dir="$1"
  local experiment_name="$2"
  local run_seed="$3"
  local checkpoint="$output_dir/checkpoints/epoch_3"
  echo "[PATHVQA_DYNAMIC_PROMPT_FIXED_VALIDATION] protocol=fixed_epoch3 split=validation checkpoint=$checkpoint"
  run_pathvqa_dynamic_prompt_eval \
    "$checkpoint" \
    validation \
    "$output_dir/eval_validation/epoch_3" \
    "$output_dir/eval_validation_epoch_3.log" || return 1

  local validation_score
  validation_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_validation/epoch_3/pathvqa_summary.json")" || return 1
  printf 'experiment\tseed\tprotocol\tvalidation_epoch\tvalidation_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" "$run_seed" fixed_epoch3_validation 3 \
    "$validation_score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_pathvqa_dynamic_prompt_interventions() {
  local source_run="${PATHVQA_DYNAMIC_PROMPT_SOURCE_RUN:-}"
  if [ -z "$source_run" ]; then
    source_run="$(ls -dt "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT"/pathvqa_dynamic_prompt_mean_ca256_len20_seed44_* 2>/dev/null | head -n 1)"
  fi
  if [ -z "$source_run" ] || [ ! -f "$source_run/selected_result.tsv" ]; then
    echo "[ERR] 找不到已完成的 Dynamic Prompt source run: $source_run" >&2
    return 1
  fi

  local checkpoint
  checkpoint="$(awk -F '\t' 'NR == 2 {print $6}' "$source_run/selected_result.tsv")"
  if [ -z "$checkpoint" ]; then
    echo "[ERR] selected_result.tsv 未记录 checkpoint: $source_run" >&2
    return 1
  fi

  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_dynamic_prompt_tuning.py
  ) || return 1

  local intervention
  local lag="${PATHVQA_DYNAMIC_PROMPT_MEMORY_LAG:-32}"
  for intervention in zero mean-residual lagged-memory; do
    local eval_output_dir="$source_run/eval_interventions_validation/$intervention"
    local eval_log="$source_run/eval_intervention_validation_${intervention}.log"
    echo "[PATHVQA_DYNAMIC_PROMPT_INTERVENTION] split=validation mode=$intervention lag=$lag checkpoint=$checkpoint"
    run_pathvqa_dynamic_prompt_eval \
      "$checkpoint" \
      validation \
      "$eval_output_dir" \
      "$eval_log" \
      --dynamic-prompt-intervention "$intervention" \
      --dynamic-prompt-memory-lag "$lag" || return 1
  done

  printf 'intervention\toverall\tyes_no\tfree_form\tsamples_changed\n'
  for intervention in zero mean-residual lagged-memory; do
    python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); i=d["dynamic_prompt_intervention"]; print("\t".join(map(str,(i["mode"],d["overall_accuracy"],d["yes_no_accuracy"],d["free_form_accuracy"],i["samples_changed"]))))' \
      "$source_run/eval_interventions_validation/$intervention/pathvqa_summary.json"
  done
}

run_pathvqa_dynamic_prompt_seed44() {
  local experiment_name="pathvqa_dynamic_prompt_mean_ca256_len20"
  local run_seed=44
  local epochs="${PATHVQA_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] Dynamic Prompt 固定评估协议要求 epochs=3，当前为: $epochs" >&2
    return 2
  fi
  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] PathVQA Dynamic Prompt 需要 datasets 和 pyarrow。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_DYNAMIC_PROMPT] experiment=$experiment_name seed=$run_seed epochs=$epochs prompt_lr=${PATHVQA_DYNAMIC_PROMPT_STATIC_LR:-0.3} dynamic_lr=${PATHVQA_DYNAMIC_PROMPT_LR:-3e-4} output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m pathvqa.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${PATHVQA_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${PATHVQA_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${PATHVQA_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${PATHVQA_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${PATHVQA_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters 2685440 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  run_pathvqa_dynamic_prompt_epoch3_validation \
    "$output_dir" "$experiment_name" "$run_seed"
}

run_pathvqa_dynamic_prompt_sparse_visual_variant_seed44() {
  local experiment_name="$1"
  local shared_s_text_mode="$2"
  local expected_trainable="$3"
  local shared_workspace="${4:-false}"
  local run_seed=44
  local epochs="${PATHVQA_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] Dynamic Prompt 固定评估协议要求 epochs=3，当前为: $epochs" >&2
    return 2
  fi
  local sparse_visual_lr="${PATHVQA_SPARSE_VISUAL_LR:-3e-5}"
  local shared_s_source=raw_unconditioned_visual_s
  local shared_s_text_merger=main_visual_merger
  local shared_s_gradient_policy=joint_or_disabled
  local workspace_args=()
  if [ "$shared_s_text_mode" = "none" ]; then
    shared_s_source=independent_text_p_and_visual_s
    shared_s_text_merger=none
  elif [ "$shared_s_text_mode" = "text_owned_visual_readonly" ]; then
    shared_s_source=text_owned_prompt_s
    shared_s_text_merger=none
    shared_s_gradient_policy=text_write_visual_readonly
  fi
  if [ "$shared_workspace" = "true" ]; then
    workspace_args=(
      --shared-workspace
      --workspace-tokens 32
      --workspace-dim 1024
      --workspace-heads 16
      --workspace-ffn-dim 4096
      --workspace-text-attention-dim 1024
      --workspace-text-heads 16
      --workspace-visual-attention-dim 1024
      --workspace-visual-heads 16
      --workspace-lr "${PATHVQA_SHARED_WORKSPACE_LR:-1e-4}"
    )
  fi
  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] PathVQA Dynamic Prompt 需要 datasets 和 pyarrow。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_DYNAMIC_PROMPT_SPARSE_VISUAL] experiment=$experiment_name seed=$run_seed epochs=$epochs anchors=5,11,17 rep_tokens=8 visual_ca=128x4 injection=single_pass_insert_strip relation=off sparse_lr=$sparse_visual_lr shared_s_text_mode=$shared_s_text_mode shared_s_source=$shared_s_source shared_s_text_merger=$shared_s_text_merger shared_s_gradient_policy=$shared_s_gradient_policy shared_workspace=$shared_workspace workspace=32x1024 workspace_blocks=3 workspace_ca=1024x16 workspace_lr=${PATHVQA_SHARED_WORKSPACE_LR:-1e-4} expected_trainable=$expected_trainable output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_dynamic_prompt_tuning.py test_sparse_visual_mmrl.py || exit 1
    python -m pathvqa.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 5 11 17 \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "$sparse_visual_lr" \
      --shared-s-text-mode "$shared_s_text_mode" \
      --shared-s-attention-dim 128 \
      --shared-s-heads 4 \
      --shared-s-visual-bottleneck-dim 128 \
      "${workspace_args[@]}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${PATHVQA_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${PATHVQA_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${PATHVQA_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${PATHVQA_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${PATHVQA_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters "$expected_trainable" \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  run_pathvqa_dynamic_prompt_epoch3_validation \
    "$output_dir" "$experiment_name" "$run_seed"
}

run_pathvqa_dynamic_prompt_sparse_visual_single_pass_seed44() {
  run_pathvqa_dynamic_prompt_sparse_visual_variant_seed44 \
    "pathvqa_dynamic_prompt_sparse_visual_single_pass_layers5_11_17_slots8_ca128_lr3e5" \
    none 3886592
}

run_pathvqa_dynamic_prompt_raw_shared_s_separate_residual_seed44() {
  run_pathvqa_dynamic_prompt_sparse_visual_variant_seed44 \
    "pathvqa_dynamic_prompt_raw_shared_s_separate_residual_keep_p_layers5_11_17_slots8_ca128_lr3e5" \
    separate_residual 5210112
}

run_pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_seed44() {
  run_pathvqa_dynamic_prompt_sparse_visual_variant_seed44 \
    "pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_layers5_11_17_slots8_ca128_lr3e5" \
    direct_prompt 3835392
}

run_pathvqa_dynamic_prompt_asymmetric_shared_s_seed44() {
  run_pathvqa_dynamic_prompt_sparse_visual_variant_seed44 \
    "pathvqa_dynamic_prompt_asymmetric_shared_s_text_owned_visual_readonly_layers5_11_17_slots8_adapter128" \
    text_owned_visual_readonly 4337312
}

run_pathvqa_dynamic_prompt_full_workspace_seed44() {
  run_pathvqa_dynamic_prompt_sparse_visual_variant_seed44 \
    "pathvqa_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8" \
    none 76896256 true
}

run_pathvqa_directional_concat_workspace_text_dynamic_only_seed44() {
  local experiment_name="pathvqa_directional_concat_workspace_text_dynamic_only_z10_d1024_l17_private_p20_s8"
  local run_seed=44
  local epochs="${PATHVQA_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] PathVQA Directional 决胜协议要求 epochs=3，当前为: $epochs" >&2
    return 2
  fi
  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] PathVQA Dynamic Prompt 需要 datasets 和 pyarrow。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_DIRECTIONAL_TEXT_DYNAMIC_ONLY] experiment=$experiment_name seed=$run_seed data_seed=42 anchor=17 private_text_prompt=20 text_workspace_anchor=10 private_visual_prompt=8 visual_workspace_anchor=10 workspace=10x1024 query=text_attention_pooling kv=full_visual cross_attention=1024x16 visual_output=static_anchor_token_concat text_output=dynamic_anchor_token_concat visual_dynamic_write=false workspace_lr=${PATHVQA_DIRECTIONAL_WORKSPACE_LR:-1e-4} expected_trainable=10625536 controlled_change=remove_z_to_visual_dynamic_mlp protocol=fixed_epoch3_validation deadline=2026-09-01T12:00:00+08:00 output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_dynamic_prompt_tuning.py test_sparse_visual_mmrl.py || exit 1
    python -m pathvqa.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 17 \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${PATHVQA_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --directional-concat-workspace \
      --no-directional-visual-dynamic-write \
      --workspace-tokens 10 \
      --workspace-dim 1024 \
      --workspace-heads 16 \
      --workspace-lr "${PATHVQA_DIRECTIONAL_WORKSPACE_LR:-1e-4}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${PATHVQA_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${PATHVQA_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${PATHVQA_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${PATHVQA_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${PATHVQA_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters 10625536 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  run_pathvqa_dynamic_prompt_epoch3_validation \
    "$output_dir" "$experiment_name" "$run_seed"
}

run_pathvqa_directional_concat_workspace_text_dynamic_only_compressed() {
  local workspace_dim="$1"
  local expected_trainable="$2"
  local run_seed="${3:-44}"
  if [ "$workspace_dim" -ne 256 ] \
    && [ "$workspace_dim" -ne 512 ] \
    && [ "$workspace_dim" -ne 768 ]; then
    echo "[ERR] Unsupported compressed Directional width: $workspace_dim" >&2
    return 2
  fi
  local experiment_name="pathvqa_directional_concat_workspace_text_dynamic_only_z10_d${workspace_dim}_l17_private_p20_s8"
  local epochs="${PATHVQA_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] PathVQA Directional d${workspace_dim} protocol requires epochs=3, got: $epochs" >&2
    return 2
  fi
  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] PathVQA Dynamic Prompt requires datasets and pyarrow. Run: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_DIRECTIONAL_TEXT_DYNAMIC_ONLY_COMPRESSED] experiment=$experiment_name seed=$run_seed data_seed=42 anchor=17 private_text_prompt=20 text_workspace_anchor=10 private_visual_prompt=8 visual_workspace_anchor=10 workspace=10x${workspace_dim} query=text_attention_pooling_2560_to_${workspace_dim} kv=full_visual_projected_1024_to_${workspace_dim} cross_attention=${workspace_dim}x16 visual_output=static_anchor_token_concat text_output=dynamic_anchor_token_concat visual_dynamic_write=false workspace_lr=${PATHVQA_DIRECTIONAL_WORKSPACE_LR:-1e-4} expected_trainable=$expected_trainable controlled_change=workspace_width_1024_to_${workspace_dim} protocol=fixed_epoch3_validation output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest \
      test_dynamic_prompt_tuning.py \
      test_sparse_visual_mmrl.py \
      test_pathvqa_directional_interventions.py || exit 1
    python -m pathvqa.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 17 \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${PATHVQA_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --directional-concat-workspace \
      --no-directional-visual-dynamic-write \
      --workspace-tokens 10 \
      --workspace-dim "$workspace_dim" \
      --workspace-heads 16 \
      --workspace-lr "${PATHVQA_DIRECTIONAL_WORKSPACE_LR:-1e-4}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${PATHVQA_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${PATHVQA_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${PATHVQA_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${PATHVQA_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${PATHVQA_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters "$expected_trainable" \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  run_pathvqa_dynamic_prompt_epoch3_validation \
    "$output_dir" "$experiment_name" "$run_seed"
}

run_pathvqa_directional_concat_workspace_text_dynamic_only_d256_seed44() {
  run_pathvqa_directional_concat_workspace_text_dynamic_only_compressed \
    256 2033408 44
}

run_pathvqa_directional_concat_workspace_text_dynamic_only_d512_seed44() {
  run_pathvqa_directional_concat_workspace_text_dynamic_only_compressed \
    512 4591616 44
}

run_pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed44() {
  run_pathvqa_directional_concat_workspace_text_dynamic_only_compressed \
    768 7805184 44
}

run_pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed() {
  local run_seed="$1"
  run_pathvqa_directional_concat_workspace_text_dynamic_only_compressed \
    768 7805184 "$run_seed"
}

run_qdpt_d768_final_dataset() {
  local dataset="$1"
  local variant="$2"
  local run_seed="${3:-44}"
  local dataset_prefix data_root output_root train_module eval_protocol
  local -a data_flags
  case "$dataset" in
    pathvqa)
      dataset_prefix="pathvqa"
      data_root="$PATHVQA_DATA_ROOT"
      output_root="$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT"
      train_module="pathvqa.train_dynamic_prompt"
      eval_protocol="fixed_epoch3_validation"
      data_flags=(--cache-dir "$PATHVQA_CACHE_ROOT")
      ;;
    slake)
      dataset_prefix="slake"
      data_root="$SLAKE_DATA_ROOT"
      output_root="$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT"
      train_module="slake.train_dynamic_prompt"
      eval_protocol="fixed_epoch3_test"
      data_flags=()
      ;;
    *)
      echo "[ERR] Unsupported QDPT final dataset: $dataset" >&2
      return 2
      ;;
  esac

  local experiment_stem expected_trainable query_source static_visual_write
  local private_visual_tokens visual_workspace_tokens
  local direct_visual_z_tokens=false
  local -a control_flags anchor_layers
  anchor_layers=(17)
  case "$variant" in
    question_static_visual)
      experiment_stem="qdpt_d768_question_q10_l17_p20_s8_av10"
      expected_trainable=7805184
      query_source="question_attention_pooling"
      static_visual_write=true
      private_visual_tokens=8
      visual_workspace_tokens=10
      control_flags=(
        --directional-query-source question_attention_pooling
        --directional-static-visual-write
      )
      ;;
    no_static_visual)
      experiment_stem="qdpt_d768_question_q10_l17_p20_no_static_visual"
      expected_trainable=7786752
      query_source="question_attention_pooling"
      static_visual_write=false
      private_visual_tokens=0
      visual_workspace_tokens=0
      control_flags=(
        --directional-query-source question_attention_pooling
        --no-directional-static-visual-write
      )
      ;;
    learned_static_query)
      experiment_stem="qdpt_d768_learned_q10_l17_p20_s8_av10"
      expected_trainable=7805184
      query_source="learned_static"
      static_visual_write=true
      private_visual_tokens=8
      visual_workspace_tokens=10
      control_flags=(
        --directional-query-source learned_static
        --directional-static-visual-write
      )
      ;;
    direct_visual_z_concat)
      experiment_stem="qdpt_d768_question_q10_l17_p20_av10_direct_zv10"
      expected_trainable=8585984
      query_source="question_attention_pooling"
      static_visual_write=true
      private_visual_tokens=0
      visual_workspace_tokens=10
      direct_visual_z_tokens=true
      control_flags=(
        --directional-query-source question_attention_pooling
        --directional-static-visual-write
        --directional-direct-visual-z-tokens
      )
      ;;
    layer18_static_visual)
      experiment_stem="qdpt_d768_question_q10_l18_p20_static_visual18"
      expected_trainable=7805184
      query_source="question_attention_pooling"
      static_visual_write=true
      private_visual_tokens=8
      visual_workspace_tokens=10
      anchor_layers=(18)
      control_flags=(
        --directional-query-source question_attention_pooling
        --directional-static-visual-write
      )
      ;;
    layers17_18_19_shared_static_visual)
      experiment_stem="qdpt_d768_question_q10_l17_18_19_shared_p20_static_visual18"
      expected_trainable=7805184
      query_source="question_attention_pooling"
      static_visual_write=true
      private_visual_tokens=8
      visual_workspace_tokens=10
      anchor_layers=(17 18 19)
      control_flags=(
        --directional-query-source question_attention_pooling
        --directional-static-visual-write
      )
      ;;
    *)
      echo "[ERR] Unsupported QDPT D768 variant: $variant" >&2
      return 2
      ;;
  esac

  local epochs="${QDPT_FINAL_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] QDPT final protocol requires epochs=3, got: $epochs" >&2
    return 2
  fi
  if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
    echo "[ERR] QDPT final protocol requires datasets and pyarrow" >&2
    return 2
  fi

  local experiment_name="${dataset_prefix}_${experiment_stem}_seed${run_seed}"
  local output_dir
  output_dir="$(available_output_dir "$output_root" "${experiment_name}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[QDPT_D768_FINAL_CONFIG] dataset=$dataset experiment=$experiment_name seed=$run_seed data_seed=42 anchors=${anchor_layers[*]} parameter_sharing=all_directional_and_visual_prompt_parameters private_text_prompt=20 text_workspace_anchor=10 private_visual_prompt=$private_visual_tokens visual_workspace_anchor=$visual_workspace_tokens workspace=10x768 query_source=$query_source visual_kv=full_current_anchor_tokens final_text_z=last_anchor static_visual_write=$static_visual_write direct_visual_z_tokens=$direct_visual_z_tokens visual_dynamic_write=false text_output=dynamic_anchor_token_concat expected_trainable=$expected_trainable epochs=3 full_evaluation=$eval_protocol intermediate_full_evaluation=disabled output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest \
      test_dynamic_prompt_tuning.py \
      test_sparse_visual_mmrl.py \
      test_pathvqa_directional_interventions.py || exit 1
    python -m "$train_module" \
      --model-path "$MODEL_PATH" \
      --data-root "$data_root" \
      "${data_flags[@]}" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers "${anchor_layers[@]}" \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${QDPT_FINAL_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --directional-concat-workspace \
      --no-directional-visual-dynamic-write \
      "${control_flags[@]}" \
      --workspace-tokens 10 \
      --workspace-dim 768 \
      --workspace-heads 16 \
      --workspace-lr "${QDPT_FINAL_WORKSPACE_LR:-1e-4}" \
      --epochs 3 \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${QDPT_FINAL_STATIC_LR:-0.3}" \
      --dynamic-lr "${QDPT_FINAL_DYNAMIC_LR:-3e-4}" \
      --batch-size "${QDPT_FINAL_BATCH_SIZE:-2}" \
      --gradient-accumulation "${QDPT_FINAL_GRAD_ACCUM:-16}" \
      --dataloader-workers "${QDPT_FINAL_WORKERS:-2}" \
      --expected-trainable-parameters "$expected_trainable" \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  if [ "$dataset" = "pathvqa" ]; then
    run_pathvqa_dynamic_prompt_epoch3_validation \
      "$output_dir" "$experiment_name" "$run_seed"
    return $?
  fi

  local checkpoint="$output_dir/checkpoints/epoch_3"
  echo "[SLAKE_DYNAMIC_PROMPT_FIXED_TEST] protocol=$eval_protocol split=test checkpoint=$checkpoint"
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_dir/eval_test/epoch_3" \
    "$output_dir/eval_test_epoch_3.log" || return 1
  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_3/slake_summary.json")" || return 1
  printf 'experiment\tseed\tprotocol\ttest_epoch\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t3\t%s\t%s\n' \
    "$experiment_name" "$run_seed" "$eval_protocol" "$test_score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_pathvqa_qdpt_d768_no_static_visual_seed44() {
  run_qdpt_d768_final_dataset pathvqa no_static_visual 44
}

run_pathvqa_qdpt_d768_no_static_visual_resume_eval() {
  local reference_run="${PATHVQA_QDPT_NO_STATIC_RUN:-}"
  if [ -z "$reference_run" ]; then
    reference_run="$(ls -dt \
      "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT"/pathvqa_qdpt_d768_question_q10_l17_p20_no_static_visual_seed44_* \
      2>/dev/null | head -1)"
  fi
  if [ -z "$reference_run" ] \
    || [ ! -f "$reference_run/checkpoints/epoch_3/dynamic_prompt_config.json" ] \
    || [ ! -f "$reference_run/checkpoints/epoch_3/dynamic_prompt.pt" ]; then
    echo "[ERR] No complete no-static-visual epoch3 checkpoint found: $reference_run" >&2
    return 1
  fi
  echo "[QDPT_NO_STATIC_VISUAL_RESUME_EVAL] training=false checkpoint=$reference_run/checkpoints/epoch_3"
  run_pathvqa_dynamic_prompt_epoch3_validation \
    "$reference_run" \
    "pathvqa_qdpt_d768_question_q10_l17_p20_no_static_visual_seed44" \
    44
}

run_pathvqa_qdpt_d768_learned_static_query_seed44() {
  run_qdpt_d768_final_dataset pathvqa learned_static_query 44
}

run_pathvqa_qdpt_d768_direct_visual_z_concat_seeds44_46() {
  local failures=0
  local run_seed
  for run_seed in 44 45 46; do
    echo "[QDPT_DIRECT_VISUAL_Z_SUITE] seed=$run_seed status=starting"
    if run_qdpt_d768_final_dataset \
      pathvqa direct_visual_z_concat "$run_seed"; then
      echo "[QDPT_DIRECT_VISUAL_Z_SUITE] seed=$run_seed status=completed"
    else
      echo "[QDPT_DIRECT_VISUAL_Z_SUITE] seed=$run_seed status=failed_continue" >&2
      failures=$((failures + 1))
    fi
  done
  if [ "$failures" -ne 0 ]; then
    echo "[ERR] Direct visual Z seed suite failures=$failures" >&2
    return 1
  fi
}

run_pathvqa_qdpt_d768_layer_sensitivity_seed44() {
  local failures=0
  local variant
  for variant in layer18_static_visual layers17_18_19_shared_static_visual; do
    echo "[QDPT_LAYER_SENSITIVITY] variant=$variant seed=44 status=starting"
    if run_qdpt_d768_final_dataset pathvqa "$variant" 44; then
      echo "[QDPT_LAYER_SENSITIVITY] variant=$variant seed=44 status=completed"
    else
      echo "[QDPT_LAYER_SENSITIVITY] variant=$variant seed=44 status=failed_continue" >&2
      failures=$((failures + 1))
    fi
  done
  if [ "$failures" -ne 0 ]; then
    echo "[ERR] QDPT layer-sensitivity failures=$failures" >&2
    return 1
  fi
}

run_electrical_qdpt_d768_seed44() {
  local experiment_name="electrical_qdpt_d768_question_q10_l17_p20_static_visual18_seed44"
  local output_dir
  output_dir="$(available_output_dir \
    "$ELECTRICAL_QDPT_OUTPUT_ROOT" \
    "${experiment_name}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[ELECTRICAL_QDPT_CONFIG] experiment=$experiment_name seed=44 data_seed=42 anchors=17 private_text_prompt=20 static_visual_prompt=18 workspace=10x768 query_source=question_attention_pooling visual_kv=full_layer17_tokens dynamic_visual_write=false expected_trainable=7805184 epochs=3 evaluation=private_fixed_holdout output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest \
      test_dynamic_prompt_tuning.py \
      test_sparse_visual_mmrl.py \
      test_electrical_qdpt.py || exit 1
    python -m electrical.train_qdpt \
      --model-path "$MODEL_PATH" \
      --data-root "$ELECTRICAL_DATA_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 17 \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${ELECTRICAL_QDPT_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --directional-concat-workspace \
      --no-directional-visual-dynamic-write \
      --directional-query-source question_attention_pooling \
      --directional-static-visual-write \
      --workspace-tokens 10 \
      --workspace-dim 768 \
      --workspace-heads 16 \
      --workspace-lr "${ELECTRICAL_QDPT_WORKSPACE_LR:-1e-4}" \
      --epochs 3 \
      --seed 44 \
      --data-seed 42 \
      --prompt-lr "${ELECTRICAL_QDPT_STATIC_LR:-0.3}" \
      --dynamic-lr 3e-4 \
      --batch-size "${ELECTRICAL_QDPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${ELECTRICAL_QDPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${ELECTRICAL_QDPT_WORKERS:-2}" \
      --max-length 1024 \
      --expected-trainable-parameters 7805184 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local checkpoint="$output_dir/checkpoints/epoch_3"
  python -m electrical.eval_qdpt \
    --checkpoint "$checkpoint" \
    --base-model "$MODEL_PATH" \
    --output-dir "$output_dir/eval_private/epoch_3" \
    2>&1 | tee "$output_dir/eval_private_epoch_3.log" || return 1
  local score
  score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["score"])' "$output_dir/eval_private/epoch_3/electrical_summary.json")" || return 1
  printf 'experiment\tseed\tepoch\tprivate_accuracy\tcheckpoint\tprotocol\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t44\t3\t%s\t%s\tprivate_fixed_holdout\n' \
    "$experiment_name" "$score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_qdpt_d768_final_pathvqa_slake_seed44() {
  local variant="${QDPT_FINAL_VARIANT:-question_static_visual}"
  run_qdpt_d768_final_dataset pathvqa "$variant" 44 || return 1
  run_qdpt_d768_final_dataset slake "$variant" 44
}

run_pathvqa_directional_width_ablation_seed44() {
  run_pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed44 || return 1
  run_pathvqa_directional_concat_workspace_text_dynamic_only_d256_seed44
}

run_slake_dynamic_prompt_eval() {
  local checkpoint="$1"
  local eval_output_dir="$2"
  local eval_log="$3"
  shift 3
  if [ ! -f "$checkpoint/dynamic_prompt_config.json" ] \
    || [ ! -f "$checkpoint/dynamic_prompt.pt" ]; then
    echo "[ERR] SLAKE Dynamic Prompt checkpoint 不完整: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python slake/slake_official_eval.py \
      --backend dynamic-prompt \
      --base-model "$MODEL_PATH" \
      --checkpoint "$checkpoint" \
      --questions "$SLAKE_DATA_ROOT/test.json" \
      --image-root "$SLAKE_DATA_ROOT/imgs" \
      --output-dir "$eval_output_dir" \
      --language all \
      --expected-split test \
      --overwrite \
      "$@" \
      2>&1 | tee "$eval_log"
  )
}

run_slake_dynamic_prompt_full_workspace_seed44() {
  local experiment_name="slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8"
  local run_seed=44
  local epochs="${SLAKE_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] SLAKE Full Workspace 泛化协议要求 epochs=3，当前为: $epochs" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[SLAKE_DYNAMIC_PROMPT_FULL_WORKSPACE] experiment=$experiment_name seed=$run_seed data_seed=42 anchors=5,11,17 private_prompt=20 private_visual_s=8 private_text_ca=256x8 private_visual_ca=128x4 workspace=32x1024 workspace_blocks=3 workspace_ca=1024x16 workspace_lr=${SLAKE_SHARED_WORKSPACE_LR:-1e-4} expected_trainable=76896256 protocol=fixed_epoch3_test_cross_dataset output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_dynamic_prompt_tuning.py test_sparse_visual_mmrl.py || exit 1
    python -m slake.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$SLAKE_DATA_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 5 11 17 \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${SLAKE_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --shared-workspace \
      --workspace-tokens 32 \
      --workspace-dim 1024 \
      --workspace-heads 16 \
      --workspace-ffn-dim 4096 \
      --workspace-text-attention-dim 1024 \
      --workspace-text-heads 16 \
      --workspace-visual-attention-dim 1024 \
      --workspace-visual-heads 16 \
      --workspace-lr "${SLAKE_SHARED_WORKSPACE_LR:-1e-4}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${SLAKE_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${SLAKE_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${SLAKE_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${SLAKE_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${SLAKE_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters 76896256 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local checkpoint="$output_dir/checkpoints/epoch_3"
  echo "[SLAKE_DYNAMIC_PROMPT_FIXED_TEST] protocol=fixed_epoch3_test_cross_dataset split=test checkpoint=$checkpoint"
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_dir/eval_test/epoch_3" \
    "$output_dir/eval_test_epoch_3.log" || return 1

  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_3/slake_summary.json")" || return 1
  printf 'experiment\tseed\tprotocol\ttest_epoch\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" "$run_seed" fixed_epoch3_test_cross_dataset 3 \
    "$test_score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_slake_dynamic_prompt_full_workspace_17only_seed44() {
  local experiment_name="slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s8"
  local run_seed=44
  local epochs="${SLAKE_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] SLAKE Full Workspace 17-only protocol requires epochs=3, got: $epochs" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[SLAKE_DYNAMIC_PROMPT_FULL_WORKSPACE_17ONLY] experiment=$experiment_name seed=$run_seed data_seed=42 anchors=17 private_prompt=20 private_visual_s=8 private_text_ca=256x8 private_visual_ca=128x4 workspace=32x1024 workspace_blocks=1 workspace_ca=1024x16 workspace_lr=${SLAKE_SHARED_WORKSPACE_LR:-1e-4} expected_trainable=34895872 protocol=fixed_epoch3_test_layer17_ablation output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_dynamic_prompt_tuning.py test_sparse_visual_mmrl.py || exit 1
    python -m slake.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$SLAKE_DATA_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 17 \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${SLAKE_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --shared-workspace \
      --workspace-tokens 32 \
      --workspace-dim 1024 \
      --workspace-heads 16 \
      --workspace-ffn-dim 4096 \
      --workspace-text-attention-dim 1024 \
      --workspace-text-heads 16 \
      --workspace-visual-attention-dim 1024 \
      --workspace-visual-heads 16 \
      --workspace-lr "${SLAKE_SHARED_WORKSPACE_LR:-1e-4}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${SLAKE_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${SLAKE_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${SLAKE_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${SLAKE_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${SLAKE_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters 34895872 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local checkpoint="$output_dir/checkpoints/epoch_3"
  echo "[SLAKE_DYNAMIC_PROMPT_FIXED_TEST] protocol=fixed_epoch3_test_layer17_ablation split=test checkpoint=$checkpoint"
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_dir/eval_test/epoch_3" \
    "$output_dir/eval_test_epoch_3.log" || return 1

  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_3/slake_summary.json")" || return 1
  printf 'experiment\tseed\tprotocol\ttest_epoch\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" "$run_seed" fixed_epoch3_test_layer17_ablation 3 \
    "$test_score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_slake_dynamic_prompt_full_workspace_17only_s20_seed44() {
  local experiment_name="slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s20"
  local run_seed=44
  local epochs="${SLAKE_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] SLAKE Full Workspace 17-only S20 protocol requires epochs=3, got: $epochs" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[SLAKE_DYNAMIC_PROMPT_FULL_WORKSPACE_17ONLY_S20] experiment=$experiment_name seed=$run_seed data_seed=42 anchors=17 private_prompt=20 private_visual_s=20 private_text_ca=256x8 private_visual_ca=128x4 workspace=32x1024 workspace_blocks=1 workspace_ca=1024x16 workspace_lr=${SLAKE_SHARED_WORKSPACE_LR:-1e-4} expected_trainable=34908160 controlled_change=private_visual_s_8_to_20 protocol=fixed_epoch3_test_layer17_s20 output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_dynamic_prompt_tuning.py test_sparse_visual_mmrl.py || exit 1
    python -m slake.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$SLAKE_DATA_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 17 \
      --sparse-visual-rep-tokens 20 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${SLAKE_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --shared-workspace \
      --workspace-tokens 32 \
      --workspace-dim 1024 \
      --workspace-heads 16 \
      --workspace-ffn-dim 4096 \
      --workspace-text-attention-dim 1024 \
      --workspace-text-heads 16 \
      --workspace-visual-attention-dim 1024 \
      --workspace-visual-heads 16 \
      --workspace-lr "${SLAKE_SHARED_WORKSPACE_LR:-1e-4}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${SLAKE_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${SLAKE_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${SLAKE_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${SLAKE_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${SLAKE_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters 34908160 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local checkpoint="$output_dir/checkpoints/epoch_3"
  echo "[SLAKE_DYNAMIC_PROMPT_FIXED_TEST] protocol=fixed_epoch3_test_layer17_s20 split=test checkpoint=$checkpoint"
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_dir/eval_test/epoch_3" \
    "$output_dir/eval_test_epoch_3.log" || return 1

  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_3/slake_summary.json")" || return 1
  printf 'experiment\tseed\tprotocol\ttest_epoch\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" "$run_seed" fixed_epoch3_test_layer17_s20 3 \
    "$test_score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_slake_directional_concat_workspace_seed44() {
  local experiment_name="slake_directional_concat_workspace_z10_d1024_l17_private_p20_s8"
  local run_seed=44
  local epochs="${SLAKE_DYNAMIC_PROMPT_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] SLAKE Directional Concat Workspace protocol requires epochs=3, got: $epochs" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[SLAKE_DIRECTIONAL_CONCAT_WORKSPACE] experiment=$experiment_name seed=$run_seed data_seed=42 anchor=17 private_text_prompt=20 text_workspace_anchor=10 private_visual_prompt=8 visual_workspace_anchor=10 workspace=10x1024 query=text_attention_pooling kv=full_visual cross_attention=1024x16 visual_output=token_concat text_output=token_concat zero_init_dynamic_mlp=true old_private_text_ca=false old_private_visual_ca=false workspace_lr=${SLAKE_DIRECTIONAL_WORKSPACE_LR:-1e-4} expected_trainable=12726784 protocol=fixed_epoch3_test output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_dynamic_prompt_tuning.py test_sparse_visual_mmrl.py || exit 1
    python -m slake.train_dynamic_prompt \
      --model-path "$MODEL_PATH" \
      --data-root "$SLAKE_DATA_ROOT" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --prompt-length 20 \
      --attention-dim 256 \
      --attention-heads 8 \
      --sparse-visual \
      --sparse-visual-anchor-layers 17 \
      --sparse-visual-rep-tokens 8 \
      --sparse-visual-attention-dim 128 \
      --sparse-visual-heads 4 \
      --sparse-visual-lr "${SLAKE_SPARSE_VISUAL_LR:-3e-5}" \
      --shared-s-text-mode none \
      --directional-concat-workspace \
      --workspace-tokens 10 \
      --workspace-dim 1024 \
      --workspace-heads 16 \
      --workspace-lr "${SLAKE_DIRECTIONAL_WORKSPACE_LR:-1e-4}" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --prompt-lr "${SLAKE_DYNAMIC_PROMPT_STATIC_LR:-0.3}" \
      --dynamic-lr "${SLAKE_DYNAMIC_PROMPT_LR:-3e-4}" \
      --batch-size "${SLAKE_DYNAMIC_PROMPT_BATCH_SIZE:-2}" \
      --gradient-accumulation "${SLAKE_DYNAMIC_PROMPT_GRAD_ACCUM:-16}" \
      --dataloader-workers "${SLAKE_DYNAMIC_PROMPT_WORKERS:-2}" \
      --expected-trainable-parameters 12726784 \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local checkpoint="$output_dir/checkpoints/epoch_3"
  echo "[SLAKE_DYNAMIC_PROMPT_FIXED_TEST] protocol=fixed_epoch3_test split=test checkpoint=$checkpoint"
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_dir/eval_test/epoch_3" \
    "$output_dir/eval_test_epoch_3.log" || return 1

  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_3/slake_summary.json")" || return 1
  printf 'experiment\tseed\tprotocol\ttest_epoch\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" "$run_seed" fixed_epoch3_test 3 \
    "$test_score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_slake_dynamic_prompt_workspace_path_interventions() {
  local reference_run="${SLAKE_WORKSPACE_REFERENCE_RUN:-$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT/slake_dynamic_prompt_full_workspace_z32_d1024_blocks3_private_p20_private_s8_seed44_20260830}"
  local checkpoint="$reference_run/checkpoints/epoch_3"
  if [ ! -f "$checkpoint/dynamic_prompt_config.json" ] \
    || [ ! -f "$checkpoint/dynamic_prompt.pt" ]; then
    echo "[ERR] SLAKE three-layer Workspace reference checkpoint not found: $checkpoint" >&2
    return 1
  fi

  local output_root
  output_root="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "slake_full_workspace_path_interventions_seed44_${RUN_DATE}")"
  mkdir -p "$output_root"
  printf 'reference_run\t%s\ncheckpoint\t%s\n' \
    "$reference_run" "$checkpoint" > "$output_root/intervention_manifest.tsv"

  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/visual_write_off_l5" \
    "$output_root/visual_write_off_l5.log" \
    --workspace-disable-visual-write-layer 5 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/visual_write_off_l11" \
    "$output_root/visual_write_off_l11.log" \
    --workspace-disable-visual-write-layer 11 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/visual_write_off_l5_l11" \
    "$output_root/visual_write_off_l5_l11.log" \
    --workspace-disable-visual-write-layer 5 \
    --workspace-disable-visual-write-layer 11 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/workspace_update_off_l5" \
    "$output_root/workspace_update_off_l5.log" \
    --workspace-bypass-update-layer 5 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/workspace_update_off_l11" \
    "$output_root/workspace_update_off_l11.log" \
    --workspace-bypass-update-layer 11 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/workspace_update_off_l5_l11" \
    "$output_root/workspace_update_off_l5_l11.log" \
    --workspace-bypass-update-layer 5 \
    --workspace-bypass-update-layer 11 || return 1

  python diagnostics/compare_slake_workspace_interventions.py \
    --baseline "$reference_run/eval_test/epoch_3" \
    --intervention-root "$output_root" || return 1
  echo "[SLAKE_WORKSPACE_PATH_INTERVENTIONS_DONE] output=$output_root"
}

run_slake_dynamic_prompt_17only_final_path_interventions() {
  local reference_run="${SLAKE_17ONLY_REFERENCE_RUN:-$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT/slake_dynamic_prompt_full_workspace_17only_z32_d1024_blocks1_private_p20_private_s8_seed44_20260830}"
  local checkpoint="$reference_run/checkpoints/epoch_3"
  if [ ! -f "$checkpoint/dynamic_prompt_config.json" ] \
    || [ ! -f "$checkpoint/dynamic_prompt.pt" ]; then
    echo "[ERR] SLAKE layer17-only Workspace reference checkpoint not found: $checkpoint" >&2
    return 1
  fi

  local output_root
  output_root="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "slake_17only_final_path_interventions_seed44_${RUN_DATE}")"
  mkdir -p "$output_root"
  printf 'reference_run\t%s\ncheckpoint\t%s\n' \
    "$reference_run" "$checkpoint" > "$output_root/intervention_manifest.tsv"

  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/workspace_visual_off_l17" \
    "$output_root/workspace_visual_off_l17.log" \
    --workspace-disable-visual-write-layer 17 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/all_visual_rep_off_l17" \
    "$output_root/all_visual_rep_off_l17.log" \
    --workspace-disable-visual-rep-write-layer 17 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/workspace_text_off" \
    "$output_root/workspace_text_off.log" \
    --workspace-disable-text-write || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/workspace_visual_text_off" \
    "$output_root/workspace_visual_text_off.log" \
    --workspace-disable-visual-write-layer 17 \
    --workspace-disable-text-write || return 1

  python diagnostics/compare_slake_workspace_interventions.py \
    --baseline "$reference_run/eval_test/epoch_3" \
    --intervention-root "$output_root" || return 1
  echo "[SLAKE_17ONLY_FINAL_PATH_INTERVENTIONS_DONE] output=$output_root"
}

run_slake_directional_concat_workspace_delta_interventions() {
  local reference_run="${SLAKE_DIRECTIONAL_CONCAT_REFERENCE_RUN:-$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT/slake_directional_concat_workspace_z10_d1024_l17_private_p20_s8_seed44_20260830}"
  local checkpoint="$reference_run/checkpoints/epoch_3"
  local baseline="$reference_run/eval_test/epoch_3"
  if [ ! -f "$checkpoint/dynamic_prompt_config.json" ] \
    || [ ! -f "$checkpoint/dynamic_prompt.pt" ]; then
    echo "[ERR] SLAKE Directional Concat checkpoint not found: $checkpoint" >&2
    return 1
  fi
  if [ ! -f "$baseline/slake_comparisons.json" ]; then
    echo "[ERR] SLAKE Directional Concat baseline predictions not found: $baseline" >&2
    return 1
  fi

  local output_root
  output_root="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "slake_directional_concat_delta_interventions_seed44_${RUN_DATE}")"
  mkdir -p "$output_root"
  printf 'reference_run\t%s\ncheckpoint\t%s\nbaseline\t%s\ncontrol\tanchors_and_token_counts_preserved\n' \
    "$reference_run" "$checkpoint" "$baseline" \
    > "$output_root/intervention_manifest.tsv"

  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/text_delta_off" \
    "$output_root/text_delta_off.log" \
    --directional-text-delta-scale 0 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/text_delta_half" \
    "$output_root/text_delta_half.log" \
    --directional-text-delta-scale 0.5 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/visual_delta_off" \
    "$output_root/visual_delta_off.log" \
    --directional-visual-delta-scale 0 || return 1
  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/visual_delta_half" \
    "$output_root/visual_delta_half.log" \
    --directional-visual-delta-scale 0.5 || return 1

  python diagnostics/compare_slake_workspace_interventions.py \
    --baseline "$baseline" \
    --intervention-root "$output_root" || return 1
  echo "[SLAKE_DIRECTIONAL_CONCAT_DELTA_INTERVENTIONS_DONE] output=$output_root"
}

run_slake_directional_concat_workspace_visual_memory_mismatch() {
  local reference_run="${SLAKE_DIRECTIONAL_CONCAT_REFERENCE_RUN:-$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT/slake_directional_concat_workspace_z10_d1024_l17_private_p20_s8_seed44_20260830}"
  local checkpoint="$reference_run/checkpoints/epoch_3"
  local baseline="$reference_run/eval_test/epoch_3"
  if [ ! -f "$checkpoint/dynamic_prompt_config.json" ] \
    || [ ! -f "$checkpoint/dynamic_prompt.pt" ]; then
    echo "[ERR] SLAKE Directional Concat checkpoint not found: $checkpoint" >&2
    return 1
  fi
  if [ ! -f "$baseline/slake_comparisons.json" ]; then
    echo "[ERR] SLAKE Directional Concat baseline predictions not found: $baseline" >&2
    return 1
  fi

  local output_root
  output_root="$(available_output_dir "$SLAKE_DYNAMIC_PROMPT_OUTPUT_ROOT" "slake_directional_concat_visual_memory_mismatch_seed44_${RUN_DATE}")"
  mkdir -p "$output_root"
  printf 'reference_run\t%s\ncheckpoint\t%s\nbaseline\t%s\ncontrol\tprevious_distinct_image_only_for_directional_ca_visual_kv\noriginal_image_input\tunchanged\n' \
    "$reference_run" "$checkpoint" "$baseline" \
    > "$output_root/intervention_manifest.tsv"

  run_slake_dynamic_prompt_eval \
    "$checkpoint" \
    "$output_root/visual_kv_previous_distinct_image" \
    "$output_root/visual_kv_previous_distinct_image.log" \
    --directional-visual-memory-mode previous-distinct-image || return 1

  python diagnostics/compare_slake_workspace_interventions.py \
    --baseline "$baseline" \
    --intervention-root "$output_root" || return 1
  echo "[SLAKE_DIRECTIONAL_CONCAT_VISUAL_MEMORY_MISMATCH_DONE] output=$output_root"
}

run_pathvqa_directional_concat_workspace_conditioning_mismatch() {
  local reference_run="${PATHVQA_DIRECTIONAL_REFERENCE_RUN:-$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT/pathvqa_directional_concat_workspace_text_dynamic_only_z10_d1024_l17_private_p20_s8_seed44_20260831}"
  local checkpoint="$reference_run/checkpoints/epoch_3"
  local baseline="$reference_run/eval_validation/epoch_3"
  if [ ! -f "$checkpoint/dynamic_prompt_config.json" ] \
    || [ ! -f "$checkpoint/dynamic_prompt.pt" ]; then
    echo "[ERR] PathVQA Directional checkpoint not found: $checkpoint" >&2
    return 1
  fi
  if [ ! -f "$baseline/pathvqa_comparisons.json" ]; then
    echo "[ERR] PathVQA Directional baseline predictions not found: $baseline" >&2
    return 1
  fi

  local output_root
  output_root="$(available_output_dir "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT" "pathvqa_directional_concat_conditioning_mismatch_seed44_${RUN_DATE}")"
  mkdir -p "$output_root"
  printf 'reference_run\t%s\ncheckpoint\t%s\nbaseline\t%s\nsplit\tvalidation\ntraining\tnone\nquestion_q_control\tprevious_distinct_question_only_for_directional_ca_q\nvisual_kv_control\tprevious_distinct_image_only_for_directional_ca_kv\noriginal_vlm_inputs\tunchanged\n' \
    "$reference_run" "$checkpoint" "$baseline" \
    > "$output_root/intervention_manifest.tsv"

  (
    cd "$ROOT_DIR" || exit 1
    CUDA_VISIBLE_DEVICES='' python -m unittest \
      test_dynamic_prompt_tuning.py \
      test_sparse_visual_mmrl.py \
      test_pathvqa_directional_interventions.py
  ) || return 1

  run_pathvqa_dynamic_prompt_eval \
    "$checkpoint" \
    validation \
    "$output_root/question_q_previous_distinct" \
    "$output_root/question_q_previous_distinct.log" \
    --directional-question-query-mode previous-distinct-question || return 1

  run_pathvqa_dynamic_prompt_eval \
    "$checkpoint" \
    validation \
    "$output_root/visual_kv_previous_distinct_image" \
    "$output_root/visual_kv_previous_distinct_image.log" \
    --directional-visual-memory-mode previous-distinct-image || return 1

  python diagnostics/compare_pathvqa_conditioning_mismatches.py \
    --baseline "$baseline" \
    --intervention-root "$output_root" || return 1
  echo "[PATHVQA_DIRECTIONAL_CONDITIONING_MISMATCH_DONE] output=$output_root"
}

run_pathvqa_dynamic_prompt_raw_shared_s_suite_seed44() {
  run_pathvqa_dynamic_prompt_raw_shared_s_separate_residual_seed44 || return 1
  run_pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_seed44 || return 1
}

run_pathvqa_lora_visual_attn_r128() {
  local experiment_name="${PATHVQA_LORA_EXPERIMENT_NAME:-pathvqa_lora_visual_all_attention_r128}"
  local last_n_layers="${PATHVQA_LORA_LAST_N_VISION_LAYERS:-24}"
  local run_seed="${PATHVQA_RUN_SEED:-$SEED}"
  local epochs="${PATHVQA_LORA_EPOCHS:-3}"
  if ! python -c 'import datasets, pyarrow, peft' >/dev/null 2>&1; then
    echo "[ERR] PathVQA LoRA 需要 datasets、pyarrow 和 peft。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_LORA_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_LORA] experiment=$experiment_name seed=$run_seed last_n_vision_layers=$last_n_layers rank=128 epochs=$epochs output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m pathvqa.train_visual_lora \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --model-path "$MODEL_PATH" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --last-n-vision-layers "$last_n_layers" \
      --epochs "$epochs" \
      --seed "$run_seed" \
      --data-seed 42 \
      --learning-rate "${PATHVQA_LORA_LR:-1e-4}" \
      --batch-size "${PATHVQA_LORA_BATCH_SIZE:-1}" \
      --gradient-accumulation "${PATHVQA_LORA_GRAD_ACCUM:-32}" \
      --dataloader-workers "${PATHVQA_LORA_WORKERS:-2}" \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local epoch_id
  for ((epoch_id = 1; epoch_id <= epochs; epoch_id++)); do
    local checkpoint="$output_dir/checkpoints/epoch_${epoch_id}"
    echo "[PATHVQA_LORA_VALIDATION] epoch=$epoch_id checkpoint=$checkpoint"
    run_pathvqa_lora_eval \
      "$checkpoint" \
      validation \
      "$output_dir/eval_validation/epoch_${epoch_id}" \
      "$output_dir/eval_validation_epoch_${epoch_id}.log" || return 1
  done

  local selection
  selection="$(python "$ROOT_DIR/pathvqa/select_best_epoch.py" --root "$output_dir" --epochs "$epochs")" || return 1
  local best_epoch
  local best_validation_score
  IFS=$'\t' read -r best_epoch best_validation_score <<< "$selection"
  local best_checkpoint="$output_dir/checkpoints/epoch_${best_epoch}"
  echo "[PATHVQA_LORA_TEST] best_epoch=$best_epoch validation=$best_validation_score"
  run_pathvqa_lora_eval \
    "$best_checkpoint" \
    test \
    "$output_dir/eval_test/epoch_${best_epoch}" \
    "$output_dir/eval_test_epoch_${best_epoch}.log" || return 1

  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_${best_epoch}/pathvqa_summary.json")" || return 1
  printf 'experiment\tseed\tbest_epoch\tvalidation_accuracy\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" "$run_seed" "$best_epoch" "$best_validation_score" \
    "$test_score" "$best_checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_pathvqa_lora_visual_attn_r128_then_base() {
  run_pathvqa_lora_visual_attn_r128 || return 1
  run_pathvqa_base
}

run_pathvqa_lora_visual_last8_attn_r128() {
  PATHVQA_LORA_EXPERIMENT_NAME=pathvqa_lora_visual_last8_attention_r128 \
  PATHVQA_LORA_LAST_N_VISION_LAYERS=8 \
    run_pathvqa_lora_visual_attn_r128
}

run_pathvqa_lora_full_model_attn_r8() {
  local experiment_name=pathvqa_lora_full_model_attention_r8
  local run_seed="$1"
  local epochs="${PATHVQA_LORA_EPOCHS:-3}"
  if [ "$epochs" -ne 3 ]; then
    echo "[ERR] Full-model LoRA 对比协议要求 epochs=3，当前为: $epochs" >&2
    return 2
  fi
  if ! python -c 'import datasets, pyarrow, peft' >/dev/null 2>&1; then
    echo "[ERR] PathVQA LoRA 需要 datasets、pyarrow 和 peft。先运行: python -m pip install -r pathvqa/requirements.txt" >&2
    return 2
  fi

  local output_dir
  output_dir="$(available_output_dir "$PATHVQA_LORA_OUTPUT_ROOT" "${experiment_name}_seed${run_seed}_${RUN_DATE}")"
  mkdir -p "$output_dir"
  echo "[PATHVQA_LORA_FULL_MODEL] experiment=$experiment_name seed=$run_seed target=visual24_qkv_proj+language36_qkvo rank=8 alpha=16 expected_trainable=7077888 epochs=3 protocol=fixed_epoch3_validation output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_pathvqa_lora_targets.py || exit 1
    python -m pathvqa.train_visual_lora \
      --data-root "$PATHVQA_DATA_ROOT" \
      --cache-dir "$PATHVQA_CACHE_ROOT" \
      --model-path "$MODEL_PATH" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --target-scope full_model \
      --rank 8 \
      --last-n-vision-layers 24 \
      --expected-trainable-parameters 7077888 \
      --epochs 3 \
      --seed "$run_seed" \
      --data-seed 42 \
      --learning-rate "${PATHVQA_LORA_LR:-1e-4}" \
      --batch-size "${PATHVQA_LORA_BATCH_SIZE:-1}" \
      --gradient-accumulation "${PATHVQA_LORA_GRAD_ACCUM:-32}" \
      --dataloader-workers "${PATHVQA_LORA_WORKERS:-2}" \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1

  local checkpoint="$output_dir/checkpoints/epoch_3"
  run_pathvqa_lora_eval \
    "$checkpoint" \
    validation \
    "$output_dir/eval_validation/epoch_3" \
    "$output_dir/eval_validation_epoch_3.log" \
    lora || return 1

  local validation_score
  validation_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_validation/epoch_3/pathvqa_summary.json")" || return 1
  printf 'experiment\tseed\tepoch\tvalidation_accuracy\tcheckpoint\tprotocol\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t3\t%s\t%s\tfixed_epoch3_validation\n' \
    "$experiment_name" "$run_seed" "$validation_score" "$checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_pathvqa_lora_full_model_attn_r8_seed44() {
  run_pathvqa_lora_full_model_attn_r8 44
}

run_pathvqa_day2_d768_lora_r8_seeds45_46() {
  local suite_failures=0

  echo "[PATHVQA_DAY2_SUITE] 1/4 D768 seed45"
  run_pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed 45 \
    || suite_failures=$((suite_failures + 1))

  echo "[PATHVQA_DAY2_SUITE] 2/4 D768 seed46"
  run_pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed 46 \
    || suite_failures=$((suite_failures + 1))

  echo "[PATHVQA_DAY2_SUITE] 3/4 Full-Attention LoRA-r8 seed45"
  run_pathvqa_lora_full_model_attn_r8 45 \
    || suite_failures=$((suite_failures + 1))

  echo "[PATHVQA_DAY2_SUITE] 4/4 Full-Attention LoRA-r8 seed46"
  run_pathvqa_lora_full_model_attn_r8 46 \
    || suite_failures=$((suite_failures + 1))

  if [ "$suite_failures" -ne 0 ]; then
    echo "[ERR] PathVQA Day 2 suite finished with failed experiments=$suite_failures; all four experiments were attempted." >&2
    return 1
  fi
  echo "[PATHVQA_DAY2_SUITE_DONE] all four experiments completed successfully"
}

run_pathvqa_last8_lora_minimal_mmrl_relation_suite() {
  local suite_seed=44
  local shared_stage1
  shared_stage1="$(available_output_dir \
    "$PATHVQA_OUTPUT_ROOT" \
    "pathvqa_mmrl_minimal_shared_stage1_seed${suite_seed}_${RUN_DATE}")"

  PATHVQA_RUN_SEED="$suite_seed" \
    run_pathvqa_lora_visual_last8_attn_r128 || return 1

  PATHVQA_EXPERIMENT_NAME=pathvqa_mmrl_minimal_shared_s_mean_relation0 \
  PATHVQA_RUN_SEED="$suite_seed" \
  MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
  MMRL_MEMORY_POOLING_MODE=mean \
  MMRL_FUSION_MODE=cross_attention \
  MMRL_SAME_INIT_LAYER_PROJECTORS=0 \
  MMRL_RELATION_LOSS_WEIGHT=0 \
  PATHVQA_EXPECTED_MMRL_PARAMETERS=7927808 \
  PATHVQA_SELECT_BEST_EPOCH=1 \
  PATHVQA_STAGE1_CHECKPOINT_OUT="$shared_stage1" \
    run_pathvqa || return 1

  PATHVQA_EXPERIMENT_NAME=pathvqa_mmrl_minimal_shared_s_mean_relation0050 \
  PATHVQA_RUN_SEED="$suite_seed" \
  MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
  MMRL_MEMORY_POOLING_MODE=mean \
  MMRL_FUSION_MODE=cross_attention \
  MMRL_SAME_INIT_LAYER_PROJECTORS=0 \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
  PATHVQA_EXPECTED_MMRL_PARAMETERS=7927808 \
  PATHVQA_SELECT_BEST_EPOCH=1 \
  PATHVQA_STAGE1_CHECKPOINT_IN="$shared_stage1" \
    run_pathvqa
}

run_pathvqa_minimal_mmrl_prompt20_seed44() {
  local shared_stage1="${PATHVQA_MINIMAL_STAGE1_CHECKPOINT:-$PATHVQA_OUTPUT_ROOT/pathvqa_mmrl_minimal_shared_stage1_seed44_20260826}"
  if [ ! -f "$shared_stage1/mmrl_delta.safetensors" ]; then
    echo "[ERR] 极简 MMRL Stage1 checkpoint 不存在: $shared_stage1" >&2
    return 1
  fi
  PATHVQA_EXPERIMENT_NAME=pathvqa_mmrl_minimal_shared_s_mean_prompt20_relation0050 \
  PATHVQA_RUN_SEED=44 \
  MMRL_QUERY_ARCHITECTURE=shared_direct_post_cross \
  MMRL_MEMORY_POOLING_MODE=mean \
  MMRL_FUSION_MODE=cross_attention \
  MMRL_SAME_INIT_LAYER_PROJECTORS=0 \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
  PATHVQA_EXPECTED_MMRL_PARAMETERS=7927808 \
  PATHVQA_SOFT_PROMPT_LENGTH=20 \
  PATHVQA_SOFT_PROMPT_INIT_SEED=44 \
  PATHVQA_PROMPT_LR=0.3 \
  PATHVQA_PROMPT_WARMUP_RATIO=0.03 \
  PATHVQA_EXPECTED_TOTAL_TRAINABLE_PARAMETERS=7979008 \
  PATHVQA_MMRL_EVAL_BACKEND=mmrl-prompt \
  PATHVQA_SELECT_BEST_EPOCH=1 \
  PATHVQA_STAGE1_CHECKPOINT_IN="$shared_stage1" \
    run_pathvqa
}

run_pathvqa_minimal_mmrl_prompt20_resume_eval() {
  local experiment_name=pathvqa_mmrl_minimal_shared_s_mean_prompt20_relation0050
  local output_dir="${PATHVQA_RESUME_OUTPUT_DIR:-$PATHVQA_OUTPUT_ROOT/${experiment_name}_seed44_20260827}"
  local epochs=3
  local epoch_id
  PATHVQA_MMRL_EVAL_BACKEND=mmrl-prompt
  for ((epoch_id = 1; epoch_id <= epochs; epoch_id++)); do
    local checkpoint="$output_dir/checkpoints/stage3_epoch_${epoch_id}"
    echo "[PATHVQA_MMRL_VALIDATION] epoch=$epoch_id checkpoint=$checkpoint"
    run_pathvqa_checkpoint_eval \
      "$checkpoint" \
      "$output_dir/eval_validation/epoch_${epoch_id}" \
      "$output_dir/eval_validation_epoch_${epoch_id}.log" \
      validation || return 1
  done
  local selection
  selection="$(python "$ROOT_DIR/pathvqa/select_best_epoch.py" --root "$output_dir" --epochs "$epochs")" || return 1
  local best_epoch
  local best_validation_score
  IFS=$'\t' read -r best_epoch best_validation_score <<< "$selection"
  local best_checkpoint="$output_dir/checkpoints/stage3_epoch_${best_epoch}"
  echo "[PATHVQA_MMRL_TEST] best_epoch=$best_epoch validation=$best_validation_score"
  run_pathvqa_checkpoint_eval \
    "$best_checkpoint" \
    "$output_dir/eval_test/epoch_${best_epoch}" \
    "$output_dir/eval_test_epoch_${best_epoch}.log" \
    test || return 1
  local test_score
  test_score="$(python -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["overall_accuracy"])' "$output_dir/eval_test/epoch_${best_epoch}/pathvqa_summary.json")" || return 1
  printf 'experiment\tseed\tbest_epoch\tvalidation_accuracy\ttest_accuracy\tcheckpoint\n' \
    > "$output_dir/selected_result.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" 44 "$best_epoch" "$best_validation_score" \
    "$test_score" "$best_checkpoint" \
    >> "$output_dir/selected_result.tsv"
  cat "$output_dir/selected_result.tsv"
}

run_final_seeds4() {
  local seed
  for seed in 44 45 46 47; do
    SLAKE_RUN_SEED="$seed" run_slake || return 1
  done
}

run_ablation_no_relation_seeds2() {
  local seed
  for seed in 44 45; do
    SLAKE_EXPERIMENT_NAME="slake_mmrl_ablation_no_relation" \
    SLAKE_RUN_SEED="$seed" \
    MMRL_SAME_INIT_LAYER_PROJECTORS=1 \
    MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
    MMRL_MEMORY_POOLING_MODE=multi_query \
    MMRL_RELATION_LOSS_WEIGHT=0.0 \
      run_slake || return 1
  done
}

run_ablation_independent_init_seeds2() {
  local seed
  for seed in 44 45; do
    SLAKE_EXPERIMENT_NAME="slake_mmrl_ablation_independent_init_relation0050" \
    SLAKE_RUN_SEED="$seed" \
    MMRL_SAME_INIT_LAYER_PROJECTORS=0 \
    MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
    MMRL_MEMORY_POOLING_MODE=multi_query \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || return 1
  done
}

run_ablation_static_query_seed45() {
  SLAKE_EXPERIMENT_NAME="slake_mmrl_ablation_static_query_relation0050" \
  SLAKE_RUN_SEED=45 \
  MMRL_SAME_INIT_LAYER_PROJECTORS=1 \
  MMRL_USE_DYNAMIC_CROSS_ATTENTION=0 \
  MMRL_MEMORY_POOLING_MODE=multi_query \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
    run_slake
}

run_ablation_mean_pooling_seed45() {
  SLAKE_EXPERIMENT_NAME="slake_mmrl_ablation_mean_pooling_relation0050" \
  SLAKE_RUN_SEED=45 \
  MMRL_SAME_INIT_LAYER_PROJECTORS=1 \
  MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
  MMRL_MEMORY_POOLING_MODE=mean \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
    run_slake
}

run_mean_pooling_final_seeds46_47() {
  local seed
  for seed in 46 47; do
    SLAKE_EXPERIMENT_NAME="slake_mmrl_ablation_mean_pooling_relation0050" \
    SLAKE_RUN_SEED="$seed" \
    MMRL_SAME_INIT_LAYER_PROJECTORS=1 \
    MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
    MMRL_MEMORY_POOLING_MODE=mean \
    MMRL_RELATION_LOSS_WEIGHT=0.05 \
      run_slake || return 1
  done
}

run_mean_pooling_no_relation_seed45() {
  SLAKE_EXPERIMENT_NAME="slake_mmrl_mean_pooling_no_relation" \
  SLAKE_RUN_SEED=45 \
  MMRL_SAME_INIT_LAYER_PROJECTORS=1 \
  MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
  MMRL_MEMORY_POOLING_MODE=mean \
  MMRL_RELATION_LOSS_WEIGHT=0.0 \
    run_slake
}

run_mean_pooling_independent_init_seed45() {
  SLAKE_EXPERIMENT_NAME="slake_mmrl_mean_pooling_independent_init_relation0050" \
  SLAKE_RUN_SEED=45 \
  MMRL_SAME_INIT_LAYER_PROJECTORS=0 \
  MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
  MMRL_MEMORY_POOLING_MODE=mean \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
    run_slake
}

run_mean_final_completion_suite() {
  run_mean_pooling_final_seeds46_47 || return 1
  run_mean_pooling_no_relation_seed45 || return 1
  run_mean_pooling_independent_init_seed45
}

run_text_guided_balanced_fusion_slots8_seed45() {
  SLAKE_EXPERIMENT_NAME="slake_mmrl_text_guided_balanced_fusion_slots8_relation0050" \
  SLAKE_RUN_SEED=45 \
  MMRL_SAME_INIT_LAYER_PROJECTORS=1 \
  MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
  MMRL_MEMORY_POOLING_MODE=text_guided \
  MMRL_MEMORY_QUERY_COUNT=8 \
  MMRL_MEMORY_ATTENTION_DIM=128 \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
    run_slake
}

run_prompt_tuning_seed44() {
  local experiment_name="slake_prompt_tuning_len20"
  local output_dir
  output_dir="$(available_output_dir "$SLAKE_OUTPUT_ROOT" "${experiment_name}_seed44_${RUN_DATE}")"
  mkdir -p "$output_dir/eval"
  echo "[SLAKE_PROMPT_TUNING] experiment=$experiment_name seed=44 output=$output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python -m slake.train_prompt_tuning \
      --model-path "$MODEL_PATH" \
      --data-root "$SLAKE_DATA_ROOT" \
      --output-dir "$output_dir" \
      --prompt-length "${PROMPT_TUNING_LENGTH:-20}" \
      --epochs "${PROMPT_TUNING_EPOCHS:-3}" \
      --seed 44 \
      --data-seed 42 \
      --learning-rate "${PROMPT_TUNING_LR:-0.3}" \
      --batch-size "${PROMPT_TUNING_BATCH_SIZE:-2}" \
      --gradient-accumulation "${PROMPT_TUNING_GRAD_ACCUM:-16}" \
      2>&1 | tee "$output_dir/train.log"
  ) || return 1
  (
    cd "$ROOT_DIR" || exit 1
    python slake/slake_official_eval.py \
      --backend prompt-tuning \
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

run_concat_mlp_fusion_seed45() {
  SLAKE_EXPERIMENT_NAME="slake_mmrl_mean_pooling_concat_mlp_relation0050" \
  SLAKE_RUN_SEED=45 \
  MMRL_SAME_INIT_LAYER_PROJECTORS=1 \
  MMRL_USE_DYNAMIC_CROSS_ATTENTION=1 \
  MMRL_MEMORY_POOLING_MODE=mean \
  MMRL_FUSION_MODE=concat_mlp \
  MMRL_RELATION_LOSS_WEIGHT=0.05 \
    run_slake
}

run_overnight_unit_tests() {
  (
    cd "$ROOT_DIR" || exit 1
    python -m unittest test_attention_pooling.py test_prompt_tuning.py
  )
}

run_overnight_prompt_and_concat_mlp() {
  local suite_failures=0
  run_overnight_unit_tests || suite_failures=$((suite_failures + 1))
  run_prompt_tuning_seed44 || suite_failures=$((suite_failures + 1))
  run_concat_mlp_fusion_seed45 || suite_failures=$((suite_failures + 1))
  if [ "$suite_failures" -ne 0 ]; then
    echo "[ERR] 今晚串行实验失败数=$suite_failures" >&2
    return 1
  fi
}

run_ablation_suite() {
  run_ablation_no_relation_seeds2 || return 1
  run_ablation_independent_init_seeds2 || return 1
  run_ablation_static_query_seed45 || return 1
  run_ablation_mean_pooling_seed45
}

failures=0
case "$RUN_TARGET" in
  train)
    run_train_dataset || failures=$((failures + 1))
    ;;
  slake)
    run_slake || failures=$((failures + 1))
    ;;
  slake_final_seeds4)
    run_final_seeds4 || failures=$((failures + 1))
    ;;
  slake_ablation_no_relation_seeds2)
    run_ablation_no_relation_seeds2 || failures=$((failures + 1))
    ;;
  slake_ablation_independent_init_seeds2)
    run_ablation_independent_init_seeds2 || failures=$((failures + 1))
    ;;
  slake_ablation_static_query_seed45)
    run_ablation_static_query_seed45 || failures=$((failures + 1))
    ;;
  slake_ablation_mean_pooling_seed45)
    run_ablation_mean_pooling_seed45 || failures=$((failures + 1))
    ;;
  slake_mean_final_completion_suite)
    run_mean_final_completion_suite || failures=$((failures + 1))
    ;;
  slake_text_guided_balanced_fusion_slots8_seed45)
    run_text_guided_balanced_fusion_slots8_seed45 || failures=$((failures + 1))
    ;;
  slake_prompt_tuning_seed44)
    run_prompt_tuning_seed44 || failures=$((failures + 1))
    ;;
  slake_concat_mlp_fusion_seed45)
    run_concat_mlp_fusion_seed45 || failures=$((failures + 1))
    ;;
  slake_overnight_prompt_and_concat_mlp)
    run_overnight_prompt_and_concat_mlp || failures=$((failures + 1))
    ;;
  slake_ablation_suite)
    run_ablation_suite || failures=$((failures + 1))
    ;;
  slake_dynamic_prompt_full_workspace_seed44)
    run_slake_dynamic_prompt_full_workspace_seed44 || failures=$((failures + 1))
    ;;
  slake_dynamic_prompt_full_workspace_17only_seed44)
    run_slake_dynamic_prompt_full_workspace_17only_seed44 || failures=$((failures + 1))
    ;;
  slake_dynamic_prompt_full_workspace_17only_s20_seed44)
    run_slake_dynamic_prompt_full_workspace_17only_s20_seed44 || failures=$((failures + 1))
    ;;
  slake_directional_concat_workspace_seed44)
    run_slake_directional_concat_workspace_seed44 || failures=$((failures + 1))
    ;;
  slake_dynamic_prompt_workspace_path_interventions)
    run_slake_dynamic_prompt_workspace_path_interventions || failures=$((failures + 1))
    ;;
  slake_dynamic_prompt_17only_final_path_interventions)
    run_slake_dynamic_prompt_17only_final_path_interventions || failures=$((failures + 1))
    ;;
  slake_directional_concat_workspace_delta_interventions)
    run_slake_directional_concat_workspace_delta_interventions || failures=$((failures + 1))
    ;;
  slake_directional_concat_workspace_visual_memory_mismatch)
    run_slake_directional_concat_workspace_visual_memory_mismatch || failures=$((failures + 1))
    ;;
  pathvqa_directional_concat_workspace_conditioning_mismatch)
    run_pathvqa_directional_concat_workspace_conditioning_mismatch || failures=$((failures + 1))
    ;;
  pathvqa)
    run_pathvqa || failures=$((failures + 1))
    ;;
  pathvqa_base)
    run_pathvqa_base || failures=$((failures + 1))
    ;;
  pathvqa_prompt_tuning_seed44)
    run_pathvqa_prompt_tuning_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_seed44)
    run_pathvqa_dynamic_prompt_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_directional_concat_workspace_text_dynamic_only_seed44)
    run_pathvqa_directional_concat_workspace_text_dynamic_only_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_directional_concat_workspace_text_dynamic_only_d512_seed44)
    run_pathvqa_directional_concat_workspace_text_dynamic_only_d512_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_directional_concat_workspace_text_dynamic_only_d256_seed44)
    run_pathvqa_directional_concat_workspace_text_dynamic_only_d256_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed44)
    run_pathvqa_directional_concat_workspace_text_dynamic_only_d768_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_qdpt_d768_no_static_visual_seed44)
    run_pathvqa_qdpt_d768_no_static_visual_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_qdpt_d768_no_static_visual_resume_eval)
    run_pathvqa_qdpt_d768_no_static_visual_resume_eval || failures=$((failures + 1))
    ;;
  pathvqa_qdpt_d768_learned_static_query_seed44)
    run_pathvqa_qdpt_d768_learned_static_query_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_qdpt_d768_direct_visual_z_concat_seeds44_46)
    run_pathvqa_qdpt_d768_direct_visual_z_concat_seeds44_46 || failures=$((failures + 1))
    ;;
  pathvqa_qdpt_d768_layer_sensitivity_seed44)
    run_pathvqa_qdpt_d768_layer_sensitivity_seed44 || failures=$((failures + 1))
    ;;
  electrical_qdpt_d768_seed44)
    run_electrical_qdpt_d768_seed44 || failures=$((failures + 1))
    ;;
  qdpt_d768_final_pathvqa_slake_seed44)
    run_qdpt_d768_final_pathvqa_slake_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_directional_width_ablation_seed44)
    run_pathvqa_directional_width_ablation_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_sparse_visual_single_pass_seed44)
    run_pathvqa_dynamic_prompt_sparse_visual_single_pass_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_raw_shared_s_separate_residual_seed44)
    run_pathvqa_dynamic_prompt_raw_shared_s_separate_residual_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_seed44)
    run_pathvqa_dynamic_prompt_raw_shared_s_direct_prompt_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_raw_shared_s_suite_seed44)
    run_pathvqa_dynamic_prompt_raw_shared_s_suite_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_asymmetric_shared_s_seed44)
    run_pathvqa_dynamic_prompt_asymmetric_shared_s_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_full_workspace_seed44)
    run_pathvqa_dynamic_prompt_full_workspace_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_dynamic_prompt_interventions)
    run_pathvqa_dynamic_prompt_interventions || failures=$((failures + 1))
    ;;
  pathvqa_minimal_mmrl_prompt20_seed44)
    run_pathvqa_minimal_mmrl_prompt20_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_minimal_mmrl_prompt20_resume_eval)
    run_pathvqa_minimal_mmrl_prompt20_resume_eval || failures=$((failures + 1))
    ;;
  pathvqa_lora_visual_attn_r128)
    run_pathvqa_lora_visual_attn_r128 || failures=$((failures + 1))
    ;;
  pathvqa_lora_visual_attn_r128_then_base)
    run_pathvqa_lora_visual_attn_r128_then_base || failures=$((failures + 1))
    ;;
  pathvqa_lora_visual_last8_attn_r128)
    run_pathvqa_lora_visual_last8_attn_r128 || failures=$((failures + 1))
    ;;
  pathvqa_lora_full_model_attn_r8_seed44)
    run_pathvqa_lora_full_model_attn_r8_seed44 || failures=$((failures + 1))
    ;;
  pathvqa_day2_d768_lora_r8_seeds45_46)
    run_pathvqa_day2_d768_lora_r8_seeds45_46 || failures=$((failures + 1))
    ;;
  pathvqa_last8_lora_minimal_mmrl_relation_suite)
    run_pathvqa_last8_lora_minimal_mmrl_relation_suite || failures=$((failures + 1))
    ;;
  all)
    run_train_dataset || failures=$((failures + 1))
    run_slake || failures=$((failures + 1))
    ;;
  *)
    echo "[ERR] 未知目标: $RUN_TARGET；新增 QDPT 目标: pathvqa_qdpt_d768_no_static_visual_seed44、pathvqa_qdpt_d768_no_static_visual_resume_eval、pathvqa_qdpt_d768_learned_static_query_seed44、pathvqa_qdpt_d768_direct_visual_z_concat_seeds44_46、pathvqa_qdpt_d768_layer_sensitivity_seed44、electrical_qdpt_d768_seed44、qdpt_d768_final_pathvqa_slake_seed44。" >&2
    exit 2
    ;;
esac

if [ "$failures" -ne 0 ]; then
  echo "[ERR] 已执行全部计划，失败实验数=$failures。" >&2
  exit 1
fi
echo "[DONE] 已完成实验目标: $RUN_TARGET"
