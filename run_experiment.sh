#!/bin/bash
set -uo pipefail

ROOT_DIR="${MMRL_ROOT_DIR:-/root/autodl-tmp/Qwen3-VL-modify-test}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
OUTPUT_ROOT="${MMRL_OUTPUT_ROOT:-$ROOT_DIR/experiment_outputs/output}"
SLAKE_DATA_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
SLAKE_OUTPUT_ROOT="${SLAKE_OUTPUT_ROOT:-$ROOT_DIR/slake/outputs/mmrl}"
PATHVQA_DATA_ROOT="${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}"
PATHVQA_CACHE_ROOT="${PATHVQA_CACHE_ROOT:-$PATHVQA_DATA_ROOT/.hf_cache}"
PATHVQA_OUTPUT_ROOT="${PATHVQA_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/mmrl}"
PATHVQA_LORA_OUTPUT_ROOT="${PATHVQA_LORA_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/lora}"
PATHVQA_BASE_OUTPUT_ROOT="${PATHVQA_BASE_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/base}"
PATHVQA_PROMPT_OUTPUT_ROOT="${PATHVQA_PROMPT_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/prompt_tuning}"
PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT="${PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT:-$ROOT_DIR/pathvqa/outputs/dynamic_prompt}"
ENV_RUN_TARGET="${RUN_TARGET:-}"
RUN_TARGET="${1:-${ENV_RUN_TARGET:-${MMRL_RUN_TARGET:-all}}}"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"
SEED="${MMRL_FIXED_SEED:-44}"
SHUTDOWN_ON_EXIT="${MMRL_SHUTDOWN_ON_EXIT:-1}"

mkdir -p "$OUTPUT_ROOT" "$SLAKE_OUTPUT_ROOT" "$PATHVQA_OUTPUT_ROOT" "$PATHVQA_LORA_OUTPUT_ROOT" "$PATHVQA_BASE_OUTPUT_ROOT" "$PATHVQA_PROMPT_OUTPUT_ROOT" "$PATHVQA_DYNAMIC_PROMPT_OUTPUT_ROOT"
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
  if [ ! -f "$checkpoint/adapter_config.json" ]; then
    echo "[ERR] LoRA checkpoint 缺少 adapter_config.json: $checkpoint" >&2
    return 1
  fi
  mkdir -p "$eval_output_dir"
  (
    cd "$ROOT_DIR" || exit 1
    python pathvqa/pathvqa_official_eval.py \
      --backend lora-vision \
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
    local eval_output_dir="$source_run/eval_interventions/$intervention"
    local eval_log="$source_run/eval_intervention_${intervention}.log"
    echo "[PATHVQA_DYNAMIC_PROMPT_INTERVENTION] mode=$intervention lag=$lag checkpoint=$checkpoint"
    run_pathvqa_dynamic_prompt_eval \
      "$checkpoint" \
      test \
      "$eval_output_dir" \
      "$eval_log" \
      --dynamic-prompt-intervention "$intervention" \
      --dynamic-prompt-memory-lag "$lag" || return 1
  done

  printf 'intervention\toverall\tyes_no\tfree_form\tsamples_changed\n'
  for intervention in zero mean-residual lagged-memory; do
    python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); i=d["dynamic_prompt_intervention"]; print("\t".join(map(str,(i["mode"],d["overall_accuracy"],d["yes_no_accuracy"],d["free_form_accuracy"],i["samples_changed"]))))' \
      "$source_run/eval_interventions/$intervention/pathvqa_summary.json"
  done
}

run_pathvqa_dynamic_prompt_seed44() {
  local experiment_name="pathvqa_dynamic_prompt_mean_ca256_len20"
  local run_seed=44
  local epochs="${PATHVQA_DYNAMIC_PROMPT_EPOCHS:-3}"
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

  local epoch_id
  for ((epoch_id = 1; epoch_id <= epochs; epoch_id++)); do
    local checkpoint="$output_dir/checkpoints/epoch_${epoch_id}"
    echo "[PATHVQA_DYNAMIC_PROMPT_VALIDATION] epoch=$epoch_id checkpoint=$checkpoint"
    run_pathvqa_dynamic_prompt_eval \
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
  echo "[PATHVQA_DYNAMIC_PROMPT_TEST] best_epoch=$best_epoch validation=$best_validation_score"
  run_pathvqa_dynamic_prompt_eval \
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
  pathvqa_last8_lora_minimal_mmrl_relation_suite)
    run_pathvqa_last8_lora_minimal_mmrl_relation_suite || failures=$((failures + 1))
    ;;
  all)
    run_train_dataset || failures=$((failures + 1))
    run_slake || failures=$((failures + 1))
    ;;
  *)
    echo "[ERR] 未知目标: $RUN_TARGET，可选 train、slake、pathvqa、pathvqa_base、pathvqa_prompt_tuning_seed44、pathvqa_dynamic_prompt_seed44、pathvqa_dynamic_prompt_interventions、pathvqa_minimal_mmrl_prompt20_seed44、pathvqa_lora_visual_attn_r128、pathvqa_lora_visual_attn_r128_then_base、pathvqa_lora_visual_last8_attn_r128、pathvqa_last8_lora_minimal_mmrl_relation_suite、slake_final_seeds4、slake_ablation_no_relation_seeds2、slake_ablation_independent_init_seeds2、slake_ablation_static_query_seed45、slake_ablation_mean_pooling_seed45、slake_mean_final_completion_suite、slake_text_guided_balanced_fusion_slots8_seed45、slake_prompt_tuning_seed44、slake_concat_mlp_fusion_seed45、slake_overnight_prompt_and_concat_mlp、slake_ablation_suite、all。" >&2
    exit 2
    ;;
esac

if [ "$failures" -ne 0 ]; then
  echo "[ERR] 已执行全部计划，失败实验数=$failures。" >&2
  exit 1
fi
echo "[DONE] 已完成实验目标: $RUN_TARGET"
