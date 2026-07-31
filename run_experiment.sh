#!/bin/bash
set -euo pipefail

SHUTDOWN_ON_EXIT="${MMRL_SHUTDOWN_ON_EXIT:-1}"

cancel_shutdown_on_interrupt() {
  SHUTDOWN_ON_EXIT=0
  trap - EXIT
  echo "[INT] 检测到 Ctrl+C，已取消自动关机。"
  exit 130
}

shutdown_on_exit() {
  local exit_code=$?
  if [ "${SHUTDOWN_ON_EXIT:-1}" != "1" ]; then
    return "$exit_code"
  fi
  echo "[EXIT] 脚本退出，exit_code=$exit_code"
  echo "[EXIT] 600 秒后自动关机。"
  echo "[EXIT] 如需取消自动关机，请在倒计时内按 Ctrl+C。"
  sleep 600
  /usr/bin/shutdown
}
trap shutdown_on_exit EXIT
trap cancel_shutdown_on_interrupt INT

ROOT_DIR="/root/autodl-tmp/Qwen3-VL-modify-test"
TRAIN_DIR="$ROOT_DIR/train"
OUTPUT_ROOT="$ROOT_DIR/experiment_outputs"
CHECKPOINT_ROOT="$OUTPUT_ROOT/output"
SLAKE_OUTPUT_ROOT="${SLAKE_OUTPUT_ROOT:-$ROOT_DIR/slake/outputs}"
SEED_STATE_FILE="${MMRL_SEED_STATE_FILE:-$OUTPUT_ROOT/next_seed.txt}"
INITIAL_SEED="${MMRL_INITIAL_SEED:-42}"
AUTO_INCREMENT_SEED="${MMRL_AUTO_INCREMENT_SEED:-0}"
FIXED_SEED="${MMRL_FIXED_SEED:-44}"
RUN_TARGET="${1:-${MMRL_RUN_TARGET:-all}}"

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT" "$SLAKE_OUTPUT_ROOT"

RUN_SUFFIX="${MMRL_RUN_SUFFIX:-}"
RUN_DATE="${MMRL_RUN_DATE:-$(date +%Y%m%d)}"

# 持久分配实验 seed；脚本重启后继续递增，不重复使用已分配 seed。
allocate_seed() {
  local seed
  if [ "$AUTO_INCREMENT_SEED" = "0" ]; then
    if ! [[ "$FIXED_SEED" =~ ^[0-9]+$ ]]; then
      echo "[ERR] 非法固定 seed: '$FIXED_SEED'" >&2
      return 1
    fi
    printf '%s\n' "$FIXED_SEED"
    return 0
  fi
  if [ "$AUTO_INCREMENT_SEED" != "1" ]; then
    echo "[ERR] MMRL_AUTO_INCREMENT_SEED 必须为 0 或 1，当前为 '$AUTO_INCREMENT_SEED'" >&2
    return 1
  fi

  if [ -f "$SEED_STATE_FILE" ]; then
    read -r seed < "$SEED_STATE_FILE"
  else
    seed="$INITIAL_SEED"
  fi

  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "[ERR] 非法 seed 状态: $SEED_STATE_FILE 内容为 '$seed'" >&2
    return 1
  fi

  printf '%s\n' "$((seed + 1))" > "${SEED_STATE_FILE}.tmp"
  mv "${SEED_STATE_FILE}.tmp" "$SEED_STATE_FILE"
  printf '%s\n' "$seed"
}

with_run_suffix() {
  local base_tag="$1"
  if [ -n "$RUN_SUFFIX" ]; then
    echo "${base_tag}_${RUN_SUFFIX}_${RUN_DATE}"
  else
    echo "${base_tag}_${RUN_DATE}"
  fi
}

# 若目录已存在，自动追加 _1, _2, ... 直到找到可用名称
find_available_tag() {
  local base_tag="$1"
  local candidate="$base_tag"
  local i=1
  while [ -d "$CHECKPOINT_ROOT/$candidate" ]; do
    candidate="${base_tag}_${i}"
    i=$((i + 1))
  done
  echo "$candidate"
}

# 只保留当前仍有模型的并列最高分和并列最低分 checkpoint。
# 第一次运行时唯一 checkpoint 同时是最高和最低，因此必定保留。
# 无有效分数的 final 不会出现在计划中，也不会被误删。
prune_middle_final_dirs() {
  local retention_plan
  if ! retention_plan="$(cd "$ROOT_DIR" && python get_score.py --checkpoint-retention-plan)"; then
    echo "[WARN] 获取 checkpoint 保留计划失败，本轮不删除任何 final 目录。"
    return 0
  fi
  if [ -z "$retention_plan" ]; then
    echo "[WARN] 没有找到可比较的有效分数，本轮不删除任何 final 目录。"
    return 0
  fi

  local action tag score final_dir
  while IFS=$'\t' read -r action tag score; do
    [ -n "$action" ] || continue
    final_dir="$CHECKPOINT_ROOT/$tag/final"
    case "$action" in
      KEEP)
        echo "[KEEP] 保留极值 checkpoint: $tag score=$score"
        ;;
      PRUNE)
        if [ -d "$final_dir" ]; then
          echo "[PRUNE] 删除中间分 checkpoint: $tag score=$score"
          rm -rf -- "$final_dir"
        fi
        ;;
      *)
        echo "[WARN] 未知保留动作: $action tag=$tag，本条跳过。"
        ;;
    esac
  done <<< "$retention_plan"
}

run_one() {
  local experiment_name="$1"
  local raw_tag="$2"
  local decouple_stage_pooling="${MMRL_DECOUPLE_STAGE_POOLING:-0}"
  local intermediate_eval_steps="${MMRL_INTERMEDIATE_EVAL_STEPS:-}"
  if [ "$decouple_stage_pooling" != "0" ] && [ "$decouple_stage_pooling" != "1" ]; then
    echo "[ERR] MMRL_DECOUPLE_STAGE_POOLING 必须为 0 或 1，当前为 '$decouple_stage_pooling'" >&2
    return 1
  fi
  local base_tag
  base_tag="$(with_run_suffix "$raw_tag")"
  local tag
  tag="$(find_available_tag "$base_tag")"
  local output_dir="$CHECKPOINT_ROOT/$tag"
  local eval_dir="$output_dir/eval"
  local experiment_seed
  experiment_seed="$(allocate_seed)"

  echo "============================================================"
  echo "[EXP] 开始实验: $tag"
  echo "[EXP] MMRL_EXPERIMENT=$experiment_name"
  echo "[EXP] seed=$experiment_seed"
  if [ "$AUTO_INCREMENT_SEED" = "1" ]; then
    echo "[EXP] seed_mode=auto_increment"
  else
    echo "[EXP] seed_mode=fixed"
  fi
  echo "[EXP] data_sampling_seed=42"
  echo "[EXP] data_order_seed=42"
  echo "[EXP] decouple_stage_pooling=$decouple_stage_pooling"
  echo "[EXP] intermediate_eval_steps=${intermediate_eval_steps:-disabled}"
  echo "[EXP] checkpoint目录: $output_dir"
  echo "============================================================"

  mkdir -p "$output_dir" "$eval_dir"

  if ! (
    cd "$TRAIN_DIR"
    MMRL_OUTPUT_DIR="$output_dir" \
    MMRL_EXPERIMENT="$experiment_name" \
    MMRL_SEED="$experiment_seed" \
    MMRL_DATA_SAMPLING_SEED="42" \
    MMRL_DATA_ORDER_SEED="42" \
    MMRL_DETERMINISTIC_SAMPLING="1" \
    MMRL_DECOUPLE_STAGE_POOLING="$decouple_stage_pooling" \
    MMRL_INTERMEDIATE_EVAL_STEPS="$intermediate_eval_steps" \
    MMRL_EVAL_EACH_EPOCH="0" \
    MMRL_LIVE_FINAL_EVAL="1" \
    MMRL_SAVE_EXTREMA_CHECKPOINTS="1" \
    python train.py 2>&1 | tee "$output_dir/train.log"
  ); then
    echo "[ERR] 实验训练或在线测评失败: $tag" >&2
    return 1
  fi

  if [ ! -f "$eval_dir/test.log" ]; then
    echo "[ERR] 训练完成后未找到在线测评日志: $eval_dir/test.log"
    return 1
  fi

  # 在线测评后只保留全局并列最高/最低，清理已经失去极值资格的 final。
  # prune_middle_final_dirs
}

run_slake_full_all() {
  local experiment_name="${1:-slake_mmrl_full_all_seed44}"
  local relation_weight="${2:-0.010}"
  local effective_delta_weight="${3:-0.0003}"
  local effective_delta_floor="${4:-0.52}"
  local run_target_label="${5:-overnight_slake_pooling_pair_seed44}"
  local data_root="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
  local model_path="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
  local slake_output_root="$SLAKE_OUTPUT_ROOT/mmrl"
  local base_tag
  base_tag="$(with_run_suffix "$experiment_name")"
  local tag="$base_tag"
  local i=1
  while [ -d "$slake_output_root/$tag" ]; do
    tag="${base_tag}_${i}"
    i=$((i + 1))
  done
  local output_dir="$slake_output_root/$tag"
  local eval_dir="$output_dir/eval"
  local status_file="$output_dir/run_status.txt"
  local latest_pointer="$SLAKE_OUTPUT_ROOT/last_overnight_run.txt"

  mkdir -p "$output_dir" "$eval_dir"
  {
    echo "status=STARTED"
    echo "target=$run_target_label"
    echo "experiment=$experiment_name"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "$status_file"
  {
    echo "target=$run_target_label"
    echo "experiment=$experiment_name"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "status_file=$status_file"
    echo "train_log=$output_dir/train.log"
    echo "eval_log=$output_dir/eval.log"
  } > "$latest_pointer"
  echo "============================================================"
  echo "[SLAKE] 开始全量训练与官方测评: $tag"
  echo "[SLAKE] data_root=$data_root"
  echo "[SLAKE] language=all"
  echo "[SLAKE] seed=44 data_seed=42"
  echo "[SLAKE] decouple_stage_pooling=0"
  echo "[SLAKE] relation_weight=$relation_weight effective_delta_weight=$effective_delta_weight effective_delta_floor=$effective_delta_floor"
  echo "[SLAKE] output_dir=$output_dir"
  echo "============================================================"

  if ! (
    cd "$ROOT_DIR"
    python slake/train_mmrl.py \
      --data-root "$data_root" \
      --model-path "$model_path" \
      --output-dir "$output_dir" \
      --experiment-name "$experiment_name" \
      --language all \
      --seed 44 \
      --data-seed 42 \
      --pooling-lr 6e-5 \
      --mmrl-lr 6e-5 \
      --router-lr 8e-5 \
      --adapter-lrs 4e-5 6e-5 8e-5 1e-4 \
      --usage-weight 0.0026 \
      --entropy-weight 0.020 \
      --entropy-target 0.72 \
      --effective-delta-weight "$effective_delta_weight" \
      --effective-delta-target-low "$effective_delta_floor" \
      --effective-delta-target-high 0.98 \
      --relation-weight "$relation_weight" \
      --scheduler constant_with_warmup \
      --warmup-ratio 0.10 \
      2>&1 | tee "$output_dir/train.log"
  ); then
    {
      echo "status=TRAIN_FAILED"
      echo "target=$run_target_label"
      echo "experiment=$experiment_name"
      echo "tag=$tag"
      echo "output_dir=$output_dir"
      echo "failed_at=$(date --iso-8601=seconds)"
    } > "$status_file"
    echo "[ERR] SLAKE 全量训练失败: $tag" >&2
    return 1
  fi

  if [ ! -d "$output_dir/final" ]; then
    {
      echo "status=CHECKPOINT_MISSING"
      echo "target=$run_target_label"
      echo "experiment=$experiment_name"
      echo "tag=$tag"
      echo "output_dir=$output_dir"
      echo "failed_at=$(date --iso-8601=seconds)"
    } > "$status_file"
    echo "[ERR] SLAKE 训练完成后未找到 checkpoint: $output_dir/final" >&2
    return 1
  fi

  {
    echo "status=EVALUATING"
    echo "target=$run_target_label"
    echo "experiment=$experiment_name"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "evaluation_started_at=$(date --iso-8601=seconds)"
  } > "$status_file"

  if ! (
    cd "$ROOT_DIR"
    python slake/slake_official_eval.py \
      --backend mmrl \
      --base-model "$model_path" \
      --checkpoint "$output_dir/final" \
      --questions "$data_root/test.json" \
      --image-root "$data_root/imgs" \
      --output-dir "$eval_dir" \
      --language all \
      --overwrite \
      2>&1 | tee "$output_dir/eval.log"
  ); then
    {
      echo "status=EVAL_FAILED"
      echo "target=$run_target_label"
      echo "experiment=$experiment_name"
      echo "tag=$tag"
      echo "output_dir=$output_dir"
      echo "failed_at=$(date --iso-8601=seconds)"
    } > "$status_file"
    echo "[ERR] SLAKE 官方测评失败: $tag" >&2
    return 1
  fi

  {
    echo "status=COMPLETED"
    echo "target=$run_target_label"
    echo "experiment=$experiment_name"
    echo "tag=$tag"
    echo "output_dir=$output_dir"
    echo "completed_at=$(date --iso-8601=seconds)"
  } > "$status_file"
  echo "[SLAKE] 全量训练与官方测评完成: $output_dir"
}

run_slake_r025_delta_pair() {
  local failures=0
  if ! run_slake_full_all \
    "slake_mmrl_r025_d058_seed44" \
    "0.025" \
    "0.0004" \
    "0.58" \
    "slake_r025_delta_pair_seed44"; then
    echo "[SLAKE-PAIR-WARN] r025/d058 失败，继续运行 r025/d070。" >&2
    failures=$((failures + 1))
  fi
  if ! run_slake_full_all \
    "slake_mmrl_r025_d070_seed44" \
    "0.025" \
    "0.0004" \
    "0.70" \
    "slake_r025_delta_pair_seed44"; then
    echo "[SLAKE-PAIR-WARN] r025/d070 失败。" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -ne 0 ]; then
    echo "[SLAKE-PAIR-ERR] 共 $failures 个 SLAKE 实验失败。" >&2
    return 1
  fi
}

run_slake_constraint_matrix_3x2_part() {
  local part="$1"
  local relation_tag
  local relation_weight
  local failures=0

  case "$part" in
    1)
      relation_tag="r0250"
      relation_weight="0.0250"
      ;;
    2)
      relation_tag="r0375"
      relation_weight="0.0375"
      ;;
    3)
      relation_tag="r0500"
      relation_weight="0.0500"
      ;;
    *)
      echo "[SLAKE-MATRIX-ERR] 未知分组: $part" >&2
      return 2
      ;;
  esac

  if ! run_slake_full_all \
    "slake_mmrl_constraint_matrix_${relation_tag}_w0004_d058_seed44" \
    "$relation_weight" \
    "0.0004" \
    "0.58" \
    "slake_constraint_matrix_3x2_part${part}"; then
    echo "[SLAKE-MATRIX-PART${part}-WARN] ${relation_tag}/w0004 失败，继续 w0008。" >&2
    failures=$((failures + 1))
  fi

  if ! run_slake_full_all \
    "slake_mmrl_constraint_matrix_${relation_tag}_w0008_d058_seed44" \
    "$relation_weight" \
    "0.0008" \
    "0.58" \
    "slake_constraint_matrix_3x2_part${part}"; then
    echo "[SLAKE-MATRIX-PART${part}-WARN] ${relation_tag}/w0008 失败。" >&2
    failures=$((failures + 1))
  fi

  echo "[SLAKE-MATRIX-PART${part}-SUMMARY] 两轮均已尝试，失败数=$failures。"
  if [ "$failures" -ne 0 ]; then
    return 1
  fi
}

# 重复跑 N 次同一实验；目录命名由 find_available_tag 自动处理，不会覆写
# 用法: run_N <experiment_name> <raw_tag> <N>
run_N() {
  local experiment_name="$1"
  local raw_tag="$2"
  local n="$3"
  local i
  for ((i = 1; i <= n; i++)); do
    run_one "$experiment_name" "$raw_tag"
  done
}

# 依次运行显式给定的固定 seed；不读取或修改自动递增状态文件。
run_fixed_seed_sequence() {
  local experiment_name="$1"
  local raw_tag_prefix="$2"
  shift 2
  local seed
  AUTO_INCREMENT_SEED=0
  for seed in "$@"; do
    FIXED_SEED="$seed"
    run_one "$experiment_name" "${raw_tag_prefix}_seed${seed}"
  done
}

run_loss_matrix_pair() {
  local relation_tag="$1"
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  if ! run_one \
    "visual_router_loss_matrix_${relation_tag}_d058_v1" \
    "visual_router_loss_matrix_${relation_tag}_d058_seed44"; then
    echo "[MATRIX-WARN] ${relation_tag}/d058 失败，继续运行 d070。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_loss_matrix_${relation_tag}_d070_v1" \
    "visual_router_loss_matrix_${relation_tag}_d070_seed44"; then
    echo "[MATRIX-WARN] ${relation_tag}/d070 失败。" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -ne 0 ]; then
    echo "[MATRIX-ERR] ${relation_tag} 共 $failures 个实验失败。" >&2
    return 1
  fi
}

run_loss_tuning_3x1() {
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  if ! run_one \
    "visual_router_loss_tuning_no_effective_delta_v1" \
    "visual_router_loss_tuning_no_effective_delta_seed44"; then
    echo "[LOSS-TUNING-WARN] no-effective-delta 失败，继续下一组。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_loss_tuning_half_usage_v1" \
    "visual_router_loss_tuning_half_usage_seed44"; then
    echo "[LOSS-TUNING-WARN] half-usage 失败，继续下一组。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_loss_tuning_relation_r0100_v1" \
    "visual_router_loss_tuning_relation_r0100_seed44"; then
    echo "[LOSS-TUNING-WARN] relation-r0100 失败。" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -ne 0 ]; then
    echo "[LOSS-TUNING-ERR] 3x1 共 $failures 个实验失败。" >&2
    return 1
  fi
}

run_final_three_experiments() {
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  MMRL_DECOUPLE_STAGE_POOLING=1
  if ! run_one \
    "visual_router_loss_matrix_r0250_d058_v1" \
    "visual_router_final_r025_d058_decoupled_pooling_seed44"; then
    echo "[FINAL-WARN] 解耦 pooling 实验失败，继续 MMRL LR 实验。" >&2
    failures=$((failures + 1))
  fi

  MMRL_DECOUPLE_STAGE_POOLING=0
  if ! run_one \
    "visual_router_final_mmrl_lr8e5_v1" \
    "visual_router_final_r025_d058_mmrl_lr8e5_seed44"; then
    echo "[FINAL-WARN] MMRL LR=8e-5 实验失败，继续 entropy 实验。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_final_entropy_target080_v1" \
    "visual_router_final_r025_d058_entropy_target080_seed44"; then
    echo "[FINAL-WARN] entropy target=0.80 实验失败。" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -ne 0 ]; then
    echo "[FINAL-ERR] 三轮收尾实验共 $failures 个失败。" >&2
    return 1
  fi
}

run_custom_r025_d058_multiseed() {
  local failures=0
  local seed
  AUTO_INCREMENT_SEED=0
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  for seed in 45 46 47; do
    FIXED_SEED="$seed"
    if ! run_one \
      "visual_router_loss_matrix_r0250_d058_v1" \
      "visual_router_custom_r025_d058_seed${seed}"; then
      echo "[CUSTOM-MULTISEED-WARN] seed=$seed 失败，继续下一 seed。" >&2
      failures=$((failures + 1))
    fi
  done

  if [ "$failures" -ne 0 ]; then
    echo "[CUSTOM-MULTISEED-ERR] 三个 seed 中有 $failures 个失败。" >&2
    return 1
  fi
}

run_custom_optimizer_adapter_sweep() {
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  if ! run_one \
    "visual_router_custom_optimizer_beta2_098_v1" \
    "visual_router_custom_r025_d058_adam_beta2_098_seed44"; then
    echo "[CUSTOM-SWEEP-WARN] Adam beta2=0.98 失败，继续 adapter LR 压缩实验。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_custom_adapter_lr_compressed_v1" \
    "visual_router_custom_r025_d058_adapter_lr_compressed_seed44"; then
    echo "[CUSTOM-SWEEP-WARN] adapter LR 压缩实验失败，继续扩张实验。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_custom_adapter_lr_expanded_v1" \
    "visual_router_custom_r025_d058_adapter_lr_expanded_seed44"; then
    echo "[CUSTOM-SWEEP-WARN] adapter LR 扩张实验失败。" >&2
    failures=$((failures + 1))
  fi

  if [ "$failures" -ne 0 ]; then
    echo "[CUSTOM-SWEEP-ERR] 三项实验中有 $failures 个失败。" >&2
    return 1
  fi
}

run_paper_ablation_seed44() {
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  if ! run_one \
    "visual_router_ablation_full_r0125_d070_v1" \
    "visual_router_ablation_full_r0125_d070_seed44"; then
    echo "[ABLATION-WARN] Full FROST-VL 复现失败，继续 w/o Rep-Token Branch。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_ablation_wo_rep_token_v1" \
    "visual_router_ablation_wo_rep_token_seed44"; then
    echo "[ABLATION-WARN] w/o Rep-Token Branch 失败，继续 Single Adapter。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_ablation_single_adapter_v1" \
    "visual_router_ablation_single_adapter_seed44"; then
    echo "[ABLATION-WARN] Single Adapter 失败，继续 Homogeneous Adapter LR。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_ablation_homogeneous_adapter_lr_v1" \
    "visual_router_ablation_homogeneous_adapter_lr_seed44"; then
    echo "[ABLATION-WARN] Homogeneous Adapter LR 失败，继续 w/o Relation Loss。" >&2
    failures=$((failures + 1))
  fi
  if ! run_one \
    "visual_router_ablation_no_relation_v1" \
    "visual_router_ablation_no_relation_seed44"; then
    echo "[ABLATION-WARN] w/o Relation Loss 失败。" >&2
    failures=$((failures + 1))
  fi

  echo "[ABLATION-SUMMARY] 五轮均已尝试，失败数=$failures。"
  if [ "$failures" -ne 0 ]; then
    return 1
  fi
}

run_paper_ablation_part_seed44() {
  local part="$1"
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  case "$part" in
    1-fast)
      if ! run_one \
        "visual_router_ablation_wo_rep_token_v1" \
        "visual_router_ablation_wo_rep_token_seed44"; then
        echo "[ABLATION-PART1-FAST-WARN] w/o Rep-Token Branch 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    1)
      if ! run_one \
        "visual_router_ablation_full_r0125_d070_v1" \
        "visual_router_ablation_full_r0125_d070_seed44"; then
        echo "[ABLATION-PART1-WARN] Full FROST-VL 复现失败，继续 w/o Rep-Token Branch。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_ablation_wo_rep_token_v1" \
        "visual_router_ablation_wo_rep_token_seed44"; then
        echo "[ABLATION-PART1-WARN] w/o Rep-Token Branch 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    2)
      if ! run_one \
        "visual_router_ablation_single_adapter_v1" \
        "visual_router_ablation_single_adapter_seed44"; then
        echo "[ABLATION-PART2-WARN] Single Adapter 失败，继续 Homogeneous Adapter LR。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_ablation_homogeneous_adapter_lr_v1" \
        "visual_router_ablation_homogeneous_adapter_lr_seed44"; then
        echo "[ABLATION-PART2-WARN] Homogeneous Adapter LR 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    3)
      if ! run_one \
        "visual_router_ablation_no_relation_v1" \
        "visual_router_ablation_no_relation_seed44"; then
        echo "[ABLATION-PART3-WARN] w/o Relation Loss 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    *)
      echo "[ABLATION-PART-ERR] 未知分组: $part" >&2
      return 2
      ;;
  esac

  echo "[ABLATION-PART${part}-SUMMARY] 本组全部实验均已尝试，失败数=$failures。"
  if [ "$failures" -ne 0 ]; then
    return 1
  fi
}

run_paper_ablation_222_part_seed44() {
  local part="$1"
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  case "$part" in
    1)
      if ! run_one \
        "visual_router_ablation_wo_rep_token_v1" \
        "visual_router_ablation_wo_rep_token_r0125_d070_seed44"; then
        echo "[ABLATION-222-PART1-WARN] w/o Rep-Token Branch 失败，继续 Single Adapter。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_ablation_single_adapter_v1" \
        "visual_router_ablation_single_adapter_r0125_d070_seed44"; then
        echo "[ABLATION-222-PART1-WARN] Single Adapter 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    2)
      if ! run_one \
        "visual_router_ablation_homogeneous_adapter_lr_v1" \
        "visual_router_ablation_homogeneous_adapter_lr_r0125_d070_seed44"; then
        echo "[ABLATION-222-PART2-WARN] Homogeneous Adapter LR 失败，继续 w/o Relation Loss。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_ablation_no_relation_v1" \
        "visual_router_ablation_no_relation_r0125_d070_seed44"; then
        echo "[ABLATION-222-PART2-WARN] w/o Relation Loss 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    3)
      if ! run_one \
        "visual_router_ablation_full_r0125_d070_v1" \
        "visual_router_ablation_full_r0125_d070_seed44_repro"; then
        echo "[ABLATION-222-PART3-WARN] 自有数据集63分配置复现失败，继续SLAKE 65分配置复现。" >&2
        failures=$((failures + 1))
      fi
      if ! run_slake_full_all \
        "slake_mmrl_r025_d058_seed44_paper_repro" \
        "0.025" \
        "0.0004" \
        "0.58" \
        "paper_ablation_222_part3_seed44"; then
        echo "[ABLATION-222-PART3-WARN] SLAKE r025/d058复现失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    *)
      echo "[ABLATION-222-ERR] 未知分组: $part" >&2
      return 2
      ;;
  esac

  echo "[ABLATION-222-PART${part}-SUMMARY] 本组两轮均已尝试，失败数=$failures。"
  if [ "$failures" -ne 0 ]; then
    return 1
  fi
}

run_stability_r0125_d058_cross_part() {
  local part="$1"
  local failures=0
  AUTO_INCREMENT_SEED=0
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  case "$part" in
    1)
      FIXED_SEED=44
      if ! run_one \
        "visual_router_stability_r0125_d058_heterogeneous_v1" \
        "visual_router_stability_r0125_d058_heterogeneous_seed44_repeat_a"; then
        echo "[STABILITY-PART1-WARN] 异构 LR seed44-A 失败，继续同质 LR seed44-B。" >&2
        failures=$((failures + 1))
      fi
      FIXED_SEED=44
      if ! run_one \
        "visual_router_stability_r0125_d058_homogeneous_v1" \
        "visual_router_stability_r0125_d058_homogeneous_seed44_repeat_b"; then
        echo "[STABILITY-PART1-WARN] 同质 LR seed44-B 失败，继续异构 LR seed45。" >&2
        failures=$((failures + 1))
      fi
      FIXED_SEED=45
      if ! run_one \
        "visual_router_stability_r0125_d058_heterogeneous_v1" \
        "visual_router_stability_r0125_d058_heterogeneous_seed45"; then
        echo "[STABILITY-PART1-WARN] 异构 LR seed45 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    2)
      FIXED_SEED=44
      if ! run_one \
        "visual_router_stability_r0125_d058_homogeneous_v1" \
        "visual_router_stability_r0125_d058_homogeneous_seed44_repeat_a"; then
        echo "[STABILITY-PART2-WARN] 同质 LR seed44-A 失败，继续异构 LR seed44-B。" >&2
        failures=$((failures + 1))
      fi
      FIXED_SEED=44
      if ! run_one \
        "visual_router_stability_r0125_d058_heterogeneous_v1" \
        "visual_router_stability_r0125_d058_heterogeneous_seed44_repeat_b"; then
        echo "[STABILITY-PART2-WARN] 异构 LR seed44-B 失败，继续同质 LR seed45。" >&2
        failures=$((failures + 1))
      fi
      FIXED_SEED=45
      if ! run_one \
        "visual_router_stability_r0125_d058_homogeneous_v1" \
        "visual_router_stability_r0125_d058_homogeneous_seed45"; then
        echo "[STABILITY-PART2-WARN] 同质 LR seed45 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    *)
      echo "[STABILITY-ERR] 未知分组: $part" >&2
      return 2
      ;;
  esac

  echo "[STABILITY-PART${part}-SUMMARY] 三轮均已尝试，失败数=$failures。"
  if [ "$failures" -ne 0 ]; then
    return 1
  fi
}

run_final_constraint_matrix_3x3_part() {
  local part="$1"
  local failures=0
  AUTO_INCREMENT_SEED=0
  FIXED_SEED=44
  MMRL_DECOUPLE_STAGE_POOLING=0
  MMRL_INTERMEDIATE_EVAL_STEPS=""

  case "$part" in
    1)
      if ! run_one \
        "visual_router_final_constraint_matrix_r0125_w0004_d070_v1" \
        "visual_router_final_constraint_matrix_r0125_w0004_d070_seed44"; then
        echo "[FINAL-MATRIX-PART1-WARN] r0125/w0004 失败，继续 r0250/w0008。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_final_constraint_matrix_r0250_w0008_d070_v1" \
        "visual_router_final_constraint_matrix_r0250_w0008_d070_seed44"; then
        echo "[FINAL-MATRIX-PART1-WARN] r0250/w0008 失败，继续 r0375/w0016。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_final_constraint_matrix_r0375_w0016_d070_v1" \
        "visual_router_final_constraint_matrix_r0375_w0016_d070_seed44"; then
        echo "[FINAL-MATRIX-PART1-WARN] r0375/w0016 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    2)
      if ! run_one \
        "visual_router_final_constraint_matrix_r0250_w0004_d070_v1" \
        "visual_router_final_constraint_matrix_r0250_w0004_d070_seed44"; then
        echo "[FINAL-MATRIX-PART2-WARN] r0250/w0004 失败，继续 r0375/w0008。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_final_constraint_matrix_r0375_w0008_d070_v1" \
        "visual_router_final_constraint_matrix_r0375_w0008_d070_seed44"; then
        echo "[FINAL-MATRIX-PART2-WARN] r0375/w0008 失败，继续 r0125/w0016。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_final_constraint_matrix_r0125_w0016_d070_v1" \
        "visual_router_final_constraint_matrix_r0125_w0016_d070_seed44"; then
        echo "[FINAL-MATRIX-PART2-WARN] r0125/w0016 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    3)
      if ! run_one \
        "visual_router_final_constraint_matrix_r0375_w0004_d070_v1" \
        "visual_router_final_constraint_matrix_r0375_w0004_d070_seed44"; then
        echo "[FINAL-MATRIX-PART3-WARN] r0375/w0004 失败，继续 r0125/w0008。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_final_constraint_matrix_r0125_w0008_d070_v1" \
        "visual_router_final_constraint_matrix_r0125_w0008_d070_seed44"; then
        echo "[FINAL-MATRIX-PART3-WARN] r0125/w0008 失败，继续 r0250/w0016。" >&2
        failures=$((failures + 1))
      fi
      if ! run_one \
        "visual_router_final_constraint_matrix_r0250_w0016_d070_v1" \
        "visual_router_final_constraint_matrix_r0250_w0016_d070_seed44"; then
        echo "[FINAL-MATRIX-PART3-WARN] r0250/w0016 失败。" >&2
        failures=$((failures + 1))
      fi
      ;;
    *)
      echo "[FINAL-MATRIX-ERR] 未知分组: $part" >&2
      return 2
      ;;
  esac

  echo "[FINAL-MATRIX-PART${part}-SUMMARY] 三轮均已尝试，失败数=$failures。"
  if [ "$failures" -ne 0 ]; then
    return 1
  fi
}

# 常用实验入口。
#   bash run_experiment.sh relation44
#   bash run_experiment.sh relation47
#   bash run_experiment.sh heterogeneous_lr_pair
#   bash run_experiment.sh heterogeneous_relation_100_102
#   bash run_experiment.sh multiturn_relation_100_102
#   bash run_experiment.sh joint_cosine_seed100
#   bash run_experiment.sh legacy_3cdf58d_seed100
#   bash run_experiment.sh fixed_stage1_constant_seed100
#   bash run_experiment.sh fixed_stage1_lr5e5_seed100
#   bash run_experiment.sh fixed_stage1_pooling_lr1e5_seed44
#   bash run_experiment.sh fixed_stage1_global_lr_half_seed44
#   bash run_experiment.sh fixed_stage1_mmrl_only_lr3e5_seed44
#   bash run_experiment.sh stage3_control_seed44
#   bash run_experiment.sh stage3_late_decay_seed44
#   bash run_experiment.sh stage3_late_decay_balanced_loss_seed44
#   bash run_experiment.sh loss_matrix_r0125_seed44
#   bash run_experiment.sh loss_matrix_r0250_seed44
#   bash run_experiment.sh loss_matrix_r0500_seed44
#   bash run_experiment.sh loss_tuning_no_effective_seed44
#   bash run_experiment.sh loss_tuning_half_usage_seed44
#   bash run_experiment.sh loss_tuning_relation_r0100_seed44
#   bash run_experiment.sh loss_tuning_3x1_seed44
#   bash run_experiment.sh slake_r025_delta_pair_seed44
#   bash run_experiment.sh slake_constraint_matrix_3x2_part1
#   bash run_experiment.sh slake_constraint_matrix_3x2_part2
#   bash run_experiment.sh slake_constraint_matrix_3x2_part3
#   bash run_experiment.sh final_three_experiments_seed44
#   bash run_experiment.sh custom_r025_d058_multiseed_45_47
#   bash run_experiment.sh custom_optimizer_adapter_sweep_seed44
#   bash run_experiment.sh paper_ablation_seed44
#   bash run_experiment.sh paper_ablation_seed44_part1
#   bash run_experiment.sh paper_ablation_seed44_wo_rep_only
#   bash run_experiment.sh paper_ablation_seed44_part2
#   bash run_experiment.sh paper_ablation_seed44_part3
#   bash run_experiment.sh paper_ablation_222_seed44_part1
#   bash run_experiment.sh paper_ablation_222_seed44_part2
#   bash run_experiment.sh paper_ablation_222_seed44_part3
#   bash run_experiment.sh stability_r0125_d058_cross_part1
#   bash run_experiment.sh stability_r0125_d058_cross_part2
#   bash run_experiment.sh final_constraint_matrix_3x3_part1
#   bash run_experiment.sh final_constraint_matrix_3x3_part2
#   bash run_experiment.sh final_constraint_matrix_3x3_part3
#   bash run_experiment.sh overnight_slake_pooling_pair_seed44
#   bash run_experiment.sh joint_cosine_1_4
#   bash run_experiment.sh joint_cosine_44_46
#   bash run_experiment.sh spatial_grounding_seed44
#   bash run_experiment.sh ce_only_seed44
#   bash run_experiment.sh usage_only_seed44
#   bash run_experiment.sh overnight_route_loss_3plus1
#   bash run_experiment.sh relation_2p5e3_repeat2
#   bash run_experiment.sh relation_5e3_repeat2
#   bash run_experiment.sh tune_mmrl_lr4e5_seed44
#   bash run_experiment.sh tune_mmrl_lr3e5_seed44
#   bash run_experiment.sh heterogeneous_no_relation_100_102
#   bash run_experiment.sh raw_adapter47
# 当前激进消融默认固定 seed 44：
#   bash run_experiment.sh direct_mmrl
#   bash run_experiment.sh two_adapter
#   bash run_experiment.sh single_adapter
case "$RUN_TARGET" in
  relation44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one "visual_router_relation_alignment_diag" "visual_router_relation_retest_seed44"
    ;;
  relation47)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=47
    run_one "visual_router_relation_alignment_diag" "visual_router_relation_retest_seed47"
    ;;
  heterogeneous_lr_pair)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one "visual_router_relation_heterogeneous_adapter_lr_v1" "visual_router_relation_heterogeneous_adapter_lr_v1_seed44"
    FIXED_SEED=48
    run_one "visual_router_relation_heterogeneous_adapter_lr_v1" "visual_router_relation_heterogeneous_adapter_lr_v1_seed48"
    ;;
  heterogeneous_relation_100_102)
    run_fixed_seed_sequence \
      "visual_router_relation_heterogeneous_adapter_lr_v1" \
      "visual_router_relation_heterogeneous_adapter_lr_v1_paired" \
      100 101 102
    ;;
  multiturn_relation_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_relation_multiturn_fixed_v1" \
      "visual_router_relation_multiturn_fixed_v1_seed100"
    ;;
  multiturn_relation_seed101)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=101
    run_one \
      "visual_router_relation_multiturn_fixed_v1" \
      "visual_router_relation_multiturn_fixed_v1_seed101"
    ;;
  multiturn_relation_100_102)
    run_fixed_seed_sequence \
      "visual_router_relation_multiturn_fixed_v1" \
      "visual_router_relation_multiturn_fixed_v1" \
      100 101 102
    ;;
  joint_cosine_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_relation_joint_multiturn_cosine_v2" \
      "visual_router_relation_joint_multiturn_cosine_v2_seed100"
    ;;
  legacy_3cdf58d_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_legacy_3cdf58d_joint_cosine_v2" \
      "visual_router_legacy_3cdf58d_joint_cosine_seed100"
    ;;
  fixed_stage1_constant_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_fixed_stage1_constant_v1" \
      "visual_router_fixed_stage1_constant_v1_seed100"
    ;;
  fixed_stage1_lr5e5_seed100)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=100
    run_one \
      "visual_router_fixed_stage1_lr5e5_v1" \
      "visual_router_fixed_stage1_lr5e5_v1_seed100"
    ;;
  fixed_stage1_pooling_lr1e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_fixed_stage1_pooling_lr1e5_v1" \
      "visual_router_fixed_stage1_pooling_lr1e5_v1_seed44"
    ;;
  fixed_stage1_global_lr_half_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_fixed_stage1_global_lr_half_v1" \
      "visual_router_fixed_stage1_global_lr_half_v1_seed44"
    ;;
  fixed_stage1_mmrl_only_lr3e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_fixed_stage1_mmrl_only_lr3e5_v1" \
      "visual_router_fixed_stage1_mmrl_only_lr3e5_v1_seed44"
    ;;
  stage3_control_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    MMRL_INTERMEDIATE_EVAL_STEPS="${MMRL_INTERMEDIATE_EVAL_STEPS:-250,500}"
    run_one \
      "visual_router_fixed_stage1_constant_v1" \
      "visual_router_fixed_stage1_constant_control_seed44"
    ;;
  stage3_late_decay_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    MMRL_INTERMEDIATE_EVAL_STEPS="${MMRL_INTERMEDIATE_EVAL_STEPS:-250,500}"
    run_one \
      "visual_router_fixed_stage1_late_decay_v1" \
      "visual_router_fixed_stage1_late_decay_seed44"
    ;;
  stage3_late_decay_balanced_loss_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    MMRL_INTERMEDIATE_EVAL_STEPS="${MMRL_INTERMEDIATE_EVAL_STEPS:-250,500}"
    run_one \
      "visual_router_fixed_stage1_late_decay_balanced_loss_v1" \
      "visual_router_fixed_stage1_late_decay_balanced_loss_seed44"
    ;;
  loss_matrix_r0125_seed44)
    run_loss_matrix_pair "r0125"
    ;;
  loss_matrix_r0250_seed44)
    run_loss_matrix_pair "r0250"
    ;;
  loss_matrix_r0500_seed44)
    run_loss_matrix_pair "r0500"
    ;;
  loss_tuning_no_effective_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    MMRL_INTERMEDIATE_EVAL_STEPS=""
    run_one \
      "visual_router_loss_tuning_no_effective_delta_v1" \
      "visual_router_loss_tuning_no_effective_delta_seed44"
    ;;
  loss_tuning_half_usage_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    MMRL_INTERMEDIATE_EVAL_STEPS=""
    run_one \
      "visual_router_loss_tuning_half_usage_v1" \
      "visual_router_loss_tuning_half_usage_seed44"
    ;;
  loss_tuning_relation_r0100_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    MMRL_INTERMEDIATE_EVAL_STEPS=""
    run_one \
      "visual_router_loss_tuning_relation_r0100_v1" \
      "visual_router_loss_tuning_relation_r0100_seed44"
    ;;
  loss_tuning_3x1_seed44)
    run_loss_tuning_3x1
    ;;
  slake_r025_delta_pair_seed44)
    run_slake_r025_delta_pair
    ;;
  slake_constraint_matrix_3x2_part1)
    run_slake_constraint_matrix_3x2_part 1
    ;;
  slake_constraint_matrix_3x2_part2)
    run_slake_constraint_matrix_3x2_part 2
    ;;
  slake_constraint_matrix_3x2_part3)
    run_slake_constraint_matrix_3x2_part 3
    ;;
  final_three_experiments_seed44)
    run_final_three_experiments
    ;;
  custom_r025_d058_multiseed_45_47)
    run_custom_r025_d058_multiseed
    ;;
  custom_optimizer_adapter_sweep_seed44)
    run_custom_optimizer_adapter_sweep
    ;;
  paper_ablation_seed44)
    run_paper_ablation_seed44
    ;;
  paper_ablation_seed44_part1)
    run_paper_ablation_part_seed44 1
    ;;
  paper_ablation_seed44_wo_rep_only)
    run_paper_ablation_part_seed44 1-fast
    ;;
  paper_ablation_seed44_part2)
    run_paper_ablation_part_seed44 2
    ;;
  paper_ablation_seed44_part3)
    run_paper_ablation_part_seed44 3
    ;;
  paper_ablation_222_seed44_part1)
    run_paper_ablation_222_part_seed44 1
    ;;
  paper_ablation_222_seed44_part2)
    run_paper_ablation_222_part_seed44 2
    ;;
  paper_ablation_222_seed44_part3)
    run_paper_ablation_222_part_seed44 3
    ;;
  stability_r0125_d058_cross_part1)
    run_stability_r0125_d058_cross_part 1
    ;;
  stability_r0125_d058_cross_part2)
    run_stability_r0125_d058_cross_part 2
    ;;
  final_constraint_matrix_3x3_part1)
    run_final_constraint_matrix_3x3_part 1
    ;;
  final_constraint_matrix_3x3_part2)
    run_final_constraint_matrix_3x3_part 2
    ;;
  final_constraint_matrix_3x3_part3)
    run_final_constraint_matrix_3x3_part 3
    ;;
  overnight_slake_pooling_pair_seed44)
    echo "[OVERNIGHT] target=$RUN_TARGET"
    echo "[OVERNIGHT] SLAKE output root=$SLAKE_OUTPUT_ROOT/mmrl"
    echo "[OVERNIGHT] latest run pointer=$SLAKE_OUTPUT_ROOT/last_overnight_run.txt"
    overnight_failures=0
    if ! run_slake_full_all; then
      echo "[OVERNIGHT-WARN] SLAKE 全量实验失败，继续共享 pooling 实验。" >&2
      overnight_failures=$((overnight_failures + 1))
    fi
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    MMRL_DECOUPLE_STAGE_POOLING=0
    if ! run_one \
      "visual_router_fixed_stage1_constant_v1" \
      "visual_router_fixed_stage1_constant_shared_pooling_seed44"; then
      echo "[OVERNIGHT-WARN] 共享 pooling 实验失败，继续解耦 pooling 实验。" >&2
      overnight_failures=$((overnight_failures + 1))
    fi
    MMRL_DECOUPLE_STAGE_POOLING=1
    if ! run_one \
      "visual_router_fixed_stage1_constant_v1" \
      "visual_router_fixed_stage1_constant_decoupled_pooling_seed44"; then
      echo "[OVERNIGHT-WARN] 解耦 pooling 实验失败。" >&2
      overnight_failures=$((overnight_failures + 1))
    fi
    if [ "$overnight_failures" -ne 0 ]; then
      echo "[OVERNIGHT-SUMMARY] 三轮均已尝试，失败数=$overnight_failures；即将按原计划自动关机。" >&2
      exit 1
    fi
    echo "[OVERNIGHT-SUMMARY] 三轮实验全部成功；即将按原计划自动关机。"
    ;;
  joint_cosine_1_4)
    run_fixed_seed_sequence \
      "visual_router_relation_joint_multiturn_cosine_v2" \
      "visual_router_relation_joint_multiturn_cosine" \
      1 2 3 4
    ;;
  joint_cosine_44_46)
    run_fixed_seed_sequence \
      "visual_router_relation_joint_multiturn_cosine_v2" \
      "visual_router_relation_joint_multiturn_cosine_clean_1_14" \
      44 45 46
    ;;
  spatial_grounding_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_spatial_grounding_v1" \
      "visual_router_spatial_grounding_v1_seed44"
    ;;
  ce_only_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_ce_only_v1" \
      "visual_router_joint_multiturn_cosine_ce_only_v1_seed44"
    ;;
  usage_only_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_usage_only_v1" \
      "visual_router_joint_multiturn_cosine_usage_only_v1_seed44"
    ;;
  overnight_route_loss_3plus1)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_usage_1e2_v1" \
      "visual_router_joint_multiturn_cosine_usage_1e2_v1_seed44"
    run_one \
      "visual_router_joint_multiturn_cosine_entropy_only_v1" \
      "visual_router_joint_multiturn_cosine_entropy_only_v1_seed44"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_v1" \
      "visual_router_joint_multiturn_cosine_route_pair_v1_seed44_repeat_a"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_v1" \
      "visual_router_joint_multiturn_cosine_route_pair_v1_seed44_repeat_b"
    ;;
  relation_2p5e3_repeat2)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_2p5e3_v1" \
      "visual_router_route_pair_relation_2p5e3_seed44_repeat_a"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_2p5e3_v1" \
      "visual_router_route_pair_relation_2p5e3_seed44_repeat_b"
    ;;
  relation_5e3_repeat2)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_5e3_v1" \
      "visual_router_route_pair_relation_5e3_seed44_repeat_a"
    run_one \
      "visual_router_joint_multiturn_cosine_route_pair_relation_5e3_v1" \
      "visual_router_route_pair_relation_5e3_seed44_repeat_b"
    ;;
  tune_mmrl_lr4e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr4e5_v1" \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr4e5_v1_seed44"
    ;;
  tune_mmrl_lr3e5_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr3e5_v1" \
      "visual_router_relation_joint_multiturn_cosine_mmrl_lr3e5_v1_seed44"
    ;;
  joint_cosine_no_usage_seed44)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=44
    run_one \
      "visual_router_relation_joint_multiturn_cosine_no_usage_v3" \
      "visual_router_relation_joint_multiturn_cosine_no_usage_v3_seed44"
    ;;
  heterogeneous_no_relation_100_102)
    run_fixed_seed_sequence \
      "visual_router_no_relation_heterogeneous_adapter_lr_v1" \
      "visual_router_no_relation_heterogeneous_adapter_lr_v1_paired" \
      100 101 102
    ;;
  raw_adapter47)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=47
    run_one "visual_router_raw_adapter_v1" "visual_router_raw_adapter_v1_seed47"
    ;;
  direct_mmrl)
    run_one "visual_router_direct_mmrl_seed44" "visual_router_direct_mmrl_seed44"
    ;;
  two_adapter)
    run_one "visual_router_two_adapter_seed44" "visual_router_two_adapter_seed44"
    ;;
  single_adapter)
    run_one "visual_router_single_adapter_seed44" "visual_router_single_adapter_seed44"
    ;;
  all)
    AUTO_INCREMENT_SEED=0
    FIXED_SEED=47
    run_one "visual_router_relation_alignment_diag" "visual_router_relation_retest_seed47"
    run_one "visual_router_raw_adapter_v1" "visual_router_raw_adapter_v1_seed47"
    ;;
  *)
    echo "[ERR] 未知实验目标: $RUN_TARGET（可选: relation44, relation47, heterogeneous_lr_pair, heterogeneous_relation_100_102, multiturn_relation_seed100, multiturn_relation_seed101, multiturn_relation_100_102, joint_cosine_seed100, legacy_3cdf58d_seed100, fixed_stage1_constant_seed100, fixed_stage1_lr5e5_seed100, fixed_stage1_pooling_lr1e5_seed44, fixed_stage1_global_lr_half_seed44, fixed_stage1_mmrl_only_lr3e5_seed44, stage3_control_seed44, stage3_late_decay_seed44, stage3_late_decay_balanced_loss_seed44, loss_matrix_r0125_seed44, loss_matrix_r0250_seed44, loss_matrix_r0500_seed44, loss_tuning_no_effective_seed44, loss_tuning_half_usage_seed44, loss_tuning_relation_r0100_seed44, loss_tuning_3x1_seed44, slake_r025_delta_pair_seed44, final_three_experiments_seed44, custom_r025_d058_multiseed_45_47, custom_optimizer_adapter_sweep_seed44, paper_ablation_seed44, paper_ablation_seed44_part1, paper_ablation_seed44_wo_rep_only, paper_ablation_seed44_part2, paper_ablation_seed44_part3, overnight_slake_pooling_pair_seed44, joint_cosine_1_4, joint_cosine_44_46, spatial_grounding_seed44, heterogeneous_no_relation_100_102, raw_adapter47, direct_mmrl, two_adapter, single_adapter, all）" >&2
    exit 2
    ;;
esac




# run_one "ablation_full_model" "ablation_full_model"
# run_one "ablation_wo_visual_gate" "ablation_wo_visual_gate"
# run_one "ablation_replace_mmrl_with_40_learnable_tokens" "ablation_replace_mmrl_with_40_learnable_tokens"

echo "[DONE] 已完成实验目标: $RUN_TARGET"
