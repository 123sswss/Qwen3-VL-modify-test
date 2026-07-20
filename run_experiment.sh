#!/bin/bash
set -euo pipefail

SHUTDOWN_ON_EXIT=1

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
  echo "[EXIT] 60 秒后自动关机。"
  echo "[EXIT] 如需取消自动关机，请在倒计时内按 Ctrl+C。"
  sleep 60
  /usr/bin/shutdown
}
trap shutdown_on_exit EXIT
trap cancel_shutdown_on_interrupt INT

ROOT_DIR="/root/autodl-tmp/Qwen3-VL-modify-test"
TRAIN_DIR="$ROOT_DIR/train"
TEST_DIR="$ROOT_DIR/test"
OUTPUT_ROOT="$ROOT_DIR/experiment_outputs"
CHECKPOINT_ROOT="$OUTPUT_ROOT/output"
SEED_STATE_FILE="${MMRL_SEED_STATE_FILE:-$OUTPUT_ROOT/next_seed.txt}"
INITIAL_SEED="${MMRL_INITIAL_SEED:-42}"
AUTO_INCREMENT_SEED="${MMRL_AUTO_INCREMENT_SEED:-0}"
FIXED_SEED="${MMRL_FIXED_SEED:-44}"

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"

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
  local base_tag
  base_tag="$(with_run_suffix "$raw_tag")"
  local tag
  tag="$(find_available_tag "$base_tag")"
  local output_dir="$CHECKPOINT_ROOT/$tag"
  local final_dir="$output_dir/final"
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
  echo "[EXP] checkpoint目录: $output_dir"
  echo "============================================================"

  mkdir -p "$output_dir" "$eval_dir"

  (
    cd "$TRAIN_DIR"
    MMRL_OUTPUT_DIR="$output_dir" \
    MMRL_EXPERIMENT="$experiment_name" \
    MMRL_SEED="$experiment_seed" \
    MMRL_DATA_SAMPLING_SEED="42" \
    MMRL_DETERMINISTIC_SAMPLING="1" \
    MMRL_EVAL_EACH_EPOCH="0" \
    python train.py 2>&1 | tee "$output_dir/train.log"
  )

  if [ ! -d "$final_dir" ]; then
    echo "[ERR] 训练完成后未找到目录: $final_dir"
    exit 1
  fi

  (
    cd "$TEST_DIR"
    MMRL_TRAINED_MODEL_PATH="$final_dir" \
    python test.py 2>&1 | tee "$eval_dir/test.log"
  )

  # 测试日志生成分数后，保留全局并列最高/最低，清理其他 final。
  prune_middle_final_dirs
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

# 固定 seed 44，只验证一路恒等专家替换一路可训练 adapter。
run_one "visual_router_routed_identity_expert_v1" "visual_router_routed_identity_expert_v1"




# run_one "ablation_full_model" "ablation_full_model"
# run_one "ablation_wo_visual_gate" "ablation_wo_visual_gate"
# run_one "ablation_replace_mmrl_with_40_learnable_tokens" "ablation_replace_mmrl_with_40_learnable_tokens"

echo "[DONE] 所有实验均已串行完成。"
