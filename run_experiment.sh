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

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"

RUN_SUFFIX="${MMRL_RUN_SUFFIX:-}"

with_run_suffix() {
  local base_tag="$1"
  if [ -n "$RUN_SUFFIX" ]; then
    echo "${base_tag}_${RUN_SUFFIX}"
  else
    echo "$base_tag"
  fi
}

# 若目录已存在，自动追加 _1, _2, ... 直到找到可用名称
find_available_tag() {
  local base_tag="$1"
  local candidate="$base_tag"
  local i=1
  while [ -d "$CHECKPOINT_ROOT/$candidate" ]; do
    candidate="${base_tag}_${i}"
    ((i++))
  done
  echo "$candidate"
}

# 只保留当前仍有模型的并列最高分和并列最低分 checkpoint。
# 已经被删除 final 的历史日志不参与比较，避免无法恢复的旧极值占位。
prune_middle_final_dirs() {
  local extrema_output
  if ! extrema_output="$(cd "$ROOT_DIR" && python get_score.py --checkpoint-extrema-names)"; then
    echo "[WARN] 获取 checkpoint 极值失败，本轮不删除任何 final 目录。"
    return 0
  fi
  if [ -z "$extrema_output" ]; then
    echo "[WARN] 没有找到可比较的有效分数，本轮不删除任何 final 目录。"
    return 0
  fi

  declare -A keep_tags=()
  local keep_tag
  while IFS= read -r keep_tag; do
    if [ -n "$keep_tag" ]; then
      keep_tags["$keep_tag"]=1
    fi
  done <<< "$extrema_output"

  local experiment_dir tag final_dir
  for experiment_dir in "$CHECKPOINT_ROOT"/*; do
    [ -d "$experiment_dir" ] || continue
    tag="$(basename "$experiment_dir")"
    [ "$tag" = "trash" ] && continue
    final_dir="$experiment_dir/final"
    [ -d "$final_dir" ] || continue

    if [[ -n "${keep_tags[$tag]+x}" ]]; then
      echo "[KEEP] 保留极值 checkpoint: $tag"
    else
      echo "[PRUNE] 删除中间分 checkpoint: $tag"
      rm -rf -- "$final_dir"
    fi
  done
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

  echo "============================================================"
  echo "[EXP] 开始实验: $tag"
  echo "[EXP] MMRL_EXPERIMENT=$experiment_name"
  echo "[EXP] checkpoint目录: $output_dir"
  echo "============================================================"

  mkdir -p "$output_dir" "$eval_dir"

  (
    cd "$TRAIN_DIR"
    MMRL_OUTPUT_DIR="$output_dir" \
    MMRL_EXPERIMENT="$experiment_name" \
    MMRL_DETERMINISTIC_SAMPLING="1" \
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

# 单边尾部保护只跑一轮，先验证能否修复低质量 Stage3 轨迹。
run_one "visual_router_expert_delta_stage3_v6_tail_guard" "visual_router_expert_delta_stage3_v6_tail_guard"




# run_one "ablation_full_model" "ablation_full_model"
# run_one "ablation_wo_visual_gate" "ablation_wo_visual_gate"
# run_one "ablation_replace_mmrl_with_40_learnable_tokens" "ablation_replace_mmrl_with_40_learnable_tokens"

echo "[DONE] 所有实验均已串行完成。"
