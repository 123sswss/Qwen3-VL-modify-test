#!/bin/bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/Qwen3-VL-modify-test"
TRAIN_DIR="$ROOT_DIR/train"
TEST_DIR="$ROOT_DIR/test"
OUTPUT_ROOT="$ROOT_DIR/experiment_outputs"
CHECKPOINT_ROOT="$OUTPUT_ROOT/output"
DATA_JSON="${1:-/root/autodl-tmp/dataset/test2_val.json}"
DATA_IMG_DIR="${2:-/root/autodl-tmp/dataset/2/train}"

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

run_one() {
  local experiment_name="$1"
  local raw_tag="$2"
  local tag
  tag="$(with_run_suffix "$raw_tag")"
  local output_dir="$CHECKPOINT_ROOT/$tag"
  local final_dir="$output_dir/final"
  local eval_dir="$OUTPUT_ROOT/eval_$tag"

  echo "============================================================"
  echo "[EXP] 开始实验: $tag"
  echo "[EXP] MMRL_EXPERIMENT=$experiment_name"
  echo "[EXP] checkpoint目录: $output_dir"
  echo "============================================================"

  rm -rf "$output_dir" "$eval_dir"
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
    python test.py "$DATA_JSON" "$DATA_IMG_DIR" 2>&1 | tee "$eval_dir/test.log"
  )

  # 测试完成后删除最终模型，保留 stage1~stage4 的日志与图表
  echo "[INFO] 测试完成，删除 final 模型目录以节省硬盘空间..."
  if [ -d "$final_dir" ]; then
    rm -rf "$final_dir"
    echo "[INFO] 已删除 final 模型目录: $final_dir"
  fi
}

# 可选实验列表（保留注释，便于后续继续切换/复用）：
# 1) dynamic_prefix：原始连续前缀法
# run_one "dynamic_prefix" "textgate_dynamic_prefixv4"
#
# 2) fixed_group20：固定四组、20个placeholder
# run_one "fixed_group20" "textgate_fixed_group20"
#
# 3) group_top4：8选4、可学习分组
# run_one "group_top4" "textgate_group_top4v4"
#
# 4) token_top20：40个rep token直接打分并固定选top20
# run_one "token_top20" "textgate_token_top20v1"

# 当前启用的实验：
# 1) 任务感知轻引导 group_top4（当前主实验）
run_one "group_top4" "textgate_group_top4_L2_v2"
# 2) dynamic_prefix 作为对照基线
# run_one "dynamic_prefix" "textgate_dynamic_prefix_v7"

echo "[DONE] 所有实验均已串行完成。"
if [ "${MMRL_DISABLE_AUTO_SHUTDOWN:-0}" = "1" ]; then
  echo "[DONE] 检测到 MMRL_DISABLE_AUTO_SHUTDOWN=1，跳过自动关机。"
else
  echo "[DONE] 60 秒后自动关机。"
  sleep 60
  /usr/bin/shutdown
fi
