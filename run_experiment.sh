#!/bin/bash
set -euo pipefail

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

  # 测试完成后删除最终模型，保留 stage1~stage4 的日志与图表
  echo "[INFO] 测试完成，删除 final 模型目录以节省硬盘空间..."
  if [ -d "$final_dir" ]; then
    rm -rf "$final_dir"
    echo "[INFO] 已删除 final 模型目录: $final_dir"
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

# 可选实验列表（保留注释，便于后续继续切换/复用）：
# 1) group_threshold_prior：8组门控 + capacity prior，主实验
# run_one "group_threshold_prior" "group_threshold_prior"
#
# 2) ablation_full_model：同主实验，用于显式消融命名
# run_one "ablation_full_model" "ablation_full_model"
#
# 3) ablation_wo_visual_gate：视觉门控消融
# run_one "ablation_wo_visual_gate" "ablation_wo_visual_gate"
#
# 4) ablation_replace_mmrl_with_40_learnable_tokens：直接可学习文本rep消融
# run_one "ablation_replace_mmrl_with_40_learnable_tokens" "ablation_replace_mmrl_with_40_learnable_tokens"

# 当前启用的文本门控实验：
# 当前代码中真实可用的实验名见 train/train.py: EXPERIMENTS
# run_one "visual_router_v2_1_text_off" "visual_router_v2_1_text_off_safe"
# run_one "visual_router_v2_text_off" "visual_router_v2_text_off_test2"

run_N "visual_router_v2_1_text_on_monitor" "visual_router_v2_1_text_on_monitor" 3

# run_one "ablation_full_model" "ablation_full_model"
# run_one "ablation_wo_visual_gate" "ablation_wo_visual_gate"
# run_one "ablation_replace_mmrl_with_40_learnable_tokens" "ablation_replace_mmrl_with_40_learnable_tokens"

echo "[DONE] 所有实验均已串行完成。"
echo "[DONE] 60 秒后自动关机。"
sleep 60
/usr/bin/shutdown
