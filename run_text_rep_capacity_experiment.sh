#!/bin/bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/Qwen3-VL-modify-test"
TRAIN_DIR="$ROOT_DIR/train"
TEST_DIR="$ROOT_DIR/test"
OUTPUT_ROOT="$ROOT_DIR/experiment_outputs"
CHECKPOINT_ROOT="$OUTPUT_ROOT/output"
DATA_JSON="${1:-/root/autodl-tmp/dataset/test2_val.json}"
DATA_IMG_DIR="${2:-/root/autodl-tmp/dataset/2/train}"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

run_one() {
  local expand_factor="$1"
  local tag="$2"
  local k_expert_target="${3:-30.0}"
  local output_dir="$CHECKPOINT_ROOT/$tag"
  local final_dir="$CHECKPOINT_ROOT/final"
  local eval_dir="$OUTPUT_ROOT/eval_$tag"
  local meta_dir="$OUTPUT_ROOT/meta_$tag"

  echo "============================================================"
  echo "[EXP] 开始实验: $tag | TEXT_REP_EXPAND_FACTOR=$expand_factor"
  echo "[EXP] checkpoint目录(按实验分别保留): $output_dir"
  echo "============================================================"

  local final_dir="$output_dir/final"

  rm -rf "$output_dir" "$eval_dir" "$meta_dir"
  mkdir -p "$output_dir" "$eval_dir" "$meta_dir"

  (
    cd "$TRAIN_DIR"
    MMRL_TEXT_REP_EXPAND_FACTOR="$expand_factor" \
    MMRL_OUTPUT_DIR="$output_dir" \
    MMRL_K_EXPERT_TARGET_S4="$k_expert_target" \
    python train.py 2>&1 | tee "$output_dir/train.log"
  )

  if [ ! -d "$final_dir" ]; then
    echo "[ERR] 训练完成后未找到目录: $final_dir"
    exit 1
  fi

  cp "$final_dir/config.json" "$meta_dir/config.json"
  cp "$final_dir/generation_config.json" "$meta_dir/generation_config.json"
  cp "$final_dir/tokenizer_config.json" "$meta_dir/tokenizer_config.json"

  (
    cd "$TEST_DIR"
    MMRL_TEXT_REP_EXPAND_FACTOR="$expand_factor" \
    MMRL_TRAINED_MODEL_PATH="$final_dir" \
    python test.py "$DATA_JSON" "$DATA_IMG_DIR" 2>&1 | tee "$eval_dir/test.log"
  )
}

# baseline: 文本 40（视觉不变）—— 已有结果，跳过
# run_one 1 "textrep40" 30.0

# ablation: 文本 80（视觉仍保持 40），K目标按比例放大 30*(80/40)=60
run_one 2 "textrep80" 60.0

echo "60秒后自动关机，如需取消，请立即执行:"
echo "sudo shutdown -c"
sleep 60
/usr/bin/shutdown