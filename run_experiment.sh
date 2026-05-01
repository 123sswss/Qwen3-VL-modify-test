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

run_one() {
  local tag="$1"
  local enable_text_gating="$2"
  local enable_k_loss_s4="$3"
  local enable_tax_loss_s4="$4"
  local output_dir="$CHECKPOINT_ROOT/$tag"
  local final_dir="$output_dir/final"
  local eval_dir="$OUTPUT_ROOT/eval_$tag"
  local meta_dir="$OUTPUT_ROOT/meta_$tag"

  echo "============================================================"
  echo "[EXP] 开始实验: $tag"
  echo "[EXP] ENABLE_TEXT_GATING=$enable_text_gating"
  echo "[EXP] ENABLE_K_LOSS_S4=$enable_k_loss_s4 | ENABLE_TAX_LOSS_S4=$enable_tax_loss_s4"
  echo "[EXP] checkpoint目录: $output_dir"
  echo "============================================================"

  rm -rf "$output_dir" "$eval_dir" "$meta_dir"
  mkdir -p "$output_dir" "$eval_dir" "$meta_dir"

  (
    cd "$TRAIN_DIR"
    MMRL_OUTPUT_DIR="$output_dir" \
    MMRL_ENABLE_TEXT_GATING="$enable_text_gating" \
    MMRL_FIXED_K_WHEN_TEXT_GATING_DISABLED="40" \
    MMRL_ENABLE_K_LOSS_S4="$enable_k_loss_s4" \
    MMRL_ENABLE_TAX_LOSS_S4="$enable_tax_loss_s4" \
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
    MMRL_TRAINED_MODEL_PATH="$final_dir" \
    python test.py "$DATA_JSON" "$DATA_IMG_DIR" 2>&1 | tee "$eval_dir/test.log"
  )
}

# 串行跑两个版本：
# 1) 当前版本：动态 K，最大池 40，text gating 存在
# 2) 消融版本：物理上不启用 text gating，固定 K=40
run_one "textgate_dynamic_pool40" 1 1 1
run_one "textgate_fixed40_no_text_gate" 0 0 0

echo "[DONE] 所有实验均已串行完成。"
sleep 60
/usr/bin/shutdown