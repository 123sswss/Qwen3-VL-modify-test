#!/bin/bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/Qwen3-VL-modify-test"
RUN_SCRIPT="$ROOT_DIR/run_experiment.sh"
OUTPUT_ROOT="$ROOT_DIR/experiment_outputs"

# 用法：
#   ./run_experiment_repeat.sh N [DATA_JSON] [DATA_IMG_DIR]
#   示例：./run_experiment_repeat.sh 5
#   示例：./run_experiment_repeat.sh 3 /path/to/val.json /path/to/images
#
#   可通过环境变量指定 GPU：
#   CUDA_VISIBLE_DEVICES=0 ./run_experiment_repeat.sh 5

REPEAT_N="${1:?用法: $0 N [DATA_JSON] [DATA_IMG_DIR]  (N=重复次数)}"
DATA_JSON="${2:-/root/autodl-tmp/dataset/test2_val.json}"
DATA_IMG_DIR="${3:-/root/autodl-tmp/dataset/2/train}"

if ! [[ "$REPEAT_N" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERR] N 必须是正整数，当前值: '$REPEAT_N'"
  exit 1
fi

SHUTDOWN_DELAY_SECONDS="${MMRL_SHUTDOWN_DELAY_SECONDS:-60}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="$OUTPUT_ROOT/repeat_run_summary_${RUN_TIMESTAMP}.log"

mkdir -p "$OUTPUT_ROOT"

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "[ERR] 未找到脚本: $RUN_SCRIPT"
  exit 1
fi

echo "[INFO] 串行重复执行 run_experiment.sh" | tee -a "$SUMMARY_LOG"
echo "[INFO] 启动时间: $RUN_TIMESTAMP" | tee -a "$SUMMARY_LOG"
echo "[INFO] 重复次数: $REPEAT_N" | tee -a "$SUMMARY_LOG"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置(默认全部可用)}" | tee -a "$SUMMARY_LOG"
echo "[INFO] DATA_JSON=$DATA_JSON" | tee -a "$SUMMARY_LOG"
echo "[INFO] DATA_IMG_DIR=$DATA_IMG_DIR" | tee -a "$SUMMARY_LOG"
echo "[INFO] 汇总日志: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"

declare -a RUN_RESULTS=()

for i in $(seq 1 "$REPEAT_N"); do
  RUN_SUFFIX="run${i}"
  echo "" | tee -a "$SUMMARY_LOG"
  echo "############################################################" | tee -a "$SUMMARY_LOG"
  echo "[INFO] >>> 第 ${i}/${REPEAT_N} 次运行 (suffix=${RUN_SUFFIX}) <<<" | tee -a "$SUMMARY_LOG"
  echo "############################################################" | tee -a "$SUMMARY_LOG"

  if (
    cd "$ROOT_DIR"
    MMRL_DISABLE_AUTO_SHUTDOWN="1" \
    MMRL_RUN_SUFFIX="$RUN_SUFFIX" \
    bash "$RUN_SCRIPT" "$DATA_JSON" "$DATA_IMG_DIR"
  ); then
    RUN_RESULTS+=("成功")
    echo "[OK] 第 ${i}/${REPEAT_N} 次运行完成 (suffix=${RUN_SUFFIX})" | tee -a "$SUMMARY_LOG"
  else
    exit_code=$?
    RUN_RESULTS+=("失败(exit_code=$exit_code)")
    echo "[ERR] 第 ${i}/${REPEAT_N} 次运行失败 (suffix=${RUN_SUFFIX}, exit_code=$exit_code)" | tee -a "$SUMMARY_LOG"
  fi
done

echo "" | tee -a "$SUMMARY_LOG"
echo "[INFO] ===== 所有运行结果汇总 =====" | tee -a "$SUMMARY_LOG"
for i in "${!RUN_RESULTS[@]}"; do
  run_idx=$((i + 1))
  echo "第${run_idx}次：${RUN_RESULTS[$i]} (suffix=run${run_idx})" | tee -a "$SUMMARY_LOG"
done

success_count=0
for r in "${RUN_RESULTS[@]}"; do
  if [[ "$r" == "成功"* ]]; then
    ((success_count++)) || true
  fi
done
echo "[INFO] 成功: ${success_count}/${REPEAT_N}" | tee -a "$SUMMARY_LOG"

echo "[DONE] 全部 ${REPEAT_N} 次运行均已结束。" | tee -a "$SUMMARY_LOG"
echo "[DONE] ${SHUTDOWN_DELAY_SECONDS} 秒后自动关机。" | tee -a "$SUMMARY_LOG"
sleep "$SHUTDOWN_DELAY_SECONDS"
/usr/bin/shutdown