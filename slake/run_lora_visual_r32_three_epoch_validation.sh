#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
SLAKE_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
RUN_TAG="lora_visual_all_attention_r32_3epoch_seed44_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/visual_peft_comparison/${RUN_TAG}"

mkdir -p "${OUTPUT_DIR}"

echo "[SLAKE_LORA_3EP] output_dir=${OUTPUT_DIR}"
if ! (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" slake/train_visual_peft.py \
        lora_visual_all_attention_r32 \
        --epochs 3 \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
); then
    echo "[SLAKE_LORA_3EP_ERR] training failed" >&2
    exit 1
fi

failures=0
evaluate_epoch() {
    local epoch_id="$1"
    local checkpoint="$2"
    local eval_dir="${OUTPUT_DIR}/eval_validation/epoch_${epoch_id}"
    mkdir -p "${eval_dir}"
    echo "[SLAKE_LORA_3EP_EVAL] epoch=${epoch_id} checkpoint=${checkpoint}"
    if ! (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" slake/slake_official_eval.py \
            --backend lora-vision \
            --base-model "${MODEL_PATH}" \
            --checkpoint "${checkpoint}" \
            --questions "${SLAKE_ROOT}/validation.json" \
            --image-root "${SLAKE_ROOT}/imgs" \
            --output-dir "${eval_dir}" \
            --language all \
            --expected-split "" \
            --max-new-tokens 32 \
            --temperature 0 \
            --answer-mode raw \
            --overwrite \
            2>&1 | tee "${OUTPUT_DIR}/eval_validation_epoch_${epoch_id}.log"
    ); then
        echo "[SLAKE_LORA_3EP_WARN] epoch=${epoch_id} evaluation failed" >&2
        failures=$((failures + 1))
    fi
}

evaluate_epoch 1 "${OUTPUT_DIR}/checkpoints/epoch_1"
evaluate_epoch 2 "${OUTPUT_DIR}/checkpoints/epoch_2"
evaluate_epoch 3 "${OUTPUT_DIR}/final"

echo "[SLAKE_LORA_3EP_SUMMARY] all epochs attempted; failures=${failures}"
if [[ ${failures} -ne 0 ]]; then
    exit 1
fi

echo "[SLAKE_LORA_3EP_DONE] output_dir=${OUTPUT_DIR}"
