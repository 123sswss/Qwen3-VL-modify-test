#!/usr/bin/env bash

set -uo pipefail

SHUTDOWN_ON_EXIT=1

cancel_shutdown_on_interrupt() {
    SHUTDOWN_ON_EXIT=0
    trap - EXIT
    echo "[INT] Ctrl+C detected; automatic shutdown cancelled."
    exit 130
}

shutdown_on_exit() {
    local exit_code=$?
    if [[ ${SHUTDOWN_ON_EXIT} -ne 1 ]]; then
        return "${exit_code}"
    fi
    echo "[EXIT] Experiment runner finished with exit_code=${exit_code}."
    echo "[EXIT] The server will shut down automatically in 600 seconds."
    echo "[EXIT] Press Ctrl+C during the countdown to cancel shutdown."
    sleep 600
    /usr/bin/shutdown
}

trap shutdown_on_exit EXIT
trap cancel_shutdown_on_interrupt INT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="/root/autodl-tmp/model"
SLAKE_ROOT="/root/autodl-tmp/dataset/slake"
QUESTIONS="${SLAKE_ROOT}/test.json"
IMAGE_ROOT="${SLAKE_ROOT}/imgs"
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/visual_peft_comparison"

mkdir -p "${OUTPUT_ROOT}"

FAILED_STEPS=()
LAST_STATUS=0

run_step() {
    local name="$1"
    local log_path="$2"
    shift 2

    mkdir -p "$(dirname -- "${log_path}")"
    printf '\n========== %s ==========\n' "${name}" | tee "${log_path}"
    "$@" 2>&1 | tee -a "${log_path}"
    LAST_STATUS=${PIPESTATUS[0]}
    if [[ ${LAST_STATUS} -eq 0 ]]; then
        printf '[PASS] %s\n' "${name}" | tee -a "${log_path}"
    else
        printf '[FAIL] %s exit_code=%s; continuing.\n' \
            "${name}" "${LAST_STATUS}" | tee -a "${log_path}"
        FAILED_STEPS+=("${name}")
    fi
}

run_eval() {
    local name="$1"
    local backend="$2"
    local output_dir="$3"
    local checkpoint="${4:-}"
    local command=(
        "${PYTHON_BIN}"
        "${SCRIPT_DIR}/slake_official_eval.py"
        --questions "${QUESTIONS}"
        --image-root "${IMAGE_ROOT}"
        --backend "${backend}"
        --base-model "${MODEL_PATH}"
        --output-dir "${output_dir}"
        --language all
        --max-new-tokens 32
        --temperature 0
        --answer-mode raw
        --overwrite
    )
    if [[ -n "${checkpoint}" ]]; then
        command+=(--checkpoint "${checkpoint}")
    fi
    run_step "${name}" "${output_dir}/eval.log" "${command[@]}"
}

run_train_and_eval() {
    local experiment="$1"
    local backend="$2"
    local experiment_dir="${OUTPUT_ROOT}/${experiment}"

    run_step \
        "${experiment}_train" \
        "${experiment_dir}/train.log" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/train_visual_peft.py" "${experiment}"
    local train_status=${LAST_STATUS}
    if [[ ${train_status} -ne 0 ]]; then
        printf '[SKIP] %s_eval because training failed.\n' "${experiment}" \
            | tee "${experiment_dir}/eval_skipped.log"
        FAILED_STEPS+=("${experiment}_eval_skipped")
        return
    fi

    run_eval \
        "${experiment}_eval" \
        "${backend}" \
        "${experiment_dir}/eval" \
        "${experiment_dir}/final"
}

cd "${REPO_ROOT}"

run_eval \
    "base_untrained_eval" \
    "base" \
    "${OUTPUT_ROOT}/base_untrained/eval"

run_train_and_eval \
    "lora_visual_all_attention_r32" \
    "lora-vision"

run_train_and_eval \
    "lora_visual_last8_attention_r32" \
    "lora-vision-last8"

run_train_and_eval \
    "dora_visual_all_attention_r32" \
    "dora-vision"

printf '\n========== SLAKE visual PEFT comparison finished ==========\n'
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    printf 'All baseline training and evaluation steps passed.\n'
    exit 0
fi

printf 'Failed or skipped steps (%s):\n' "${#FAILED_STEPS[@]}"
printf '  - %s\n' "${FAILED_STEPS[@]}"
exit 1
