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
    echo "[EXIT] Last-8 LoRA R128 experiment finished with exit_code=${exit_code}."
    echo "[EXIT] The server will shut down automatically in 600 seconds."
    echo "[EXIT] Press Ctrl+C during the countdown to cancel shutdown."
    sleep 600
    /usr/bin/shutdown
    return "${exit_code}"
}

trap shutdown_on_exit EXIT
trap cancel_shutdown_on_interrupt INT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
SLAKE_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
EXPERIMENT="lora_visual_last8_attention_r128"
RUN_TAG="${EXPERIMENT}_3epoch_seed44_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/visual_peft_comparison/${RUN_TAG}"

mkdir -p "${OUTPUT_DIR}"
echo "[SLAKE_LAST8_R128] output_dir=${OUTPUT_DIR}"

if ! (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" slake/train_visual_peft.py \
        "${EXPERIMENT}" \
        --epochs 3 \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
); then
    echo "[SLAKE_LAST8_R128_ERR] training failed" >&2
    exit 1
fi

run_eval() {
    local checkpoint="$1"
    local questions="$2"
    local expected_split="$3"
    local output_dir="$4"
    local log_path="$5"
    mkdir -p "${output_dir}"
    (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" slake/slake_official_eval.py \
            --backend lora-vision-last8 \
            --base-model "${MODEL_PATH}" \
            --checkpoint "${checkpoint}" \
            --questions "${questions}" \
            --image-root "${SLAKE_ROOT}/imgs" \
            --output-dir "${output_dir}" \
            --language all \
            --expected-split "${expected_split}" \
            --max-new-tokens 32 \
            --temperature 0 \
            --answer-mode raw \
            --overwrite \
            2>&1 | tee "${log_path}"
    )
}

for epoch_id in 1 2 3; do
    checkpoint="${OUTPUT_DIR}/checkpoints/epoch_${epoch_id}"
    if [[ "${epoch_id}" = "3" ]]; then
        checkpoint="${OUTPUT_DIR}/final"
    fi
    echo "[SLAKE_LAST8_R128_VAL] epoch=${epoch_id} checkpoint=${checkpoint}"
    if ! run_eval \
        "${checkpoint}" \
        "${SLAKE_ROOT}/validation.json" "" \
        "${OUTPUT_DIR}/eval_validation/epoch_${epoch_id}" \
        "${OUTPUT_DIR}/eval_validation_epoch_${epoch_id}.log"; then
        echo "[SLAKE_LAST8_R128_ERR] epoch ${epoch_id} validation failed" >&2
        exit 1
    fi
done

selection="$("${PYTHON_BIN}" -c '
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for epoch in (1, 2, 3):
    path = root / "eval_validation" / f"epoch_{epoch}" / "slake_summary.json"
    with path.open("r", encoding="utf-8") as handle:
        rows.append((epoch, float(json.load(handle)["overall_accuracy"])))
epoch, score = max(rows, key=lambda item: (item[1], -item[0]))
print(f"{epoch}\t{score}")
' "${OUTPUT_DIR}")"
IFS=$'\t' read -r best_epoch best_validation_score <<< "${selection}"
best_checkpoint="${OUTPUT_DIR}/checkpoints/epoch_${best_epoch}"
if [[ "${best_epoch}" = "3" ]]; then
    best_checkpoint="${OUTPUT_DIR}/final"
fi

echo "[SLAKE_LAST8_R128_TEST] best_epoch=${best_epoch} validation=${best_validation_score}"
if ! run_eval \
    "${best_checkpoint}" \
    "${SLAKE_ROOT}/test.json" "test" \
    "${OUTPUT_DIR}/eval_test/epoch_${best_epoch}" \
    "${OUTPUT_DIR}/eval_test_epoch_${best_epoch}.log"; then
    echo "[SLAKE_LAST8_R128_ERR] selected test failed" >&2
    exit 1
fi

test_score="$("${PYTHON_BIN}" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["overall_accuracy"])' \
    "${OUTPUT_DIR}/eval_test/epoch_${best_epoch}/slake_summary.json")"
printf 'experiment\tbest_epoch\tvalidation_accuracy\ttest_accuracy\tcheckpoint\n' \
    > "${OUTPUT_DIR}/selected_result.tsv"
printf '%s\t%s\t%s\t%s\t%s\n' \
    "${EXPERIMENT}" "${best_epoch}" "${best_validation_score}" \
    "${test_score}" "${best_checkpoint}" \
    >> "${OUTPUT_DIR}/selected_result.tsv"

cat "${OUTPUT_DIR}/selected_result.tsv"
echo "[SLAKE_LAST8_R128_DONE] output_dir=${OUTPUT_DIR}"
