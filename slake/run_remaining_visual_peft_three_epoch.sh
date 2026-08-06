#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
SLAKE_ROOT="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
RUN_TAG="remaining_visual_peft_3epoch_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/outputs/visual_peft_comparison/${RUN_TAG}"
SUMMARY_TSV="${RUN_DIR}/selected_results.tsv"

mkdir -p "${RUN_DIR}"
printf 'method\tbest_epoch\tvalidation_accuracy\ttest_accuracy\tcheckpoint\n' > "${SUMMARY_TSV}"

failures=0

run_eval() {
    local backend="$1"
    local checkpoint="$2"
    local questions="$3"
    local expected_split="$4"
    local output_dir="$5"
    local log_path="$6"
    local command=(
        "${PYTHON_BIN}" slake/slake_official_eval.py
        --backend "${backend}"
        --base-model "${MODEL_PATH}"
        --questions "${questions}"
        --image-root "${SLAKE_ROOT}/imgs"
        --output-dir "${output_dir}"
        --language all
        --expected-split "${expected_split}"
        --max-new-tokens 32
        --temperature 0
        --answer-mode raw
        --overwrite
    )
    if [[ -n "${checkpoint}" ]]; then
        command+=(--checkpoint "${checkpoint}")
    fi
    (
        cd "${REPO_ROOT}"
        "${command[@]}" 2>&1 | tee "${log_path}"
    )
    return "${PIPESTATUS[0]}"
}

read_accuracy() {
    "${PYTHON_BIN}" -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["overall_accuracy"])' \
        "$1"
}

echo "[SLAKE_PEFT_REMAINING] run_dir=${RUN_DIR}"

base_dir="${RUN_DIR}/base_untrained"
mkdir -p "${base_dir}/eval_test"
if run_eval \
    "base" "" "${SLAKE_ROOT}/test.json" "test" \
    "${base_dir}/eval_test" "${base_dir}/eval_test.log"; then
    base_score="$(read_accuracy "${base_dir}/eval_test/slake_summary.json")"
    printf 'base_untrained\t-\t-\t%s\t%s\n' \
        "${base_score}" "${MODEL_PATH}" >> "${SUMMARY_TSV}"
else
    echo "[SLAKE_PEFT_REMAINING_WARN] base evaluation failed; continuing." >&2
    failures=$((failures + 1))
fi

run_method() {
    local experiment="$1"
    local backend="$2"
    local method_dir="${RUN_DIR}/${experiment}"
    local epoch_id
    local checkpoint
    local eval_dir

    mkdir -p "${method_dir}"
    echo "[SLAKE_PEFT_REMAINING_TRAIN] experiment=${experiment}"
    if ! (
        cd "${REPO_ROOT}"
        "${PYTHON_BIN}" slake/train_visual_peft.py \
            "${experiment}" \
            --epochs 3 \
            --output-dir "${method_dir}" \
            2>&1 | tee "${method_dir}/train.log"
    ); then
        echo "[SLAKE_PEFT_REMAINING_WARN] ${experiment} training failed; continuing." >&2
        failures=$((failures + 1))
        return
    fi

    for epoch_id in 1 2 3; do
        checkpoint="${method_dir}/checkpoints/epoch_${epoch_id}"
        if [[ "${epoch_id}" = "3" ]]; then
            checkpoint="${method_dir}/final"
        fi
        eval_dir="${method_dir}/eval_validation/epoch_${epoch_id}"
        mkdir -p "${eval_dir}"
        echo "[SLAKE_PEFT_REMAINING_VAL] experiment=${experiment} epoch=${epoch_id}"
        if ! run_eval \
            "${backend}" "${checkpoint}" \
            "${SLAKE_ROOT}/validation.json" "" \
            "${eval_dir}" "${method_dir}/eval_validation_epoch_${epoch_id}.log"; then
            echo "[SLAKE_PEFT_REMAINING_WARN] ${experiment} epoch ${epoch_id} validation failed." >&2
            failures=$((failures + 1))
            return
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
' "${method_dir}")"
    IFS=$'\t' read -r best_epoch best_validation_score <<< "${selection}"
    best_checkpoint="${method_dir}/checkpoints/epoch_${best_epoch}"
    if [[ "${best_epoch}" = "3" ]]; then
        best_checkpoint="${method_dir}/final"
    fi

    test_dir="${method_dir}/eval_test/epoch_${best_epoch}"
    mkdir -p "${test_dir}"
    echo "[SLAKE_PEFT_REMAINING_TEST] experiment=${experiment} best_epoch=${best_epoch} validation=${best_validation_score}"
    if ! run_eval \
        "${backend}" "${best_checkpoint}" \
        "${SLAKE_ROOT}/test.json" "test" \
        "${test_dir}" "${method_dir}/eval_test_epoch_${best_epoch}.log"; then
        echo "[SLAKE_PEFT_REMAINING_WARN] ${experiment} selected test failed." >&2
        failures=$((failures + 1))
        return
    fi

    test_score="$(read_accuracy "${test_dir}/slake_summary.json")"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${experiment}" "${best_epoch}" "${best_validation_score}" \
        "${test_score}" "${best_checkpoint}" >> "${SUMMARY_TSV}"
}

# lora_visual_all_attention_r32 was already completed separately.
run_method "lora_visual_last8_attention_r32" "lora-vision-last8"
run_method "dora_visual_all_attention_r32" "dora-vision"
run_method "lora_visual_all_attention_r64" "lora-vision"
run_method "lora_full_model_attention_r16" "lora"

echo "[SLAKE_PEFT_REMAINING_SUMMARY] failures=${failures} results=${SUMMARY_TSV}"
cat "${SUMMARY_TSV}"
if [[ ${failures} -ne 0 ]]; then
    exit 1
fi

echo "[SLAKE_PEFT_REMAINING_DONE] run_dir=${RUN_DIR}"
