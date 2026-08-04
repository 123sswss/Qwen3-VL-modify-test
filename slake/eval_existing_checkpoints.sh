#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/model}"
SLAKE_ROOT="${SLAKE_ROOT:-/root/autodl-tmp/dataset/slake}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/outputs/visual_peft_comparison}"
RESULT_ROOT="${RESULT_ROOT:-${CHECKPOINT_ROOT}/timing_results}"
TIMING_WARMUP_RUNS="${TIMING_WARMUP_RUNS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

QUESTIONS="${SLAKE_ROOT}/test.json"
IMAGE_ROOT="${SLAKE_ROOT}/imgs"
FAILED=()

mkdir -p "${RESULT_ROOT}"

resolve_final() {
    local directory="$1"
    if [[ -f "${directory}/final/adapter_config.json" ]]; then
        printf '%s\n' "${directory}/final"
    elif [[ -f "${directory}/adapter_config.json" ]]; then
        printf '%s\n' "${directory}"
    else
        return 1
    fi
}

run_eval() {
    local name="$1"
    local backend="$2"
    local checkpoint="${3:-}"
    local output_dir="${RESULT_ROOT}/${name}"
    local command=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/slake_official_eval.py"
        --backend "${backend}"
        --base-model "${BASE_MODEL}"
        --questions "${QUESTIONS}"
        --image-root "${IMAGE_ROOT}"
        --output-dir "${output_dir}"
        --language all
        --max-new-tokens "${MAX_NEW_TOKENS}"
        --temperature 0
        --answer-mode raw
        --timing-warmup-runs "${TIMING_WARMUP_RUNS}"
        --overwrite
    )
    if [[ -n "${checkpoint}" ]]; then
        command+=(--checkpoint "${checkpoint}")
    fi
    if [[ -n "${EVAL_LIMIT}" ]]; then
        command+=(--limit "${EVAL_LIMIT}")
    fi

    mkdir -p "${output_dir}"
    echo "============================================================"
    echo "[EVAL] method=${name} backend=${backend} checkpoint=${checkpoint:-BASE}"
    echo "============================================================"
    if ! "${command[@]}" 2>&1 | tee "${output_dir}/eval.log"; then
        echo "[WARN] ${name} failed; continuing." >&2
        FAILED+=("${name}")
    fi
}

run_peft() {
    local name="$1"
    local backend="$2"
    local directory="${CHECKPOINT_ROOT}/${name}"
    local checkpoint
    if ! checkpoint="$(resolve_final "${directory}")"; then
        echo "[WARN] Missing PEFT checkpoint: ${directory}/final" >&2
        FAILED+=("${name}")
        return
    fi
    run_eval "${name}" "${backend}" "${checkpoint}"
}

cd "${REPO_ROOT}"
run_eval "base_untrained" "base"
run_peft "lora_visual_all_attention_r32" "lora-vision"
run_peft "lora_visual_all_attention_r64" "lora-vision"
run_peft "lora_visual_last8_attention_r32" "lora-vision-last8"
run_peft "dora_visual_all_attention_r32" "dora-vision"
run_peft "lora_full_model_attention_r16" "lora"

if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_checkpoint_timing.py" "${RESULT_ROOT}"; then
    echo "[WARN] Failed to build comparison summary." >&2
    FAILED+=("summary")
fi

if (( ${#FAILED[@]} > 0 )); then
    printf '[DONE-WITH-WARNINGS] failed=%s\n' "${FAILED[*]}" >&2
    exit 1
fi
echo "[DONE] All existing checkpoints evaluated. Results: ${RESULT_ROOT}"
