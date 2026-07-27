#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
RUN_STAMP="${BASELINE_RUN_STAMP:-${BASELINE_RUN_DATE:-$(date +%Y%m%d_%H%M%S)}}"
LOG_ROOT="${BASELINE_LOG_ROOT:-log}"
mkdir -p "${LOG_ROOT}"

FAILED_STEPS=()
LOG_FILTER_PATTERN='Loading weights:|Materializing param='

finish_with_shutdown() {
    local exit_code="$1"

    echo "[EXIT] script finished, exit_code=${exit_code}"
    echo "[EXIT] auto shutdown in 60 seconds."
    echo "[EXIT] press Ctrl+C during countdown to cancel shutdown."

    trap 'echo "[EXIT] auto shutdown cancelled."; exit "${exit_code}"' INT
    sleep 60
    local sleep_status=$?
    trap - INT

    if [ "${sleep_status}" -ne 0 ]; then
        echo "[EXIT] auto shutdown cancelled."
        exit "${exit_code}"
    fi

    /usr/bin/shutdown
    exit "${exit_code}"
}

run_step() {
    local name="$1"
    local experiment="$2"
    local phase="$3"
    shift 3

    local log_dir="${LOG_ROOT}/${experiment}"
    local logfile="${log_dir}/${phase}_${RUN_STAMP}.log"
    local suffix=2
    mkdir -p "${log_dir}"
    while [ -e "${logfile}" ]; do
        logfile="${log_dir}/${phase}_${RUN_STAMP}_${suffix}.log"
        ((suffix++))
    done

    echo "========== ${name} =========="
    echo "[LOG] ${logfile}"
    "$@" 2>&1 \
        | tr '\r' '\n' \
        | grep -v -E "${LOG_FILTER_PATTERN}" \
        | tee "${logfile}"
    local status=${PIPESTATUS[0]}

    if [ "${status}" -ne 0 ]; then
        echo "========== ${name} FAILED with exit code ${status}; continuing =========="
        FAILED_STEPS+=("${name}:${status}")
    else
        echo "========== ${name} finished =========="
    fi
}

# run_step "LoRA full-attention train: rank8/rank16/rank32" "lora" "train" python trainLora.py
# run_step "LoRA full-attention eval: rank8/rank16/rank32" "lora" "eval" python loraTest.py

run_step "LoRA vision-attention train: rank8/rank16/rank32" "lora_vision_attn" "train" python trainLoraVision.py
run_step "LoRA vision-attention eval: rank8/rank16/rank32" "lora_vision_attn" "eval" python loraVisionTest.py

run_step "LoRA last-8 vision train+eval: rank32 full-linear/attention" "lora_vision_last8" "train_eval" python loraLast8VisionExperiments.py

run_step "DoRA full-attention train: rank8/rank16" "dora" "train" python trainDora.py
run_step "DoRA full-attention eval: rank8/rank16" "dora" "eval" python doraTest.py

run_step "DoRA vision-attention train: rank8/rank16" "dora_vision_attn" "train" python trainDoraVision.py
run_step "DoRA vision-attention eval: rank8/rank16" "dora_vision_attn" "eval" python doraVisionTest.py

# run_step "IA3 train" "ia3" "train" python trainIA3.py
# run_step "IA3 eval" "ia3" "eval" python ia3Test.py

# run_step "Adapter train" "adapter" "train" python trainAdapter.py
# run_step "Adapter eval" "adapter" "eval" python adapterTest.py

echo "========== All PEFT experiments attempted =========="
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    echo "Failed steps:"
    printf '  %s\n' "${FAILED_STEPS[@]}"
    finish_with_shutdown 1
fi

echo "All steps finished successfully."
finish_with_shutdown 0
