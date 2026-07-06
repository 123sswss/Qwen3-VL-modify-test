#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs

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
    local logfile="$2"
    shift 2

    echo "========== ${name} =========="
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

run_step "LoRA full-attention train: rank8/rank16/rank32" logs/train_lora.log python trainLora.py
run_step "LoRA full-attention eval: rank8/rank16/rank32" logs/eval_lora.log python loraTest.py

run_step "LoRA vision-attention train: rank8/rank16/rank32" logs/train_lora_vision_attn.log python trainLoraVision.py
run_step "LoRA vision-attention eval: rank8/rank16/rank32" logs/eval_lora_vision_attn.log python loraVisionTest.py

run_step "DoRA full-attention train: rank8/rank16" logs/train_dora.log python trainDora.py
run_step "DoRA full-attention eval: rank8/rank16" logs/eval_dora.log python doraTest.py

run_step "DoRA vision-attention train: rank8/rank16" logs/train_dora_vision_attn.log python trainDoraVision.py
run_step "DoRA vision-attention eval: rank8/rank16" logs/eval_dora_vision_attn.log python doraVisionTest.py

run_step "IA3 train" logs/train_ia3.log python trainIA3.py
run_step "IA3 eval" logs/eval_ia3.log python ia3Test.py

run_step "Adapter train" logs/train_adapter.log python trainAdapter.py
run_step "Adapter eval" logs/eval_adapter.log python adapterTest.py

echo "========== All PEFT experiments attempted =========="
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    echo "Failed steps:"
    printf '  %s\n' "${FAILED_STEPS[@]}"
    finish_with_shutdown 1
fi

echo "All steps finished successfully."
finish_with_shutdown 0
