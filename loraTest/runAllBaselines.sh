#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs

FAILED_STEPS=()

finish_with_shutdown() {
    local exit_code="$1"

    echo "[EXIT] 脚本退出，exit_code=${exit_code}"
    echo "[EXIT] 60 秒后自动关机。"
    echo "[EXIT] 如需取消自动关机，请在倒计时内按 Ctrl+C。"

    trap 'echo "[EXIT] 已取消自动关机。"; exit "${exit_code}"' INT
    sleep 60
    local sleep_status=$?
    trap - INT

    if [ "${sleep_status}" -ne 0 ]; then
        echo "[EXIT] 已取消自动关机。"
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
    "$@" 2>&1 | tee "${logfile}"
    local status=${PIPESTATUS[0]}

    if [ "${status}" -ne 0 ]; then
        echo "========== ${name} FAILED with exit code ${status}; continuing =========="
        FAILED_STEPS+=("${name}:${status}")
    else
        echo "========== ${name} finished =========="
    fi
}

run_step "LoRA train: rank8/rank16/rank32" logs/train_lora.log python trainLora.py
run_step "LoRA eval: rank8/rank16/rank32" logs/eval_lora.log python loraTest.py

run_step "DoRA train: rank8/rank16" logs/train_dora.log python trainDora.py
run_step "DoRA eval: rank8/rank16" logs/eval_dora.log python doraTest.py

run_step "IA3 train" logs/train_ia3.log python trainIA3.py
run_step "IA3 eval" logs/eval_ia3.log python ia3Test.py

run_step "Adapter train" logs/train_adapter.log python trainAdapter.py
run_step "Adapter eval" logs/eval_adapter.log python adapterTest.py

echo "========== All seven PEFT experiments attempted =========="
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    echo "Failed steps:"
    printf '  %s\n' "${FAILED_STEPS[@]}"
    finish_with_shutdown 1
fi

echo "All steps finished successfully."
finish_with_shutdown 0
