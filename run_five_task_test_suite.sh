#!/bin/bash
set -euo pipefail
# No shutdown trap, no retry. User executes all GPU child processes.
cd "${MMRL_ROOT_DIR:-/root/autodl-tmp/Qwen3-VL-modify-test}"
export MMRL_SHUTDOWN_ON_EXIT=0
python -m diagnostics.run_five_task_test_suite --root "$PWD" "$@"
