#!/usr/bin/env bash
# Single user-started experiment; fail-stop, no retries/shutdown.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
"${V10_PYTHON:-/root/miniconda3/bin/python}" -m diagnostics.run_pathvqa_v10_head_fixed "$@"
