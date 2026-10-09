#!/usr/bin/env bash
# User-started fresh7epochs -> curves -> Validation5/6/7 -> paired reports.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
"${V10_PYTHON:-/root/miniconda3/bin/python}" -m diagnostics.run_pathvqa_v10_head_fixed_7ep "$@"
