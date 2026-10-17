#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
"${V1_PYTHON:-/root/miniconda3/bin/python}" -m diagnostics.run_pathvqa_read_write_three_seed_test
