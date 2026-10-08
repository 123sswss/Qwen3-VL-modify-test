#!/usr/bin/env bash
# User-started GPU queue; no retry. Detached CPU timer waits 600s before shutdown.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
export PYTHONUNBUFFERED=1
"${V10_PYTHON:-/root/miniconda3/bin/python}" -m diagnostics.run_pathvqa_v10_seeds --shutdown-after "$@"
