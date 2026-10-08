#!/usr/bin/env bash
# User personally launches every GPU preflight, train and Test operation.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
"${V10_PYTHON:-/root/miniconda3/bin/python}" -m diagnostics.run_slake_v10_seed44 "$@"
