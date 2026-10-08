#!/usr/bin/env bash
# Independent user-launched experiment; never enables automatic shutdown.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
python -m diagnostics.run_slake_cocoop_seed44 "$@"
