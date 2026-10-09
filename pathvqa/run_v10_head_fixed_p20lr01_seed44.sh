#!/usr/bin/env bash
# User-started single run: fail-stop, no retry or shutdown.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
"${V10_PYTHON:-/root/miniconda3/bin/python}" -c 'from diagnostics.run_pathvqa_v10_head_fixed import main; raise SystemExit(main(p20_low_lr=True))' "$@"
