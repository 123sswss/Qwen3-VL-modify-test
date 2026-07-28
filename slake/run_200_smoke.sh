#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python slake/train_mmrl.py \
  --smoke-test \
  --no-save-final \
  --output-dir ./slake_outputs/mmrl_smoke_200 \
  "$@"
