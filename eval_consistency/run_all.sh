#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python eval_consistency/run_all.py "$@"
