#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_lora_r2_dataset_seed44.sh rsvqa_lr
