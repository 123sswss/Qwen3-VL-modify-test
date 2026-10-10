#!/usr/bin/env bash
# Existing seed44 epoch5 only; no training, retries or shutdown.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${LORA_PYTHON:-/root/miniconda3/bin/python}"
RUN="${PATHVQA_LORA_R2_RUN:-pathvqa/outputs/lora/pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44_20261001_1}"
OUT="$RUN/test_epoch5_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$OUT/exit_status.txt"' EXIT
"$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
d=json.loads((root/'train_report.json').read_text(encoding='utf-8'))
c=json.loads((root/'checkpoints/epoch_5/adapter_config.json').read_text(encoding='utf-8'))
assert d['experiment']=='pathvqa_lora_full_model_attention_r2_norm_fixed_5ep_b1a32_seed44'
assert d['seed']==44 and d['data_seed']==42 and d['epochs']==5
assert d['trainer_model_accepts_loss_kwargs'] is False and d['parameter_counts']['trainable']==1769472
assert c['r']==2 and c['lora_alpha']==4 and not c['use_rslora']
print('[LORA_R2_TEST_CHECKPOINT]',(root/'checkpoints/epoch_5').resolve())
PY
printf 'evaluation_commit=%s\nrun=%s\nepoch=5\nsplit=test\n' "$(git rev-parse HEAD)" "$RUN" > "$OUT/config.txt"
"$PY" -m pathvqa.pathvqa_official_eval \
    --backend lora --base-model "${MMRL_MODEL_PATH:-/root/autodl-tmp/model}" \
    --checkpoint "$RUN/checkpoints/epoch_5" \
    --data-root "${PATHVQA_DATA_ROOT:-/root/autodl-tmp/dataset/pathVQA}" \
    --split test --output-dir "$OUT" 2>&1 | tee "$OUT/eval.log"
echo "[LORA_R2_TEST_DONE] output=$OUT"
