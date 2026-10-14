#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
PY="${V1_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MMRL_MODEL_PATH:-/root/autodl-tmp/model}"
DATA="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
BASELINE=slake/outputs/visual_selection_prefix/slake_v1_norm_fixed_5ep_seed44_20260928
EXPERIMENT=slake_v1_read_write_lr3e4_norm_fixed_5ep_seed44
OUT="slake/outputs/visual_selection_prefix/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
STAGE=baseline_check
trap 'code=$?; printf "stage=%s\nexit_code=%s\n" "$STAGE" "$code" > "$OUT/exit_status.txt"' EXIT
"$PY" - "$BASELINE" "$DATA" "$MODEL" <<'PY'
import sys
from pathlib import Path
from diagnostics.run_slake_cocoop_seed44 import references,baseline_audit
baseline_audit(Path(sys.argv[1]),references(Path(sys.argv[2])/'test.json'),Path(sys.argv[3]))
PY
printf 'experiment=%s\ncommit=%s\nvalue_projection_lr=0.0003\nprefix_output_lr=0.0003\nexpected_trainable=1864963\n' \
    "$EXPERIMENT" "$(git rev-parse HEAD)" > "$OUT/config.txt"
echo "[SLAKE_V1_READ_WRITE_OUTPUT] $OUT"
STAGE=train
"$PY" -m pathvqa.train_visual_selection_prefix --dataset slake \
    --model-path "$MODEL" --data-root "$DATA" --output-dir "$OUT" --experiment-name "$EXPERIMENT" \
    --visual-prompt-mode split18 --read-write-learning-rate 3e-4 \
    --model-seed 44 --epochs 5 --save-epochs 3 4 5 \
    --batch-size 2 --gradient-accumulation 16 --correct-loss-accumulation 2>&1 | tee "$OUT/train.log"
STAGE=test_epoch5
"$PY" -m slake.slake_official_eval --backend visual-selection-prefix --base-model "$MODEL" \
    --checkpoint "$OUT/checkpoints/epoch_5" --questions "$DATA/test.json" --image-root "$DATA/imgs" \
    --language all --expected-split test --max-new-tokens 32 --temperature 0 --answer-mode raw \
    --output-dir "$OUT/eval_test/epoch_5" 2>&1 | tee "$OUT/eval_test.log"
STAGE=paired_vs_v1
"$PY" - "$BASELINE" "$OUT" "$DATA" <<'PY' 2>&1 | tee "$OUT/paired_vs_v1.log"
import sys,json
from pathlib import Path
from diagnostics.run_slake_cocoop_seed44 import references,paired,write,read
baseline,out,data=map(Path,sys.argv[1:])
result=paired(baseline/'eval_test/epoch_5',out/'eval_test/epoch_5',references(data/'test.json'))
result['budget_note']='Both original V1 and read/write-LR variant: seed44/data42, five epochs, fixed epoch5 bilingual Test; only value_blocks/prefix_output base LR changed.'
write(out/'paired_vs_v1.json',result)
summary=read(out/'eval_test/epoch_5/slake_summary.json')
print(json.dumps({'summary':summary,'paired':result},ensure_ascii=False))
(out/'ledger_fragment.md').write_text(
    'SLAKE V1 read/write LR3e-4, seed44/data42, fixed epoch5 bilingual Test.\n\n'
    +json.dumps({'output':str(out),'summary':summary,'paired':result},ensure_ascii=False,indent=2),encoding='utf-8')
PY
STAGE=completed
echo "[SLAKE_V1_READ_WRITE_DONE] $OUT validation=false shutdown=false"
