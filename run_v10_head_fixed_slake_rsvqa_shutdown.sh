#!/usr/bin/env bash
# User-only GPU execution: SLAKE then RSVQA-LR, fail-stop, shutdown on exit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
PY="${V10_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MODEL_PATH:-/root/autodl-tmp/model}"
SLAKE="${SLAKE_DATA_ROOT:-/root/autodl-tmp/dataset/slake}"
RSVQA="${RSVQA_DATA_ROOT:-/root/autodl-tmp/dataset/RSVQA/6344334}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SUITE="outputs/v10_head_fixed_slake_rsvqa/$STAMP"
SLAKE_OUT="slake/outputs/v10_head_fixed/slake_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_$STAMP"
RSVQA_OUT="RSVQA/outputs/v10_head_fixed/rsvqa_lr_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_$STAMP"
mkdir -p "$SUITE" "$SLAKE_OUT" "$RSVQA_OUT"
STAGE="starting"
SLAKE_STATUS="not_run"
RSVQA_STATUS="not_run"
finish() {
    local code=$?
    trap - EXIT
    set +e
    printf 'exit_code=%s\nstage=%s\nslake=%s\nrsvqa_lr=%s\n' \
        "$code" "$STAGE" "$SLAKE_STATUS" "$RSVQA_STATUS" > "$SUITE/final_status.txt"
    printf 'calling /usr/bin/shutdown at %s\n' "$(date -Is)" > "$SUITE/shutdown.log"
    sync
    # Invoke in Bash, not Python exec: AutoDL shutdown may be a script without shebang.
    /usr/bin/shutdown >> "$SUITE/shutdown.log" 2>&1
    local shutdown_code=$?
    printf 'shutdown_exit_code=%s\n' "$shutdown_code" >> "$SUITE/shutdown.log"
    sync
    exit "$code"
}
trap finish EXIT
printf 'commit=%s\nslake_output=%s\nrsvqa_output=%s\nmodel_seed=44\ndata_seed=42\nepochs=5\nprimary_epoch=5\nbatch=2\naccumulation=16\n' \
    "$(git rev-parse HEAD)" "$SLAKE_OUT" "$RSVQA_OUT" > "$SUITE/config.txt"
step() {
    STAGE="$1"
    local log="$2"
    shift 2
    printf '%s started %s\n' "$STAGE" "$(date -Is)" >> "$SUITE/stages.log"
    local code=0
    "$@" 2>&1 | tee "$log" || code=$?
    printf '%s exit_code=%s %s\n' "$STAGE" "$code" "$(date -Is)" >> "$SUITE/stages.log"
    if (( code != 0 )); then
        if [[ "$STAGE" == slake_* ]]; then SLAKE_STATUS=failed; fi
        if [[ "$STAGE" == rsvqa_* ]]; then RSVQA_STATUS=failed; fi
        exit "$code"
    fi
}
echo "[V10_HEAD_FIXED_SUITE] $SUITE (SLAKE -> RSVQA-LR; shutdown on success/failure)"
SLAKE_STATUS=running
step slake_train "$SLAKE_OUT/train.log" "$PY" -m slake.train_v10_head_fixed \
    --dataset slake --model-seed 44 --model-path "$MODEL" --data-root "$SLAKE" \
    --output-dir "$SLAKE_OUT" --expected-train-count 9834 --epochs 5 --save-epochs 3 4 5
step slake_test "$SLAKE_OUT/eval_test.log" "$PY" -m slake.slake_official_eval \
    --backend v10-head-fixed --base-model "$MODEL" --checkpoint "$SLAKE_OUT/checkpoints/epoch_5" \
    --questions "$SLAKE/test.json" --image-root "$SLAKE/imgs" --language all --expected-split test \
    --max-new-tokens 32 --temperature 0 --answer-mode raw --output-dir "$SLAKE_OUT/eval_test/epoch_5"
SLAKE_STATUS=completed
RSVQA_STATUS=running
step rsvqa_train "$RSVQA_OUT/train.log" "$PY" -m RSVQA.train_v10_head_fixed \
    --dataset rsvqa_lr --model-seed 44 --model-path "$MODEL" --data-root "$RSVQA" \
    --output-dir "$RSVQA_OUT" --expected-train-count 57223 --epochs 5 --save-epochs 3 4 5
step rsvqa_test "$RSVQA_OUT/eval_test.log" "$PY" -m RSVQA.rsvqa_lr_official_eval \
    --backend v10-head-fixed --base-model "$MODEL" --checkpoint "$RSVQA_OUT/checkpoints/epoch_5" \
    --data-root "$RSVQA" --split test --max-new-tokens 32 --output-dir "$RSVQA_OUT/eval_test/epoch_5"
RSVQA_STATUS=completed
step summarize "$SUITE/summary.log" "$PY" - "$SLAKE_OUT" "$RSVQA_OUT" "$SUITE" <<'PY'
import json,sys
from pathlib import Path
runs={}
for name,path,filename in zip(('slake','rsvqa_lr'),sys.argv[1:3],('slake_summary.json','rsvqa_summary.json')):
    root=Path(path)
    runs[name]={'output':str(root.resolve()),'train_report':json.loads((root/'train_report.json').read_text()),
                'summary':json.loads((root/'eval_test/epoch_5'/filename).read_text())}
out=Path(sys.argv[3])
(out/'final_report.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'ledger_fragment.md').write_text('# V10 head-fixed SLAKE / RSVQA-LR seed44, fixed epoch5 Test\n'+
    json.dumps(runs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(runs,ensure_ascii=False,indent=2))
PY
STAGE=completed
