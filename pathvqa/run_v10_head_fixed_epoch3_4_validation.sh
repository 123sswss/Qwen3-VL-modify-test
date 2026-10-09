#!/usr/bin/env bash
# Evaluation only. Epoch5 remains the original primary result.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${V10_PYTHON:-/root/miniconda3/bin/python}"
RUN="pathvqa/outputs/v10_head_fixed/pathvqa_v10_condition_head_ln_default_metanet_h160_lr3e4_norm_fixed_5ep_seed44_20261009_102158_643548"
OUTPUT="$RUN/epoch3_4_validation_$(date +%Y%m%d_%H%M%S)"
mkdir "$OUTPUT"
echo "[EPOCH3_4_OUTPUT] $OUTPUT"
git rev-parse HEAD > "$OUTPUT/execution_commit.txt"
epoch=0
trap 'code=$?; printf "exit_code=%s\nlast_epoch=%s\n" "$code" "$epoch" > "$OUTPUT/exit_status.txt"' EXIT
export PYTHONUNBUFFERED=1
for epoch in 3 4; do
  checkpoint="$RUN/checkpoints/epoch_$epoch"
  test -s "$checkpoint/v10_head_fixed.pt"
  test -f "$checkpoint/v10_head_fixed_config.json"
  "$PYTHON" -m pathvqa.pathvqa_official_eval \
    --backend v10-head-fixed --base-model /root/autodl-tmp/model \
    --checkpoint "$checkpoint" --data-root /root/autodl-tmp/dataset/pathVQA \
    --split validation --max-new-tokens 32 --temperature 0 --answer-mode raw \
    --output-dir "$OUTPUT/epoch_$epoch" \
    2>&1 | tee "$OUTPUT/epoch_${epoch}.log"
done
"$PYTHON" - "$OUTPUT" "$RUN" <<'PY'
import json,sys
from pathlib import Path
output,run = map(Path,sys.argv[1:])
rows = []
for epoch in (3,4,5):
    p = (output/f"epoch_{epoch}" if epoch != 5 else run/"eval_validation/epoch_5")/"pathvqa_summary.json"
    if not p.exists():
        continue
    summary = json.loads(p.read_text(encoding="utf-8"))
    assert summary["count"] == 6259 and summary["split"] == "validation"
    rows.append({"epoch":epoch,"summary_path":str(p),"summary":summary,"reused":epoch==5})
    print("epoch{}: Overall={:.4f} Yes/No={:.4f} Free-form={:.4f}".format(
        epoch,summary["overall_accuracy"],summary["yes_no_accuracy"],summary["free_form_accuracy"]))
    print("Question types:",summary["per_question_type_accuracy"])
(output/"epoch_scores.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
PY
