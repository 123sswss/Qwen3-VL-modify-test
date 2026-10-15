#!/usr/bin/env bash
# User personally launches the three GPU exports; CPU analysis follows separately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${V1_PYTHON:-/root/miniconda3/bin/python}"
OUT="${1:-outputs/v1_three_branch_audit_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$OUT"
STAGE=precheck
trap 'code=$?; printf "stage=%s\nexit_code=%s\n" "$STAGE" "$code" > "$OUT/exit_status.txt"' EXIT
echo "[THREE_BRANCH_AUDIT_OUTPUT] $OUT"
if [[ ! -f "$OUT/binding.json" ]]; then
    "$PY" -m diagnostics.audit_v1_three_branches precheck --output "$OUT" 2>&1 | tee "$OUT/precheck.log"
fi
for NAME in v1 lr3e4 v20; do
    STAGE="export_$NAME"
    "$PY" -m diagnostics.audit_v1_three_branches export --model "$NAME" --output "$OUT" 2>&1 | tee "$OUT/export_$NAME.log"
done
STAGE=exports_completed_cpu_analysis_pending
echo "[THREE_BRANCH_EXPORTS_DONE] $OUT"
