#!/usr/bin/env bash
# Explicit opt-in shutdown suite; GPU operations are launched by the user.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="outputs/v1_read_write_three_tasks_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
STAGE=starting
finish() {
  CODE=$?
  trap - EXIT
  set +e
  printf 'stage=%s\nexit_code=%s\nfinished_at=%s\n' "$STAGE" "$CODE" "$(date -Is)" > "$OUT/exit_status.txt"
  echo "[SUITE_FINISHED] stage=$STAGE exit_code=$CODE; shutdown in 600 seconds" | tee -a "$OUT/suite.log"
  sync
  sleep 600
  printf 'called_at=%s\n' "$(date -Is)" > "$OUT/shutdown_status.txt"
  /usr/bin/shutdown >> "$OUT/shutdown.log" 2>&1
  SHUTDOWN_CODE=$?
  printf 'shutdown_exit_code=%s\n' "$SHUTDOWN_CODE" >> "$OUT/shutdown_status.txt"
  sync
  exit "$CODE"
}
trap finish EXIT
printf 'commit=%s\norder=rsvqa_lr:44,pathvqa:45,slake:45\nshutdown_delay_seconds=600\n' "$(git rev-parse HEAD)" > "$OUT/config.txt"
echo "[SUITE_OUTPUT] $OUT"
for TASK in rsvqa_lr:44 pathvqa:45 slake:45; do
  STAGE="$TASK"
  echo "[TASK_START] $TASK" | tee -a "$OUT/suite.log"
  bash run_v1_read_write_lr3e4_single.sh "${TASK%:*}" "${TASK#*:}" 2>&1 | tee "$OUT/${TASK/:/_}.log"
  echo "[TASK_DONE] $TASK" | tee -a "$OUT/suite.log"
done
STAGE=completed
