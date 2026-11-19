#!/usr/bin/env bash
# Wait for Go1 same-body eval (#18) to finish, then: build dataset → train → 3-axis eval → report.
# Does not touch the running #18 processes; only polls status / process presence.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
EXP="$ROOT/dr_narrow_vintix_bodyscale"
SAME="$ROOT/same_body_scale_zeroshot"
STATUS="$SAME/eval_go1_status.json"
LOG="$EXP/logs/wait_and_run.log"
mkdir -p "$EXP/logs"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

wait_for_go1_eval() {
  log "waiting for #18 Go1 eval to finish (status=$STATUS)"
  while true; do
    # Prefer status file when present and consistent.
    if [[ -f "$STATUS" ]]; then
      if python - "$STATUS" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
done, total = int(d.get("done", 0)), int(d.get("total", -1))
skipped = int(d.get("n_skipped", 0) or 0)
if skipped <= 0 and d.get("skipped"):
    skipped = len(d.get("skipped") or [])
running = d.get("running") or []
print(
    f"status done={done}/{total} skipped={skipped} "
    f"accounted={done + skipped}/{total} running={len(running)}",
    flush=True,
)
# finished when done+skipped cover the grid and nothing running
# (early-stop leaves far scales as skipped, so done alone never reaches total)
sys.exit(0 if total > 0 and (done + skipped) >= total and not running else 1)
PY
      then
        log "#18 status says finished (done+skipped>=total, not running)"
        return 0
      fi
    fi
    # Fallback: no eval_go1_pipeline / worker processes
    if ! pgrep -af 'same_body_scale_zeroshot/eval_go1_(pipeline|worker)\.py' >/dev/null 2>&1; then
      if [[ -f "$STATUS" ]]; then
        log "no eval process; re-check status once more"
        sleep 5
        continue
      fi
      log "no status file and no eval process — assuming #18 not running; proceed"
      return 0
    fi
    sleep 60
  done
}

wait_for_go1_eval
log "#18 Go1 eval finished (or not running). starting DR-narrow pipeline"

log "build dataset"
python -u "$EXP/build_dataset.py" 2>&1 | tee -a "$LOG"

log "train Vintix"
bash "$EXP/train.sh" 2>&1 | tee -a "$LOG"

MODEL="$EXP/models/dr_narrow_success_5epochs/dr_narrow_success_5epochs/0004_epoch"
if [[ ! -d "$MODEL" ]]; then
  log "ERROR: missing $MODEL after train"
  exit 1
fi

log "Go1 3-axis eval"
python -u "$EXP/eval_go1_pipeline.py" 2>&1 | tee -a "$EXP/logs/eval_go1.log" | tee -a "$LOG"

log "numberline + REPORT.md"
python -u "$EXP/make_numberline_report.py" 2>&1 | tee -a "$LOG"

log "pipeline complete — sync Notion REPORT next (manual/agent)"
echo done > "$EXP/logs/PIPELINE_DONE"
