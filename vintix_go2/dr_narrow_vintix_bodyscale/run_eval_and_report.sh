#!/usr/bin/env bash
# Resume Go1 3-axis eval (+ numberline report) after train is done.
# Used when wait_and_run is restarted mid-eval (e.g. early-stop backport).
set -euo pipefail
cd "$(dirname "$0")/.."
EXP="$(pwd)/dr_narrow_vintix_bodyscale"
MODEL="$EXP/models/dr_narrow_success_5epochs/dr_narrow_success_5epochs/0004_epoch"
LOG="$EXP/logs/eval_go1.log"
mkdir -p "$EXP/logs"

if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: missing $MODEL" >&2
  exit 1
fi

rm -f "$EXP/STOP"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval_go1_pipeline (early-stop) start" | tee -a "$LOG"
python -u "$EXP/eval_go1_pipeline.py" 2>&1 | tee -a "$LOG"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] make_numberline_report" | tee -a "$LOG"
python -u "$EXP/make_numberline_report.py" 2>&1 | tee -a "$LOG"
echo done > "$EXP/logs/PIPELINE_DONE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline complete" | tee -a "$LOG"
