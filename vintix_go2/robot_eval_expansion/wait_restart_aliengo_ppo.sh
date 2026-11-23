#!/usr/bin/env bash
# Wait for failed Aliengo PPO (PID) to exit, then start v2 with fixed cfgs.
set -euo pipefail
OLD_PID="${1:-1929430}"
ROOT=/workspace
LOGDIR="$ROOT/vintix_go2/robot_eval_expansion/logs"
OUT="$LOGDIR/train_ppo/aliengo_seed1_v2.out"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for PID $OLD_PID to exit..."
while kill -0 "$OLD_PID" 2>/dev/null; do
  sleep 15
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] old train exited; starting v2"
mkdir -p "$LOGDIR/train_ppo" "$LOGDIR/aliengo-walking-v2"
cd "$ROOT/Genesis/examples/locomotion"
# shellcheck disable=SC2086
nohup python train.py -r aliengo -e aliengo-walking-v2 \
  --log_dir "$LOGDIR/aliengo-walking-v2" \
  --max_iterations 301 \
  --seed 1 > "$OUT" 2>&1 &
echo $! > "$LOGDIR/train_ppo/aliengo_v2.pid"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] started v2 PID=$(cat "$LOGDIR/train_ppo/aliengo_v2.pid") log=$OUT"
