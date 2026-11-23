#!/usr/bin/env bash
# After current Aliengo PPO exits, restart with latest get_aliengo_cfgs().
# Archive note: kp20/z0.55 failed (len≈5); kp40/z0.40 failed worse (len≈1.7).
# Next try: kp40 / init z=0.55 / height target 0.45.
set -euo pipefail
OLD_PID="${1:?usage: $0 <pid-to-wait>}"
ROOT=/workspace
LOGDIR="$ROOT/vintix_go2/robot_eval_expansion/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$LOGDIR/train_ppo/aliengo_seed1_kp40_z55_${STAMP}.out"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for PID $OLD_PID to exit..."
while kill -0 "$OLD_PID" 2>/dev/null; do
  sleep 20
done
# If another train already holds the GPU, do not start a third copy.
if pgrep -f "train.py -r aliengo" >/dev/null 2>&1; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another aliengo train already running; skip"
  exit 0
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting kp40/z55 restart"
# Archive previous attempt if present
if [ -d "$LOGDIR/aliengo-walking" ] && [ ! -d "$LOGDIR/aliengo-walking_kp40_z40_failed_${STAMP}" ]; then
  mv "$LOGDIR/aliengo-walking" "$LOGDIR/aliengo-walking_kp40_z40_failed_${STAMP}" || true
fi
mkdir -p "$LOGDIR/train_ppo" "$LOGDIR/aliengo-walking"
cd "$ROOT/Genesis/examples/locomotion"
nohup python train.py -r aliengo -e aliengo-walking \
  --log_dir "$LOGDIR/aliengo-walking" \
  --max_iterations 301 \
  --seed 1 > "$OUT" 2>&1 &
echo $! > "$LOGDIR/train_ppo/aliengo_latest.pid"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] started PID=$(cat "$LOGDIR/train_ppo/aliengo_latest.pid") log=$OUT"
