#!/usr/bin/env bash
# After current Aliengo PPO exits, restart with latest get_aliengo_cfgs().
# Attempt history:
#   kp20/z0.55 → len≈5; kp40/z0.40 → len≈1.7; planned z0.55 rejected after FK.
# Current: kp40 / init z=0.50 / height target 0.46 (FK standing ≈0.4607).
# Handles zombie PIDs (container init may not reap; kill -0 still succeeds on Z).
set -euo pipefail
OLD_PID="${1:?usage: $0 <pid-to-wait>}"
ROOT=/workspace
LOGDIR="$ROOT/vintix_go2/robot_eval_expansion/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$LOGDIR/train_ppo/aliengo_seed1_fk050_${STAMP}.out"

is_alive_non_zombie() {
  local pid="$1"
  [[ -r "/proc/$pid/stat" ]] || return 1
  # /proc/PID/stat field 3 is state; Z = zombie
  local state
  state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || echo Z)
  [[ "$state" != "Z" ]]
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for PID $OLD_PID to exit (ignore zombies)..."
while is_alive_non_zombie "$OLD_PID"; do
  sleep 20
done

if pgrep -f "train.py -r aliengo" >/dev/null 2>&1; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another aliengo train already running; skip"
  exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting kp40/z050 (FK-based) restart"
if [ -d "$LOGDIR/aliengo-walking" ] && [ ! -d "$LOGDIR/aliengo-walking_prev_${STAMP}" ]; then
  mv "$LOGDIR/aliengo-walking" "$LOGDIR/aliengo-walking_prev_${STAMP}" || true
fi
mkdir -p "$LOGDIR/train_ppo" "$LOGDIR/aliengo-walking"
cd "$ROOT/Genesis/examples/locomotion"
nohup python train.py -r aliengo -e aliengo-walking \
  --log_dir "$LOGDIR/aliengo-walking" \
  --max_iterations 301 \
  --seed 1 > "$OUT" 2>&1 &
echo $! > "$LOGDIR/train_ppo/aliengo_latest.pid"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] started PID=$(cat "$LOGDIR/train_ppo/aliengo_latest.pid") log=$OUT"
