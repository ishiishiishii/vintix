#!/usr/bin/env bash
# Restart Aliengo PPO after config fix (MorFiC-aligned height/PD).
# Old failed run used z=0.55/kp=20; do not collect AD from that checkpoint.
set -euo pipefail
ROOT=/workspace
LOGDIR="$ROOT/vintix_go2/robot_eval_expansion/logs"
OUTDIR="$LOGDIR/aliengo-walking-v2"
mkdir -p "$LOGDIR/train_ppo" "$OUTDIR"
cd "$ROOT/Genesis/examples/locomotion"
exec python train.py -r aliengo -e aliengo-walking-v2 \
  --log_dir "$OUTDIR" \
  --max_iterations 301 \
  --seed 1
