#!/bin/bash
set -euo pipefail
PID_FILE=/workspace/vintix_go2/robot_eval_expansion/logs/train_ppo/spot_isaac_seed1.pid
LOG=/workspace/vintix_go2/robot_eval_expansion/logs/train_ppo/spot_isaac_eval.out
CKPT=/workspace/vintix_go2/robot_eval_expansion/logs/spot-walking-isaac/model_300.pt
JSON=/workspace/vintix_go2/robot_eval_expansion/results/spot_isaac_expert_quality.json
VIDEO=/workspace/vintix_go2/robot_eval_expansion/figures/videos/spot_isaac_walk_model300.mp4
PID=$(cat "$PID_FILE")
echo "[$(date -Is)] waiting for Spot Isaac train pid=$PID" | tee "$LOG"
# Prefer /proc/$PID status: zombies (Z) count as done so we never hang on defunct.
while true; do
  if [[ ! -d /proc/$PID ]]; then
    break
  fi
  stat=$(awk '{print $3}' /proc/$PID/stat 2>/dev/null || true)
  if [[ "$stat" == "Z" ]]; then
    echo "[$(date -Is)] train pid=$PID is zombie; treating as exited" | tee -a "$LOG"
    break
  fi
  sleep 30
done
echo "[$(date -Is)] train exited; waiting for $CKPT" | tee -a "$LOG"
for i in $(seq 1 90); do
  [[ -f "$CKPT" ]] && break
  sleep 10
done
if [[ ! -f "$CKPT" ]]; then
  echo "[$(date -Is)] ERROR: $CKPT missing" | tee -a "$LOG"
  exit 1
fi
cd /workspace/Genesis/examples/locomotion
/opt/conda/bin/python /workspace/vintix_go2/robot_eval_expansion/eval_expert_quality.py \
  --robot spot \
  --model_path "$CKPT" \
  --cmd_vx 0.7 \
  --num_envs 10 \
  --rollout_steps 1000 \
  --follow_cam \
  --out_json "$JSON" \
  --video "$VIDEO" \
  2>&1 | tee -a "$LOG"
echo "[$(date -Is)] eval done" | tee -a "$LOG"
