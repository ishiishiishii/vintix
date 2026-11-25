#!/bin/bash
set -euo pipefail
PID_FILE=/workspace/vintix_go2/robot_eval_expansion/logs/train_ppo/laikago_raisim_pose_seed1.pid
LOG=/workspace/vintix_go2/robot_eval_expansion/logs/train_ppo/laikago_raisim_pose_eval.out
CKPT=/workspace/vintix_go2/robot_eval_expansion/logs/laikago-walking-raisim-pose/model_300.pt
JSON=/workspace/vintix_go2/robot_eval_expansion/results/laikago_raisim_pose_expert_quality.json
VIDEO=/workspace/vintix_go2/robot_eval_expansion/figures/videos/laikago_raisim_pose_walk_model300.mp4
PID=$(cat "$PID_FILE")
echo "[$(date -Is)] waiting for Laikago Raisim-pose 301 train pid=$PID" | tee "$LOG"
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
  --robot laikago \
  --model_path "$CKPT" \
  --cmd_vx 0.4 \
  --num_envs 10 \
  --rollout_steps 1000 \
  --follow_cam \
  --out_json "$JSON" \
  --video "$VIDEO" \
  2>&1 | tee -a "$LOG"
echo "[$(date -Is)] eval done" | tee -a "$LOG"
