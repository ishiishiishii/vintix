#!/usr/bin/env bash
# Wait for Aliengo PPO model_300.pt, then collect Adaptation (AD) trajectories.
# Usage: setsid nohup bash wait_collect_aliengo_ad.sh > logs/train_ppo/wait_collect_ad.out 2>&1 &
set -euo pipefail
ROOT="/workspace/vintix_go2/robot_eval_expansion"
CKPT="${ROOT}/logs/aliengo-walking/model_300.pt"
TRAIN_PID_FILE="${ROOT}/logs/train_ppo/aliengo_morfic.pid"
OUT_DIR="${ROOT}/data/aliengo_ad_parallel"
LOG="${ROOT}/logs/train_ppo/aliengo_ad_collect.out"
PY="/opt/conda/bin/python"
COLLECT="/workspace/vintix_go2/scripts/collect_ad_data_parallel.py"

mkdir -p "${ROOT}/logs/train_ppo" "${OUT_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${CKPT}"
while [[ ! -f "${CKPT}" ]]; do
  sleep 30
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] checkpoint present"

# If a train PID was recorded, wait for it to exit (avoid reading a partial write).
# Zombies (<defunct>) still pass kill -0 — treat them as done.
is_alive_non_zombie() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 1
  local state
  state="$(ps -o state= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  [[ -n "$state" && "$state" != Z* ]]
}
if [[ -f "${TRAIN_PID_FILE}" ]]; then
  TPID="$(cat "${TRAIN_PID_FILE}" || true)"
  if [[ -n "${TPID}" ]] && is_alive_non_zombie "${TPID}"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for train PID ${TPID} to exit"
    while is_alive_non_zombie "${TPID}"; do
      sleep 20
    done
  fi
fi

# Extra settle in case the final save is still flushing.
sleep 5
ls -la "${CKPT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching AD collect → ${OUT_DIR}"
cd /workspace
# Wave-1: 10 envs × 200k steps (verifies pipeline; full 1M can follow).
exec "${PY}" "${COLLECT}" \
  -r aliengo \
  --model_path "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --num_envs 10 \
  --target_steps_per_env 200000 \
  --max_steps 1000 \
  > "${LOG}" 2>&1
