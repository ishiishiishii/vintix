#!/usr/bin/env bash
# After B1 Adaptation-history collect finishes: write wave1 summary, then start ANYmal-B PPO
# and the AD wait/collect chain. Does not touch the running B1 collect process.
#
# Usage:
#   setsid nohup bash wait_after_b1_ad_start_anymalb.sh \
#     > logs/train_ppo/wait_after_b1_ad_start_anymalb.out 2>&1 &
#   echo $! > logs/train_ppo/wait_after_b1_ad_start_anymalb.pid
set -euo pipefail
ROOT="/workspace/vintix_go2/robot_eval_expansion"
GENESIS_LOCO="/workspace/Genesis/examples/locomotion"
PY="/opt/conda/bin/python"
B1_AD_PID_FILE="${ROOT}/logs/train_ppo/b1_ad_collect.pid"
B1_AD_DIR="${ROOT}/data/b1_ad_parallel"
B1_SUMMARY="${ROOT}/results/b1_wave1_summary.json"
ANYMALB_LOGDIR="${ROOT}/logs/anymalb-walking"
ANYMALB_TRAIN_OUT="${ROOT}/logs/train_ppo/anymalb_seed1.out"
ANYMALB_PID_FILE="${ROOT}/logs/train_ppo/anymalb_latest.pid"
CHAIN_LOG="${ROOT}/logs/train_ppo/wait_after_b1_ad_start_anymalb.out"

mkdir -p "${ROOT}/logs/train_ppo" "${ROOT}/results" "${ANYMALB_LOGDIR}"

is_alive_non_zombie() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 1
  local state
  state="$(ps -o state= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  [[ -n "$state" && "$state" != Z* ]]
}

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

B1_PID=""
if [[ -f "${B1_AD_PID_FILE}" ]]; then
  B1_PID="$(cat "${B1_AD_PID_FILE}" || true)"
fi
if [[ -z "${B1_PID}" ]]; then
  # Fallback: discover from process list
  B1_PID="$(pgrep -f "collect_ad_data_parallel.py -r b1" | head -n1 || true)"
fi

if [[ -n "${B1_PID}" ]] && is_alive_non_zombie "${B1_PID}"; then
  log "waiting for B1 AD collect PID ${B1_PID} to exit"
  while is_alive_non_zombie "${B1_PID}"; do
    sleep 60
  done
  log "B1 AD collect PID ${B1_PID} exited"
else
  log "B1 AD collect not running (pid=${B1_PID:-none}); proceed if data present"
fi

# Summarize B1 wave1 (h5 lengths + prior smoke). Prefer SWMR/read without blocking writers.
"${PY}" - <<'PY'
import json, glob, os
from pathlib import Path
root = Path("/workspace/vintix_go2/robot_eval_expansion")
ad = root / "data" / "b1_ad_parallel"
files = sorted(ad.glob("trajectories_env_*.h5"))
# file size proxy if h5 locked; try open
steps_sum = 0
per_env = []
try:
    import h5py
    for f in files:
        with h5py.File(f, "r") as h:
            n = None
            for k in ("rewards", "actions", "observations", "obs", "dones"):
                if k in h:
                    n = int(h[k].shape[0]); break
            if n is None:
                n = 0
            per_env.append({"file": f.name, "steps": n})
            steps_sum += n
except Exception as e:
    per_env = [{"error": str(e), "n_files": len(files)}]
    steps_sum = -1

smoke_path = root / "results" / "b1_ckpt_smoke_eval.json"
smoke = {}
if smoke_path.exists():
    smoke = json.loads(smoke_path.read_text())

out = {
    "robot": "b1",
    "expert_ckpt": "logs/b1-walking/model_40.pt",
    "smoke_eval": "results/b1_ckpt_smoke_eval.json",
    "smoke_note": "8env×500; model_40 completion 100%, mean return ≈2.3 (see smoke json)",
    "smoke": smoke,
    "ad_data_dir": "data/b1_ad_parallel",
    "ad_num_envs": 10,
    "ad_target_steps_per_env": 200000,
    "ad_final_steps_sum": steps_sum,
    "ad_per_env": per_env,
    "status": "wave1_ad_complete" if steps_sum >= 1_900_000 else "wave1_ad_partial_or_unreadable",
}
(root / "results" / "b1_wave1_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"ad_final_steps_sum": steps_sum, "status": out["status"]}, indent=2))
PY

# Do not start ANYmal-B if another heavy job already holds the GPU.
BUSY="$(pgrep -af "python.*(train\.py|collect_ad_data_parallel\.py|save_vintix|run_experiment)" || true)"
if [[ -n "${BUSY}" ]]; then
  log "GPU still busy after B1 AD; NOT starting ANYmal-B. Busy:"
  log "${BUSY}"
  exit 0
fi

if [[ -f "${ANYMALB_LOGDIR}/model_300.pt" ]]; then
  log "ANYmal-B model_300.pt already exists; skip PPO, ensure AD wait is running"
else
  # B1 lesson: mid-train peak then collapse can miss walking weights if save_interval=100.
  # Pre-seed cfgs.pkl with save_interval=20 so train.py loads it (resume_existing path)
  # without editing common train.py. Empty log dir except cfgs → fresh run, denser saves.
  if [[ ! -f "${ANYMALB_LOGDIR}/cfgs.pkl" ]]; then
    log "pre-seeding ${ANYMALB_LOGDIR}/cfgs.pkl with save_interval=20"
    "${PY}" - <<'PY'
import os, pickle, sys
sys.path.insert(0, "/workspace/Genesis/examples/locomotion")
from train import get_anymalb_cfgs, get_train_cfg
log_dir = "/workspace/vintix_go2/robot_eval_expansion/logs/anymalb-walking"
os.makedirs(log_dir, exist_ok=True)
env_cfg, obs_cfg, reward_cfg, command_cfg = get_anymalb_cfgs()
train_cfg = get_train_cfg("anymalb-walking", 301, seed=1)
train_cfg["save_interval"] = 20
with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
print("wrote cfgs.pkl save_interval=", train_cfg["save_interval"])
PY
  fi
  log "starting ANYmal-B PPO (301 iter, seed=1, dense save via cfgs) → ${ANYMALB_TRAIN_OUT}"
  (
    cd "${GENESIS_LOCO}"
    exec "${PY}" train.py -r anymalb -e anymalb-walking \
      --log_dir "${ANYMALB_LOGDIR}" \
      --max_iterations 301 \
      --seed 1
  ) > "${ANYMALB_TRAIN_OUT}" 2>&1 &
  TRAIN_PID=$!
  echo "${TRAIN_PID}" > "${ANYMALB_PID_FILE}"
  log "ANYmal-B PPO PID ${TRAIN_PID}"
fi

# Always (re)start AD wait/collect chain so history is collected after PPO.
log "starting wait_collect_anymalb_ad.sh"
setsid nohup bash "${ROOT}/wait_collect_anymalb_ad.sh" \
  > "${ROOT}/logs/train_ppo/wait_collect_anymalb_ad.out" 2>&1 &
echo $! > "${ROOT}/logs/train_ppo/wait_collect_anymalb_ad.pid"
log "wait_collect_anymalb_ad PID $(cat "${ROOT}/logs/train_ppo/wait_collect_anymalb_ad.pid")"
log "done"
