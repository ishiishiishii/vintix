#!/usr/bin/env bash
# Wait for B1 PPO model_300.pt, quality-gate on final episode length, then collect AD.
# Usage: setsid nohup bash wait_collect_b1_ad.sh > logs/train_ppo/wait_collect_b1_ad.out 2>&1 &
set -euo pipefail
ROOT="/workspace/vintix_go2/robot_eval_expansion"
CKPT="${ROOT}/logs/b1-walking/model_300.pt"
TRAIN_PID_FILE="${ROOT}/logs/train_ppo/b1_latest.pid"
TRAIN_LOG="${ROOT}/logs/train_ppo/b1_seed1_bh2.out"
OUT_DIR="${ROOT}/data/b1_ad_parallel"
LOG="${ROOT}/logs/train_ppo/b1_ad_collect.out"
GATE_JSON="${ROOT}/results/b1_ad_quality_gate.json"
MIN_FINAL_LENGTH="${B1_AD_MIN_LENGTH:-200}"
PY="/opt/conda/bin/python"
COLLECT="/workspace/vintix_go2/scripts/collect_ad_data_parallel.py"

mkdir -p "${ROOT}/logs/train_ppo" "${OUT_DIR}" "${ROOT}/results"
export TRAIN_LOG GATE_JSON MIN_FINAL_LENGTH

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${CKPT}"
while [[ ! -f "${CKPT}" ]]; do
  sleep 30
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] checkpoint present"

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

sleep 5
ls -la "${CKPT}"

# Quality gate: do not collect AD from a collapsed / non-walking expert.
set +e
GATE_OUT="$("${PY}" - <<'PY'
import json, re, time, os
from pathlib import Path
log = Path(os.environ["TRAIN_LOG"])
gate_json = Path(os.environ["GATE_JSON"])
min_len = float(os.environ["MIN_FINAL_LENGTH"])
text = log.read_text(errors="replace") if log.exists() else ""
rows = []
for m in re.finditer(r"Learning iteration\s+(\d+)/301", text):
    chunk = text[m.start(): m.start() + 700]
    lm = re.search(r"Mean episode length:\s*([0-9.]+)", chunk)
    rm = re.search(r"Mean total reward:\s*([-0-9.]+)", chunk)
    if lm and rm:
        rows.append((int(m.group(1)), float(lm.group(1)), float(rm.group(1))))
final_len = rows[-1][1] if rows else 0.0
final_iter = rows[-1][0] if rows else -1
peak = max(rows, key=lambda x: x[1]) if rows else (0, 0.0, 0.0)
ok = final_len >= min_len
payload = {
    "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "train_log": str(log),
    "final_iteration": final_iter,
    "final_mean_episode_length": final_len,
    "peak_iteration": peak[0],
    "peak_mean_episode_length": peak[1],
    "min_required_length": min_len,
    "pass": ok,
}
gate_json.write_text(json.dumps(payload, indent=2) + "\n")
print(("PASS" if ok else "FAIL") + f" final_len={final_len:.2f} peak={peak[1]:.2f}@{peak[0]}")
raise SystemExit(0 if ok else 2)
PY
)"
GATE_RC=$?
set -e
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] quality gate: ${GATE_OUT}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wrote ${GATE_JSON}"

if [[ "${GATE_RC}" -ne 0 ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP AD collect (expert not walking enough). See ${GATE_JSON}"
  exit 2
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching AD collect → ${OUT_DIR}"
cd /workspace
exec "${PY}" "${COLLECT}" \
  -r b1 \
  --model_path "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --num_envs 10 \
  --target_steps_per_env 200000 \
  --max_steps 1000 \
  > "${LOG}" 2>&1
