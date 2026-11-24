#!/usr/bin/env bash
# MorFiC official Isaac Gym playback (#22). Does not touch the Genesis conda env.
#
# Usage:
#   bash wrappers/morfic/run_morfic_official_eval.sh
#   bash wrappers/morfic/run_morfic_official_eval.sh --robot go2 --steps 800 --num-envs 1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_NAME="${ISAAC_ENV:-isaacgym_py38}"
PY="/opt/conda/envs/${ENV_NAME}/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "missing conda env: $ENV_NAME (run install_isaac_gym.sh first)" >&2
  exit 1
fi
# gym_*.so needs libpython from the conda env (conda run does not always source activate.d)
export LD_LIBRARY_PATH="/opt/conda/envs/${ENV_NAME}/lib:${LD_LIBRARY_PATH:-}"

MORFIC="$ROOT/third_party/MorFiC"
OUT_DIR="$ROOT/results/morfic_official"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUT_DIR/run_${STAMP}.out"
JSON="$OUT_DIR/morfic_go2_${STAMP}.json"

ROBOT=go2
STEPS=800
NUM_ENVS=1
VX=1.0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot) ROBOT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --vx) VX="$2"; shift 2 ;;
    --out-json) JSON="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

cd "$MORFIC"
echo "logging to $LOG"
set +e
"$PY" scripts/play.py \
  --robot "$ROBOT" \
  --steps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --vx "$VX" \
  --headless \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG"
PLAY_RC=${PIPESTATUS[0]}
set -e

"$PY" - "$LOG" "$JSON" "$ROBOT" "$STEPS" "$NUM_ENVS" "$VX" "$PLAY_RC" <<'PY'
import json, re, sys
from pathlib import Path
log_path, json_path, robot, steps, num_envs, vx, rc = sys.argv[1:8]
text = Path(log_path).read_text(errors="replace")
def grab(pat, default=None, cast=float):
    m = re.search(pat, text)
    if not m:
        return default
    return cast(m.group(1))
summary = {
    "method": "MorFiC",
    "simulator": "Isaac Gym Preview 4",
    "robot": robot,
    "steps": int(steps),
    "num_envs": int(num_envs),
    "cmd_vx": float(vx),
    "play_exit_code": int(rc),
    "log": str(log_path),
    "mean_vx": grab(r"vx: target=[-\d.]+, mean=([-\d.]+)"),
    "rmse_vx": grab(r"vx: target=[-\d.]+, mean=[-\d.]+, RMSE=([-\d.]+)"),
    "mean_vy": grab(r"vy: target=[-\d.]+, mean=([-\d.]+)"),
    "rmse_vy": grab(r"vy: target=[-\d.]+, mean=[-\d.]+, RMSE=([-\d.]+)"),
    "mean_yaw": grab(r"yaw: target=[-\d.]+, mean=([-\d.]+)"),
    "rmse_yaw": grab(r"yaw: target=[-\d.]+, mean=[-\d.]+, RMSE=([-\d.]+)"),
    "mean_time_to_done_s": grab(r"Mean time to first done or horizon: ([-\d.]+) s"),
    "no_done_fraction_pct": grab(r"No-done fraction: ([-\d.]+)%"),
}
Path(json_path).write_text(json.dumps(summary, indent=2) + "\n")
print("wrote", json_path)
print(json.dumps(summary, indent=2))
sys.exit(0 if int(rc) == 0 else int(rc))
PY
