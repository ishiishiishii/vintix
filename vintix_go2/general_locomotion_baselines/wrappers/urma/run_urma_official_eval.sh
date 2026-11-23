#!/usr/bin/env bash
# Bounded URMA official-sim smoke (fixes infinite-loop upstream test).
# Usage:
#   bash wrappers/urma/run_urma_official_eval.sh
#   bash wrappers/urma/run_urma_official_eval.sh --num_episodes 3 --max_steps 2000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_NAME="${URMA_CONDA_ENV:-urma_eval}"
PY="/opt/conda/envs/${ENV_NAME}/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "missing conda env python: $PY" >&2
  exit 1
fi

# Prefer vendored NVIDIA libs if present (GPU may still fail: jaxlib cudnn8.9 vs pip cudnn9).
NVIDIA_ROOT="/opt/conda/envs/${ENV_NAME}/lib/python3.11/site-packages/nvidia"
for d in cudnn cublas cuda_runtime cuda_nvrtc cufft curand cusolver cusparse nccl; do
  lib="${NVIDIA_ROOT}/${d}/lib"
  if [[ -d "$lib" ]]; then
    export LD_LIBRARY_PATH="${lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done

mkdir -p "$ROOT/results/urma_official"
LOG="$ROOT/results/urma_official/run_$(date -u +%Y%m%dT%H%M%SZ).out"
cd "$ROOT"
echo "logging to $LOG"
"$PY" wrappers/urma/run_urma_official_eval.py "$@" 2>&1 | tee "$LOG"
