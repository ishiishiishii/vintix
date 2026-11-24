#!/usr/bin/env bash
# Install NVIDIA Isaac Gym Preview into an isolated conda env for MorFiC (#22).
# Does NOT touch the main Genesis conda env.
#
# Usage:
#   1) Download Isaac Gym Preview 4 from NVIDIA Developer (license required)
#   2) Place the tarball somewhere, e.g. /workspace/downloads/IsaacGym_Preview_4_Package.tar.gz
#   3) bash install_isaac_gym.sh /path/to/IsaacGym_Preview_*.tar.gz
set -euo pipefail
TARBALL="${1:-}"
if [[ -z "${TARBALL}" || ! -f "${TARBALL}" ]]; then
  echo "ERROR: pass path to Isaac Gym Preview tarball (NVIDIA download; not redistributable)."
  echo "Example: $0 /workspace/downloads/IsaacGym_Preview_4_Package.tar.gz"
  exit 1
fi
ENV_NAME="${ISAAC_ENV:-isaacgym_py38}"
PREFIX="/opt/conda/envs/${ENV_NAME}"
WORK="/workspace/vintix_go2/general_locomotion_baselines/third_party/isaacgym_preview"
mkdir -p "$(dirname "${WORK}")"
rm -rf "${WORK}"
mkdir -p "${WORK}"
case "${TARBALL}" in
  *.tar.gz|*.tgz) tar -xzf "${TARBALL}" -C "${WORK}" ;;
  *.tar)          tar -xf  "${TARBALL}" -C "${WORK}" ;;
  *) echo "ERROR: unsupported archive type: ${TARBALL}"; exit 1 ;;
esac
# tarball usually unpacks to isaacgym/
IG_ROOT="$(find "${WORK}" -maxdepth 2 -type d -name isaacgym | head -1)"
if [[ -z "${IG_ROOT}" ]]; then
  echo "Could not find isaacgym/ after extract"; ls -la "${WORK}"; exit 1
fi
echo "Isaac Gym root: ${IG_ROOT}"
# Create py3.8 env if missing (MorFiC README: py3.8 + torch 1.10 era)
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.8
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
# Preview 4 works with older torch; prefer cu117 then fall back.
python -m pip install 'numpy<2' pyyaml
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  || python -m pip install torch==1.10.2 torchvision==0.11.3
python -m pip install -e "${IG_ROOT}/python"
# gym_*.so needs libpython from the conda env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# Persist for conda run / interactive shells
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/isaacgym_ld.sh" <<'EOF'
export _ISAACGYM_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/isaacgym_ld.sh" <<'EOF'
export LD_LIBRARY_PATH="${_ISAACGYM_OLD_LD_LIBRARY_PATH:-}"
unset _ISAACGYM_OLD_LD_LIBRARY_PATH
EOF
# Import order: isaacgym before torch (NVIDIA requirement). Smoke without creating sim.
python - <<'PY'
import isaacgym  # noqa: F401 — must import before torch
import torch
print("isaacgym import OK", isaacgym.__file__)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY
echo "Installed into conda env: ${ENV_NAME}"
echo "Activate with: conda activate ${ENV_NAME}  # sets LD_LIBRARY_PATH via activate.d"
echo "Next: cd third_party/MorFiC && conda run -n ${ENV_NAME} python scripts/play.py --robot go2 --vx 1.0 --steps 800 --headless"
# Optional: install MorFiC Python package into this env (after Isaac Gym OK)
#   cd third_party/MorFiC
#   pip install 'setuptools<70' 'numpy==1.23.5' 'gym==0.23.1' tqdm matplotlib \
#     'params-proto==2.10.5' 'ml_logger==0.8.117' 'ml_dash==0.3.20' 'jaynes>=0.9.2'
#   pip install -e . --no-deps
# Then: bash wrappers/morfic/run_morfic_official_eval.sh
