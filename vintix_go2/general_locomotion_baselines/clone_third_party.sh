#!/usr/bin/env bash
# Shallow-clone adopted baselines into third_party/ (not committed).
# Usage: bash clone_third_party.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${ROOT}/third_party"
mkdir -p "${DEST}"

clone_one() {
  local url="$1"
  local name="$2"
  local dir="${DEST}/${name}"
  if [[ -d "${dir}/.git" ]]; then
    echo "[skip] ${name} already present"
    return 0
  fi
  echo "[clone] ${name} <- ${url}"
  git clone --depth 1 "${url}" "${dir}"
  # Record tip SHA for provenance
  git -C "${dir}" rev-parse HEAD > "${dir}/.clone_sha"
  echo "  SHA=$(cat "${dir}/.clone_sha")"
}

clone_one "https://github.com/HybridRobotics/GenLoco.git" "GenLoco"
clone_one "https://github.com/nico-bohlinger/one_policy_to_run_them_all.git" "URMA_one_policy_to_run_them_all"
clone_one "https://github.com/prakharmishra27/morfic-rl.git" "MorFiC"
clone_one "https://github.com/prakharmishra27/McARL.git" "McARL"

echo "Done. Licenses: see SOURCE_LICENSES.md"
