#!/usr/bin/env python3
"""List local third_party clone status without loading simulators or GPUs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "third_party"

EXPECTED = {
    "GenLoco": {
        "markers": [
            "LICENSE.txt",
            "motion_imitation",
            "motion_imitation/data/policies/morphology_generator_pace_model.zip",
        ],
    },
    "URMA_one_policy_to_run_them_all": {
        "markers": [
            "LICENSE",
            "experiments/pre_trained_model",
            "one_policy_to_run_them_all",
        ],
    },
    "MorFiC": {
        "markers": [
            "LICENSE",
            "go2_gym",
            "runs/gait-conditioned-agility/2026-05-15/train/003739.836224/checkpoints/ac_weights_last.pt",
        ],
    },
    "McARL": {
        "markers": ["LICENSE", "README.md"],
        "code_expected": False,
        "note": "stub repo (docs only); no training code at clone time",
    },
}


def main() -> None:
    rows = []
    for name, spec in EXPECTED.items():
        d = ROOT / name
        sha_path = d / ".clone_sha"
        sha = sha_path.read_text().strip() if sha_path.exists() else None
        missing = [m for m in spec["markers"] if not (d / m).exists()]
        rows.append(
            {
                "name": name,
                "present": d.is_dir(),
                "sha": sha,
                "missing_markers": missing,
                "ok": d.is_dir() and not missing,
                "note": spec.get("note"),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if not all(r["ok"] for r in rows if r["name"] != "McARL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
