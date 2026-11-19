#!/usr/bin/env python3
"""MorFiC → Genesis evaluation entry (skeleton / dry-run).

Does **not** launch Genesis or allocate GPU by default. ``--dry_run`` (default)
reuses the CPU dimension smoke. Full sim evaluation waits until #18 / #21
release the GPU.

  cd /workspace/vintix_go2/general_locomotion_baselines
  python wrappers/morfic/run_morfic_genesis_eval.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]  # general_locomotion_baselines/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wrappers.morfic.obs_adapter import GenesisToMorficObs, MORFIC_HISTORY_FLAT, MORFIC_OBS_DIM  # noqa: E402
from wrappers.morfic.policy_loader import (  # noqa: E402
    DEFAULT_CKPT,
    assert_dims_match_checkpoint,
    load_ac_weight_shapes,
    smoke_forward_shapes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true", default=True, help="CPU dim check only (default)")
    p.add_argument("--run", action="store_true", help="Reserved: full Genesis eval (blocked until GPU free)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--robot", default="go2")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.run:
        print(
            "ERROR: full Genesis MorFiC eval is reserved until #18/#21 free the GPU.\n"
            "Re-run without --run for CPU-only dry-run.",
            file=sys.stderr,
        )
        return 2

    print("=== MorFiC Genesis eval entry (dry-run) ===")
    print(f"robot={args.robot} num_envs={args.num_envs} max_steps={args.max_steps}")
    print(f"ckpt={args.ckpt} exists={args.ckpt.is_file()}")

    shapes = load_ac_weight_shapes(args.ckpt)
    dims = assert_dims_match_checkpoint(args.ckpt)
    adapter = GenesisToMorficObs(batch_size=args.num_envs, device="cpu")
    fwd = smoke_forward_shapes(ckpt_path=args.ckpt, batch_size=2, steps=5)
    report = {
        "ok": True,
        "mode": "dry_run",
        "planned_eval": {
            "robot": args.robot,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "obs_path": "Genesis45 → GenesisToMorficObs → morph/adapt/actor",
            "metrics": ["completion_rate", "tracking_success_rate", "avg_forward_speed_mps"],
        },
        "dims": dims.__dict__,
        "adapter": {
            "genesis_obs": 45,
            "morfic_step": MORFIC_OBS_DIM,
            "history_flat": MORFIC_HISTORY_FLAT,
            "batch_size": adapter.batch_size,
        },
        "checkpoint_shapes": {k: list(v) for k, v in shapes.items()},
        "forward_shapes": {k: list(v) for k, v in fwd.items()},
        "next": "Implement --run with Genesis Go2 env once GPU is free (#18→#21).",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
