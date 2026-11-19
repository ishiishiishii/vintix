#!/usr/bin/env python3
"""GenLoco → Genesis evaluation entry (skeleton / dry-run).

Default ``--dry_run`` stays CPU-only (no Genesis / GPU). Full ``--run`` waits
until #18 / #21 release the GPU.

  cd /workspace/vintix_go2/general_locomotion_baselines
  python wrappers/genloco/run_genloco_genesis_eval.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wrappers.genloco.obs_adapter import (  # noqa: E402
    GENLOCO_OBS_DIM,
    GenesisToGenlocoObs,
)
from wrappers.genloco.policy_loader import (  # noqa: E402
    DEFAULT_CKPT,
    assert_dims_match_checkpoint,
    load_weight_shapes,
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
    p.add_argument(
        "--policy",
        choices=("pace", "spin", "inverse"),
        default="pace",
        help="Which GenLoco morphology_generator zip to plan for",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.run:
        print(
            "ERROR: full Genesis GenLoco eval is reserved until #18/#21 free the GPU.\n"
            "Re-run without --run for CPU-only dry-run.",
            file=sys.stderr,
        )
        return 2

    print("=== GenLoco Genesis eval entry (dry-run) ===")
    print(f"robot={args.robot} policy={args.policy} num_envs={args.num_envs} max_steps={args.max_steps}")
    print(f"ckpt={args.ckpt} exists={args.ckpt.is_file()}")

    shapes = load_weight_shapes(args.ckpt)
    dims = assert_dims_match_checkpoint(args.ckpt)
    adapter = GenesisToGenlocoObs(batch_size=args.num_envs, device="cpu")
    fwd = smoke_forward_shapes(ckpt_path=args.ckpt, batch_size=2, steps=5)
    report = {
        "ok": True,
        "mode": "dry_run",
        "planned_eval": {
            "robot": args.robot,
            "policy_zip": args.policy,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "obs_path": "Genesis45 → GenesisToGenlocoObs → SB PPO actor (451→1024→512→12)",
            "metrics": ["completion_rate", "tracking_success_rate", "avg_forward_speed_mps"],
            "caveat": "Official GenLoco is motion imitation (phase); velocity-tracking metrics need remapping",
        },
        "dims": dims.__dict__,
        "adapter": {
            "genesis_obs": 45,
            "genloco_flat": GENLOCO_OBS_DIM,
            "batch_size": adapter.batch_size,
        },
        "checkpoint_shapes": {k: list(v) for k, v in shapes.items()},
        "forward_shapes": {k: list(v) for k, v in fwd.items()},
        "next": "Implement --run with Genesis Go2 env once GPU is free (#18→#21). Compare to official PyBullet play.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
