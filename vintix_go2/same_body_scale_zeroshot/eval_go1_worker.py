#!/usr/bin/env python3
"""Single-point Go1 same-body Vintix eval worker for eval_go1_pipeline.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
RB_DIR = VINTIX_ROOT / "robot_body_scale_zeroshot"
sys.path.insert(0, str(RB_DIR))

import run_experiment as rx  # noqa: E402
from robots import ROBOTS  # noqa: E402

GO1_SAME = EXP_DIR / "models" / "go1_data1M_5epochs" / "go1_data1M_5epochs" / "0004_epoch"
rx.RESULTS_DIR = EXP_DIR / "results"
rx.LOGS_DIR = EXP_DIR / "logs" / "eval"
ROBOTS["go1"]["vintix"] = GO1_SAME


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=["mass", "size", "size_density"])
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--max_episodes", type=int, default=10)
    args = p.parse_args()
    rx.worker_eval("go1", "vintix", args.axis, args.scale, args.seed, args.num_envs, args.max_episodes)


if __name__ == "__main__":
    main()
