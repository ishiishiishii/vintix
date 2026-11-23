#!/usr/bin/env python3
"""GenLoco → Genesis evaluation entry.

Default ``--dry_run`` stays CPU-only (no Genesis / GPU). ``--run`` opens a
Genesis Go2 env, maps obs through ``GenesisToGenlocoObs``, and rolls out the
pretrained SB PPO mean-actor (approximate obs — see NOTES.md).

  cd /workspace/vintix_go2/general_locomotion_baselines
  python wrappers/genloco/run_genloco_genesis_eval.py --dry_run
  python wrappers/genloco/run_genloco_genesis_eval.py --run --num_envs 4 --max_steps 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]
_REPO = _PKG_ROOT.parents[1]
_LOCO = _REPO / "Genesis" / "examples" / "locomotion"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wrappers.genloco.obs_adapter import (  # noqa: E402
    GENLOCO_OBS_DIM,
    GenesisToGenlocoObs,
)
from wrappers.genloco.policy_loader import (  # noqa: E402
    DEFAULT_CKPT,
    POLICY_ZIPS,
    GenlocoActorPolicy,
    assert_dims_match_checkpoint,
    load_weight_shapes,
    smoke_forward_shapes,
)

OUT_DIR = _PKG_ROOT / "results" / "genloco_genesis"

_POLICY_TO_ZIP = {
    "pace": "morphology_generator_pace_model.zip",
    "spin": "morphology_generator_spin_model.zip",
    "inverse": "morphology_generator_inverse.zip",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true", default=True, help="CPU dim check only (default)")
    p.add_argument("--run", action="store_true", help="Full Genesis Go2 eval (GPU)")
    p.add_argument("--ckpt", type=Path, default=None, help="Override policy zip path")
    p.add_argument("--robot", default="go2", choices=("go2",), help="Only Go2 wired for now")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument(
        "--policy",
        choices=("pace", "spin", "inverse"),
        default="pace",
        help="Which GenLoco morphology_generator zip",
    )
    p.add_argument("--out", type=Path, default=None, help="JSON report path")
    return p.parse_args()


def _resolve_ckpt(args: argparse.Namespace) -> Path:
    if args.ckpt is not None:
        return args.ckpt
    return DEFAULT_CKPT.parent / _POLICY_TO_ZIP[args.policy]


def _dry_run(args: argparse.Namespace) -> int:
    ckpt = _resolve_ckpt(args)
    print("=== GenLoco Genesis eval entry (dry-run) ===")
    print(f"robot={args.robot} policy={args.policy} num_envs={args.num_envs} max_steps={args.max_steps}")
    print(f"ckpt={ckpt} exists={ckpt.is_file()}")

    shapes = load_weight_shapes(ckpt)
    dims = assert_dims_match_checkpoint(ckpt)
    adapter = GenesisToGenlocoObs(batch_size=args.num_envs, device="cpu")
    fwd = smoke_forward_shapes(ckpt_path=ckpt, batch_size=2, steps=5)
    report = {
        "ok": True,
        "mode": "dry_run",
        "planned_eval": {
            "robot": args.robot,
            "policy_zip": args.policy,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "obs_path": "Genesis45 → GenesisToGenlocoObs → SB PPO actor (451→1024→512→12)",
            "metrics": ["completion_rate", "mean_episode_length", "mean_reward_per_step"],
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
        "available_zips": list(POLICY_ZIPS),
        "next": "python wrappers/genloco/run_genloco_genesis_eval.py --run",
    }
    print(json.dumps(report, indent=2))
    return 0


def _run_genesis(args: argparse.Namespace) -> int:
    if str(_LOCO) not in sys.path:
        sys.path.insert(0, str(_LOCO))

    import genesis as gs  # noqa: WPS433
    from env import Go2Env  # noqa: WPS433
    from train import get_go2_cfgs  # noqa: WPS433

    ckpt = _resolve_ckpt(args)
    if not gs._initialized:
        gs.init(backend=gs.gpu)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    env_cfg = dict(env_cfg)
    env_cfg["episode_length_s"] = max(args.max_steps * 0.02, 1.0)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )
    device = gs.device
    policy = GenlocoActorPolicy(ckpt_path=ckpt, device=device)
    adapter = GenesisToGenlocoObs(batch_size=args.num_envs, device=device)

    obs, _extras = env.get_observations()
    adapter.reset()
    episode_len = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    ep_lengths: list[int] = []
    reward_sum = 0.0
    reward_count = 0
    max_ep = int(env.max_episode_length)

    for _step in range(args.max_steps):
        packed = adapter.step(obs)
        actions = policy.act(packed["obs_flat"])
        obs, rewards, dones, _infos = env.step(actions)
        reward_sum += float(rewards.mean().item())
        reward_count += 1
        episode_len += 1

        done_mask = dones.bool() if dones.dtype != torch.bool else dones
        if done_mask.any():
            idxs = done_mask.nonzero(as_tuple=False).flatten()
            for i in idxs.tolist():
                ep_lengths.append(int(episode_len[i].item()))
                episode_len[i] = 0
            adapter.reset()
            packed = adapter.step(obs)

    for i in range(args.num_envs):
        length = int(episode_len[i].item())
        if length > 0:
            ep_lengths.append(length)

    n_eps = max(len(ep_lengths), 1)
    completion_rate = float(sum(1 for L in ep_lengths if L >= max_ep) / n_eps)
    mean_ep_len = float(sum(ep_lengths) / n_eps) if ep_lengths else 0.0
    report = {
        "ok": True,
        "mode": "run",
        "robot": args.robot,
        "policy": args.policy,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "max_episode_length": max_ep,
        "ckpt": str(ckpt),
        "caveat": (
            "Genesis45→451 obs is approximate (IMU R/P from gravity, Y=0; synthetic phase). "
            "Official GenLoco is motion imitation in PyBullet — not binary-parity; "
            "velocity-tracking metrics are remapped to completion/length only."
        ),
        "metrics": {
            "n_episodes": len(ep_lengths),
            "completion_rate": completion_rate,
            "mean_episode_length": mean_ep_len,
            "mean_reward_per_step": reward_sum / max(reward_count, 1),
            "episode_lengths": ep_lengths,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (
        OUT_DIR / f"genloco_{args.policy}_go2_ne{args.num_envs}_s{args.max_steps}.json"
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


def main() -> int:
    args = parse_args()
    if args.run:
        return _run_genesis(args)
    return _dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
