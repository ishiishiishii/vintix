#!/usr/bin/env python3
"""MorFiC → Genesis evaluation entry.

Default ``--dry_run`` stays CPU-only (no Genesis / GPU). ``--run`` opens a
Genesis Go2 env, maps obs through ``GenesisToMorficObs``, and rolls out the
pretrained actor path (approximate obs reconstruction — see NOTES.md).

  cd /workspace/vintix_go2/general_locomotion_baselines
  python wrappers/morfic/run_morfic_genesis_eval.py --dry_run
  python wrappers/morfic/run_morfic_genesis_eval.py --run --num_envs 4 --max_steps 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]  # general_locomotion_baselines/
_REPO = _PKG_ROOT.parents[1]
_LOCO = _REPO / "Genesis" / "examples" / "locomotion"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wrappers.morfic.obs_adapter import GenesisToMorficObs, MORFIC_HISTORY_FLAT, MORFIC_OBS_DIM  # noqa: E402
from wrappers.morfic.policy_loader import (  # noqa: E402
    DEFAULT_CKPT,
    MorficActorPolicy,
    assert_dims_match_checkpoint,
    load_ac_weight_shapes,
    smoke_forward_shapes,
)

OUT_DIR = _PKG_ROOT / "results" / "morfic_genesis"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true", default=True, help="CPU dim check only (default)")
    p.add_argument("--run", action="store_true", help="Full Genesis Go2 eval (GPU)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--robot", default="go2", choices=("go2",), help="Only Go2 wired for now")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None, help="JSON report path")
    return p.parse_args()


def _dry_run(args: argparse.Namespace) -> int:
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
            "metrics": ["completion_rate", "mean_episode_length", "mean_reward_per_step"],
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
        "next": "python wrappers/morfic/run_morfic_genesis_eval.py --run",
    }
    print(json.dumps(report, indent=2))
    return 0


def _run_genesis(args: argparse.Namespace) -> int:
    if str(_LOCO) not in sys.path:
        sys.path.insert(0, str(_LOCO))

    import genesis as gs  # noqa: WPS433
    from env import Go2Env  # noqa: WPS433
    from train import get_go2_cfgs  # noqa: WPS433

    if not gs._initialized:
        gs.init(backend=gs.gpu)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    # Shorten episode to max_steps for a bounded smoke/eval.
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
    policy = MorficActorPolicy(ckpt_path=args.ckpt, device=device)
    adapter = GenesisToMorficObs(batch_size=args.num_envs, device=device)

    obs, _extras = env.get_observations()
    adapter.reset()
    episode_len = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    ep_lengths: list[int] = []
    reward_sum = 0.0
    reward_count = 0
    max_ep = int(env.max_episode_length)

    for _step in range(args.max_steps):
        packed = adapter.step(obs)
        actions = policy.act(packed["obs_history_flat"])
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
            # Full history reset on any done (simple; per-env reset can refine later).
            adapter.reset()
            packed = adapter.step(obs)

    # Envs still running at cutoff: count current length as an episode.
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
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "max_episode_length": max_ep,
        "ckpt": str(args.ckpt),
        "caveat": (
            "Genesis45→70 obs is approximate (gait cmds zero-padded; synthetic clock). "
            "Not binary-parity with Isaac Gym MorFiC play."
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
    out = args.out or (OUT_DIR / f"morfic_go2_ne{args.num_envs}_s{args.max_steps}.json")
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
