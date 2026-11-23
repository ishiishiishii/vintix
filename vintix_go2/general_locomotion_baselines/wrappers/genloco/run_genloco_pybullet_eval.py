#!/usr/bin/env python3
"""GenLoco official-ish PyBullet evaluation (no TensorFlow / Stable-Baselines).

Uses GenLoco's PyBullet imitation env + our NPZ mean-actor
(``GenlocoActorPolicy``). This avoids TF 1.15 (Python 3.7/3.8 only) while
keeping the official simulator and observation pipeline.

  cd /workspace/vintix_go2/general_locomotion_baselines
  python wrappers/genloco/run_genloco_pybullet_eval.py --robot go1 --num_episodes 3

Notes
-----
- Task is motion imitation (phase), not velocity tracking.
- ``go2`` is not in GenLoco's robot list; default robot is ``go1``.
- TensorFlow is stubbed only so ``env_builder`` can import unused randomizers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]
_GENLOCO_ROOT = _PKG_ROOT / "third_party" / "GenLoco"
_OUT_DIR = _PKG_ROOT / "results" / "genloco_pybullet"

if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_GENLOCO_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENLOCO_ROOT))

from wrappers.genloco.policy_loader import (  # noqa: E402
    DEFAULT_CKPT,
    GenlocoActorPolicy,
    assert_dims_match_checkpoint,
)

_POLICY_TO_ZIP = {
    "pace": "morphology_generator_pace_model.zip",
    "spin": "morphology_generator_spin_model.zip",
    "inverse": "morphology_generator_inverse.zip",
}

_ROBOTS = (
    "go1",
    "a1",
    "aliengo",
    "mini_cheetah",
    "anymal_b",
    "anymal_c",
    "spot",
    "spotmicro",
    "laikago",
)


def _install_tensorflow_stub() -> None:
    """env_builder imports TF only for unused train-time randomizers."""
    if "tensorflow" in sys.modules:
        return
    tf = types.ModuleType("tensorflow")
    compat = types.ModuleType("tensorflow.compat")
    v1 = types.ModuleType("tensorflow.compat.v1")
    tf.compat = compat
    compat.v1 = v1
    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.compat"] = compat
    sys.modules["tensorflow.compat.v1"] = v1


def _patch_collections_sequence() -> None:
    """GenLoco uses collections.Sequence (removed in Python 3.10+)."""
    import collections
    import collections.abc

    if not hasattr(collections, "Sequence"):
        collections.Sequence = collections.abc.Sequence  # type: ignore[attr-defined]
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]


def _patch_imitation_task_randint() -> None:
    """Old GenLoco calls ``env.np_random.randint``; gym 0.26 uses Generator."""
    from motion_imitation.envs.env_wrappers import imitation_task

    def _randint(self, val_min, val_max, size=None):  # noqa: ANN001
        rng = self._env.np_random
        if hasattr(rng, "integers"):
            return rng.integers(val_min, val_max, size=size)
        return rng.randint(val_min, val_max, size=size)

    imitation_task.ImitationTask._randint = _randint  # type: ignore[method-assign]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", default="go1", choices=_ROBOTS)
    p.add_argument("--policy", default="pace", choices=tuple(_POLICY_TO_ZIP))
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument(
        "--motion_file",
        default="motion_imitation/data/motions/a1_pace.txt",
        help="Relative to GenLoco root (default matches official run.py)",
    )
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=300, help="Per-episode cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _resolve_ckpt(args: argparse.Namespace) -> Path:
    if args.ckpt is not None:
        return args.ckpt
    return DEFAULT_CKPT.parent / _POLICY_TO_ZIP[args.policy]


def _build_env(args: argparse.Namespace):
    _patch_collections_sequence()
    _install_tensorflow_stub()
    os.chdir(_GENLOCO_ROOT)
    from motion_imitation.envs import env_builder
    from motion_imitation.robots import (  # noqa: F401
        a1,
        aliengo,
        anymal_b_simple,
        anymal_c_simple,
        base_robot,
        go1,
        mini_cheetah,
        spot,
        spotmicro,
    )
    _patch_imitation_task_randint()

    robot_classes = {
        "laikago": base_robot.Base_robot,
        "a1": a1.A1,
        "anymal_b": anymal_b_simple.Anymal_b,
        "anymal_c": anymal_c_simple.Anymal_c,
        "mini_cheetah": mini_cheetah.Mini_cheetah,
        "go1": go1.Go1,
        "aliengo": aliengo.Aliengo,
        "spot": spot.Spot,
        "spotmicro": spotmicro.SpotMicro,
    }
    motion = Path(args.motion_file)
    if not motion.is_file():
        raise FileNotFoundError(f"motion file not found under GenLoco root: {motion}")

    env = env_builder.build_imitation_env(
        motion_files=[str(motion)],
        num_parallel_envs=1,
        mode="test",
        enable_randomizer=False,
        enable_sync_root_rotation=False,
        enable_rendering=False,
        enable_phase_only=True,
        enable_randomized_robot=False,
        robot_class=robot_classes[args.robot],
        visualize=False,
    )
    return env


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = _resolve_ckpt(args)
    assert_dims_match_checkpoint(ckpt)
    policy = GenlocoActorPolicy(ckpt_path=ckpt, device="cpu")

    print("=== GenLoco PyBullet eval (official sim, TF-free actor) ===")
    print(f"robot={args.robot} policy={args.policy} episodes={args.num_episodes}")
    print(f"ckpt={ckpt}")
    print(f"motion={args.motion_file}")

    env = _build_env(args)
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_alive_frac: list[float] = []

    for ep in range(args.num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_r = 0.0
        steps = 0
        done = False
        while not done and steps < args.max_steps:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).view(1, -1)
            if obs_t.shape[-1] != 451:
                raise RuntimeError(f"expected obs dim 451, got {tuple(obs_t.shape)}")
            with torch.no_grad():
                action = policy.act(obs_t).cpu().numpy()[0]
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _info = step_out
                done = bool(done)
            total_r += float(reward)
            steps += 1
        ep_returns.append(total_r)
        ep_lengths.append(steps)
        ep_alive_frac.append(steps / float(args.max_steps))
        print(f"  episode {ep}: return={total_r:.3f} length={steps}")

    report = {
        "ok": True,
        "mode": "pybullet_official_sim",
        "robot": args.robot,
        "policy_zip": args.policy,
        "ckpt": str(ckpt),
        "motion_file": args.motion_file,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "mean_return": float(np.mean(ep_returns)),
        "mean_episode_length": float(np.mean(ep_lengths)),
        "mean_alive_frac_of_max_steps": float(np.mean(ep_alive_frac)),
        "episode_returns": ep_returns,
        "episode_lengths": ep_lengths,
        "task": "motion imitation (phase_only)",
        "notes": (
            "Actor loaded from SB zip NPZ without TensorFlow. "
            "Simulator and obs pipeline are GenLoco PyBullet. "
            "Not velocity-tracking; compare carefully with Vintix metrics."
        ),
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (
        _OUT_DIR
        / f"genloco_{args.policy}_{args.robot}_ep{args.num_episodes}_s{args.max_steps}.json"
    )
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print(
        f"mean_return={report['mean_return']:.3f} "
        f"mean_len={report['mean_episode_length']:.1f} "
        f"alive_frac={report['mean_alive_frac_of_max_steps']:.3f}"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
