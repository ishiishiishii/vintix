#!/usr/bin/env python3
"""Bounded official-sim eval for URMA pretrained weights (Issue #22).

Upstream ``PPO.test`` has a ``while True`` that never breaks on ``done``,
so ``experiments/test.sh`` runs forever. This entrypoint monkeypatches
``test`` to stop on termination / truncation and on ``--max_steps``.

Default: ``multi_robot`` with **Go2 only** (1 env), 3 episodes, 4000-step cap
(official episode_length is 20 s × 0.005 s = 4000).

  cd /workspace/vintix_go2/general_locomotion_baselines
  bash wrappers/urma/run_urma_official_eval.sh
  # or:
  /opt/conda/envs/urma_eval/bin/python wrappers/urma/run_urma_official_eval.py \\
      --num_episodes 3 --max_steps 4000

Writes ``results/urma_official/urma_<tag>_epN_sM.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

_PKG_ROOT = Path("/workspace/vintix_go2/general_locomotion_baselines")
_URMA_ROOT = _PKG_ROOT / "third_party" / "URMA_one_policy_to_run_them_all"
_RLX_ROOT = _PKG_ROOT / "third_party" / "RL-X"
_OUT_DIR = _PKG_ROOT / "results" / "urma_official"
_EXPERIMENTS = _URMA_ROOT / "experiments"


def _prepend_sys_path() -> None:
    for p in (_RLX_ROOT, _URMA_ROOT, _PKG_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _restrict_multi_robot_types(robots: tuple[str, ...]) -> None:
    """Shrink default 16-type list so a Go2-only CPU smoke is feasible.

    Must re-register: RL-X stores the get_config callable at import time.
    """
    # Trigger registration of multi_robot (and siblings) first.
    import one_policy_to_run_them_all.environments.multi_robot  # noqa: F401
    from one_policy_to_run_them_all.environments.multi_robot import default_config as dc
    from one_policy_to_run_them_all.environments.multi_robot.create_env import create_env
    from one_policy_to_run_them_all.environments.multi_robot.general_properties import (
        GeneralProperties,
    )
    from rl_x.environments.environment_manager import register_environment

    _orig = dc.get_config

    def get_config(environment_name):
        cfg = _orig(environment_name)
        cfg.train_robot_types = robots
        cfg.eval_robot_types = ()
        return cfg

    dc.get_config = get_config
    register_environment("multi_robot", get_config, create_env, GeneralProperties)


def _patch_test(max_steps: int, metrics_out: dict) -> None:
    """Replace infinite-loop ``PPO.test`` with a bounded version."""
    import jax
    import jax.numpy as jnp
    from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.ppo import PPO

    def bounded_test(self, episodes):
        @jax.jit
        def get_action(policy_state, state):
            @partial(jax.jit, static_argnums=(1, 2))
            def get_action_single(state, env_id_0, env_id_1):
                state_b = state[env_id_0:env_id_1]
                dynamic_joint_combined_state_b = state_b[
                    :, : self.dynamic_joint_observation_lengths[env_id_0]
                ].reshape(
                    (
                        -1,
                        self.nr_dynamic_joint_observations[env_id_0],
                        self.single_dynamic_joint_observation_length,
                    )
                )
                dynamic_joint_description_b = dynamic_joint_combined_state_b[
                    :, :, : self.dynamic_joint_description_size
                ]
                dynamic_joint_state_b = dynamic_joint_combined_state_b[
                    :, :, self.dynamic_joint_description_size :
                ]
                dynamic_foot_combined_state_b = state_b[
                    :,
                    self.dynamic_joint_observation_lengths[env_id_0] : self.dynamic_joint_observation_lengths[
                        env_id_0
                    ]
                    + self.dynamic_foot_observation_lengths[env_id_0],
                ].reshape(
                    (
                        -1,
                        self.nr_dynamic_foot_observations[env_id_0],
                        self.single_dynamic_foot_observation_length,
                    )
                )
                dynamic_foot_description_b = dynamic_foot_combined_state_b[
                    :, :, : self.dynamic_foot_description_size
                ]
                dynamic_foot_state_b = dynamic_foot_combined_state_b[
                    :, :, self.dynamic_foot_description_size :
                ]
                action_mean_b, _ = self.policy.apply(
                    policy_state.params,
                    dynamic_joint_description_b,
                    dynamic_joint_state_b,
                    dynamic_foot_description_b,
                    dynamic_foot_state_b,
                    state_b[:, self.policy_general_state_mask[env_id_0]],
                )
                action = jnp.concatenate(
                    [
                        action_mean_b,
                        jnp.zeros(
                            (self.nr_envs_per_env_type, self.missing_nr_of_actions[env_id_0])
                        ),
                    ],
                    axis=1,
                )
                return self.get_processed_action(action)

            processed_actions = []
            for i in range(self.env_ids.shape[0] - 1):
                processed_actions.append(
                    get_action_single(state, self.env_ids[i], self.env_ids[i + 1])
                )
            return jnp.concatenate(processed_actions, axis=0)

        self.set_eval_mode()
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        for i in range(episodes):
            episode_return = 0.0
            episode_length = 0
            state, _ = self.env.reset()
            while episode_length < max_steps:
                processed_action = get_action(self.policy_state, state)
                state, reward, terminated, truncated, _info = self.env.step(
                    jax.device_get(processed_action)
                )
                r = float(np.asarray(reward).mean())
                episode_return += r
                episode_length += 1
                done = np.asarray(terminated) | np.asarray(truncated)
                if bool(np.asarray(done).all()):
                    break
            ep_returns.append(float(episode_return))
            ep_lengths.append(int(episode_length))
            print(
                f"[urma_bounded] episode {i + 1}/{episodes} "
                f"return={episode_return:.3f} length={episode_length}",
                flush=True,
            )

        metrics_out["episode_returns"] = ep_returns
        metrics_out["episode_lengths"] = ep_lengths
        metrics_out["mean_return"] = float(np.mean(ep_returns)) if ep_returns else None
        metrics_out["mean_length"] = float(np.mean(ep_lengths)) if ep_lengths else None
        metrics_out["survived_full_horizon_frac"] = (
            float(np.mean([L >= max_steps for L in ep_lengths])) if ep_lengths else None
        )
        # Write immediately: vector-env teardown can hang after test returns.
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        mid = metrics_out.get("_out_path")
        if mid:
            Path(mid).write_text(json.dumps(metrics_out, indent=2) + "\n")
            print(f"[urma_bounded] wrote {mid}", flush=True)

    PPO.test = bounded_test


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="multi_robot", help="RL-X environment name")
    p.add_argument(
        "--robots",
        default="unitree_go2",
        help="Comma-separated robot types (patches multi_robot default list)",
    )
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument(
        "--nr_envs",
        type=int,
        default=0,
        help="Total envs; 0 = one env per robot in --robots",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--load_model",
        default="pre_trained_model",
        help="Relative to experiments/ (official pretrained dir name)",
    )
    args = p.parse_args()

    robots = tuple(r.strip() for r in args.robots.split(",") if r.strip())
    if not robots:
        raise SystemExit("--robots must list at least one robot type")
    nr_envs = args.nr_envs if args.nr_envs > 0 else len(robots)

    _prepend_sys_path()
    if args.env == "multi_robot":
        _restrict_multi_robot_types(robots)
    os.chdir(_EXPERIMENTS)

    tag = "go2" if robots == ("unitree_go2",) else args.env
    out = _OUT_DIR / f"urma_{tag}_ep{args.num_episodes}_s{args.max_steps}.json"
    metrics: dict = {
        "method": "URMA",
        "eval_place": "official RL-X / MuJoCo (bounded wrapper)",
        "environment": args.env,
        "robots": list(robots),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "nr_envs": nr_envs,
        "seed": args.seed,
        "device_note": "Forced algorithm.device=cpu (urma_eval jaxlib wants cudnn8.9; pip has cudnn9)",
        "upstream_bug": (
            "PPO.test while True never breaks on done; this script monkeypatches it"
        ),
        "note_horizon": (
            "Go2 control step ≈0.02 s; official 20 s episode ≈1000 steps "
            "(physics dt 0.005 with decimation)"
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_out_path": str(out),
    }

    _patch_test(args.max_steps, metrics)

    argv = [
        "run_urma_official_eval.py",
        "--algorithm.name=uni_ppo.ppo",
        f"--environment.name={args.env}",
        "--runner.track_console=True",
        f"--runner.load_model={args.load_model}",
        "--algorithm.determine_fastest_cpu_for_gpu=False",
        "--runner.mode=test",
        "--environment.mode=test",
        f"--environment.nr_envs={nr_envs}",
        "--environment.render=False",
        "--environment.add_goal_arrow=False",
        f"--environment.seed={args.seed}",
        f"--runner.nr_test_episodes={args.num_episodes}",
        "--algorithm.device=cpu",
    ]
    if args.env == "multi_robot":
        argv.append("--environment.multi_render=False")
    sys.argv = argv

    t0 = time.time()
    from rl_x.runner.runner import Runner

    runner = Runner(implementation_package_names=["rl_x", "one_policy_to_run_them_all"])
    runner.run()
    metrics["wall_time_sec"] = round(time.time() - t0, 1)
    metrics["finished_at"] = datetime.now(timezone.utc).isoformat()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"urma_{tag}_ep{args.num_episodes}_s{args.max_steps}.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[urma_bounded] wrote {out}", flush=True)
    print(
        json.dumps(
            {
                k: metrics.get(k)
                for k in ("mean_return", "mean_length", "survived_full_horizon_frac")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
