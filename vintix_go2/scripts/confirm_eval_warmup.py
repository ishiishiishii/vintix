#!/usr/bin/env python3
"""
Auxiliary evaluation script with PPO warm-up.

- Runs PPO expert for the first N steps to populate the Vintix history buffer.
- Afterwards switches control to a Vintix policy for the remaining steps.
- Optionally records the entire trajectory (including warm-up) to HDF5 so that
  it can be analysed with the existing confirm_* tooling.

This script is strictly for verification/diagnostics; it does not modify any
core training/evaluation code paths.
"""

import argparse
import os
import pickle
import sys
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as exc:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from exc

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
if GENESIS_LOCOMOTION_PATH not in sys.path:
    sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

from env import Go2Env, MiniCheetahEnv, LaikagoEnv
from vintix.vintix import Vintix


class VintixHistoryBuffer:
    """Maintains the rolling observation/action/reward history for Vintix."""

    def __init__(self, max_len: int = 1024) -> None:
        from collections import deque

        self.max_len = max_len
        self.observations = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.step_nums = deque(maxlen=max_len)
        self.current_step = 0

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.step_nums.append(self.current_step)
        self.current_step += 1

    def reset_episode(self, observation: np.ndarray, action_dim: int) -> None:
        """Called when the environment resets to seed history with zeros."""
        zero_action = np.zeros(action_dim, dtype=np.float32)
        self.add(observation, zero_action, 0.0)

    def get_context(self, context_len: int) -> list | None:
        if len(self.observations) == 0:
            return None

        obs_list = list(self.observations)[-context_len:]
        act_list = list(self.actions)[-context_len:]
        rew_list = list(self.rewards)[-context_len:]
        step_list = list(self.step_nums)[-context_len:]

        batch = [{
            "observation": torch.tensor(np.array(obs_list), dtype=torch.float32),
            "prev_action": torch.tensor(np.array(act_list), dtype=torch.float32),
            "prev_reward": torch.tensor(np.array(rew_list), dtype=torch.float32).unsqueeze(1),
            "step_num": torch.tensor(step_list, dtype=torch.int32),
            "task_name": "go2_walking_ad",
        }]
        return batch


class TrajectoryRecorder:
    """Buffers evaluation steps and flushes them to HDF5 on completion."""

    def __init__(self, output_path: Path, warmup_steps: int) -> None:
        self.output_path = output_path
        self.warmup_steps = warmup_steps
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.step_nums: list[int] = []
        self.episode_ids: list[int] = []
        self.warmup_flags: list[bool] = []

    def append(self,
               observation: np.ndarray,
               action: np.ndarray,
               reward: float,
               step_num: int,
               episode_id: int,
               episode_step: int) -> None:
        self.observations.append(observation.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.rewards.append(float(reward))
        self.step_nums.append(int(step_num))
        self.episode_ids.append(int(episode_id))
        self.warmup_flags.append(episode_step < self.warmup_steps if self.warmup_steps > 0 else False)

    def save(self) -> None:
        if len(self.observations) == 0:
            print("Recorder: no steps captured, skipping save.")
            return

        obs = np.stack(self.observations, axis=0)
        act = np.stack(self.actions, axis=0)
        rew = np.array(self.rewards, dtype=np.float32)
        step_nums = np.array(self.step_nums, dtype=np.int32)
        episode_ids = np.array(self.episode_ids, dtype=np.int32)
        warmup_mask = np.array(self.warmup_flags, dtype=np.bool_)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        group_name = f"0-{obs.shape[0] - 1}"
        with h5py.File(self.output_path, "w") as h5:
            grp = h5.create_group(group_name)
            grp.create_dataset("proprio_observation", data=obs)
            grp.create_dataset("action", data=act)
            grp.create_dataset("reward", data=rew)
            grp.create_dataset("step_num", data=step_nums)
            grp.create_dataset("episode_id", data=episode_ids)
            grp.attrs["source"] = "confirm_eval_warmup"

        warmup_path = self.output_path.with_suffix(".warmup.npy")
        np.save(warmup_path, warmup_mask)
        print(f"✓ Recorded trajectory: {self.output_path}")
        print(f"✓ Warmup mask saved : {warmup_path}")


def build_env(robot_type: str,
              env_cfg,
              obs_cfg,
              reward_cfg,
              command_cfg,
              show_viewer: bool) -> Go2Env | MiniCheetahEnv | LaikagoEnv:
    env_kwargs = dict(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    if robot_type == "go2":
        return Go2Env(**env_kwargs)
    if robot_type == "minicheetah":
        return MiniCheetahEnv(**env_kwargs)
    if robot_type == "laikago":
        return LaikagoEnv(**env_kwargs)
    raise ValueError(f"Unknown robot type: {robot_type}")


def load_configs(exp_name: str):
    genesis_root = Path(__file__).parents[2] / "Genesis"
    log_dir = genesis_root / "logs" / exp_name
    cfgs_path = log_dir / "cfgs.pkl"
    if not cfgs_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfgs_path}")
    return log_dir, pickle.load(open(cfgs_path, "rb"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Confirm Vintix evaluation with PPO warm-up (auxiliary tool)"
    )
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                        help="Experiment name under Genesis/logs")
    parser.add_argument("-r", "--robot_type", type=str,
                        choices=["go2", "minicheetah", "laikago"],
                        default="go2", help="Robot type")
    parser.add_argument("--vintix_path", type=str, required=True,
                        help="Directory of the trained Vintix checkpoint")
    parser.add_argument("--ppo_ckpt", type=int, default=300,
                        help="PPO checkpoint index (model_<idx>.pt)")
    parser.add_argument("--context_len", type=int, default=1024,
                        help="Vintix context length")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Total number of steps to simulate (including warm-up)")
    parser.add_argument("--warmup_steps", type=int, default=128,
                        help="Number of PPO warm-up steps before handing control to Vintix")
    parser.add_argument("--record_path", type=str,
                        help="Optional HDF5 output path for trajectory logging")
    parser.add_argument("--show_viewer", action="store_true",
                        help="Display Genesis viewer during evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    gs.init()

    log_dir, cfg_tuple = load_configs(args.exp_name)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg_tuple
    reward_cfg["reward_scales"] = {}

    env = build_env(args.robot_type, env_cfg, obs_cfg, reward_cfg, command_cfg, args.show_viewer)
    print("============================================================")
    print("Confirm Evaluation with PPO Warm-Up")
    print("============================================================")
    print(f"Robot           : {args.robot_type}")
    print(f"PPO checkpoint  : model_{args.ppo_ckpt}.pt")
    print(f"Warm-up steps   : {args.warmup_steps}")
    print(f"Max steps total : {args.max_steps}")
    print(f"Vintix model    : {args.vintix_path}")
    print("============================================================")
    print(f"Observation dim : {env.num_obs}")
    print(f"Action dim      : {env.num_actions}")
    print("============================================================")

    # PPO expert policy for warm-up
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    ppo_path = Path(log_dir) / f"model_{args.ppo_ckpt}.pt"
    if not ppo_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {ppo_path}")
    runner.load(str(ppo_path))
    ppo_policy = runner.get_inference_policy(device=gs.device)

    # Vintix policy
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model.eval()

    history_buffer = VintixHistoryBuffer(max_len=args.context_len)
    recorder = TrajectoryRecorder(Path(args.record_path), args.warmup_steps) if args.record_path else None

    obs, _ = env.reset()
    history_buffer.reset_episode(obs[0].cpu().numpy(), env.num_actions)

    step_count = 0
    episode_id = 0
    episode_step = 0

    def handle_reset():
        nonlocal obs, episode_id, episode_step
        obs, _ = env.reset()
        history_buffer.reset_episode(obs[0].cpu().numpy(), env.num_actions)
        episode_id += 1
        episode_step = 0

    with torch.no_grad():
        # PPO warm-up phase
        warmup_target = min(args.warmup_steps, args.max_steps)
        while step_count < warmup_target:
            action = ppo_policy(obs)
            obs, rew, dones, infos = env.step(action)

            obs_np = obs[0].cpu().numpy()
            act_np = action[0].cpu().numpy()
            rew_val = float(rew.cpu().numpy()[0])

            history_buffer.add(obs_np, act_np, rew_val)

            if recorder:
                recorder.append(obs_np, act_np, rew_val, step_count, episode_id, episode_step)

            step_count += 1
            episode_step += 1

            if dones[0]:
                handle_reset()

        # Switch to Vintix control
        while step_count < args.max_steps:
            context = history_buffer.get_context(args.context_len)
            if context is None:
                action = torch.zeros(1, env.num_actions, device=gs.device)
            else:
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_actions, _ = vintix_model(context)
                if isinstance(pred_actions, list):
                    pred_actions = pred_actions[0]
                if pred_actions.dim() == 3:
                    action = pred_actions[0, -1, :].unsqueeze(0).float()
                elif pred_actions.dim() == 2:
                    action = pred_actions[-1, :].unsqueeze(0).float()
                else:
                    raise ValueError(f"Unexpected pred_actions shape: {pred_actions.shape}")

            obs, rew, dones, infos = env.step(action)

            obs_np = obs[0].cpu().numpy()
            act_np = action[0].cpu().numpy()
            rew_val = float(rew.cpu().numpy()[0])

            history_buffer.add(obs_np, act_np, rew_val)

            if recorder:
                recorder.append(obs_np, act_np, rew_val, step_count, episode_id, episode_step)

            if step_count % 100 == 0 or step_count == args.max_steps - 1:
                print(f"Progress: {step_count + 1}/{args.max_steps} steps "
                      f"(episode {episode_id}, step {episode_step})")

            step_count += 1
            episode_step += 1

            if dones[0]:
                handle_reset()

    if recorder:
        recorder.save()


if __name__ == "__main__":
    main()

