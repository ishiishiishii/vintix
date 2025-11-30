#!/usr/bin/env python3
"""
Train PPO (Genesis locomotion) while recording the actual training trajectory.

This script is based on Genesis/examples/locomotion/train.py but extends the
runner so that every environment step used during PPO training is written to an
HDF5 file. The resulting dataset mirrors Algorithm Distillation data, except it
captures the genuine learning trajectory (no synthetic epsilon interpolation).

Usage example:

    PYTHONPATH=/workspace/vintix_go2:/workspace/Genesis/examples/locomotion \\
    python scripts/train_with_history.py \\
        -e go2-walking-history \\
        -r go2 \\
        --num_envs 1024 \\
        --max_iterations 301 \\
        --record_output data/go2_trajectories/go2_ppo_history
"""

import argparse
import os
import pickle
import shutil
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Optional

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
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from exc

from rsl_rl.runners import OnPolicyRunner

import genesis as gs  # type: ignore

from env import Go2Env  # type: ignore
from env import LaikagoEnv  # type: ignore
from env import MiniCheetahEnv  # type: ignore


# -----------------------------------------------------------------------------
# PPO config helpers (copied from Genesis/examples/locomotion/train.py)
# -----------------------------------------------------------------------------

def get_train_cfg(exp_name, max_iterations, seed=1):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 10,
        "empirical_normalization": None,
        "seed": seed,
    }

    return train_cfg_dict


def get_go2_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_minicheetah_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "torso_to_abduct_fl_j": 0.0,
            "torso_to_abduct_fr_j": 0.0,
            "torso_to_abduct_hr_j": 0.0,
            "torso_to_abduct_hl_j": 0.0,
            "abduct_fl_to_thigh_fl_j": -0.8,
            "abduct_fr_to_thigh_fr_j": -0.8,
            "abduct_hr_to_thigh_hr_j": -0.8,
            "abduct_hl_to_thigh_hl_j": -0.8,
            "thigh_fl_to_knee_fl_j": 1.5,
            "thigh_fr_to_knee_fr_j": 1.5,
            "thigh_hr_to_knee_hr_j": 1.5,
            "thigh_hl_to_knee_hl_j": 1.5,
        },
        "joint_names": [
            "torso_to_abduct_fr_j",
            "abduct_fr_to_thigh_fr_j",
            "thigh_fr_to_knee_fr_j",
            "torso_to_abduct_fl_j",
            "abduct_fl_to_thigh_fl_j",
            "thigh_fl_to_knee_fl_j",
            "torso_to_abduct_hr_j",
            "abduct_hr_to_thigh_hr_j",
            "thigh_hr_to_knee_hr_j",
            "torso_to_abduct_hl_j",
            "abduct_hl_to_thigh_hl_j",
            "thigh_hl_to_knee_hl_j",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.45],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_laikago_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FR_hip_motor_2_chassis_joint": 0.0,
            "FR_upper_leg_2_hip_motor_joint": -0.8,
            "FR_lower_leg_2_upper_leg_joint": 1.5,
            "FL_hip_motor_2_chassis_joint": 0.0,
            "FL_upper_leg_2_hip_motor_joint": -0.8,
            "FL_lower_leg_2_upper_leg_joint": 1.5,
            "RR_hip_motor_2_chassis_joint": 0.0,
            "RR_upper_leg_2_hip_motor_joint": -0.8,
            "RR_lower_leg_2_upper_leg_joint": 1.5,
            "RL_hip_motor_2_chassis_joint": 0.0,
            "RL_upper_leg_2_hip_motor_joint": -0.8,
            "RL_lower_leg_2_upper_leg_joint": 1.5,
        },
        "joint_names": [
            "FR_hip_motor_2_chassis_joint",
            "FR_upper_leg_2_hip_motor_joint",
            "FR_lower_leg_2_upper_leg_joint",
            "FL_hip_motor_2_chassis_joint",
            "FL_upper_leg_2_hip_motor_joint",
            "FL_lower_leg_2_upper_leg_joint",
            "RR_hip_motor_2_chassis_joint",
            "RR_upper_leg_2_hip_motor_joint",
            "RR_lower_leg_2_upper_leg_joint",
            "RL_hip_motor_2_chassis_joint",
            "RL_upper_leg_2_hip_motor_joint",
            "RL_lower_leg_2_upper_leg_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.45],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.45,
        "feet_height_target": 0.095,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


# -----------------------------------------------------------------------------
# History recorder
# -----------------------------------------------------------------------------

class TrainingHistoryRecorder:
    """Collect and save PPO training rollouts as HDF5 groups."""

    def __init__(
        self,
        output_dir: Path,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        group_size: int = 50000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = group_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs

        self.h5_path = self.output_dir / "trajectories_0000.h5"
        self.h5_file = h5py.File(self.h5_path, "w")

        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_steps = []
        self.buffer_env_ids = []
        self.buffer_episode_ids = []
        self.buffer_iterations = []
        self.buffer_size = 0

        self.step_counters = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_ids = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_reward_running = np.zeros(self.num_envs, dtype=np.float64)

        self.episode_lengths = []
        self.episode_rewards = []

        self.total_transitions_written = 0
        self.total_transitions_seen = 0
        self.max_iteration = -1

        self.obs_sum = np.zeros(self.obs_dim, dtype=np.float64)
        self.obs_sq_sum = np.zeros(self.obs_dim, dtype=np.float64)
        self.action_sum = np.zeros(self.action_dim, dtype=np.float64)
        self.action_sq_sum = np.zeros(self.action_dim, dtype=np.float64)
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

        self.closed = False

    def record_batch(self, iteration: int, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """Record one PPO rollout step for all envs."""
        if self.closed:
            raise RuntimeError("Recorder is already closed.")

        # Ensure shapes: obs/actions -> (num_envs, dim), rewards/dones -> (num_envs, 1) or (num_envs,)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        if obs.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {obs.shape[0]}")

        step_nums = self.step_counters.copy()
        episode_ids = self.episode_ids.copy()
        env_ids = np.arange(self.num_envs, dtype=np.int32)
        iter_array = np.full((self.num_envs, 1), iteration, dtype=np.int32)

        # Update running sums for stats
        self.obs_sum += obs.sum(axis=0)
        self.obs_sq_sum += np.square(obs).sum(axis=0)
        self.action_sum += actions.sum(axis=0)
        self.action_sq_sum += np.square(actions).sum(axis=0)
        reward_scalar = rewards.astype(np.float64).flatten()
        self.reward_sum += reward_scalar.sum()
        self.reward_sq_sum += np.square(reward_scalar).sum()

        # Episode bookkeeping
        self.episode_reward_running += reward_scalar

        self.buffer_obs.append(obs.astype(np.float32))
        self.buffer_actions.append(actions.astype(np.float32))
        self.buffer_rewards.append(rewards.astype(np.float32))
        self.buffer_steps.append(step_nums.astype(np.int32))
        self.buffer_env_ids.append(env_ids.astype(np.int32))
        self.buffer_episode_ids.append(episode_ids.astype(np.int32))
        self.buffer_iterations.append(iter_array.astype(np.int32))

        self.buffer_size += obs.shape[0]
        self.total_transitions_seen += obs.shape[0]
        self.max_iteration = max(self.max_iteration, iteration)

        # Increment step counters AFTER storing current step
        self.step_counters += 1

        # Handle done environments
        done_indices = np.where(dones.squeeze(-1) > 0.0)[0]
        if done_indices.size > 0:
            for idx in done_indices:
                self.episode_lengths.append(int(self.step_counters[idx]))
                self.episode_rewards.append(float(self.episode_reward_running[idx]))
                self.step_counters[idx] = 0
                self.episode_reward_running[idx] = 0.0
                self.episode_ids[idx] += 1

        if self.buffer_size >= self.group_size:
            self._flush_buffer()

    def _flush_buffer(self):
        if self.buffer_size == 0:
            return

        obs = np.concatenate(self.buffer_obs, axis=0)
        acts = np.concatenate(self.buffer_actions, axis=0)
        rews = np.concatenate(self.buffer_rewards, axis=0)
        steps = np.concatenate(self.buffer_steps, axis=0)
        env_ids = np.concatenate(self.buffer_env_ids, axis=0)
        episode_ids = np.concatenate(self.buffer_episode_ids, axis=0)
        iterations = np.concatenate(self.buffer_iterations, axis=0)

        idx = 0
        total = obs.shape[0]
        while idx < total:
            chunk_end = min(idx + self.group_size, total)
            start_idx = self.total_transitions_written
            end_idx = start_idx + (chunk_end - idx) - 1
            group_name = f"{start_idx}-{end_idx}"
            group = self.h5_file.create_group(group_name)
            group.create_dataset("proprio_observation", data=obs[idx:chunk_end], compression="gzip")
            group.create_dataset("action", data=acts[idx:chunk_end], compression="gzip")
            group.create_dataset("reward", data=rews[idx:chunk_end], compression="gzip")
            group.create_dataset("step_num", data=steps[idx:chunk_end], compression="gzip")
            group.create_dataset("env_id", data=env_ids[idx:chunk_end], compression="gzip")
            group.create_dataset("episode_id", data=episode_ids[idx:chunk_end], compression="gzip")
            group.create_dataset("iteration", data=iterations[idx:chunk_end], compression="gzip")

            self.total_transitions_written += chunk_end - idx
            idx = chunk_end

        # Reset buffers with leftover data (should be zero after loop)
        leftover = obs[idx:]
        if leftover.size:
            self.buffer_obs = [obs[idx:]]
            self.buffer_actions = [acts[idx:]]
            self.buffer_rewards = [rews[idx:]]
            self.buffer_steps = [steps[idx:]]
            self.buffer_env_ids = [env_ids[idx:]]
            self.buffer_episode_ids = [episode_ids[idx:]]
            self.buffer_iterations = [iterations[idx:]]
            self.buffer_size = leftover.shape[0]
        else:
            self.buffer_obs = []
            self.buffer_actions = []
            self.buffer_rewards = []
            self.buffer_steps = []
            self.buffer_env_ids = []
            self.buffer_episode_ids = []
            self.buffer_iterations = []
            self.buffer_size = 0

        self.h5_file.flush()

    def finalize(self):
        if self.closed:
            return
        # Flush buffers
        self._flush_buffer()
        # Close HDF5
        self.h5_file.close()
        self.closed = True

        # Compute statistics
        n = max(self.total_transitions_written, 1)
        obs_mean = (self.obs_sum / n).tolist()
        obs_var = (self.obs_sq_sum / n) - np.square(self.obs_sum / n)
        obs_std = np.sqrt(np.maximum(obs_var, 1e-12)).tolist()

        action_mean = (self.action_sum / n).tolist()
        action_var = (self.action_sq_sum / n) - np.square(self.action_sum / n)
        action_std = np.sqrt(np.maximum(action_var, 1e-12)).tolist()

        reward_mean = float(self.reward_sum / n)
        reward_var = float(self.reward_sq_sum / n - (self.reward_sum / n) ** 2)
        reward_std = float(np.sqrt(max(reward_var, 1e-12)))

        episode_lengths = np.array(self.episode_lengths, dtype=np.float64)
        episode_rewards = np.array(self.episode_rewards, dtype=np.float64)

        metadata = {
            "task_name": "go2_walking_ad",
            "group_name": "go2_locomotion",
            "observation_shape": {"proprio": [self.obs_dim]},
            "action_dim": self.action_dim,
            "action_type": "continuous",
            "reward_scale": 1.0,
            "algorithm_distillation": False,
            "num_envs": self.num_envs,
            "num_trajectories": int(len(episode_lengths)),
            "total_transitions": int(self.total_transitions_written),
            "max_iteration": int(self.max_iteration),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "acs_mean": action_mean,
            "acs_std": action_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": float(episode_rewards.min() if episode_rewards.size else 0.0),
            "reward_max": float(episode_rewards.max() if episode_rewards.size else 0.0),
            "episode_length_mean": float(episode_lengths.mean() if episode_lengths.size else 0.0),
            "episode_length_std": float(episode_lengths.std() if episode_lengths.size else 0.0),
        }

        meta_path_pickle = self.output_dir / f"{self.output_dir.name}.pkl"
        with meta_path_pickle.open("wb") as f_pickle:
            pickle.dump(metadata, f_pickle)

        import json

        meta_path_json = self.output_dir / f"{self.output_dir.name}.json"
        with meta_path_json.open("w") as f_json:
            json.dump(metadata, f_json, indent=2)


# -----------------------------------------------------------------------------
# Custom runner that records history
# -----------------------------------------------------------------------------

class OnPolicyRunnerWithHistory(OnPolicyRunner):
    def __init__(
        self,
        env,
        train_cfg,
        log_dir,
        device,
        recorder: Optional[TrainingHistoryRecorder] = None,
    ):
        super().__init__(env, train_cfg, log_dir, device=device)
        self.history_recorder = recorder

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        recorder = self.history_recorder
        try:
            # The body below is copied from OnPolicyRunner.learn with hooks inserted

            if self.log_dir is not None and self.writer is None:
                self.logger_type = self.cfg.get("logger", "tensorboard").lower()
                if self.logger_type == "neptune":
                    from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                    self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                    self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
                elif self.logger_type == "wandb":
                    from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                    self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                    self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
                elif self.logger_type == "tensorboard":
                    from torch.utils.tensorboard import SummaryWriter

                    self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
                else:
                    raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )

            obs, extras = self.env.get_observations()
            critic_obs = extras["observations"].get("critic", obs)
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
            self.train_mode()

            ep_infos = []
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            if self.alg.rnd:
                erewbuffer = deque(maxlen=100)
                irewbuffer = deque(maxlen=100)
                cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

            start_iter = self.current_learning_iteration
            tot_iter = start_iter + num_learning_iterations
            for it in range(start_iter, tot_iter):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                with torch.inference_mode():
                    for _ in range(self.num_steps_per_env):
                        obs_before_step = obs.detach().to("cpu").numpy() if recorder else None
                        actions = self.alg.act(obs, critic_obs)
                        actions_device = actions.to(self.env.device)
                        actions_cpu = actions.detach().to("cpu").numpy() if recorder else None

                        obs_env, rewards_env, dones_env, infos = self.env.step(actions_device)
                        if recorder:
                            rewards_cpu = rewards_env.detach().to("cpu").numpy()
                            dones_cpu = dones_env.detach().to("cpu").numpy()
                            recorder.record_batch(
                                iteration=it,
                                obs=obs_before_step,
                                actions=actions_cpu,
                                rewards=rewards_cpu,
                                dones=dones_cpu,
                            )

                        obs, rewards, dones = (
                            obs_env.to(self.device),
                            rewards_env.to(self.device),
                            dones_env.to(self.device),
                        )

                        obs = self.obs_normalizer(obs)
                        if "critic" in infos["observations"]:
                            critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                        else:
                            critic_obs = obs

                        self.alg.process_env_step(rewards, dones, infos)
                        intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                        if self.log_dir is not None:
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                            if self.alg.rnd:
                                cur_ereward_sum += rewards
                                cur_ireward_sum += intrinsic_rewards  # type: ignore
                                cur_reward_sum += rewards + intrinsic_rewards
                            else:
                                cur_reward_sum += rewards
                            cur_episode_length += 1
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                            if self.alg.rnd:
                                erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                cur_ereward_sum[new_ids] = 0
                                cur_ireward_sum[new_ids] = 0

                    end.record()
                    torch.cuda.synchronize()
                    collection_time = start.elapsed_time(end) / 1000.0

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self.alg.compute_returns(critic_obs)

                mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
                end.record()
                torch.cuda.synchronize()
                learn_time = start.elapsed_time(end) / 1000.0
                self.current_learning_iteration = it

                if self.log_dir is not None:
                    self.log(
                        {
                            "it": it,
                            "num_learning_iterations": num_learning_iterations,
                            "start_iter": start_iter,
                            "tot_iter": tot_iter,
                            "collection_time": collection_time,
                            "learn_time": learn_time,
                            "mean_value_loss": mean_value_loss,
                            "mean_surrogate_loss": mean_surrogate_loss,
                            "mean_entropy": mean_entropy,
                            "mean_rnd_loss": mean_rnd_loss,
                            "mean_symmetry_loss": mean_symmetry_loss,
                            "ep_infos": ep_infos,
                            "rewbuffer": rewbuffer,
                            "lenbuffer": lenbuffer,
                            "erewbuffer": erewbuffer if self.alg.rnd else [],
                            "irewbuffer": irewbuffer if self.alg.rnd else [],
                        }
                    )
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

                ep_infos.clear()

                if it == start_iter:
                    from rsl_rl.utils import store_code_state

                    git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                    if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                        for path in git_file_paths:
                            self.writer.save_file(path)

            if self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        finally:
            if recorder is not None:
                recorder.finalize()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO and record training trajectory.")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago"], default="go2")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=301)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--domain_randomization", action="store_true", default=False)
    parser.add_argument("--mass_range_min", type=float, default=0.9)
    parser.add_argument("--mass_range_max", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--record_output", type=str, required=True, help="Output directory for recorded trajectories.")
    parser.add_argument("--group_size", type=int, default=50000, help="Transitions per HDF5 group.")
    args = parser.parse_args()

    gs.init(logging_level="warning", precision="64")

    log_dir = Path("logs") / args.exp_name

    if args.robot_type == "go2":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_go2_cfgs()
    elif args.robot_type == "minicheetah":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_minicheetah_cfgs()
    elif args.robot_type == "laikago":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_laikago_cfgs()
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")

    dr_cfg = {
        "domain_randomization": args.domain_randomization,
        "mass_range": (args.mass_range_min, args.mass_range_max),
    }

    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, seed=args.seed)

    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )
    else:
        env = LaikagoEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_randomization=dr_cfg["domain_randomization"],
            mass_range=dr_cfg["mass_range"],
        )

    recorder = TrainingHistoryRecorder(
        output_dir=Path(args.record_output),
        num_envs=env.num_envs,
        obs_dim=obs_cfg["num_obs"],
        action_dim=env_cfg["num_actions"],
        group_size=args.group_size,
    )

    runner = OnPolicyRunnerWithHistory(env, train_cfg, str(log_dir), device=gs.device, recorder=recorder)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Fine-tuning: loading {args.pretrained_path}")
        runner.load(args.pretrained_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()


