#!/usr/bin/env python3
"""
並列環境を使用したAlgorithm Distillationデータ収集

各環境が独立してデータを収集し、各環境1つのファイルに保存
"""
import argparse
import json
import math
import sys
from pathlib import Path
import pickle

import h5py
import numpy as np
import torch
from tqdm import tqdm

GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

from importlib import metadata
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from env import Go2Env
from env import MiniCheetahEnv
from env import LaikagoEnv


class PerEnvADDataCollector:
    """各環境ごとのADデータ収集器（各環境1つのファイル）"""
    
    def __init__(self, output_dir, env_idx, group_size=50000, robot_type="go2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_idx = env_idx
        self.group_size = group_size
        self.robot_type = robot_type
        
        filename = self.output_dir / f"trajectories_env_{env_idx:04d}.h5"
        self.h5_file = h5py.File(filename, 'w')
        self.filename = filename
        
        self.buffer_transitions = []
        self.total_transitions = 0
        self.global_written = 0
        self.group_count = 0
        
        print(f"[Env {env_idx} Collector] Created: {filename}")
    
    def add_transitions_batch(self, transitions):
        """複数のトランジションをまとめて追加
        
        Args:
            transitions: list of transitions, each transition is a dict with keys:
                'obs', 'action', 'next_obs', 'reward', 'step'
        """
        self.buffer_transitions.extend(transitions)
        self.total_transitions += len(transitions)
        
        if len(self.buffer_transitions) >= self.group_size:
            self._flush_buffer()
        
        if self.total_transitions % 10000 == 0:
            self.h5_file.flush()
    
    def _flush_buffer(self):
        """バッファをHDF5に書き込み"""
        if len(self.buffer_transitions) == 0:
            return
        
        chunk_size = min(len(self.buffer_transitions), self.group_size)
        chunk_transitions = self.buffer_transitions[:chunk_size]
        
        obs_list = []
        for i, t in enumerate(chunk_transitions):
            if i == 0:
                obs_list.append(t['obs'])
            obs_list.append(t['next_obs'])
        
        action_list = [t['action'] for t in chunk_transitions]
        reward_list = [t['reward'] for t in chunk_transitions]
        step_num_list = [t['step'] for t in chunk_transitions]
        
        obs_chunk = np.array(obs_list, dtype=np.float32)
        act_chunk = np.array(action_list, dtype=np.float32)
        rew_chunk = np.array(reward_list, dtype=np.float32)
        step_chunk = np.array(step_num_list, dtype=np.int32)
        
        start_idx = self.global_written
        end_idx = start_idx + chunk_size - 1
        group_name = f"{start_idx}-{end_idx}"
        
        group = self.h5_file.create_group(group_name)
        group.create_dataset('proprio_observation', data=obs_chunk)
        group.create_dataset('action', data=act_chunk)
        group.create_dataset('reward', data=rew_chunk)
        group.create_dataset('step_num', data=step_chunk)
        
        del self.buffer_transitions[:chunk_size]
        
        self.global_written += chunk_size
        self.group_count += 1
        
        self.h5_file.flush()
    
    def finalize(self):
        """残りのデータを保存して終了"""
        if len(self.buffer_transitions) > 0:
            self._flush_buffer()
        
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        print(f"[Env {self.env_idx} Collector] Total: {self.total_transitions:,} transitions")
        self._save_metadata()
    
    def _save_metadata(self):
        """メタデータを保存"""
        if not self.filename.exists():
            return
        
        obs_sum = None
        obs_sq_sum = None
        obs_count = 0
        
        acs_sum = None
        acs_sq_sum = None
        
        reward_sum = 0.0
        reward_sq_sum = 0.0
        reward_min = float('inf')
        reward_max = float('-inf')
        
        with h5py.File(self.filename, 'r') as f:
            for gname in f.keys():
                obs_chunk = np.array(f[gname]['proprio_observation'])
                acs_chunk = np.array(f[gname]['action'])
                rew_chunk = np.array(f[gname]['reward'])
                
                if obs_sum is None:
                    obs_sum = obs_chunk.sum(axis=0)
                    obs_sq_sum = np.square(obs_chunk).sum(axis=0)
                    acs_sum = acs_chunk.sum(axis=0)
                    acs_sq_sum = np.square(acs_chunk).sum(axis=0)
                else:
                    obs_sum += obs_chunk.sum(axis=0)
                    obs_sq_sum += np.square(obs_chunk).sum(axis=0)
                    acs_sum += acs_chunk.sum(axis=0)
                    acs_sq_sum += np.square(acs_chunk).sum(axis=0)
                
                obs_count += obs_chunk.shape[0]
                reward_sum += rew_chunk.sum()
                reward_sq_sum += np.square(rew_chunk).sum()
                reward_min = min(reward_min, float(rew_chunk.min()))
                reward_max = max(reward_max, float(rew_chunk.max()))
        
        if obs_count == 0:
            return
        
        obs_mean = obs_sum / obs_count
        obs_var = np.maximum(obs_sq_sum / obs_count - np.square(obs_mean), 0.0)
        obs_std = np.sqrt(obs_var) + 1e-8
        
        acs_mean = acs_sum / obs_count
        acs_var = np.maximum(acs_sq_sum / obs_count - np.square(acs_mean), 0.0)
        acs_std = np.sqrt(acs_var) + 1e-8
        
        reward_mean = reward_sum / obs_count
        reward_var = max(reward_sq_sum / obs_count - reward_mean ** 2, 0.0)
        reward_std = math.sqrt(reward_var)
        
        # robot_typeに基づいてtask_nameとgroup_nameを設定
        if self.robot_type == "minicheetah":
            task_name = "minicheetah_walking_ad"
            group_name = "minicheetah_locomotion"
        elif self.robot_type == "laikago":
            task_name = "laikago_walking_ad"
            group_name = "laikago_locomotion"
        else:  # go2
            task_name = "go2_walking_ad"
            group_name = "go2_locomotion"
        
        metadata = {
            "task_name": task_name,
            "group_name": group_name,
            "observation_shape": {"proprio": [33]},  # 45次元から行動12次元を除外
            "action_dim": 12,
            "action_type": "continuous",
            "reward_scale": 1.0,
            "algorithm_distillation": True,
            "env_idx": int(self.env_idx),
            "obs_mean": obs_mean.tolist(),
            "obs_std": obs_std.tolist(),
            "acs_mean": acs_mean.tolist(),
            "acs_std": acs_std.tolist(),
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "reward_min": float(reward_min),
            "reward_max": float(reward_max),
            "total_transitions": int(self.total_transitions),
        }
        
        metadata_path = self.output_dir / f"trajectories_env_{self.env_idx:04d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect AD data with parallel environments")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained PPO model")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago"],
                        default="go2", help="Robot type")
    parser.add_argument("--output_dir", type=str, default="data/go2_trajectories/go2_ad_parallel",
                        help="Output directory for AD data")
    parser.add_argument("--target_steps_per_env", type=int, default=1_000_000,
                        help="Total trajectory steps to collect per environment")
    parser.add_argument("--num_envs", type=int, default=10,
                        help="Number of parallel environments (trajectories)")
    parser.add_argument("--max_perf", type=float, default=1.0,
                        help="Maximum performance level (0.0=random, 1.0=expert)")
    parser.add_argument("--noise_free_fraction", type=float, default=0.05,
                        help="Fraction f of trajectory where epsilon=0 (final f*N_s steps are expert-only, default=0.05)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--decay_power", type=float, default=0.5,
                        help="Power parameter p for epsilon decay (lower -> quicker drop)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Parallel Algorithm Distillation Data Collection")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Robot: {args.robot_type}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Target steps per env: {args.target_steps_per_env}")
    print(f"Max performance: {args.max_perf}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Decay power (p): {args.decay_power}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    gs.init(logging_level="warning", precision="64")
    
    model_dir = Path(args.model_path).parent
    cfgs_path = model_dir / "cfgs.pkl"
    
    if not cfgs_path.exists():
        genesis_logs = Path(__file__).parents[2] / "Genesis" / "logs" / model_dir.name
        cfgs_path = genesis_logs / "cfgs.pkl"
        if not cfgs_path.exists():
            raise FileNotFoundError(f"cfgs.pkl not found at {cfgs_path}")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))
    
    print("Creating parallel environments...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    print(f"✓ Created {args.num_envs} parallel {args.robot_type} environments")
    
    runner = OnPolicyRunner(env, train_cfg, str(model_dir), device=gs.device)
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"✓ Loaded model from {args.model_path}\n")
    
    collectors = [PerEnvADDataCollector(args.output_dir, env_idx, robot_type=args.robot_type) 
                  for env_idx in range(args.num_envs)]
    
    base_seed = 42
    env_generators = [torch.Generator(device=gs.device) for _ in range(args.num_envs)]
    for env_idx, gen in enumerate(env_generators):
        gen.manual_seed(base_seed + env_idx)
    print(f"✓ Set independent random seeds for action generation (base_seed={base_seed}, env_seeds={base_seed}..{base_seed + args.num_envs - 1})")
    
    default_joint_angles = torch.tensor([
        0.0,   # FR_hip
        0.8,   # FR_thigh
        -1.5,  # FR_calf
        0.0,   # FL_hip
        0.8,   # FL_thigh
        -1.5,  # FL_calf
        0.0,   # RR_hip
        1.0,   # RR_thigh
        -1.5,  # RR_calf
        0.0,   # RL_hip
        1.0,   # RL_thigh
        -1.5,  # RL_calf
    ], device=gs.device)
    
    joint_limits = torch.tensor([
        [-1.0472,  1.0472],   # FR_hip
        [-1.5708,  3.4907],   # FR_thigh
        [-2.7227, -0.83776],  # FR_calf
        [-1.0472,  1.0472],   # FL_hip
        [-1.5708,  3.4907],   # FL_thigh
        [-2.7227, -0.83776],  # FL_calf
        [-1.0472,  1.0472],   # RR_hip
        [-0.5236,  4.5379],   # RR_thigh
        [-2.7227, -0.83776],  # RR_calf
        [-1.0472,  1.0472],   # RL_hip
        [-0.5236,  4.5379],   # RL_thigh
        [-2.7227, -0.83776],  # RL_calf
    ], device=gs.device)
    
    action_scale = env_cfg["action_scale"]
    action_limits = (joint_limits - default_joint_angles.unsqueeze(1)) / action_scale
    
    print(f"✓ Computed action ranges from joint limits (action_scale={action_scale})")
    print(f"  Action ranges per joint:")
    joint_names = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
                   "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
    for i, name in enumerate(joint_names):
        print(f"    {i:2d}: {name:12s} [{action_limits[i, 0]:7.2f}, {action_limits[i, 1]:7.2f}]")
    print()
    
    f = args.noise_free_fraction
    p = args.decay_power
    target_steps_tensor = torch.tensor(float(args.target_steps_per_env), device=gs.device)
    raw_threshold = (1.0 - f) * args.target_steps_per_env
    if raw_threshold <= 0:
        threshold_steps_tensor = None
        threshold_valid = False
    else:
        threshold_steps_tensor = torch.tensor(float(raw_threshold), device=gs.device)
        threshold_valid = True
    
    print(f"Collecting {args.num_envs} independent trajectories with ε schedule...")
    print(f"  Target steps per env: {args.target_steps_per_env:,}")
    print(f"  f (noise-free fraction): {f:.2f}")
    print(f"  p (decay power): {p:.3f}")
    if not threshold_valid:
        print("  ε is 0 from the start (noise-free trajectory)")
    else:
        print(f"  ε becomes 0 for each env after step: {int(threshold_steps_tensor.item()):,}")
    print(f"  Method: action = ε * random + (1-ε) * expert")
    print(f"  Each environment saves to: trajectories_env_XXXX.h5")
    print()
    
    obs, _ = env.reset()
    # obsは45次元のまま（policyに渡すため）
    
    episode_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=gs.device)
    env_active = torch.ones(args.num_envs, dtype=torch.bool, device=gs.device)
    env_completed = torch.zeros(args.num_envs, dtype=torch.bool, device=gs.device)
    target_reached_flag = torch.zeros(args.num_envs, dtype=torch.bool, device=gs.device)
    completed_env_count = 0
    
    episode_rewards_list = []
    episode_lengths_list = []
    current_episode_rewards = torch.zeros(args.num_envs, device=gs.device)
    env_total_steps = torch.zeros(args.num_envs, dtype=torch.float32, device=gs.device)
    
    env_episode_data = [[] for _ in range(args.num_envs)]
    
    pbar = tqdm(total=args.num_envs, desc="Collecting AD trajectories")
    
    with torch.no_grad():
        while bool(env_active.any().item()):
            if not threshold_valid:
                eps_values = torch.zeros(args.num_envs, device=gs.device)
            else:
                ratios = torch.clamp(env_total_steps / threshold_steps_tensor, 0.0, 1.0)
                ratio_term = torch.pow(ratios, p)
                eps_values = torch.pow(torch.clamp(1.0 - ratio_term, 0.0), 1.0 / p)
            eps_values = torch.where(env_active, eps_values, torch.zeros_like(eps_values))

            expert_actions = policy(obs)
            
            random_actions = torch.zeros_like(expert_actions)
            for env_idx in range(args.num_envs):
                random_actions[env_idx] = action_limits[:, 0] + torch.rand(
                    expert_actions[env_idx].shape,
                    device=gs.device,
                    generator=env_generators[env_idx]
                ) * (action_limits[:, 1] - action_limits[:, 0])
            
            eps_expanded = eps_values.unsqueeze(1)
            actions = eps_expanded * random_actions + (1.0 - eps_expanded) * expert_actions
            actions = torch.clamp(actions, action_limits[:, 0], action_limits[:, 1])
            
            obs_without_actions = obs[:, :-12]
            next_obs, rewards, dones, infos = env.step(actions)
            next_obs_without_actions = next_obs[:, :-12]
            
            for env_idx in range(args.num_envs):
                if env_active[env_idx]:
                    transition = {
                        'obs': obs_without_actions[env_idx].cpu().numpy(),
                        'action': actions[env_idx].cpu().numpy(),
                        'next_obs': next_obs_without_actions[env_idx].cpu().numpy(),
                        'reward': float(rewards[env_idx].cpu()),
                        'step': int(episode_steps[env_idx].cpu())
                    }
                    env_episode_data[env_idx].append(transition)
            
            current_episode_rewards += rewards
            episode_steps += 1
            env_total_steps += env_active.to(env_total_steps.dtype)
            
            done_mask = (dones | (episode_steps >= args.max_steps)) & env_active
            
            if done_mask.any():
                done_indices = torch.where(done_mask)[0]
                
                for idx in done_indices:
                    idx_int = int(idx.cpu())
                    episode_transitions = env_episode_data[idx_int]
                    collectors[idx_int].add_transitions_batch(episode_transitions)
                    
                    ep_reward = float(current_episode_rewards[idx].cpu())
                    ep_length = int(episode_steps[idx].cpu())
                    
                    episode_rewards_list.append(ep_reward)
                    episode_lengths_list.append(ep_length)
                    
                    env_episode_data[idx_int] = []
                    current_episode_rewards[idx] = 0.0
                    episode_steps[idx] = 0
                    
                    if target_reached_flag[idx] or env_total_steps[idx] >= target_steps_tensor:
                        env_active[idx] = False
                        target_reached_flag[idx] = False
                        if not env_completed[idx]:
                            env_completed[idx] = True
                            completed_env_count += 1
                            pbar.update(1)
                            total_steps_so_far = float(env_total_steps.sum().cpu())
                            print(f"\n[Env {completed_env_count}/{args.num_envs} completed] active_envs={int(env_active.sum().cpu())}")
                            print(f"  Collected steps (sum/env): {int(total_steps_so_far):,}")
                
                if len(done_indices) > 0:
                    env.reset_idx(done_indices)
                    # リセット後の観測値からも行動を除外してnext_obs_without_actionsを更新
                    next_obs_without_actions[done_indices] = env.obs_buf[done_indices][:, :-12]
                
                recent_rews = episode_rewards_list[-100:] if len(episode_rewards_list) >= 100 else episode_rewards_list
                recent_lens = episode_lengths_list[-100:] if len(episode_lengths_list) >= 100 else episode_lengths_list
                if env_active.any():
                    active_eps = eps_values[env_active]
                    eps_min = float(active_eps.min().cpu())
                    eps_max = float(active_eps.max().cpu())
                else:
                    eps_min = eps_max = 0.0
                total_steps_so_far = float(env_total_steps.sum().cpu())
                print(f"\n[Progress] ε-range=[{eps_min:.4f}, {eps_max:.4f}], active_envs={int(env_active.sum().cpu())}")
                print(f"  Collected steps (sum/env): {int(total_steps_so_far):,}")
                if recent_rews:
                    print(f"  Last {len(recent_rews)} eps: len={np.mean(recent_lens):.1f}, reward={np.mean(recent_rews):.4f}")
            
            target_reached_mask = (env_total_steps >= target_steps_tensor) & env_active & ~target_reached_flag
            if target_reached_mask.any():
                reached_indices = torch.where(target_reached_mask)[0]
                for idx in reached_indices:
                    target_reached_flag[idx] = True
                    idx_int = int(idx.cpu())
                    print(f"\n[Env {idx_int}] Target steps reached ({int(env_total_steps[idx].cpu()):,} >= {int(target_steps_tensor):,})")
                    print(f"  Waiting for current episode to complete (episode step: {int(episode_steps[idx].cpu())})")
            
            # 次のループで使用する観測値（policyに渡すため45次元のまま）
            obs = next_obs
    
    pbar.close()
    
    for collector in collectors:
        collector.finalize()
    
    print(f"\n{'='*80}")
    print("Parallel AD Data Collection Complete!")
    print(f"{'='*80}")
    print(f"Completed trajectories: {int(env_completed.sum().cpu())} / {args.num_envs}")
    print(f"Total episodes recorded: {len(episode_lengths_list)}")
    total_transitions = sum(c.total_transitions for c in collectors)
    print(f"Total transitions: {total_transitions:,}")
    if episode_lengths_list:
        print(f"\nEpisode Statistics:")
        print(f"  Mean length: {np.mean(episode_lengths_list):.2f}")
        print(f"  Mean reward: {np.mean(episode_rewards_list):.4f}")
        print(f"  Reward std: {np.std(episode_rewards_list):.4f}")
        print(f"  Min reward: {np.min(episode_rewards_list):.4f}")
        print(f"  Max reward: {np.max(episode_rewards_list):.4f}")
    if not threshold_valid:
        final_eps_values = torch.zeros(args.num_envs, device=gs.device)
    else:
        final_ratios = torch.clamp(env_total_steps / threshold_steps_tensor, 0.0, 1.0)
        final_ratio_term = torch.pow(final_ratios, p)
        final_eps_values = torch.pow(torch.clamp(1.0 - final_ratio_term, 0.0), 1.0 / p)
    print(f"\nFinal ε range: [{float(final_eps_values.min().cpu()):.4f}, {float(final_eps_values.max().cpu()):.4f}]")
    print(f"Final steps (sum/env): {int(env_total_steps.sum().cpu()):,}")
    print(f"\nOutput files:")
    for collector in collectors:
        print(f"  - {collector.filename}")
        print(f"  - {collector.output_dir / f'trajectories_env_{collector.env_idx:04d}.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


"""
使用例:

# 10環境、各環境 1,000,000 ステップ
python scripts/collect_ad_data_parallel.py \
    --model_path /workspace/Genesis/logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_ad_parallel \
    --target_steps_per_env 1000000 \
    --num_envs 10 \
    --max_perf 1.0 \
    --decay_power 0.4

# 各環境のデータは独立したファイルに保存される:
#   - trajectories_env_0000.h5
#   - trajectories_env_0001.h5
#   - ...
#   - trajectories_env_0009.h5
"""
