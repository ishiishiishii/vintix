#!/usr/bin/env python3
"""
Algorithm Distillation用データ収集スクリプト

訓練済みPPOモデルにノイズを加えながら実行し、
学習過程（下手→上手）のデータを収集します。
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Genesis locomotion環境のインポート用
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


class ADDataCollector:
    """Algorithm Distillation用データ収集"""
    
    def __init__(self, output_dir: str, group_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = group_size
        
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_steps = []
        self.buffer_eps = []  # ノイズレベルも記録
        self.total_transitions = 0
        
        self.h5_file = h5py.File(self.output_dir / "trajectories_0000.h5", 'w')
        print(f"[AD Collector] Created: {self.output_dir / 'trajectories_0000.h5'}")
    
    def collect_episode(self, env, policy, device, eps, action_min, action_max, max_steps=1000):
        """1エピソードを収集（線形補間でノイズ追加）
        
        Algorithm Distillation論文に従い、ランダム行動と専門家行動を線形補間します。
        
        Args:
            env: Genesis環境
            policy: 訓練済みPPOポリシー
            device: torch device
            eps: ノイズレベル (0.0=専門家, 1.0=ランダム)
                 action = eps * random_action + (1 - eps) * expert_action
            action_min: 各関節のアクション下限（可動域から計算）
            action_max: 各関節のアクション上限（可動域から計算）
            max_steps: 最大ステップ数
            
        Returns:
            (エピソード長, 累積報酬)
        """
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        
        episode_obs, episode_acts, episode_rews, episode_steps = [], [], [], []
        
        while not done and step < max_steps:
            with torch.no_grad():
                # 専門家の行動を取得
                expert_action = policy(obs)
                
                # ランダム行動を生成（論文のアルゴリズムに従って一様分布からサンプリング）
                # u ~ Uniform(amin, amax) - 各関節の可動域で一様分布
                uniform_samples = torch.rand(1, env.num_actions, device=device)
                # 各関節の可動域範囲で一様分布: [amin, amax]
                random_action = action_min + uniform_samples * (action_max - action_min)
                
                # 線形補間: action = eps * random + (1 - eps) * expert
                action = eps * random_action + (1.0 - eps) * expert_action
            
            # データを記録
            episode_obs.append(obs.cpu().numpy()[0])
            episode_acts.append(action.cpu().numpy()[0])
            episode_steps.append(step)
            
            # 環境をステップ
            obs, reward, done, _ = env.step(action)
            reward_value = float(reward.cpu().numpy()[0])
            episode_rews.append(reward_value)
            total_reward += reward_value
            step += 1
        
        # バッファに追加
        self.buffer_obs.extend(episode_obs)
        self.buffer_actions.extend(episode_acts)
        self.buffer_rewards.extend(episode_rews)
        self.buffer_steps.extend(episode_steps)
        self.total_transitions += len(episode_obs)
        
        # グループサイズに達したらフラッシュ
        if len(self.buffer_obs) >= self.group_size:
            self._flush_buffer()
        
        # 10エピソードごとにもflush（安全のため）
        if len(episode_obs) > 0 and self.total_transitions % 10000 == 0:
            self.h5_file.flush()
        
        return step, total_reward
    
    def _flush_buffer(self):
        """バッファをHDF5に書き込み"""
        if len(self.buffer_obs) == 0:
            return
        
        while len(self.buffer_obs) >= self.group_size:
            start_idx = self.total_transitions - len(self.buffer_obs)
            end_idx = start_idx + self.group_size - 1
            group_name = f"{start_idx}-{end_idx}"
            
            group = self.h5_file.create_group(group_name)
            group.create_dataset(
                'proprio_observation',
                data=np.array(self.buffer_obs[:self.group_size], dtype=np.float32)
            )
            group.create_dataset(
                'action',
                data=np.array(self.buffer_actions[:self.group_size], dtype=np.float32)
            )
            group.create_dataset(
                'reward',
                data=np.array(self.buffer_rewards[:self.group_size], dtype=np.float32)
            )
            group.create_dataset(
                'step_num',
                data=np.array(self.buffer_steps[:self.group_size], dtype=np.int32)
            )
            
            self.buffer_obs = self.buffer_obs[self.group_size:]
            self.buffer_actions = self.buffer_actions[self.group_size:]
            self.buffer_rewards = self.buffer_rewards[self.group_size:]
            self.buffer_steps = self.buffer_steps[self.group_size:]
        
        # ファイルをflushして確実に書き込む
        self.h5_file.flush()
    
    def finalize(self):
        """残りデータを保存"""
        if len(self.buffer_obs) > 0:
            start_idx = self.total_transitions - len(self.buffer_obs)
            end_idx = self.total_transitions - 1
            group_name = f"{start_idx}-{end_idx}"
            
            group = self.h5_file.create_group(group_name)
            group.create_dataset(
                'proprio_observation',
                data=np.array(self.buffer_obs, dtype=np.float32)
            )
            group.create_dataset(
                'action',
                data=np.array(self.buffer_actions, dtype=np.float32)
            )
            group.create_dataset(
                'reward',
                data=np.array(self.buffer_rewards, dtype=np.float32)
            )
            group.create_dataset(
                'step_num',
                data=np.array(self.buffer_steps, dtype=np.int32)
            )
        
        self.h5_file.close()
        print(f"[AD Collector] Total: {self.total_transitions:,} transitions")
        self._save_metadata()
    
    def _save_metadata(self):
        """メタデータ保存"""
        h5_path = self.output_dir / "trajectories_0000.h5"
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                all_obs, all_acts, all_rews = [], [], []
                for gname in f.keys():
                    all_obs.append(np.array(f[gname]['proprio_observation']))
                    all_acts.append(np.array(f[gname]['action']))
                    all_rews.append(np.array(f[gname]['reward']))
                
                if all_obs:
                    all_obs = np.vstack(all_obs)
                    all_acts = np.vstack(all_acts)
                    all_rews = np.concatenate(all_rews)
                    
                    metadata = {
                        "task_name": "go2_walking_ad",
                        "group_name": "go2_locomotion",
                        "observation_shape": {"proprio": [45]},
                        "action_dim": 12,
                        "action_type": "continuous",
                        "reward_scale": 1.0,
                        "algorithm_distillation": True,
                        "obs_mean": all_obs.mean(axis=0).tolist(),
                        "obs_std": (all_obs.std(axis=0) + 1e-8).tolist(),
                        "acs_mean": all_acts.mean(axis=0).tolist(),
                        "acs_std": (all_acts.std(axis=0) + 1e-8).tolist(),
                        "reward_mean": float(all_rews.mean()),
                        "reward_std": float(all_rews.std()),
                        "reward_min": float(all_rews.min()),
                        "reward_max": float(all_rews.max()),
                    }
                    
                    with open(self.output_dir / f"{self.output_dir.name}.json", 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"[AD Collector] Metadata saved")


def main():
    parser = argparse.ArgumentParser(description="Collect AD data from trained PPO model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained PPO model (e.g., logs/go2-walking/model_300.pt)")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago"],
                        default="go2", help="Robot type")
    parser.add_argument("--output_dir", type=str, default="data/go2_trajectories/go2_ad_data",
                        help="Output directory for AD data")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--max_perf", type=float, default=0.4,
                        help="Maximum performance level (0.0=random, 1.0=expert)")
    parser.add_argument("--decay_power", type=float, default=0.5,
                        help="Power parameter p for epsilon decay (lower -> quicker drop)")
    parser.add_argument("--show_viewer", action="store_true",
                        help="Show viewer")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Algorithm Distillation Data Collection")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Robot: {args.robot_type}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max performance: {args.max_perf}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    gs.init(logging_level="warning", precision="64")
    
    # モデルと設定をロード
    model_dir = Path(args.model_path).parent
    cfgs_path = model_dir / "cfgs.pkl"
    
    if not cfgs_path.exists():
        raise FileNotFoundError(f"cfgs.pkl not found at {cfgs_path}")
    
    with open(cfgs_path, 'rb') as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    
    print("✓ Loaded configuration")
    
    # 環境を作成（1環境のみ）
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=args.show_viewer
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=args.show_viewer
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=args.show_viewer
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Created {args.robot_type} environment")
    
    # PPOモデルをロード
    runner = OnPolicyRunner(env, train_cfg, str(model_dir), device=gs.device)
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"✓ Loaded model from {args.model_path}\n")
    
    # Go2の各関節の可動域（URDFから）をアクション空間にマッピング
    # 関節名の順序: ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
    #                "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
    if args.robot_type == "go2":
        # デフォルト関節角度
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
        
        # 関節の可動域（URDFから）[rad]
        joint_limits_lower = torch.tensor([
            -1.0472,  # FR_hip: [-1.0472, 1.0472]
            -1.5708,  # FR_thigh: [-1.5708, 3.4907]
            -2.7227,  # FR_calf: [-2.7227, -0.83776]
            -1.0472,  # FL_hip: [-1.0472, 1.0472]
            -1.5708,  # FL_thigh: [-1.5708, 3.4907]
            -2.7227,  # FL_calf: [-2.7227, -0.83776]
            -1.0472,  # RR_hip: [-1.0472, 1.0472]
            -0.5236,  # RR_thigh: [-0.5236, 4.5379]
            -2.7227,  # RR_calf: [-2.7227, -0.83776]
            -1.0472,  # RL_hip: [-1.0472, 1.0472]
            -0.5236,  # RL_thigh: [-0.5236, 4.5379]
            -2.7227,  # RL_calf: [-2.7227, -0.83776]
        ], device=gs.device)
        
        joint_limits_upper = torch.tensor([
            1.0472,   # FR_hip
            3.4907,   # FR_thigh
            -0.83776, # FR_calf
            1.0472,   # FL_hip
            3.4907,   # FL_thigh
            -0.83776, # FL_calf
            1.0472,   # RR_hip
            4.5379,   # RR_thigh
            -0.83776, # RR_calf
            1.0472,   # RL_hip
            4.5379,   # RL_thigh
            -0.83776, # RL_calf
        ], device=gs.device)
        
        # アクション空間での範囲を計算
        # target_dof_pos = action * action_scale + default_dof_pos
        # より: action = (target_dof_pos - default_dof_pos) / action_scale
        action_scale = env_cfg["action_scale"]
        action_min = (joint_limits_lower - default_joint_angles) / action_scale
        action_max = (joint_limits_upper - default_joint_angles) / action_scale
        
        print(f"✓ Computed action ranges from joint limits (action_scale={action_scale})")
        print(f"  Action ranges per joint:")
        joint_names = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
                       "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
        for i, name in enumerate(joint_names):
            print(f"    {i:2d}: {name:12s} [{action_min[i]:7.2f}, {action_max[i]:7.2f}]")
        print()
    else:
        # 他のロボットタイプの場合は、暫定的に広い範囲を使用
        # TODO: 他のロボットタイプの可動域を実装
        action_min = torch.full((env.num_actions,), -100.0, device=gs.device)
        action_max = torch.full((env.num_actions,), 100.0, device=gs.device)
        print(f"⚠ Using default action range [-100, 100] for {args.robot_type} (not implemented)")
        print()
    
    # データ収集開始
    # group_size: 50,000 transitions ≈ 50エピソード/グループ
    collector = ADDataCollector(args.output_dir, group_size=50000)
    
    # εスケジュール設定（Algorithm Distillation論文の式）
    # ε(n_s) = (1 - (n_s / ((1-f)N_s)))^p  if n_s <= (1-f)N_s
    # ε(n_s) = 0                            if n_s > (1-f)N_s
    # 
    # n_s: 現在のステップ数（軌道の開始からの累積ステップ）
    # N_s: 軌道全体の最大ステップ数
    # f: 軌道の最後でノイズを0にする部分の割合（max_perf=1.0なら f=0）
    # p: 減衰曲線の形状パラメータ
    
    total_steps = args.num_episodes * 1000  # N_s: 想定される総ステップ数
    f = 1.0 - args.max_perf  # f=0 なら最後まで専門家
    p = args.decay_power
    threshold_step = (1.0 - f) * total_steps  # (1-f)N_s
    if threshold_step <= 0:
        threshold_step = None
    
    current_step = 0
    episode_lengths = []
    episode_rewards = []
    
    print(f"Collecting {args.num_episodes} episodes with ε schedule...")
    print(f"  Total steps: {total_steps}")
    print(f"  f (noise-free fraction): {f:.2f}")
    print(f"  p (decay power): {p:.3f}")
    if threshold_step is None:
        print("  ε stays at 1.0 for the entire run (max_perf <= 0)\n")
    else:
        print(f"  ε becomes 0 after step: {int(threshold_step):,}\n")
    
    for episode in tqdm(range(args.num_episodes), desc="Collecting AD data"):
        # 現在のεを計算
        if threshold_step is None:
            eps = 1.0
        elif current_step <= threshold_step:
            eps = (1.0 - (current_step / threshold_step)) ** p
        else:
            eps = 0.0
        
        ep_len, ep_reward = collector.collect_episode(env, policy, gs.device, eps, action_min, action_max, max_steps=1000)
        episode_lengths.append(ep_len)
        episode_rewards.append(ep_reward)
        current_step += ep_len  # 実際のステップ数を累積
        
        if (episode + 1) % 10 == 0:
            recent_lens = episode_lengths[-10:]
            recent_rews = episode_rewards[-10:]
            print(f"\n[Episode {episode+1}/{args.num_episodes}] ε={eps:.3f}")
            print(f"  Last 10: len={np.mean(recent_lens):.1f}, reward={np.mean(recent_rews):.4f}")
    
    # データ保存
    collector.finalize()
    
    print(f"\n{'='*80}")
    print("AD Data Collection Complete!")
    print(f"{'='*80}")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Total transitions: {collector.total_transitions:,}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    print(f"Mean episode reward: {np.mean(episode_rewards):.4f}")
    print(f"\nOutput:")
    print(f"  - {collector.output_dir / 'trajectories_0000.h5'}")
    print(f"  - {collector.output_dir / f'{Path(args.output_dir).name}.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


"""
使用例:

# 基本的な使い方
python scripts/collect_ad_data.py \
    --model_path logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_ad_data \
    --num_episodes 100

# パラメータ調整
python scripts/collect_ad_data.py \
    --model_path logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_ad_v2 \
    --num_episodes 200 \
    --max_perf 0.6

# ビジュアライザーあり
python scripts/collect_ad_data.py \
    --model_path logs/go2-walking/model_300.pt \
    --output_dir data/go2_trajectories/go2_ad_visual \
    --num_episodes 50 \
    --show_viewer

# MiniCheetah
python scripts/collect_ad_data.py \
    --model_path logs/minicheetah-walking/model_300.pt \
    -r minicheetah \
    --output_dir data/go2_trajectories/minicheetah_ad_data
"""

