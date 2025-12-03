#!/usr/bin/env python3
"""
Vintix モデルの動作を動画として保存するスクリプト

eval_vintix.pyをベースに録画機能を追加

Usage:
    python scripts/save_vintix.py --vintix_path models/vintix_go2/vintix_go2_ad/0095_epoch --output movie/vintix_expert_only.mp4
"""
import argparse
import os
import pickle
import sys
from pathlib import Path
from collections import deque
from importlib import metadata

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Genesis locomotion環境のインポート用
GENESIS_LOCOMOTION_PATH = str(Path(__file__).parents[2] / "Genesis" / "examples" / "locomotion")
sys.path.insert(0, GENESIS_LOCOMOTION_PATH)

# rsl_rl バージョンチェック
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

import genesis as gs
from env import Go2Env
from env import MiniCheetahEnv
from env import LaikagoEnv

# Vintixモジュールのインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from vintix.vintix import Vintix


class VintixHistoryBuffer:
    """Vintix用の履歴バッファ（環境リセット後も保持）"""
    
    def __init__(self, max_len=1024):
        self.max_len = max_len
        self.observations = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.step_nums = deque(maxlen=max_len)
        self.current_step = 0
    
    def add(self, obs, action, reward):
        """履歴に追加"""
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.step_nums.append(self.current_step)
        self.current_step += 1
    
    def get_context(self, context_len=1024):
        """Vintix用のコンテキストを取得（eval_vintix.pyと同じ）"""
        if len(self.observations) == 0:
            return None
        
        # 最新のcontext_len分を取得
        obs_list = list(self.observations)[-context_len:]
        act_list = list(self.actions)[-context_len:]
        rew_list = list(self.rewards)[-context_len:]
        step_list = list(self.step_nums)[-context_len:]
        
        # Vintixの入力形式：eval_vintix.pyと同じ
        batch = [{
            'observation': torch.tensor(np.array(obs_list), dtype=torch.float32),
            'prev_action': torch.tensor(np.array(act_list), dtype=torch.float32),
            'prev_reward': torch.tensor(np.array(rew_list), dtype=torch.float32).unsqueeze(1),
            'step_num': torch.tensor(step_list, dtype=torch.int32),
            'task_name': 'go2_walking_ad',
        }]
        
        return batch


def main():
    parser = argparse.ArgumentParser(description="Save Vintix model behavior as video")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                        help="Experiment name (for loading env config)")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago"], 
                        default="go2", help="Robot type")
    parser.add_argument("--vintix_path", type=str, required=True,
                        help="Path to Vintix model directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output video file path (e.g., movie/vintix_expert.mp4)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps to record (default: 500 = 10 seconds at 50Hz)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS")
    parser.add_argument("--context_len", type=int, default=1024,
                        help="Context length for Vintix")
    args = parser.parse_args()

    print("=" * 80)
    print("Vintix Go2 Video Recording")
    print("=" * 80)
    print(f"Vintix model: {args.vintix_path}")
    print(f"Output video: {args.output}")
    print(f"Max steps: {args.max_steps}")
    print(f"FPS: {args.fps}")
    print("=" * 80)
    print()

    # Genesis初期化
    gs.init()

    # 環境設定の読み込み（eval_vintix.pyと同じ）
    genesis_root = Path(__file__).parents[2] / "Genesis"
    log_dir = genesis_root / "logs" / args.exp_name
    cfgs_path = log_dir / "cfgs.pkl"
    
    if not cfgs_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfgs_path}")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))
    
    # 環境作成（eval_vintix.pyと同じ）
    print("Creating environment...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,  # 録画時はビューアー非表示
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Created {args.robot_type} environment")

    # Vintixモデルのロード（eval_vintix.pyと同じ）
    print(f"Loading Vintix model from {args.vintix_path}...")
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model.eval()
    print("✓ Vintix model loaded")

    # 履歴バッファの初期化
    history_buffer = VintixHistoryBuffer(max_len=args.context_len)

    # 環境リセット
    obs, _ = env.reset()
    # 観測値から行動を除外（訓練時と同じ33次元にする）
    obs = obs[:, :-12]
    
    # 初期履歴の追加（ゼロアクション、ゼロ報酬）
    initial_action = np.zeros(env.num_actions)
    initial_reward = 0.0
    history_buffer.add(obs[0].cpu().numpy(), initial_action, initial_reward)

    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting video recording...")
    print(f"Recording {args.max_steps} steps at {args.fps} FPS...")
    
    # 録画開始
    env.cam.start_recording()
    
    step_count = 0
    episode_count = 0
    episode_reward = 0.0
    episode_step_count = 0
    total_reward = 0.0
    
    # ステップごとの統計を記録（グラフの横軸をステップ数にするため）
    step_rewards = []  # 各ステップの報酬
    cumulative_rewards = []  # 各ステップまでの累積報酬
    episode_starts = []  # エピソード開始位置（グラフで区切りを表示するため）
    
    # エピソード統計の記録（サマリー表示
    episode_rewards = []
    episode_lengths = []
    episode_avg_rewards = []
    
    # 最初のエピソード開始位置
    episode_starts.append(0)
    
    with torch.no_grad():
        while step_count < args.max_steps:
            # Vintixから行動予測（eval_vintix.pyと同じロジック）
            context = history_buffer.get_context(args.context_len)
            
            if context is not None:
                # デバイスに転送
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)
                
                # Vintixで予測（eval_vintix.pyと同じ）
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_actions, metadata = vintix_model(context)
                
                # 最新の予測行動を取得（fp32に変換）
                # pred_actionsはリストの場合があるので、最初の要素を取得
                if isinstance(pred_actions, list):
                    pred_actions = pred_actions[0]
                
                # pred_actionsの次元を確認
                if pred_actions.dim() == 3:  # [batch, seq, act_dim]
                    action = pred_actions[0, -1, :].unsqueeze(0).float()
                elif pred_actions.dim() == 2:  # [seq, act_dim]
                    action = pred_actions[-1, :].unsqueeze(0).float()
                else:
                    raise ValueError(f"Unexpected pred_actions shape: {pred_actions.shape}")
            else:
                # 履歴がない場合はゼロアクション
                action = torch.zeros(1, env.num_actions, device=gs.device)
            
            # 環境ステップ
            obs, rewards, dones, infos = env.step(action)
            # 観測値から行動を除外（訓練時と同じ33次元にする）
            obs = obs[:, :-12]
            env.cam.render()
            
            # 報酬とアクション履歴に追加（行動を除外した観測値）
            reward_value = float(rewards.cpu().numpy()[0])
            history_buffer.add(
                obs[0].cpu().numpy(),
                action[0].cpu().numpy(),
                reward_value
            )
            
            # ステップごとのデータを記録
            step_rewards.append(reward_value)
            total_reward += reward_value
            cumulative_rewards.append(total_reward)
            
            episode_reward += reward_value
            step_count += 1
            episode_step_count += 1
            
            # 進捗表示（100ステップごと）
            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(f"Step {step_count:5d} / {args.max_steps} | Episode {episode_count + 1} | "
                      f"Ep Step: {episode_step_count:4d} | Ep Reward: {episode_reward:7.3f} | "
                      f"Avg Reward: {avg_reward:7.5f}")
            
            # 環境リセット判定
            if dones[0]:
                print(f"Episode {episode_count + 1} completed | Reward: {episode_reward:.3f} | Steps: {episode_step_count}")
                
                # エピソード統計を記録
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step_count)
                episode_avg_rewards.append(episode_reward / episode_step_count if episode_step_count > 0 else 0.0)
                
                # 次のエピソード開始位置を記録
                episode_starts.append(step_count)
                
                episode_count += 1
                episode_reward = 0.0
                episode_step_count = 0
                obs, _ = env.reset()
                # 観測値から行動を除外（訓練時と同じ33次元にする）
                obs = obs[:, :-12]
    
    # 最後のエピソードが完了していない場合でも、その報酬を記録
    if episode_step_count > 0:
        print(f"Final episode (incomplete) | Reward: {episode_reward:.3f} | Steps: {episode_step_count}")
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step_count)
        episode_avg_rewards.append(episode_reward / episode_step_count if episode_step_count > 0 else 0.0)
        episode_starts.append(step_count)
    
    # 録画停止と保存
    print(f"\nStopping recording and saving to {args.output}...")
    env.cam.stop_recording(save_to_filename=str(args.output), fps=args.fps)
    
    # 最終統計
    avg_reward_per_step = total_reward / step_count if step_count > 0 else 0.0
    
    # グラフの作成
    if len(step_rewards) > 0:
        print(f"\nCreating performance graphs...")
        
        # グラフのファイル名（動画と同じディレクトリに保存）
        graph_path = output_path.with_suffix('.png')
        
        # ステップ数配列
        steps = np.arange(1, len(step_rewards) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Vintix Model Performance - {output_path.stem}', fontsize=16, fontweight='bold')
        
        # 1. 累積報酬の推移（ステップ数ベース）
        ax1 = axes[0, 0]
        ax1.plot(steps, cumulative_rewards, linewidth=2, label='Cumulative Reward')
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('Cumulative Reward', fontsize=11)
        ax1.set_title('Cumulative Reward vs Steps', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # エピソードの区切りを縦線で表示
        for ep_start in episode_starts[1:]:  # 最初の0は除く
            if ep_start < len(cumulative_rewards):
                ax1.axvline(x=ep_start, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax1.legend()
        
        # 2. ステップごとの報酬（ステップ数ベース）
        ax2 = axes[0, 1]
        ax2.plot(steps, step_rewards, linewidth=1, alpha=0.6, label='Reward per Step')
        # 移動平均を追加（見やすくするため）
        if len(step_rewards) > 50:
            window = min(50, len(step_rewards) // 10)
            moving_avg = np.convolve(step_rewards, np.ones(window)/window, mode='valid')
            moving_avg_steps = steps[window-1:]
            ax2.plot(moving_avg_steps, moving_avg, linewidth=2, color='orange', label=f'Moving Avg (window={window})')
        ax2.set_xlabel('Step', fontsize=11)
        ax2.set_ylabel('Reward', fontsize=11)
        ax2.set_title('Reward per Step', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        # エピソードの区切りを縦線で表示
        for ep_start in episode_starts[1:]:  # 最初の0は除く
            if ep_start < len(step_rewards):
                ax2.axvline(x=ep_start, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
        ax2.legend()
        
        # 3. 平均報酬の推移（移動平均、ステップ数ベース）
        ax3 = axes[1, 0]
        if len(step_rewards) > 10:
            # 累積平均を計算
            cumulative_avg = np.cumsum(step_rewards) / steps
            ax3.plot(steps, cumulative_avg, linewidth=2, color='green', label='Running Average')
            ax3.axhline(y=avg_reward_per_step, color='r', linestyle='--', label=f'Overall Mean: {avg_reward_per_step:.6f}')
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Average Reward', fontsize=11)
        ax3.set_title('Running Average Reward', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        # エピソードの区切りを縦線で表示
        for ep_start in episode_starts[1:]:  # 最初の0は除く
            if ep_start < len(step_rewards):
                ax3.axvline(x=ep_start, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax3.legend()
        
        # 4. 統計サマリー（テキスト）
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 統計計算（空配列の場合はN/Aを表示）
        if len(episode_rewards) > 0:
            ep_reward_mean = np.mean(episode_rewards)
            ep_reward_std = np.std(episode_rewards)
            ep_reward_min = np.min(episode_rewards)
            ep_reward_max = np.max(episode_rewards)
        else:
            ep_reward_mean = ep_reward_std = ep_reward_min = ep_reward_max = float('nan')
        
        if len(episode_lengths) > 0:
            ep_length_mean = np.mean(episode_lengths)
            ep_length_std = np.std(episode_lengths)
            ep_length_min = np.min(episode_lengths)
            ep_length_max = np.max(episode_lengths)
        else:
            ep_length_mean = ep_length_std = ep_length_min = ep_length_max = float('nan')
        
        # 統計値を文字列に変換
        def format_stat(value, fmt):
            if np.isnan(value):
                return 'N/A'
            return f"{value:{fmt}}"
        
        summary_text = f"""
Performance Summary

Total Episodes: {len(episode_rewards)}
Total Steps: {step_count}

Cumulative Reward:
  Mean: {format_stat(ep_reward_mean, '.3f')}
  Std: {format_stat(ep_reward_std, '.3f')}
  Min: {format_stat(ep_reward_min, '.3f')}
  Max: {format_stat(ep_reward_max, '.3f')}

Episode Length:
  Mean: {format_stat(ep_length_mean, '.1f')}
  Std: {format_stat(ep_length_std, '.1f')}
  Min: {format_stat(ep_length_min, '.0f')}
  Max: {format_stat(ep_length_max, '.0f')}

Reward per Step:
  Overall: {avg_reward_per_step:.6f}
  Mean: {np.mean(step_rewards):.6f}
  Std: {np.std(step_rewards):.6f}
  Min: {np.min(step_rewards):.6f}
  Max: {np.max(step_rewards):.6f}

Model: {Path(args.vintix_path).name}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
        print(f"✓ Graph saved: {graph_path}")
        
        # CSVファイルにも保存（ステップごとのデータ）
        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w') as f:
            f.write("step,reward,cumulative_reward,episode\n")
            current_episode = 0
            for i, (reward, cum_reward) in enumerate(zip(step_rewards, cumulative_rewards), 1):
                # 現在のエピソード番号を判定
                if current_episode + 1 < len(episode_starts) and i >= episode_starts[current_episode + 1]:
                    current_episode += 1
                f.write(f"{i},{reward:.6f},{cum_reward:.6f},{current_episode + 1}\n")
        print(f"✓ CSV saved: {csv_path}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ Video saved successfully!")
    print(f"  Output: {output_path.absolute()}")
    print(f"  Total steps: {step_count}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Average reward per step: {avg_reward_per_step:.6f}")
    if output_path.exists():
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
