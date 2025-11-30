#!/usr/bin/env python3
"""
Vintix モデルの評価スクリプト

Genesis/examples/locomotion/eval.pyをベースに、Vintixモデル用に改良。
環境リセット後も履歴を保持し、報酬計算も含む完全な評価を実行。
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
        """Vintix用のコンテキストを取得"""
        if len(self.observations) == 0:
            return None
        
        # 最新のcontext_len分を取得
        obs_list = list(self.observations)[-context_len:]
        act_list = list(self.actions)[-context_len:]
        rew_list = list(self.rewards)[-context_len:]
        step_list = list(self.step_nums)[-context_len:]
        
        # Vintixの入力形式：リストの辞書（bf16対応）
        batch = [{
            'observation': torch.tensor(np.array(obs_list), dtype=torch.bfloat16),  # [T, obs_dim] bf16
            'prev_action': torch.tensor(np.array(act_list), dtype=torch.bfloat16),  # [T, act_dim] bf16
            'prev_reward': torch.tensor(np.array(rew_list), dtype=torch.bfloat16).unsqueeze(1),  # [T, 1] bf16
            'step_num': torch.tensor(step_list, dtype=torch.int32),  # [T]
            'task_name': 'go2_walking_ad',  # タスク名
        }]
        
        return batch
    
    def reset(self):
        """履歴を完全にクリア（通常は使わない）"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.step_nums.clear()
        self.current_step = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                        help="Experiment name (for loading env config)")
    parser.add_argument("-r", "--robot_type", type=str, choices=["go2", "minicheetah", "laikago"], 
                        default="go2", help="Robot type")
    parser.add_argument("--vintix_path", type=str, required=True,
                        help="Path to trained Vintix model directory")
    parser.add_argument("--context_len", type=int, default=2048,
                        help="Context length for Vintix (should match training context_len)")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Maximum evaluation steps")
    parser.add_argument("--reset_threshold", type=int, default=1000,
                        help="Reset environment every N steps")
    args = parser.parse_args()
    
    print("="*80)
    print("Vintix Go2 Evaluation")
    print("="*80)
    print(f"Vintix model: {args.vintix_path}")
    print(f"Robot: {args.robot_type}")
    print(f"Context length: {args.context_len}")
    print("="*80 + "\n")
    
    # Genesis初期化
    gs.init()
    
    # 環境設定をロード（PPO訓練時の設定を使用）
    log_dir = f"{GENESIS_LOCOMOTION_PATH}/logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    
    # 報酬計算を有効にする（PPOと同じ設定）
    # reward_cfg["reward_scales"] = {}  # この行をコメントアウトして報酬計算を有効化
    
    print("Creating environment...")
    if args.robot_type == "go2":
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    elif args.robot_type == "minicheetah":
        env = MiniCheetahEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    elif args.robot_type == "laikago":
        env = LaikagoEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
    else:
        raise ValueError(f"Unknown robot type: {args.robot_type}")
    
    print(f"✓ Environment created")
    print(f"  Observation dim: {env.num_obs}")
    print(f"  Action dim: {env.num_actions}\n")
    
    # Vintixモデルをロード
    print("Loading Vintix model...")
    vintix_model = Vintix()
    vintix_model.load_model(args.vintix_path)
    vintix_model = vintix_model.to(gs.device)
    vintix_model = vintix_model.to(torch.bfloat16)  # bf16に変換（訓練時と同じ）
    
    # ALiBi slopes を bf16 に変換（FlashAttention互換性のため）
    for module in vintix_model.modules():
        if hasattr(module, 'alibi_slopes'):
            module.alibi_slopes = module.alibi_slopes.to(torch.bfloat16)
    
    vintix_model.eval()
    
    print(f"✓ Vintix model loaded")
    print(f"  Parameters: {sum(p.numel() for p in vintix_model.parameters()):,}\n")
    
    # 履歴バッファを作成
    history_buffer = VintixHistoryBuffer(max_len=args.context_len)
    
    # 評価ループ
    print("="*80)
    print("Starting Evaluation (Press Ctrl+C to stop)")
    print("="*80)
    
    obs, _ = env.reset()
    total_reward = 0.0
    episode_reward = 0.0
    step_count = 0  # 総ステップ数
    episode_step_count = 0  # 現在のエピソード内のステップ数
    episode_count = 0
    
    with torch.no_grad():
        while step_count < args.max_steps:
            # Vintixモデルで行動を予測
            context = history_buffer.get_context(context_len=args.context_len)
            
            if context is not None:
                # デバイスに移動
                for key in context[0]:
                    if isinstance(context[0][key], torch.Tensor):
                        context[0][key] = context[0][key].to(gs.device)
                
                # Vintixで予測（訓練時と同じautocastを使用）
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
                # 初回のみランダム行動
                action = torch.randn(1, env.num_actions, device=gs.device) * 0.1
            
            # 環境をステップ
            obs_next, reward, done, info = env.step(action)
            
            # 履歴に追加
            history_buffer.add(
                obs.cpu().numpy()[0],
                action.cpu().numpy()[0],
                float(reward.cpu().numpy()[0])
            )
            
            # 統計更新
            total_reward += float(reward.cpu().numpy()[0])
            episode_reward += float(reward.cpu().numpy()[0])
            step_count += 1
            episode_step_count += 1
            
            # 進捗表示
            if step_count % 100 == 0:
                print(f"Step {step_count:5d} | Episode {episode_count+1} | "
                      f"Episode Step: {episode_step_count:4d} | "
                      f"Episode Reward: {episode_reward:8.3f} | "
                      f"History Length: {len(history_buffer.observations):4d} | "
                      f"Current Step Num: {history_buffer.current_step}")
            
            # 環境リセット条件
            if (done.any() or 
                episode_step_count >= args.reset_threshold):
                
                print(f"Episode {episode_count + 1} completed | "
                      f"Reward: {episode_reward:.3f} | Steps: {episode_step_count}")
                
                # 環境をリセット（履歴は保持、ダミーデータは追加しない）
                obs, _ = env.reset()
                episode_count += 1
                episode_reward = 0.0
                episode_step_count = 0  # エピソード内ステップをリセット
            else:
                obs = obs_next
    
    # 最終統計
    print(f"\nFinal Statistics:")
    print(f"  Total steps: {step_count}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Average reward per step: {total_reward / step_count:.6f}")
    print(f"  History buffer length: {len(history_buffer.observations)}")


if __name__ == "__main__":
    main()