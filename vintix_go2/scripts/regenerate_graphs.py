#!/usr/bin/env python3
"""
既存のCSVファイルからグラフを再生成するスクリプト
縦軸を固定したグラフを生成します。
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def regenerate_graphs_from_csv(csv_path: Path, episode_csv_path: Path = None, max_episodes: int = None):
    """CSVファイルからグラフを再生成"""
    # ステップごとのグラフを再生成
    if csv_path.exists():
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        steps = data[:, 0]
        mean_rewards = data[:, 1]
        std_rewards = data[:, 2]
        
        graph_path = csv_path.with_suffix('.png')
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        fig1.suptitle(f'Vintix Model Performance (Parallel) - {graph_path.stem}', fontsize=16, fontweight='bold')
        
        ax1.plot(steps, mean_rewards, linewidth=2, label='Mean Reward', color='blue')
        ax1.fill_between(steps,
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.3, color='blue', label='±1 Std')
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('Reward', fontsize=11)
        ax1.set_title('Reward per Step (Mean ± Std)', fontsize=12, fontweight='bold')
        ax1.set_ylim(-0.03, 0.03)  # 縦軸を固定
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)
        ax1.legend()
        
        plt.tight_layout()
        plt.savefig(str(graph_path), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"✓ Graph saved: {graph_path}")
    
    # エピソードごとのグラフを再生成
    if episode_csv_path and episode_csv_path.exists():
        # CSVファイルのヘッダーを確認
        with open(episode_csv_path, 'r') as f:
            header = f.readline().strip()
        
        # 新しい形式（episode_number,env_index,...）か古い形式（episode_number,...）かを判定
        if 'env_index' in header:
            # 新しい形式：episode_number,env_index,cumulative_steps,cumulative_reward,episode_length
            # episode_numberは各環境ごとのエピソード番号（0始まり）
            # env_indexは環境のインデックス（0始まり）
            data = np.loadtxt(episode_csv_path, delimiter=',', skiprows=1)
            episode_numbers = data[:, 0].astype(int)  # 各環境ごとのエピソード番号
            env_indices = data[:, 1].astype(int)  # 環境インデックス
            cumulative_rewards = data[:, 3]
            
            # エピソード番号ごとにグループ化
            # 同じエピソード番号（各環境でのエピソード番号）を持つ全環境の累積報酬をグループ化
            episode_num_to_rewards = {}
            for ep_num, reward in zip(episode_numbers, cumulative_rewards):
                # ep_numはその環境でのエピソード番号（0始まり）
                # 同じep_numを持つ全環境の累積報酬をグループ化
                if ep_num not in episode_num_to_rewards:
                    episode_num_to_rewards[ep_num] = []
                episode_num_to_rewards[ep_num].append(reward)
        else:
            # 古い形式：episode_number,cumulative_steps,cumulative_reward,episode_length
            # 後方互換性のため、エピソード番号を環境数で割って計算
            data = np.loadtxt(episode_csv_path, delimiter=',', skiprows=1)
            num_envs = 10  # デフォルト値
            # 古い形式では、episode_numberは全環境を通しての通し番号（0, 1, 2, ..., 9, 10, 11, ...）
            # 各環境が順番にエピソードを完了したと仮定：
            # - 0-9: 1エピソード目（環境0-9）
            # - 10-19: 2エピソード目（環境0-9）
            # したがって、episode_number // num_envsが各環境でのエピソード番号
            episode_numbers = (data[:, 0] // num_envs).astype(int)
            cumulative_rewards = data[:, 2]
            
            # エピソード番号ごとにグループ化
            # 同じエピソード番号（各環境でのエピソード番号）を持つ全環境の累積報酬をグループ化
            episode_num_to_rewards = {}
            for ep_num, reward in zip(episode_numbers, cumulative_rewards):
                # ep_numは各環境でのエピソード番号（0始まり）
                # 同じep_numを持つ全環境の累積報酬をグループ化
                if ep_num not in episode_num_to_rewards:
                    episode_num_to_rewards[ep_num] = []
                episode_num_to_rewards[ep_num].append(reward)
        
        # エピソード番号ごとの平均と標準偏差を計算
        episode_nums = sorted(episode_num_to_rewards.keys())
        # max_episodesが指定されている場合は、その数までに制限
        if max_episodes is not None:
            episode_nums = [ep for ep in episode_nums if ep < max_episodes]
        episode_means = [np.mean(episode_num_to_rewards[ep]) for ep in episode_nums]
        episode_stds = [np.std(episode_num_to_rewards[ep]) if len(episode_num_to_rewards[ep]) > 1 else 0.0 
                       for ep in episode_nums]
        
        episode_graph_path = episode_csv_path.with_suffix('.png')
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        fig2.suptitle(f'Vintix Model Performance - Episodes (Parallel) - {episode_graph_path.stem}', fontsize=16, fontweight='bold')
        
        # エピソード番号ごとの平均と標準偏差をプロット
        episode_nums_1based = [ep + 1 for ep in episode_nums]  # エピソード番号は1から始まる
        ax2.plot(episode_nums_1based, episode_means, linewidth=2, label='Mean Cumulative Reward per Episode', color='green')
        ax2.fill_between(episode_nums_1based,
                         np.array(episode_means) - np.array(episode_stds),
                         np.array(episode_means) + np.array(episode_stds),
                         alpha=0.3, color='green', label='±1 Std')
        ax2.set_xlabel('Episode Number', fontsize=11)
        ax2.set_ylabel('Cumulative Reward per Episode', fontsize=11)
        ax2.set_title('Cumulative Reward per Episode (Mean ± Std)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 27)  # 縦軸を固定
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(str(episode_graph_path), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"✓ Episode graph saved: {episode_graph_path}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate graphs from CSV files with fixed y-axis")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to step-based CSV file")
    parser.add_argument("--episode_csv_path", type=str, default=None, help="Path to episode-based CSV file")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum number of episodes to plot (0-indexed, e.g., 20 means episodes 0-19)")
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    episode_csv_path = Path(args.episode_csv_path) if args.episode_csv_path else None
    
    regenerate_graphs_from_csv(csv_path, episode_csv_path, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
