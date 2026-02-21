#!/usr/bin/env python3
"""
Generate the Episode Cumulative Reward vs Cumulative Steps plot 
from trajectory data (only the top-left subplot).
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_trajectory_data(h5_path: Path):
    """Load trajectory data preserving order: episodes and step-wise rewards."""
    episodes_data = []  # List of dicts: {cumulative_reward, length, cumulative_steps}
    step_rewards = []  # List of (step_num, reward) tuples in order
    
    with h5py.File(h5_path, "r") as f:
        # グループ名を数値順にソート（データの順序を保持）
        group_names = sorted(f.keys(), key=lambda x: int(x.split('-')[0]))
        
        all_rewards = []
        all_step_nums = []
        
        for group_name in group_names:
            rewards = np.array(f[group_name]["reward"])
            step_nums = np.array(f[group_name]["step_num"])
            all_rewards.append(rewards)
            all_step_nums.append(step_nums)
        
        if not all_rewards:
            return episodes_data, step_rewards
        
        # データを順番に結合（順序を保持）
        all_rewards = np.concatenate(all_rewards)
        all_step_nums = np.concatenate(all_step_nums)
        
        # エピソードを分割（step_num == 0で区切る）
        current_episode_rewards = []
        cumulative_steps = 0
        global_step = 0
        
        for reward, step_num in zip(all_rewards, all_step_nums):
            # ステップごとのデータを保存（順序を保持）
            step_rewards.append((global_step, float(reward)))
            global_step += 1
            
            # エピソードの区切り（step_num == 0）
            if step_num == 0 and len(current_episode_rewards) > 0:
                episodes_data.append({
                    "cumulative_reward": float(sum(current_episode_rewards)),
                    "length": len(current_episode_rewards),
                    "cumulative_steps": cumulative_steps,  # エピソード開始時の累積ステップ数
                })
                cumulative_steps += len(current_episode_rewards)
                current_episode_rewards = []
            
            current_episode_rewards.append(float(reward))
        
        # 最後のエピソード
        if len(current_episode_rewards) > 0:
            episodes_data.append({
                "cumulative_reward": float(sum(current_episode_rewards)),
                "length": len(current_episode_rewards),
                "cumulative_steps": cumulative_steps,
            })
    
    return episodes_data, step_rewards


def generate_cumulative_reward_plot(data_dir, output_path, title, target_steps_per_env=1000000):
    """
    Generate only the Episode Cumulative Reward vs Cumulative Steps plot.
    """
    data_dir = Path(data_dir)
    trajectory_files = sorted(data_dir.glob("trajectory_*.h5"))
    if not trajectory_files:
        trajectory_files = sorted(data_dir.glob("trajectories_*.h5"))
    if not trajectory_files:
        trajectory_files = sorted(data_dir.glob("trajectories_env_*.h5"))
    num_trajectories = len(trajectory_files)

    print(f"Loading {num_trajectories} trajectory files from: {data_dir}")

    if num_trajectories == 0:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")

    # 各トラジェクトリのデータを読み込み（順序を保持）
    all_episodes_data = []  # List of lists: each inner list is episodes for one trajectory
    
    for traj_file in tqdm(trajectory_files, desc="Loading trajectories"):
        episodes_data, _ = load_trajectory_data(traj_file)
        all_episodes_data.append(episodes_data)
    
    episodes_per_traj = [len(episodes) for episodes in all_episodes_data]
    max_episodes = max(episodes_per_traj) if episodes_per_traj else 0
    min_episodes = min(episodes_per_traj) if episodes_per_traj else 0
    
    print("\nTrajectory info:")
    print(f"  Total trajectories: {num_trajectories}")
    print(f"  Episodes per trajectory: {min_episodes} ~ {max_episodes}")
    
    # 累積ステップ数でビン分割（100分割）
    max_steps = target_steps_per_env
    num_bins = 100
    step_bins = np.linspace(0, max_steps, num_bins + 1)
    
    # 各ビンごとの累積報酬を集計
    bin_cumulative_rewards = []  # List of lists: rewards in each bin
    
    for i in range(num_bins):
        step_min = step_bins[i]
        step_max = step_bins[i + 1]
        cum_rewards = []
        
        for traj_episodes in all_episodes_data:
            for ep in traj_episodes:
                # エピソード開始時の累積ステップ数がこのビンに含まれる場合
                if step_min <= ep['cumulative_steps'] < step_max:
                    cum_rewards.append(ep['cumulative_reward'])
        
        bin_cumulative_rewards.append(cum_rewards)
    
    # 平均と標準偏差を計算
    mean_cum_rewards = [np.mean(rews) if rews else np.nan for rews in bin_cumulative_rewards]
    std_cum_rewards = [np.std(rews) if rews and len(rews) > 1 else (0.0 if rews else np.nan) for rews in bin_cumulative_rewards]
    
    # ビンの中央値をx軸に使用
    bin_centers = [(step_bins[i] + step_bins[i + 1]) / 2 for i in range(num_bins)]
    
    # 可視化（左上のサブプロットのみ）
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    # 有効なデータのみをプロット
    valid_mask = ~np.isnan(mean_cum_rewards)
    valid_bin_centers = np.array(bin_centers)[valid_mask]
    valid_mean_rewards = np.array(mean_cum_rewards)[valid_mask]
    valid_std_rewards = np.array(std_cum_rewards)[valid_mask]
    
    # 平均と標準偏差をプロット
    ax.plot(valid_bin_centers, valid_mean_rewards, linewidth=2, color="blue", label="Mean")
    ax.fill_between(
        valid_bin_centers,
        valid_mean_rewards - valid_std_rewards,
        valid_mean_rewards + valid_std_rewards,
        alpha=0.3,
        color="blue",
        label="±1 std",
    )
    
    ax.set_xlabel("Cumulative Steps", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title("Episode Cumulative Reward vs Cumulative Steps", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Ensure PDF format
    if not str(output_path).endswith('.pdf'):
        output_path = output_path.with_suffix('.pdf')
    
    # Save to temporary location first, then move with sudo if needed
    import tempfile
    import shutil
    import subprocess
    
    temp_file = Path(tempfile.gettempdir()) / f"temp_{output_path.name}"
    plt.savefig(temp_file, format='pdf', bbox_inches="tight")
    plt.close()
    
    # Try to move the file to the final location
    try:
        shutil.move(str(temp_file), str(output_path))
        print(f"\n✅ Visualization saved: {output_path}")
    except PermissionError:
        # If permission denied, use sudo to move
        subprocess.run(['sudo', 'mv', str(temp_file), str(output_path)], check=True)
        print(f"\n✅ Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Episode Cumulative Reward vs Cumulative Steps plot from trajectory data"
    )
    parser.add_argument("data_dir", type=str, help="Directory containing trajectory HDF5 files")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: <data_dir>/<title>_cumulative_reward.pdf)",
    )
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Title for the graph (e.g., 'go2_trajectories')",
    )
    parser.add_argument(
        "--target_steps_per_env",
        type=int,
        default=1000000,
        help="Target steps per environment (default: 1000000)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if args.output is None:
        output_path = data_dir / f"{args.title}_cumulative_reward.pdf"
    else:
        output_path = Path(args.output)
        # Ensure PDF extension
        if not str(output_path).endswith('.pdf'):
            output_path = output_path.with_suffix('.pdf')
    
    generate_cumulative_reward_plot(
        data_dir,
        output_path,
        args.title,
        target_steps_per_env=args.target_steps_per_env,
    )


if __name__ == "__main__":
    main()
