#!/usr/bin/env python3
"""
Visualize multi-trajectory Algorithm Distillation datasets with step-based x-axis.
Plots episode-wise statistics with cumulative steps on x-axis instead of episode numbers.
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


def compute_epsilon_schedule(total_steps, f, p):
    """
    Compute epsilon schedule using the same formula as collect_ad_data_parallel.py
    ε(n_s) = (1 - (n_s / ((1-f)N_s))^p)^{1/p}  if n_s <= (1-f)N_s
    ε(n_s) = 0                                  if n_s > (1-f)N_s
    """
    N_s = total_steps
    threshold = (1.0 - f) * N_s
    
    epsilon_schedule = []
    for n_s in range(total_steps):
        if threshold > 0 and n_s <= threshold:
            ratio = n_s / threshold
            ratio_term = ratio ** p
            eps = (max(0.0, 1.0 - ratio_term)) ** (1.0 / p)
        else:
            eps = 0.0
        epsilon_schedule.append(eps)
    
    return epsilon_schedule


def visualize_multi_trajectories_by_steps(data_dir, output_path, f=0.0, p=0.4, target_steps_per_env=1000000):
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
    all_step_rewards = []  # List of lists: each inner list is (step, reward) tuples
    
    for traj_file in tqdm(trajectory_files, desc="Loading trajectories"):
        episodes_data, step_rewards = load_trajectory_data(traj_file)
        all_episodes_data.append(episodes_data)
        all_step_rewards.append(step_rewards)
    
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
    
    # 各ビンごとの累積報酬とエピソード長を集計
    bin_cumulative_rewards = []  # List of lists: rewards in each bin
    bin_episode_lengths = []  # List of lists: episode lengths in each bin
    
    for i in range(num_bins):
        step_min = step_bins[i]
        step_max = step_bins[i + 1]
        cum_rewards = []
        lengths = []
        
        for traj_episodes in all_episodes_data:
            for ep in traj_episodes:
                # エピソード開始時の累積ステップ数がこのビンに含まれる場合
                if step_min <= ep['cumulative_steps'] < step_max:
                    cum_rewards.append(ep['cumulative_reward'])
                    lengths.append(ep['length'])
        
        bin_cumulative_rewards.append(cum_rewards)
        bin_episode_lengths.append(lengths)
    
    # 平均と標準偏差を計算
    mean_cum_rewards = [np.mean(rews) if rews else np.nan for rews in bin_cumulative_rewards]
    std_cum_rewards = [np.std(rews) if rews and len(rews) > 1 else (0.0 if rews else np.nan) for rews in bin_cumulative_rewards]
    mean_lengths = [np.mean(lens) if lens else np.nan for lens in bin_episode_lengths]
    std_lengths = [np.std(lens) if lens and len(lens) > 1 else (0.0 if lens else np.nan) for lens in bin_episode_lengths]
    
    # ビンの中央値をx軸に使用
    bin_centers = [(step_bins[i] + step_bins[i + 1]) / 2 for i in range(num_bins)]
    
    # 2. ステップごとの報酬（横軸=ステップ数）
    # 全トラジェクトリの最大ステップ数を取得
    max_steps_actual = max([len(steps) for steps in all_step_rewards]) if all_step_rewards else 0
    
    step_wise_rewards = []  # List of lists: [traj0_step0, traj1_step0, ..., traj0_step1, ...]
    
    for step_idx in range(max_steps_actual):
        rewards_at_step = []
        for traj_steps in all_step_rewards:
            if step_idx < len(traj_steps):
                rewards_at_step.append(traj_steps[step_idx][1])  # reward value
        step_wise_rewards.append(rewards_at_step)
    
    mean_step_rewards = [np.mean(rews) if rews else 0.0 for rews in step_wise_rewards]
    std_step_rewards = [np.std(rews) if rews else 0.0 for rews in step_wise_rewards]
    
    # 3. Epsilon schedule（収集プログラムと同じ式を使用）
    epsilon_schedule = compute_epsilon_schedule(target_steps_per_env, f, p)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Algorithm Distillation - Multi-Trajectory Analysis (Step-based)\n"
        f"{data_dir.name} ({num_trajectories} trajectories)",
        fontsize=16,
        fontweight="bold",
    )
    
    # グラフ1: エピソードごとの累積報酬（横軸=累積ステップ数）
    ax1 = axes[0, 0]
    
    # 有効なデータのみをプロット
    valid_mask = ~np.isnan(mean_cum_rewards)
    valid_bin_centers = np.array(bin_centers)[valid_mask]
    valid_mean_rewards = np.array(mean_cum_rewards)[valid_mask]
    valid_std_rewards = np.array(std_cum_rewards)[valid_mask]
    
    # 平均と標準偏差をプロット
    ax1.plot(valid_bin_centers, valid_mean_rewards, linewidth=2, color="blue", label="Mean")
    ax1.fill_between(
        valid_bin_centers,
        valid_mean_rewards - valid_std_rewards,
        valid_mean_rewards + valid_std_rewards,
        alpha=0.3,
        color="blue",
        label="±1 std",
    )
    
    # 中央値も表示
    median_cum_rewards = []
    for i in range(num_bins):
        rews = bin_cumulative_rewards[i]
        if rews:
            median_cum_rewards.append(np.median(rews))
        else:
            median_cum_rewards.append(np.nan)
    
    valid_median = np.array(median_cum_rewards)[valid_mask]
    ax1.plot(valid_bin_centers, valid_median, linewidth=1.5, color="orange", linestyle="--", label="Median", alpha=0.7)
    
    ax1.set_xlabel("Cumulative Steps", fontsize=12)
    ax1.set_ylabel("Cumulative Reward", fontsize=12)
    ax1.set_title("Episode Cumulative Reward vs Cumulative Steps", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # グラフ2: エピソード長（横軸=累積ステップ数）
    ax2 = axes[0, 1]
    
    valid_mean_lengths = np.array(mean_lengths)[valid_mask]
    valid_std_lengths = np.array(std_lengths)[valid_mask]
    
    ax2.plot(valid_bin_centers, valid_mean_lengths, linewidth=2, color="green", label="Mean")
    ax2.fill_between(
        valid_bin_centers,
        valid_mean_lengths - valid_std_lengths,
        valid_mean_lengths + valid_std_lengths,
        alpha=0.3,
        color="green",
        label="±1 std",
    )
    ax2.set_xlabel("Cumulative Steps", fontsize=12)
    ax2.set_ylabel("Episode Length (steps)", fontsize=12)
    ax2.set_title("Episode Length vs Cumulative Steps", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # グラフ3: ステップごとの報酬
    ax3 = axes[1, 0]
    step_indices = np.arange(max_steps_actual)
    ax3.plot(step_indices, mean_step_rewards, linewidth=1, color="red", label="Mean", alpha=0.7)
    ax3.fill_between(
        step_indices,
        np.array(mean_step_rewards) - np.array(std_step_rewards),
        np.array(mean_step_rewards) + np.array(std_step_rewards),
        alpha=0.2,
        color="red",
        label="±1 std",
    )
    ax3.set_xlabel("Step Number", fontsize=12)
    ax3.set_ylabel("Reward per Step", fontsize=12)
    ax3.set_title("Step-wise Reward (averaged across trajectories)", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # グラフ4: Epsilon schedule
    ax4 = axes[1, 1]
    epsilon_steps = np.arange(len(epsilon_schedule))
    ax4.plot(epsilon_steps, epsilon_schedule, linewidth=2, color="purple", label="ε schedule")
    ax4.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="ε = 0 (expert)")
    ax4.axhline(y=1, color="orange", linestyle="--", alpha=0.5, label="ε = 1 (random)")
    ax4.set_xlabel("Step Number", fontsize=12)
    ax4.set_ylabel("Epsilon (ε)", fontsize=12)
    ax4.set_title(f"Epsilon Schedule (p={p}, f={f:.3f})", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(-0.05, 1.05)
    
    # ε=0になる最初のステップを表示
    zero_eps_idx = np.where(np.array(epsilon_schedule) == 0.0)[0]
    if zero_eps_idx.size > 0 and zero_eps_idx[0] > 0:
        first_zero = int(zero_eps_idx[0])
        ax4.axvline(x=first_zero, color="red", linestyle=":", alpha=0.5)
        ax4.text(
            first_zero,
            0.5,
            f"ε=0 at step {first_zero}",
            rotation=90,
            verticalalignment="center",
            fontsize=10,
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n✅ Visualization saved: {output_path}")
    print("\n" + "=" * 80)
    print("Multi-Trajectory Statistics Summary (Step-based)")
    print("=" * 80)
    print(f"Total Trajectories: {num_trajectories}")
    print(f"Episodes per Trajectory: {min_episodes} ~ {max_episodes}")
    print(f"Total Episodes: {sum(episodes_per_traj)}")
    print(f"Max Steps per Trajectory: {max_steps_actual}")
    if mean_cum_rewards:
        valid_rewards = [r for r in mean_cum_rewards if not np.isnan(r)]
        valid_stds = [s for s in std_cum_rewards if not np.isnan(s) and s > 0]
        if valid_rewards:
            print("\nCumulative Reward per Episode (by cumulative steps):")
            if valid_stds:
                print(f"  Mean: {np.mean(valid_rewards):.4f} ± {np.mean(valid_stds):.4f}")
            else:
                print(f"  Mean: {np.mean(valid_rewards):.4f} ± N/A")
    if mean_lengths:
        valid_lengths = [l for l in mean_lengths if not np.isnan(l)]
        valid_stds_lengths = [s for s in std_lengths if not np.isnan(s) and s > 0]
        if valid_lengths:
            print("\nEpisode Length (by cumulative steps):")
            if valid_stds_lengths:
                print(f"  Mean: {np.mean(valid_lengths):.1f} ± {np.mean(valid_stds_lengths):.1f}")
            else:
                print(f"  Mean: {np.mean(valid_lengths):.1f} ± N/A")
    print("\nEpsilon Schedule:")
    if epsilon_schedule:
        print(f"  Initial: {epsilon_schedule[0]:.4f}")
        print(f"  Final: {epsilon_schedule[-1]:.4f}")
        if zero_eps_idx.size > 0 and zero_eps_idx[0] > 0:
            print(f"  ε=0 from step: {int(zero_eps_idx[0])}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-trajectory AD datasets with step-based x-axis"
    )
    parser.add_argument("data_dir", type=str, help="Directory containing trajectory HDF5 files")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: <data_dir>/<data_dir>_step_based_analysis.png)",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=0.05,
        help="Noise-free fraction f (default: 0.05)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.4,
        help="Decay power p (default: 0.4)",
    )
    parser.add_argument(
        "--target_steps_per_env",
        type=int,
        default=1000000,
        help="Target steps per environment for epsilon schedule (default: 1000000)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if args.output is None:
        output_path = data_dir / f"{data_dir.name}_step_based_analysis.png"
    else:
        output_path = Path(args.output)
    
    visualize_multi_trajectories_by_steps(
        data_dir,
        output_path,
        f=args.f,
        p=args.p,
        target_steps_per_env=args.target_steps_per_env,
    )


if __name__ == "__main__":
    main()

