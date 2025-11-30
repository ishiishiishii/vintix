#!/usr/bin/env python3
"""
Visualize multi-trajectory Algorithm Distillation datasets.
Plots episode-wise and step-wise statistics with proper ordering.
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
    episodes_data = []  # List of dicts: {cumulative_reward, length, episode_idx}
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
        episode_idx = 0
        global_step = 0
        
        for reward, step_num in zip(all_rewards, all_step_nums):
            # ステップごとのデータを保存（順序を保持）
            step_rewards.append((global_step, float(reward)))
            global_step += 1
            
            # エピソードの区切り（step_num == 0）
            if step_num == 0 and len(current_episode_rewards) > 0:
                episodes_data.append({
                    "episode_idx": episode_idx,
                    "cumulative_reward": float(sum(current_episode_rewards)),
                    "length": len(current_episode_rewards),
                    "avg_reward": float(np.mean(current_episode_rewards)),
                })
                episode_idx += 1
                current_episode_rewards = []
            
            current_episode_rewards.append(float(reward))
        
        # 最後のエピソード
        if len(current_episode_rewards) > 0:
            episodes_data.append({
                "episode_idx": episode_idx,
                "cumulative_reward": float(sum(current_episode_rewards)),
                "length": len(current_episode_rewards),
                "avg_reward": float(np.mean(current_episode_rewards)),
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


def visualize_multi_trajectories(data_dir, output_path, f=0.0, p=0.4, target_steps_per_env=1000000):
    data_dir = Path(data_dir)
    trajectory_files = sorted(data_dir.glob("trajectory_*.h5"))
    if not trajectory_files:
        trajectory_files = sorted(data_dir.glob("trajectories_*.h5"))
    num_trajectories = len(trajectory_files)

    print(f"Loading {num_trajectories} trajectory files from: {data_dir}")

    if num_trajectories == 0:
        raise FileNotFoundError(f"No trajectory_*.h5 files found in {data_dir}")

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
    
    # 1. エピソードごとの累積報酬（横軸=エピソード番号）
    episode_cumulative_rewards = []  # List of lists: [traj0_ep0, traj1_ep0, ..., traj0_ep1, ...]
    episode_lengths = []  # List of lists
    
    for ep_idx in range(max_episodes):
        cum_rewards = []
        lengths = []
        for traj_episodes in all_episodes_data:
            if ep_idx < len(traj_episodes):
                cum_rewards.append(traj_episodes[ep_idx]["cumulative_reward"])
                lengths.append(traj_episodes[ep_idx]["length"])
        episode_cumulative_rewards.append(cum_rewards)
        episode_lengths.append(lengths)
    
    # 平均と標準偏差を計算
    mean_cum_rewards = [np.mean(rews) if rews else 0.0 for rews in episode_cumulative_rewards]
    std_cum_rewards = [np.std(rews) if rews else 0.0 for rews in episode_cumulative_rewards]
    mean_lengths = [np.mean(lens) if lens else 0.0 for lens in episode_lengths]
    std_lengths = [np.std(lens) if lens else 0.0 for lens in episode_lengths]
    
    # 2. ステップごとの報酬（横軸=ステップ数）
    # 全トラジェクトリの最大ステップ数を取得
    max_steps = max([len(steps) for steps in all_step_rewards]) if all_step_rewards else 0
    
    step_wise_rewards = []  # List of lists: [traj0_step0, traj1_step0, ..., traj0_step1, ...]
    
    for step_idx in range(max_steps):
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
        f"Algorithm Distillation - Multi-Trajectory Analysis\n"
        f"{data_dir.name} ({num_trajectories} trajectories)",
        fontsize=16,
        fontweight="bold",
    )
    
    # グラフ1: エピソードごとの累積報酬
    ax1 = axes[0, 0]
    episode_indices = np.arange(max_episodes)
    
    # 平均と標準偏差をプロット
    ax1.plot(episode_indices, mean_cum_rewards, linewidth=2, color="blue", label="Mean")
    ax1.fill_between(
        episode_indices,
        np.array(mean_cum_rewards) - np.array(std_cum_rewards),
        np.array(mean_cum_rewards) + np.array(std_cum_rewards),
        alpha=0.3,
        color="blue",
        label="±1 std",
    )
    
    # 中央値も表示（外れ値の影響を減らすため）
    median_cum_rewards = []
    for ep_idx in range(max_episodes):
        rews = episode_cumulative_rewards[ep_idx] if ep_idx < len(episode_cumulative_rewards) else []
        if rews:
            median_cum_rewards.append(np.median(rews))
        else:
            median_cum_rewards.append(0.0)
    ax1.plot(episode_indices, median_cum_rewards, linewidth=1.5, color="orange", linestyle="--", label="Median", alpha=0.7)
    
    ax1.set_xlabel("Episode Number", fontsize=12)
    ax1.set_ylabel("Cumulative Reward", fontsize=12)
    ax1.set_title("Episode Cumulative Reward (averaged across trajectories)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 最初のエピソードの詳細をテキストで表示
    if episode_cumulative_rewards and len(episode_cumulative_rewards[0]) > 0:
        first_ep_rewards = episode_cumulative_rewards[0]
        stats_text = (
            f"Ep 0: Mean={np.mean(first_ep_rewards):.2f}, "
            f"Median={np.median(first_ep_rewards):.2f}, "
            f"Std={np.std(first_ep_rewards):.2f}\n"
            f"Range: [{np.min(first_ep_rewards):.2f}, {np.max(first_ep_rewards):.2f}]"
        )
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )
    
    # グラフ2: エピソード長
    ax2 = axes[0, 1]
    ax2.plot(episode_indices, mean_lengths, linewidth=2, color="green", label="Mean")
    ax2.fill_between(
        episode_indices,
        np.array(mean_lengths) - np.array(std_lengths),
        np.array(mean_lengths) + np.array(std_lengths),
        alpha=0.3,
        color="green",
        label="±1 std",
    )
    ax2.set_xlabel("Episode Number", fontsize=12)
    ax2.set_ylabel("Episode Length (steps)", fontsize=12)
    ax2.set_title("Episode Length (averaged across trajectories)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # グラフ3: ステップごとの報酬
    ax3 = axes[1, 0]
    step_indices = np.arange(max_steps)
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
    print("Multi-Trajectory Statistics Summary")
    print("=" * 80)
    print(f"Total Trajectories: {num_trajectories}")
    print(f"Episodes per Trajectory: {min_episodes} ~ {max_episodes}")
    print(f"Total Episodes: {sum(episodes_per_traj)}")
    print(f"Max Steps per Trajectory: {max_steps}")
    if mean_cum_rewards:
        print("\nCumulative Reward per Episode (averaged):")
        print(f"  Mean: {np.mean(mean_cum_rewards):.4f} ± {np.mean(std_cum_rewards):.4f}")
        print(f"  Initial (ep 0): {mean_cum_rewards[0]:.4f} ± {std_cum_rewards[0]:.4f}")
        if len(mean_cum_rewards) > 1:
            print(f"  Final (ep {max_episodes-1}): {mean_cum_rewards[-1]:.4f} ± {std_cum_rewards[-1]:.4f}")
    if mean_lengths:
        print("\nEpisode Length (averaged):")
        print(f"  Mean: {np.mean(mean_lengths):.1f} ± {np.mean(std_lengths):.1f}")
        print(f"  Initial (ep 0): {mean_lengths[0]:.1f} ± {std_lengths[0]:.1f}")
        if len(mean_lengths) > 1:
            print(f"  Final (ep {max_episodes-1}): {mean_lengths[-1]:.1f} ± {std_lengths[-1]:.1f}")
    print("\nEpsilon Schedule:")
    if epsilon_schedule:
        print(f"  Initial: {epsilon_schedule[0]:.4f}")
        print(f"  Final: {epsilon_schedule[-1]:.4f}")
        if zero_eps_idx.size > 0 and zero_eps_idx[0] > 0:
            print(f"  ε=0 from step: {int(zero_eps_idx[0])}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Algorithm Distillation trajectory datasets",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Directory containing trajectory files or a single trajectory HDF5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG file path (default: auto-generated)",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=None,
        help="Noise-free fraction parameter f (if provided, overrides --max_perf)",
    )
    parser.add_argument(
        "--max_perf",
        type=float,
        default=1.0,
        help="Maximum performance level (0.0=random, 1.0=expert). f = 1.0 - max_perf (default: 1.0)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.4,
        help="Decay power parameter (default: 0.4)",
    )
    parser.add_argument(
        "--target_steps_per_env",
        type=int,
        default=1000000,
        help="Target steps per environment for epsilon schedule (default: 1000000)",
    )

    args = parser.parse_args()
    
    # f が指定されていない場合は max_perf から計算
    if args.f is None:
        args.f = 1.0 - args.max_perf

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        data_dir = input_path.parent
        trajectory_files = [input_path]
    else:
        data_dir = input_path
        trajectory_files = sorted(data_dir.glob("trajectory_*.h5"))
        if not trajectory_files:
            trajectory_files = sorted(data_dir.glob("trajectories_*.h5"))

    if not trajectory_files:
        raise FileNotFoundError(
            f"No trajectory files found in {input_path}. Expected names like "
            "trajectory_XXXX.h5 or trajectories_XXXX.h5"
        )

    if args.output is None:
        output_filename = f"{data_dir.name}_multi_trajectory_analysis.png"
        output_path = data_dir / output_filename
    else:
        output_path = Path(args.output)

    print("=" * 80)
    print("Multi-Trajectory Algorithm Distillation Data Visualization")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Parameters: max_perf={args.max_perf}, f={args.f:.3f}, p={args.p}, target_steps={args.target_steps_per_env}")
    print("=" * 80 + "\n")

    visualize_multi_trajectories(
        data_dir, 
        output_path, 
        f=args.f, 
        p=args.p,
        target_steps_per_env=args.target_steps_per_env
    )


if __name__ == "__main__":
    main()
