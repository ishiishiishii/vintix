#!/usr/bin/env python3
"""
補助解析ツール:
  - 評価ログ (confirm_eval_history_recorder.py で生成) を用いて、
    Mahalanobis 距離がしきい値を超えた "OOD スパイク" のタイミングと
    報酬推移を可視化する。

本研究の中核ではなく、データ診断のための確認目的コード。
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def load_expert_observations(path: Path,
                             key: str = "proprio_observation",
                             max_samples: int = 200_000,
                             seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    total = 0
    with h5py.File(path, "r") as f:
        group_names = sorted(f.keys(), key=lambda s: int(s.split("-")[0]))
        for gname in group_names:
            data = f[gname][key][:]
            if data.size == 0:
                continue
            remaining = max_samples - total
            if remaining <= 0:
                break
            if data.shape[0] <= remaining:
                samples.append(data)
                total += data.shape[0]
            else:
                idx = rng.choice(data.shape[0], size=remaining, replace=False)
                samples.append(data[idx])
                total += remaining
    if not samples:
        raise ValueError("No expert data found.")
    return np.concatenate(samples, axis=0)


def compute_mahalanobis(data: np.ndarray,
                        mean: np.ndarray,
                        cov: np.ndarray,
                        eps: float = 1e-6) -> np.ndarray:
    cov_reg = cov + np.eye(cov.shape[0]) * eps
    inv_cov = np.linalg.inv(cov_reg)
    diff = data - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))


def main():
    parser = argparse.ArgumentParser(description="OOD スパイク解析（確認用）")
    parser.add_argument("--expert_path", type=str, required=True, help="専門家データ HDF5")
    parser.add_argument("--eval_path", type=str, required=True, help="評価ログ HDF5")
    parser.add_argument("--output_prefix", type=str, required=True, help="出力先ファイル接頭辞")
    parser.add_argument("--threshold", type=float, default=20.0, help="Mahalanobis しきい値")
    parser.add_argument("--max_expert_samples", type=int, default=200_000, help="専門家サンプル数上限")
    parser.add_argument("--seed", type=int, default=0, help="サンプリング seed")
    parser.add_argument("--warmup_steps", type=int, default=15, help="各エピソード先頭の除外ステップ数")
    args = parser.parse_args()

    expert_path = Path(args.expert_path)
    eval_path = Path(args.eval_path)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Expert statistics
    expert_obs = load_expert_observations(
        expert_path, max_samples=args.max_expert_samples, seed=args.seed
    )
    expert_mean = expert_obs.mean(axis=0)
    expert_cov = np.cov((expert_obs - expert_mean), rowvar=False)

    # Evaluation data
    with h5py.File(eval_path, "r") as f:
        if len(f.keys()) != 1:
            raise RuntimeError("評価 HDF5 には単一グループのみを想定しています。")
        grp = f[list(f.keys())[0]]
        obs_eval = grp["proprio_observation"][:]
        rewards_eval = grp["reward"][:]
        step_nums = grp["step_num"][:]
        episode_ids = grp.get("episode_id")
        if episode_ids is None:
            raise RuntimeError("evaluation データに episode_id が存在しません。recorder を更新してください。")
        episode_ids = episode_ids[:]

    warmup_mask_path = eval_path.with_suffix(".warmup.npy")
    warmup_mask = None
    if warmup_mask_path.exists():
        warmup_mask = np.load(warmup_mask_path)
        if warmup_mask.shape[0] != rewards_eval.shape[0]:
            warmup_mask = None

    mahal_dist = compute_mahalanobis(obs_eval, expert_mean, expert_cov)

    unique_episodes = np.unique(episode_ids)
    records = []
    spike_steps = []

    for ep in unique_episodes:
        idx = np.where(episode_ids == ep)[0]
        if idx.size == 0:
            continue

        # Apply warmup exclusion
        warmup_cut = min(args.warmup_steps, idx.size)
        valid_idx = idx[warmup_cut:]
        if valid_idx.size == 0:
            continue

        maha_ep = mahal_dist[valid_idx]
        reward_ep = rewards_eval[valid_idx]
        total_reward_ep = reward_ep.sum()

        max_maha = float(maha_ep.max()) if maha_ep.size else float("nan")
        spike_mask = maha_ep > args.threshold
        first_spike = int(valid_idx[np.argmax(spike_mask)]) if np.any(spike_mask) else -1

        if first_spike >= 0:
            post_idx = valid_idx[valid_idx >= first_spike]
            pre_idx = valid_idx[valid_idx < first_spike]
            reward_post = rewards_eval[post_idx].sum()
            reward_pre = rewards_eval[pre_idx].sum()
            spike_steps.append(first_spike)
        else:
            reward_post = float("nan")
            reward_pre = total_reward_ep

        records.append({
            "episode_id": int(ep),
            "episode_length": int(idx.size),
            "effective_length": int(valid_idx.size),
            "total_reward": float(total_reward_ep),
            "max_mahalanobis": max_maha,
            "first_spike_step": first_spike,
            "reward_pre_spike": float(reward_pre),
            "reward_post_spike": float(reward_post),
        })

    # Save CSV
    csv_path = output_prefix.with_suffix(".csv")
    with open(csv_path, "w") as f:
        headers = [
            "episode_id",
            "episode_length",
            "effective_length",
            "total_reward",
            "max_mahalanobis",
            "first_spike_step",
            "reward_pre_spike",
            "reward_post_spike",
        ]
        f.write(",".join(headers) + "\n")
        for rec in records:
            row = ",".join(str(rec[h]) for h in headers)
            f.write(row + "\n")

    # Plot time series and episode summaries
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle("OOD Spike Analysis (auxiliary)", fontsize=16, fontweight="bold")

    ax0 = axes[0]
    ax0.plot(step_nums, mahal_dist, color="#DD8452", linewidth=1)
    ax0.axhline(args.threshold, color="red", linestyle="--", label=f"Threshold={args.threshold}")
    if warmup_mask is not None:
        warmup_steps_idx = np.where(warmup_mask)[0]
        if warmup_steps_idx.size > 0:
            ax0.scatter(step_nums[warmup_steps_idx], mahal_dist[warmup_steps_idx],
                        s=4, color="#4C72B0", alpha=0.3, label="Warmup steps")
    ax0.set_title("Mahalanobis Distance over Steps", fontweight="bold")
    ax0.set_ylabel("Distance")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(step_nums, np.cumsum(rewards_eval), color="#55A868", linewidth=1)
    ax1.set_title("Cumulative Reward over Steps", fontweight="bold")
    ax1.set_ylabel("Cum. Reward")
    ax1.grid(True, alpha=0.3)
    if spike_steps:
        ax1.scatter(step_nums[np.array(spike_steps, dtype=int)],
                    np.cumsum(rewards_eval)[np.array(spike_steps, dtype=int)],
                    color="#C44E52", s=20, label="First spikes")
        ax1.legend()

    ax2 = axes[2]
    episode_ids_sorted = [rec["episode_id"] for rec in records]
    max_maha_sorted = [rec["max_mahalanobis"] for rec in records]
    total_reward_sorted = [rec["total_reward"] for rec in records]
    ax2.bar(episode_ids_sorted, max_maha_sorted, color="#DD8452", alpha=0.7, label="Max Mahalanobis")
    ax22 = ax2.twinx()
    ax22.plot(episode_ids_sorted, total_reward_sorted, color="#55A868", marker="o",
              linestyle="-", label="Total Reward")
    ax2.axhline(args.threshold, color="red", linestyle="--")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Max Mahalanobis", color="#DD8452")
    ax22.set_ylabel("Total Reward", color="#55A868")
    ax2.set_title("Episode Summary", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax22.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    plot_path = output_prefix.with_suffix(".png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary text
    summary_path = output_prefix.with_suffix(".txt")
    num_spike_episodes = sum(1 for rec in records if rec["first_spike_step"] >= 0)
    with open(summary_path, "w") as f:
        f.write("OOD Spike Analysis Summary\n")
        f.write(f"Eval steps: {len(step_nums)}\n")
        f.write(f"Episodes: {len(records)}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Episodes with spike: {num_spike_episodes}\n")
        if num_spike_episodes > 0:
            post_rewards = [rec["reward_post_spike"]
                            for rec in records if rec["first_spike_step"] >= 0 and not np.isnan(rec["reward_post_spike"])]
            f.write(f"Mean post-spike reward sum: {np.mean(post_rewards):.4f}\n")

    print(f"✓ Analysis saved: {plot_path.name}, {csv_path.name}, {summary_path.name}")


if __name__ == "__main__":
    main()

