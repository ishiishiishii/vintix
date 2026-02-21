#!/usr/bin/env python3
"""
各履歴データディレクトリから「累積ステップ vs エピソードごと累積報酬」のグラフを、
題名なし・軸ラベル・軸数値を5倍大きくした見やすい版として新規出力する。
既存のグラフは削除せず、各データフォルダに cumulative_reward_readable.pdf で保存する。
"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 既存スクリプトのロジックを再利用
sys.path.insert(0, str(Path(__file__).parent))
from generate_cumulative_reward_plot import load_trajectory_data


# 軸ラベル・軸数値のフォントサイズ（すべて同じ大きさ）
FONT_SIZE_LABEL = 34   # xlabel, ylabel
FONT_SIZE_TICK = 28    # 軸の目盛り数字（1e6 なども同じサイズ）


def generate_readable_plot(data_dir: Path, output_path: Path, target_steps_per_env: int = 1_000_000):
    """
    累積ステップ vs エピソード累積報酬のグラフを、題名なし・軸を大きくして描画し保存する。
    データの読み込み・ビン集計は generate_cumulative_reward_plot と同じ。
    """
    data_dir = Path(data_dir)
    trajectory_files = sorted(data_dir.glob("trajectory_*.h5"))
    if not trajectory_files:
        trajectory_files = sorted(data_dir.glob("trajectories_*.h5"))
    if not trajectory_files:
        trajectory_files = sorted(data_dir.glob("trajectories_env_*.h5"))

    if not trajectory_files:
        raise FileNotFoundError(f"No trajectory HDF5 files found in {data_dir}")

    all_episodes_data = []
    for traj_file in tqdm(trajectory_files, desc="Loading", leave=False):
        episodes_data, _ = load_trajectory_data(traj_file)
        all_episodes_data.append(episodes_data)

    num_bins = 100
    max_steps = target_steps_per_env
    step_bins = np.linspace(0, max_steps, num_bins + 1)
    bin_cumulative_rewards = []

    for i in range(num_bins):
        step_min, step_max = step_bins[i], step_bins[i + 1]
        cum_rewards = []
        for traj_episodes in all_episodes_data:
            for ep in traj_episodes:
                if step_min <= ep["cumulative_steps"] < step_max:
                    cum_rewards.append(ep["cumulative_reward"])
        bin_cumulative_rewards.append(cum_rewards)

    mean_cum_rewards = [np.mean(rews) if rews else np.nan for rews in bin_cumulative_rewards]
    std_cum_rewards = [np.std(rews) if rews and len(rews) > 1 else (0.0 if rews else np.nan) for rews in bin_cumulative_rewards]
    bin_centers = [(step_bins[i] + step_bins[i + 1]) / 2 for i in range(num_bins)]

    valid_mask = ~np.isnan(mean_cum_rewards)
    valid_bin_centers = np.array(bin_centers)[valid_mask]
    valid_mean_rewards = np.array(mean_cum_rewards)[valid_mask]
    valid_std_rewards = np.array(std_cum_rewards)[valid_mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    # 題名は一切つけない（suptitle / title なし）
    ax.plot(valid_bin_centers, valid_mean_rewards, linewidth=2, color="blue", label="Mean")
    ax.fill_between(
        valid_bin_centers,
        valid_mean_rewards - valid_std_rewards,
        valid_mean_rewards + valid_std_rewards,
        alpha=0.3,
        color="blue",
        label="±1 std",
    )

    ax.set_xlabel("Cumulative Steps", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Cumulative Reward", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    # y軸レンジを統一（下限−1.5、上限25）
    ax.set_ylim(-1.5, 25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_TICK)

    plt.tight_layout()
    # 科学表記のオフセット（1e6 など）は別オブジェクトのため、ここでフォントサイズを揃える
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    output_path = Path(output_path)
    if not str(output_path).lower().endswith(".pdf"):
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate readable cumulative reward plots (no title, 5x larger axis).")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory containing go2_trajectories, go1_trajectories, etc. (default: vintix_go2/data)")
    args = parser.parse_args()

    if args.data_root is not None:
        base_data_dir = Path(args.data_root)
    else:
        base_data_dir = Path(__file__).resolve().parents[1] / "data"

    dirs_and_output_names = [
        (base_data_dir / "go2_trajectories" / "data_1M", "cumulative_reward_readable.pdf"),
        (base_data_dir / "go1_trajectories", "cumulative_reward_readable.pdf"),
        (base_data_dir / "a1_trajectories", "cumulative_reward_readable.pdf"),
        (base_data_dir / "minicheetah_trajectories", "cumulative_reward_readable.pdf"),
    ]

    for data_dir, out_name in dirs_and_output_names:
        if not data_dir.exists():
            print(f"⚠ Skip (not found): {data_dir}")
            continue
        out_path = data_dir / out_name
        print(f"Generating: {data_dir} -> {out_path}")
        try:
            generate_readable_plot(data_dir, out_path)
            print(f"  Saved: {out_path}")
        except FileNotFoundError as e:
            print(f"  Skip: {e}")
        except Exception as e:
            print(f"  Error: {e}")
            raise

    print("\nDone. Existing graphs were not modified.")


if __name__ == "__main__":
    main()
