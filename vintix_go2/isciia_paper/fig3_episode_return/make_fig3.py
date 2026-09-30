#!/usr/bin/env python3
"""
論文 Fig. 3（連続ノイズ蒸留中の平均エピソードリターン）を 4 ロボット分 PDF で出力する。

投稿版の図（旧 ``scripts/generate_readable_cumulative_reward_plots.py`` で作成）と同じデータ・
集計・体裁で、縦軸名だけ本文に合わせて "Average episode return" にしている。
集計は ``scripts/visualize_trajectories.py`` の関数をそのまま使う。

使い方（コンテナ内、vintix_go2 直下で）:
  python isciia_paper/fig3_episode_return/make_fig3.py
出力: このディレクトリに ``<robot>cumulative_reward_readable.pdf``（論文の figs/ と同じファイル名）
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_OUT_DIR = Path(__file__).resolve().parent
_VINTIX_ROOT = _OUT_DIR.parents[1]
sys.path.insert(0, str(_VINTIX_ROOT / "scripts"))
from visualize_trajectories import (  # noqa: E402
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    compute_bin_episode_stats,
    discover_trajectory_files,
    load_trajectory_data,
)

DATA_DIRS = {
    "go1": _VINTIX_ROOT / "data/go1_trajectories",
    "go2": _VINTIX_ROOT / "data/go2_trajectories/data_1M",
    "a1": _VINTIX_ROOT / "data/a1_trajectories",
    "minicheetah": _VINTIX_ROOT / "data/minicheetah_trajectories",
}
MAX_STEPS = 1_000_000
YLABEL = "Average episode return"


def make_plot(data_dir: Path, out_path: Path) -> None:
    all_episodes = [
        load_trajectory_data(f, quiet=True)[0]
        for f in tqdm(discover_trajectory_files(data_dir), desc=data_dir.name, leave=False)
    ]
    centers, means, stds, *_ = compute_bin_episode_stats(all_episodes, max_steps=MAX_STEPS, num_bins=100)
    means = np.asarray(means, dtype=np.float64)
    valid = ~np.isnan(means)
    x = np.asarray(centers)[valid]
    m = means[valid]
    s = np.asarray(stds, dtype=np.float64)[valid]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, m, linewidth=2, color="blue", label="Mean")
    ax.fill_between(x, m - s, m + s, alpha=0.3, color="blue", label="±1 std")
    ax.set_xlabel("Cumulative Steps", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(-1.5, 25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_TICK)
    plt.tight_layout()
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for robot, data_dir in DATA_DIRS.items():
        out_path = _OUT_DIR / f"{robot}cumulative_reward_readable.pdf"
        make_plot(data_dir, out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
