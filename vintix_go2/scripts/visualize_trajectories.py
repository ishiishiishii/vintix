#!/usr/bin/env python3
"""
visualize_trajectories.py — 軌道 HDF5 の可視化（累積ステップ横軸）
================================================================

【役割】
  指定ディレクトリ内の **全ての** 軌道 HDF5（``trajectory_*.h5`` / ``trajectories_*.h5`` /
  ``trajectories_env_*.h5`` のいずれかにマッチした集合）を読み込み、データを **プール** して
  **横軸＝累積ステップ数**の次の **3 種類の PNG** を **1 セットだけ** 出力する（従来の 2×2 のうち
  エピソード系・ステップ系と同じ統計: **ビン内または系列ごとの平均と標準偏差**）。

  1. **累積報酬** … 累積ステップのビンごとに、そのビンに含まれる全エピソード（全 HDF5 合算）の
     累積報酬の平均と標準偏差（±1 std で塗り）。
  2. **エピソード長** … 同じビンごとのエピソード長の平均と標準偏差。
  3. **ステップごとの報酬** … HDF5 が複数あるときは各ファイルのステップ平均報酬をインデックスで
     揃え、**ファイル間**の平均と標準偏差。1 ファイルのみのときは従来どおりそのファイル内の
     平均±std（PPO 並列なら環境間）。

  図の体裁は ``generate_readable_eval_graphs.py`` / ``eval_vintix.py`` と同系
  （ラベル 34 / 目盛 28、題名なし、``figsize=(14,10)``）。

【入力】
  第 1 引数 ``DATA`` は **ディレクトリ**、または **.h5 ファイルのパス**（そのファイルが置いてある
  **ディレクトリ内の全軌道 HDF5** を対象にする）。

【出力】
  軌道データと **同じディレクトリ**に、ディレクトリ名（または単一 .h5 を渡したときはその親フォルダ名）を
  接頭辞とした 3 ファイル（英語ファイル名。同じ接頭辞の旧日本語名 PNG があれば削除して置き換え）:

  - ``<dir_name>_episode_cumulative_reward.png``
  - ``<dir_name>_episode_length.png``
  - ``<dir_name>_reward_per_step.png``

  横軸（累積ステップ）は科学表記（目盛は係数、``×10ⁿ`` 形式のオフセット表示）を使用。

【使用例】
  python scripts/visualize_trajectories.py data/ppo_history
  python scripts/visualize_trajectories.py data/ppo_history/trajectories_env_0000.h5

【主なオプション】
  --target_steps_per_env  エピソード用ビンの横軸上限の下限（実データの最大累積ステップと大きい方を採用）。
  --relax_output_permissions  保存後に chmod（環境変数 VINTIX_RELAX_OUTPUT_PERMISSIONS=1 でも可）。

【データ形式】
  ``load_trajectory_data`` 参照。Collect（env_id なし）はエピソード境界を step_num==0 で検出。
  PPO 並列（env_id あり）はエピソード集計なしでステップ曲線のみ平均±std。
"""

import argparse
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_VINTIX_ROOT = _SCRIPT_DIR.parent


def resolve_vintix_path(p: str) -> Path:
    """相対パスは vintix_go2 リポジトリルート基準。絶対パスはそのまま。"""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_VINTIX_ROOT / path).resolve()


def load_trajectory_data(h5_path: Path, *, quiet: bool = False):
    """Load trajectory data: episodes (Collect) and step-wise (cumulative_steps, mean_r, std_r).

    Returns:
        episodes_data: Collect のみ有効。PPO 並列集約時は空リスト。
        step_rewards: list of (cumulative_steps, reward, std) — std は単一軌道時 0.0。
        ppo_num_envs: PPO 形式のとき並列環境数。それ以外は None。
    """
    episodes_data = []
    step_rewards = []
    ppo_num_envs = None

    with h5py.File(h5_path, "r") as f:
        group_names = sorted(f.keys(), key=lambda x: int(x.split("-")[0]))

        all_rewards = []
        all_step_nums = []
        all_env_ids = []
        has_env_id = False

        for group_name in group_names:
            g = f[group_name]
            rewards = np.array(g["reward"]).reshape(-1)
            step_nums = np.array(g["step_num"]).reshape(-1)
            all_rewards.append(rewards)
            all_step_nums.append(step_nums)
            if "env_id" in g:
                has_env_id = True
                all_env_ids.append(np.array(g["env_id"]).reshape(-1))

        if not all_rewards:
            return episodes_data, step_rewards, ppo_num_envs

        all_rewards = np.concatenate(all_rewards)
        all_step_nums = np.concatenate(all_step_nums)

        if has_env_id:
            env_ids = np.concatenate(all_env_ids)
            num_envs = int(env_ids.max()) + 1
            ppo_num_envs = num_envs
            n = (len(all_rewards) // num_envs) * num_envs
            if n == 0:
                return episodes_data, step_rewards, ppo_num_envs
            all_rewards = all_rewards[:n]
            rmat = all_rewards.reshape(-1, num_envs)
            mean_r = rmat.mean(axis=1)
            std_r = rmat.std(axis=1)
            n_parallel = mean_r.shape[0]
            for k in range(n_parallel):
                cum_steps = float((k + 1) * num_envs)
                step_rewards.append((cum_steps, float(mean_r[k]), float(std_r[k])))
            if not quiet:
                print(
                    f"  [PPO parallel] {h5_path.name}: num_envs={num_envs}, parallel_steps={n_parallel}, "
                    f"transitions_used={n} (mean/std over envs per parallel step)"
                )
            return episodes_data, step_rewards, ppo_num_envs

        # Collect: 単一時系列（1 行 = 1 遷移）。累積ステップ数 = 1..N
        cumulative_steps = 0
        global_step = 0
        current_episode_rewards = []

        for reward, step_num in zip(all_rewards, all_step_nums):
            step_rewards.append((float(global_step + 1), float(reward), 0.0))
            global_step += 1

            if step_num == 0 and len(current_episode_rewards) > 0:
                episodes_data.append(
                    {
                        "cumulative_reward": float(sum(current_episode_rewards)),
                        "length": len(current_episode_rewards),
                        "cumulative_steps": cumulative_steps,
                    }
                )
                cumulative_steps += len(current_episode_rewards)
                current_episode_rewards = []

            current_episode_rewards.append(float(reward))

        if len(current_episode_rewards) > 0:
            episodes_data.append(
                {
                    "cumulative_reward": float(sum(current_episode_rewards)),
                    "length": len(current_episode_rewards),
                    "cumulative_steps": cumulative_steps,
                }
            )

    return episodes_data, step_rewards, ppo_num_envs


# generate_readable_eval_graphs / eval_vintix と同系の軸まわり
FONT_SIZE_LABEL = 34
FONT_SIZE_TICK = 28

# 縦軸範囲（全出力で統一）
YLIM_EPISODE_CUMULATIVE_REWARD = (-4.0, 25.0)
YLIM_EPISODE_LENGTH = (0.0, 1150.0)
YLIM_REWARD_PER_STEP = (-0.035, 0.023)


def discover_trajectory_files(data_dir: Path) -> list[Path]:
    """Return sorted HDF5 paths under ``data_dir`` (directory only).

    優先順: ``trajectory_*.h5`` → ``trajectories_*.h5`` → ``trajectories_env_*.h5``
    （先にマッチしたパターンのみ採用）。
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {data_dir}")
    for pat in ("trajectory_*.h5", "trajectories_*.h5", "trajectories_env_*.h5"):
        found = sorted(data_dir.glob(pat))
        if found:
            return [p.resolve() for p in found]
    return []


def _max_cumulative_step(all_episodes_data: list[list[dict]], all_step_rewards: list[list]) -> int:
    m = 1
    for traj_eps in all_episodes_data:
        for ep in traj_eps:
            m = max(m, int(ep["cumulative_steps"]) + int(ep["length"]))
    for traj_steps in all_step_rewards:
        for t in traj_steps:
            m = max(m, int(float(t[0])))
    return m


def compute_bin_episode_stats(
    all_episodes_data: list[list[dict]], max_steps: int, num_bins: int = 100
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[list],
    list[list],
]:
    step_bins = np.linspace(0, max(max_steps, 1), num_bins + 1)
    bin_cumulative_rewards: list[list[float]] = []
    bin_episode_lengths: list[list[float]] = []

    for i in range(num_bins):
        step_min = step_bins[i]
        step_max = step_bins[i + 1]
        cum_rewards: list[float] = []
        lengths: list[float] = []
        for traj_episodes in all_episodes_data:
            for ep in traj_episodes:
                if step_min <= ep["cumulative_steps"] < step_max:
                    cum_rewards.append(float(ep["cumulative_reward"]))
                    lengths.append(float(ep["length"]))
        bin_cumulative_rewards.append(cum_rewards)
        bin_episode_lengths.append(lengths)

    mean_cum_rewards = [np.mean(rews) if rews else np.nan for rews in bin_cumulative_rewards]
    std_cum_rewards = [
        np.std(rews) if rews and len(rews) > 1 else (0.0 if rews else np.nan) for rews in bin_cumulative_rewards
    ]
    mean_lengths = [np.mean(lens) if lens else np.nan for lens in bin_episode_lengths]
    std_lengths = [
        np.std(lens) if lens and len(lens) > 1 else (0.0 if lens else np.nan) for lens in bin_episode_lengths
    ]
    bin_centers = [float((step_bins[i] + step_bins[i + 1]) / 2) for i in range(num_bins)]
    return (
        bin_centers,
        mean_cum_rewards,
        std_cum_rewards,
        mean_lengths,
        std_lengths,
        bin_cumulative_rewards,
        bin_episode_lengths,
    )


def _format_xaxis_scientific(ax) -> None:
    """横軸を係数＋×10ⁿ の科学表記に（大きい累積ステップ向け）。目盛は 2,4,6,8 のような係数になりやすい。"""
    ax.xaxis.set_major_locator(
        mticker.MaxNLocator(nbins=8, steps=[1, 2, 4, 5, 10], integer=False)
    )
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


def _finalize_readable_axes(
    fig,
    ax,
    xlabel: str,
    ylabel: str,
    legend_loc: str = "lower right",
    *,
    scientific_x: bool = True,
) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    if scientific_x:
        _format_xaxis_scientific(ax)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=FONT_SIZE_TICK, loc=legend_loc)
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)


def aggregate_step_curves_across_files(all_step_rewards: list[list]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """各 HDF5 のステップ平均報酬をインデックスで揃え、ファイル間の平均・標準偏差を返す。"""
    nonempty = [t for t in all_step_rewards if t]
    if not nonempty:
        return None
    min_len = min(len(t) for t in nonempty)
    if min_len == 0:
        return None
    cum_x = np.array([nonempty[0][k][0] for k in range(min_len)], dtype=np.float64)
    means = np.zeros(min_len, dtype=np.float64)
    stds = np.zeros(min_len, dtype=np.float64)
    for k in range(min_len):
        vals = np.array([float(t[k][1]) for t in nonempty if len(t) > k], dtype=np.float64)
        means[k] = float(np.mean(vals))
        stds[k] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return cum_x, means, stds


def save_three_aggregate_plots(
    out_dir: Path,
    base_name: str,
    all_episodes_data: list[list[dict]],
    all_step_rewards: list[list],
    any_ppo: bool,
    max_steps: int,
) -> list[Path]:
    """Write 3 PNGs under ``out_dir`` with prefix ``base_name`` (readable-style)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 旧日本語ファイル名は残ると紛らわしいので削除してから英語名で保存
    for legacy_suffix in ("_累積報酬.png", "_エピソード長.png", "_ステップごとの報酬.png"):
        legacy = out_dir / f"{base_name}{legacy_suffix}"
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass
    out_paths = [
        out_dir / f"{base_name}_episode_cumulative_reward.png",
        out_dir / f"{base_name}_episode_length.png",
        out_dir / f"{base_name}_reward_per_step.png",
    ]

    (
        bin_centers,
        mean_cum_rewards,
        std_cum_rewards,
        mean_lengths,
        std_lengths,
        _bin_cumulative_rewards,
        _bin_episode_lengths,
    ) = compute_bin_episode_stats(all_episodes_data, max_steps=max_steps, num_bins=100)

    valid_mask = np.array([not np.isnan(m) for m in mean_cum_rewards], dtype=bool)
    valid_bin_centers = np.array(bin_centers, dtype=np.float64)[valid_mask]
    valid_mean_rewards = np.array(mean_cum_rewards, dtype=np.float64)[valid_mask]
    valid_std_rewards = np.array(std_cum_rewards, dtype=np.float64)[valid_mask]

    # --- 1: episode cumulative reward vs cumulative steps (bin center) ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    if valid_bin_centers.size and np.any(np.isfinite(valid_mean_rewards)):
        ax.plot(
            valid_bin_centers,
            valid_mean_rewards,
            linewidth=2,
            color="green",
            label="Mean cumulative reward per episode",
        )
        ax.fill_between(
            valid_bin_centers,
            valid_mean_rewards - valid_std_rewards,
            valid_mean_rewards + valid_std_rewards,
            alpha=0.3,
            color="green",
            label="±1 std",
        )
    else:
        msg = "No episode data (PPO parallel mode)" if any_ppo else "No episode data"
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=FONT_SIZE_TICK, transform=ax.transAxes)

    ax.set_ylim(*YLIM_EPISODE_CUMULATIVE_REWARD)
    _finalize_readable_axes(fig, ax, "Cumulative steps", "Cumulative reward per episode")
    fig.savefig(str(out_paths[0]), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2: episode length vs cumulative steps ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    valid_mean_lengths = np.array(mean_lengths, dtype=np.float64)[valid_mask]
    valid_std_lengths = np.array(std_lengths, dtype=np.float64)[valid_mask]
    if valid_bin_centers.size and np.any(np.isfinite(valid_mean_lengths)):
        ax.plot(valid_bin_centers, valid_mean_lengths, linewidth=2, color="green", label="Mean episode length")
        ax.fill_between(
            valid_bin_centers,
            valid_mean_lengths - valid_std_lengths,
            valid_mean_lengths + valid_std_lengths,
            alpha=0.3,
            color="green",
            label="±1 std",
        )
    else:
        msg = "No episode data (PPO parallel mode)" if any_ppo else "No episode data"
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=FONT_SIZE_TICK, transform=ax.transAxes)

    ax.set_ylim(*YLIM_EPISODE_LENGTH)
    _finalize_readable_axes(fig, ax, "Cumulative steps", "Episode length (steps)")
    fig.savefig(str(out_paths[1]), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 3: step-wise reward vs cumulative steps ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    max_curve_points = 25_000
    plotted = False
    if len(all_step_rewards) == 1 and all_step_rewards[0]:
        traj_steps = all_step_rewards[0]
        cum_x = np.array([t[0] for t in traj_steps], dtype=np.float64)
        mean_y = np.array([t[1] for t in traj_steps], dtype=np.float64)
        std_y = np.array([t[2] for t in traj_steps], dtype=np.float64)
        if cum_x.size > max_curve_points:
            step = max(1, cum_x.size // max_curve_points)
            sel = np.s_[::step]
            cum_x = cum_x[sel]
            mean_y = mean_y[sel]
            std_y = std_y[sel]
        label = "Mean reward ± std (over envs)" if any_ppo else "Mean reward per step"
        ax.plot(cum_x, mean_y, linewidth=2, color="blue", label=label, alpha=0.95)
        ax.fill_between(cum_x, mean_y - std_y, mean_y + std_y, alpha=0.3, color="blue", label="±1 std")
        plotted = True
    elif len(all_step_rewards) > 1:
        agg = aggregate_step_curves_across_files(all_step_rewards)
        if agg is not None:
            cum_x, mean_y, std_y = agg
            if cum_x.size > max_curve_points:
                step = max(1, cum_x.size // max_curve_points)
                sel = np.s_[::step]
                cum_x = cum_x[sel]
                mean_y = mean_y[sel]
                std_y = std_y[sel]
            ax.plot(cum_x, mean_y, linewidth=2, color="blue", label="Mean reward (across trajectory files)")
            ax.fill_between(cum_x, mean_y - std_y, mean_y + std_y, alpha=0.3, color="blue", label="±1 std (across files)")
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No step reward data", ha="center", va="center", fontsize=FONT_SIZE_TICK, transform=ax.transAxes)

    ax.set_ylim(*YLIM_REWARD_PER_STEP)
    _finalize_readable_axes(fig, ax, "Cumulative steps", "Reward per step")
    fig.savefig(str(out_paths[2]), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_paths


def visualize_directory_aggregate(data_dir: Path, base_name: str, target_steps_floor: int) -> list[Path]:
    """Load all trajectory HDF5 under ``data_dir``; one set of 3 aggregate PNGs."""
    trajectory_files = discover_trajectory_files(data_dir)
    if not trajectory_files:
        raise FileNotFoundError(f"No trajectory HDF5 files under: {data_dir}")

    all_episodes_data: list[list[dict]] = []
    all_step_rewards: list[list] = []
    any_ppo = False

    for traj_file in tqdm(trajectory_files, desc="Loading trajectories"):
        episodes_data, step_rewards, ppo_num_envs = load_trajectory_data(traj_file, quiet=True)
        all_episodes_data.append(episodes_data)
        all_step_rewards.append(step_rewards)
        if ppo_num_envs is not None:
            any_ppo = True

    data_max = _max_cumulative_step(all_episodes_data, all_step_rewards)
    max_steps = max(int(target_steps_floor), int(data_max))

    n_ep = sum(len(e) for e in all_episodes_data)
    n_files = len(trajectory_files)
    print(f"Aggregated {n_files} file(s), {n_ep} episodes (Collect), max cumulative step ≈ {data_max}")

    return save_three_aggregate_plots(
        data_dir,
        base_name,
        all_episodes_data,
        all_step_rewards,
        any_ppo=any_ppo,
        max_steps=max_steps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pool all trajectory HDF5 in a directory; 3 readable-style aggregate PNGs."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Directory of trajectory HDF5, or path to one .h5 (then all HDF5 in that folder are pooled)",
    )
    parser.add_argument(
        "--target_steps_per_env",
        type=int,
        default=1_000_000,
        help="Minimum upper bound for episode binning on x-axis; actual max uses max(this, data extent).",
    )
    parser.add_argument(
        "--relax_output_permissions",
        action="store_true",
        help="chmod 664 on each output PNG. Also VINTIX_RELAX_OUTPUT_PERMISSIONS=1.",
    )

    args = parser.parse_args()

    data_path = resolve_vintix_path(args.data_path)
    if data_path.is_file() and data_path.suffix.lower() == ".h5":
        data_dir = data_path.parent.resolve()
    elif data_path.is_dir():
        data_dir = data_path.resolve()
    else:
        raise FileNotFoundError(f"Not a directory or .h5 file: {data_path}")

    base_name = data_dir.name
    paths = visualize_directory_aggregate(data_dir, base_name, args.target_steps_per_env)
    for p in paths:
        print(f"✅ {p}")

    relax = args.relax_output_permissions or os.environ.get("VINTIX_RELAX_OUTPUT_PERMISSIONS", "").strip() in (
        "1",
        "true",
        "yes",
    )
    if relax:
        for p in paths:
            if p.exists():
                try:
                    p.chmod(0o664)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
