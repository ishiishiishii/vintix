#!/usr/bin/env python3
"""
All one group 等の評価結果（CSV）から、見やすいグラフを生成する。
・題名なし（モデル名・タイトル不要）
・軸ラベル・軸数値は generate_readable_cumulative_reward_plots に合わせる（FONT_SIZE_LABEL=34, FONT_SIZE_TICK=28）
・Go1, Go2, Unitree A1, Mini Cheetah の各ロボットについて episode 用のみ出力（ステップグラフは出力しない）
・既存ファイルは一切削除・上書きせず、{ロボット名}_episodes_readable.pdf のみ新規追加。
・エピソードグラフは先頭10エピソードのみ表示。
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ラベル・軸数値のフォントサイズ（generate_readable_cumulative_reward_plots に合わせる）
FONT_SIZE_LABEL = 34
FONT_SIZE_TICK = 28

# 各ロボット名（ディレクトリ名またはファイルプレフィックス）
# ネスト構成: result_root/go1/go1_10envs_*episodes.csv。flat構成: result_root/go1_10envs_*episodes.csv
ROBOT_NAMES = ["go1", "go2", "unitreea1", "minicheetah"]


def find_csv_stem(search_dir: Path, prefix: str, stem_suffix: str = None):
    """search_dir 内で prefix_10envs_*episodes[ stem_suffix].csv に一致する stem を1つ返す。見つからなければ None。"""
    if not search_dir.is_dir():
        return None
    if stem_suffix:
        pattern = re.compile(re.escape(prefix) + r"_10envs_\d+episodes_" + re.escape(stem_suffix) + r"\.csv")
    else:
        pattern = re.compile(re.escape(prefix) + r"_10envs_\d+episodes\.csv")
    for f in search_dir.iterdir():
        if f.suffix == ".csv" and pattern.fullmatch(f.name):
            return f.stem
    return None


# エピソードグラフに表示するエピソード数（CSVに20エピソードあっても先頭この数だけ使用）
MAX_EPISODES_PLOT = 10


def load_episode_data(episode_csv_path: Path, max_episodes: int = MAX_EPISODES_PLOT):
    """エピソードCSVを読み、エピソード番号ごとの平均・標準偏差を返す。max_episodes で先頭 N エピソードのみ使用。"""
    with open(episode_csv_path, "r") as f:
        header = f.readline().strip()
    if "env_index" in header:
        data = np.loadtxt(episode_csv_path, delimiter=",", skiprows=1)
        episode_numbers = data[:, 0].astype(int)
        cumulative_rewards = data[:, 3]
    else:
        data = np.loadtxt(episode_csv_path, delimiter=",", skiprows=1)
        num_envs = 10
        episode_numbers = (data[:, 0] // num_envs).astype(int)
        cumulative_rewards = data[:, 2]
    episode_num_to_rewards = {}
    for ep_num, reward in zip(episode_numbers, cumulative_rewards):
        if ep_num not in episode_num_to_rewards:
            episode_num_to_rewards[ep_num] = []
        episode_num_to_rewards[ep_num].append(reward)
    episode_nums = sorted(episode_num_to_rewards.keys())
    # 先頭 max_episodes エピソードのみ
    episode_nums = [ep for ep in episode_nums if ep < max_episodes]
    means = [np.mean(episode_num_to_rewards[ep]) for ep in episode_nums]
    stds = [np.std(episode_num_to_rewards[ep]) if len(episode_num_to_rewards[ep]) > 1 else 0.0 for ep in episode_nums]
    return episode_nums, means, stds


def generate_readable_step_graph(csv_path: Path, output_path: Path):
    """ステップ用CSVから題名なし・大フォントのグラフをPDFで保存。"""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    steps = data[:, 0]
    mean_rewards = data[:, 1]
    std_rewards = data[:, 2]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.plot(steps, mean_rewards, linewidth=2, label="Mean Reward", color="blue")
    ax.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, color="blue", label="±1 Std")
    ax.set_xlabel("Step", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Reward", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.03, 0.03)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=FONT_SIZE_TICK, loc="lower right")
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    out = output_path if str(output_path).lower().endswith(".pdf") else output_path.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    try:
        os.chmod(out, 0o644)
    except OSError:
        pass
    return out


def generate_readable_episode_graph(episode_csv_path: Path, output_path: Path, robot_name: str = None, legend_upper_right_robot: str = None):
    """エピソード用CSVから題名なし・大フォントのグラフをPDFで保存。legend_upper_right_robot で指定したロボットのみ凡例を右上に。"""
    episode_nums, means, stds = load_episode_data(episode_csv_path)
    if not episode_nums:
        return None
    # 1エピソードからプロット（1-based）
    x_episodes = [ep + 1 for ep in episode_nums]

    legend_loc = "upper right" if (legend_upper_right_robot and robot_name == legend_upper_right_robot) else "lower right"

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.plot(x_episodes, means, linewidth=2, label="Mean Cumulative Reward per Episode", color="green")
    ax.fill_between(x_episodes, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                    alpha=0.3, color="green", label="±1 Std")
    ax.set_xlabel("Episode Number", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Cumulative Reward per Episode", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(-5, 28)
    ax.set_xlim(0, 11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_TICK, loc=legend_loc)
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    out = output_path if str(output_path).lower().endswith(".pdf") else output_path.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    try:
        os.chmod(out, 0o644)
    except OSError:
        pass
    return out


def load_episode_mean_std(episode_csv_path: Path):
    """エピソードCSVの全 cumulative_reward の平均・標準偏差を返す。(mean, std)"""
    with open(episode_csv_path, "r") as f:
        header = f.readline().strip()
    if "env_index" in header:
        data = np.loadtxt(episode_csv_path, delimiter=",", skiprows=1)
        cumulative_rewards = data[:, 3]
    else:
        data = np.loadtxt(episode_csv_path, delimiter=",", skiprows=1)
        cumulative_rewards = data[:, 2]
    mean = float(np.mean(cumulative_rewards))
    std = float(np.std(cumulative_rewards)) if len(cumulative_rewards) > 1 else 0.0
    return mean, std


def write_summary_table(save_dir: Path, robot_stats: list, name_suffix: str = None):
    """各ロボットの平均値を表にまとめて Markdown と CSV で保存。robot_stats: [(robot_name, mean, std), ...]。name_suffix 指定時はファイル名に含める。"""
    suf = f"_{name_suffix}" if name_suffix else ""
    md_path = save_dir / f"episode_reward_summary{suf}.md"
    csv_path = save_dir / f"episode_reward_summary{suf}.csv"
    lines_md = [
        "| Robot | Mean Cumulative Reward | Std |",
        "|-------|------------------------|-----|",
    ]
    rows_csv = [["Robot", "Mean Cumulative Reward", "Std"]]
    for name, mean, std in robot_stats:
        display_name = {"unitreea1": "Unitree A1", "minicheetah": "Mini Cheetah"}.get(name, name.capitalize())
        lines_md.append(f"| {display_name} | {mean:.4f} | {std:.4f} |")
        rows_csv.append([name, f"{mean:.6f}", f"{std:.6f}"])
    md_path.write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(rows_csv[0]) + "\n")
        for row in rows_csv[1:]:
            f.write(",".join(row) + "\n")
    print(f"\nSummary table saved: {md_path}")
    print(f"Summary table saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate readable evaluation graphs (no title, large fonts) for go1/go2/a1/minicheetah. PDFs are saved in the same directory as the CSV data. Existing files are never deleted. Episode graph uses first 10 episodes only.")
    parser.add_argument("result_root", type=str,
                        help="Root directory containing go1/, go2/, unitreea1/, minicheetah/ with CSV files (e.g. Allonegroup/result/Allonegroup0000epoch)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Optional: directory to save PDFs (default: same as result_root)")
    parser.add_argument("--legend-upper-right", type=str, default=None, choices=["go1", "go2", "unitreea1", "minicheetah"],
                        help="Robot whose graph should have legend in upper right (default: none)")
    parser.add_argument("--robots", type=str, default=None,
                        help="Comma-separated robot names to process only (e.g. go1 or go1,go2). Default: all.")
    parser.add_argument("--stem-suffix", type=str, default=None,
                        help="Optional stem suffix for CSV (e.g. norm_from_training) to use prefix_10envs_Nepisodes_SUFFIX.csv")
    args = parser.parse_args()

    base = Path(args.result_root)
    out_base = Path(args.output_dir) if args.output_dir else None
    if not base.exists():
        print(f"⚠ Result root not found: {base}")
        return

    robots_to_process = ROBOT_NAMES
    if args.robots:
        robots_to_process = [r.strip().lower() for r in args.robots.split(",") if r.strip()]
        for r in robots_to_process:
            if r not in ROBOT_NAMES:
                print(f"⚠ Unknown robot in --robots: {r}. Using: {ROBOT_NAMES}")
                robots_to_process = ROBOT_NAMES
                break

    robot_stats = []  # (robot_name, mean, std) のリスト
    stem_suffix = args.stem_suffix.strip() if getattr(args, "stem_suffix", None) and args.stem_suffix else None

    for robot_name in robots_to_process:
        # ネスト構成: base/go1/, base/go2/ ... の下に CSV。flat 構成: base 直下に go1_10envs_10episodes.csv 等
        robot_path = base / robot_name
        if robot_path.is_dir():
            search_dir = robot_path
            prefix = robot_name if robot_name != "unitreea1" else "a1"  # ネスト時 unitreea1 は a1_ のことがある
        else:
            search_dir = base
            prefix = robot_name  # flat では go1_ / go2_ / unitreea1_ / minicheetah_ で保存
        csv_stem = find_csv_stem(search_dir, prefix, stem_suffix)
        if csv_stem is None:
            # unitreea1 で unitreea1 が見つからなかったら a1 で再検索（ネスト時）
            if robot_name == "unitreea1" and search_dir == base:
                csv_stem = find_csv_stem(base, "unitreea1", stem_suffix)
            if robot_name == "unitreea1" and csv_stem is None:
                csv_stem = find_csv_stem(search_dir, "a1", stem_suffix)
            if csv_stem is None:
                print(f"⚠ Skip {robot_name}: no CSV like {prefix}_10envs_*episodes*.csv in {search_dir}")
                continue
        step_csv = search_dir / f"{csv_stem}.csv"
        episode_csv = search_dir / f"{csv_stem}_episodes.csv"

        # エピソードグラフのみ出力するため、_episodes.csv があればOK（step CSV は必須にしない）
        if not episode_csv.exists():
            print(f"⚠ Skip {robot_name}: not found {episode_csv}")
            continue

        # 保存場所: --output-dir 指定時はそこへ、なければデータ（CSV）と同じディレクトリ
        save_dir = out_base if out_base is not None else search_dir
        # stem_suffix 指定時は出力名に含める（例: go1_episodes_norm_from_training_readable.pdf）
        if stem_suffix:
            ep_out = save_dir / f"{robot_name}_episodes_{stem_suffix}_readable.pdf"
        else:
            ep_out = save_dir / f"{robot_name}_episodes_readable.pdf"

        print(f"Generating readable graphs for {robot_name}...")
        if True:  # episode_csv は上で存在確認済み
            try:
                generate_readable_episode_graph(episode_csv, ep_out, robot_name=robot_name, legend_upper_right_robot=args.legend_upper_right)
                print(f"  Saved: {ep_out}")
            except Exception as e:
                print(f"  Error (episode): {e}")
        try:
            mean, std = load_episode_mean_std(episode_csv)
            robot_stats.append((robot_name, mean, std))
        except Exception as e:
            print(f"  (skip summary for {robot_name}: {e})")

        print(f"✓ {robot_name} done.")

    save_dir = out_base if out_base is not None else base
    if robot_stats:
        write_summary_table(save_dir, robot_stats, name_suffix=stem_suffix)

    print("\nDone. Readable graphs saved (first 10 episodes).")


if __name__ == "__main__":
    main()
