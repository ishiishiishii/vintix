#!/usr/bin/env python3
"""
Without モデルについて、ファインチューニングなし vs デコーダ FT 後の評価を
1 枚の readable スタイルグラフに重ねて表示する。

- ファインチューニングなし: ``*_10envs_10episodes_norm_from_training_episodes.csv``
  （各 ``*_0001_epoch`` Result）
- デコーダ FT 後: ``*_10envs_10episodes_episodes.csv``（各 FT Result）

既存の ``*_episodes_readable.pdf`` や CSV は上書きしない。
体裁は ``generate_readable_eval_graphs.py`` / ``save_vintix.py`` に準拠。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from plot_style import (  # noqa: E402
    FIGSIZE,
    LINE_WIDTH,
    ROBOT_COLORS,
    STD_FILL_ALPHA,
    Y_LABEL,
    Y_LIM,
    finetune_legend_label,
    shade_mean_std,
)

FONT_SIZE_LABEL = 34
FONT_SIZE_TICK = 28
FONT_SIZE_LEGEND = 18
MAX_EPISODES_PLOT = 10
# (表示名, ベース Result ディレクトリ, FT Result ディレクトリ, 比較対象ロボット, CSV プレフィックス)
DEFAULT_COMPARISONS: list[tuple[str, Path, Path, str, str]] = [
    (
        "go1_without",
        Path("models/go1_without/go1_without/Result/go1_without_0001_epoch"),
        Path(
            "models/go1_without/go1_without/go1_without_epoch1_0p1data_finetune"
            "/Result/go1_without_epoch1_0p1data_finetune_epoch0"
        ),
        "go1",
        "go1",
    ),
    (
        "go2_without",
        Path("models/go2_without/go2_without/Result/go2_without_0001_epoch"),
        Path(
            "models/go2_without/go2_without/go2_without_tenthdata_finetune"
            "/Result/go2_without_tenthdata_finetune_epoch0"
        ),
        "go2",
        "go2",
    ),
    (
        "a1_without",
        Path("models/a1_without/a1_without/Result/a1_without_0001_epoch"),
        Path(
            "models/a1_without/a1_without/a1_without_epoch1_0p1data_finetune"
            "/Result/a1_without_epoch1_0p1data_finetune_epoch0"
        ),
        "unitreea1",
        "unitreea1",
    ),
    (
        "minicheetah_without",
        Path("models/minicheetah_without/minicheetah_without/Result/minicheetah_without_0001_epoch"),
        Path(
            "models/minicheetah_without/minicheetah_without"
            "/minicheetah_without_epoch1_0p7data_finetune"
            "/Result/minicheetah_without_epoch1_0p7data_finetune_epoch0"
        ),
        "minicheetah",
        "minicheetah",
    ),
]

NORM_EPISODES_CSV = "{prefix}_10envs_10episodes_norm_from_training_episodes.csv"
FINETUNE_EPISODES_CSV = "{prefix}_10envs_10episodes_episodes.csv"

# experience_decoder_ft_data_fraction_* experiment layout
EXP_MODEL_SPECS: list[tuple[str, str, str]] = [
    ("go1_without", "go1", "go1"),
    ("go2_without", "go2", "go2"),
    ("a1_without", "a1", "a1"),
    ("minicheetah_without", "minicheetah", "minicheetah"),
]
EXP_BASE_SUBDIR = "p00"
EXP_FINETUNE_SUBDIR = "p10"


def resolve_norm_from_training_csv(result_dir: Path, csv_prefix: str) -> Path | None:
    path = Path(result_dir) / NORM_EPISODES_CSV.format(prefix=csv_prefix)
    return path if path.is_file() else None


def resolve_finetune_episodes_csv(result_dir: Path, csv_prefix: str) -> Path | None:
    path = Path(result_dir) / FINETUNE_EPISODES_CSV.format(prefix=csv_prefix)
    return path if path.is_file() else None


def resolve_experiment_episodes_csv(eval_dir: Path, csv_prefix: str) -> Path | None:
    path = eval_dir / FINETUNE_EPISODES_CSV.format(prefix=csv_prefix)
    return path if path.is_file() else None


def load_episode_data(episode_csv_path: Path, max_episodes: int = MAX_EPISODES_PLOT):
    with open(episode_csv_path, "r", encoding="utf-8") as f:
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
    episode_num_to_rewards: dict[int, list[float]] = {}
    for ep_num, reward in zip(episode_numbers, cumulative_rewards):
        episode_num_to_rewards.setdefault(int(ep_num), []).append(float(reward))
    episode_nums = sorted(ep for ep in episode_num_to_rewards if ep < max_episodes)
    means = [float(np.mean(episode_num_to_rewards[ep])) for ep in episode_nums]
    stds = [
        float(np.std(episode_num_to_rewards[ep])) if len(episode_num_to_rewards[ep]) > 1 else 0.0
        for ep in episode_nums
    ]
    return episode_nums, means, stds


def save_comparison_graph(
    *,
    base_csv: Path,
    finetune_csv: Path,
    out_path: Path,
    eval_robot: str,
    show_legend: bool,
    label_without: str | None = None,
    label_finetune: str | None = None,
) -> Path:
    robot_color = ROBOT_COLORS.get(eval_robot, "#1f77b4")
    without_label = label_without or "Without finetune (0% data)"
    ft_label = label_finetune or finetune_legend_label(eval_robot)

    series: list[tuple[list[int], list[float], list[float], str, str, str]] = []
    for csv_path, label, color, linestyle in (
        (base_csv, without_label, "#7f7f7f", "--"),
        (finetune_csv, ft_label, robot_color, "-"),
    ):
        episode_nums, means, stds = load_episode_data(csv_path)
        if not episode_nums:
            raise ValueError(f"No episode data in {csv_path}")
        x_episodes = [ep + 1 for ep in episode_nums]
        series.append((x_episodes, means, stds, label, color, linestyle))

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    for x_episodes, means, stds, label, color, linestyle in series:
        line_label = label if show_legend else "_nolegend_"
        line, = ax.plot(
            x_episodes,
            means,
            linewidth=LINE_WIDTH,
            label=line_label,
            color=color,
            linestyle=linestyle,
        )
        shade_mean_std(ax, x_episodes, means, stds, color=color, alpha=STD_FILL_ALPHA)
        if not show_legend:
            line.set_label("_nolegend_")

    ax.set_xlabel("Episode Number", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(Y_LABEL, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(*Y_LIM)
    ax.set_xlim(0, 11)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(
            fontsize=FONT_SIZE_LEGEND,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.02),
            framealpha=0.92,
            borderaxespad=0.0,
            handlelength=1.6,
            labelspacing=0.35,
        )
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

    out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    try:
        os.chmod(out_path, 0o644)
    except OSError:
        pass
    return out_path


def run_comparison(
    model_label: str,
    base_dir: Path,
    finetune_dir: Path,
    target_robot: str,
    csv_prefix: str,
    output_dir: Path,
    *,
    use_norm_from_training_base: bool = True,
    label_without: str | None = None,
    label_finetune: str | None = None,
    write_poster: bool = False,
) -> list[Path]:
    if use_norm_from_training_base:
        base_csv = resolve_norm_from_training_csv(base_dir, csv_prefix)
        if base_csv is None:
            print(
                f"⚠ [{model_label}] skip: not found {NORM_EPISODES_CSV.format(prefix=csv_prefix)} in {base_dir}"
            )
            return []
    else:
        base_csv = resolve_finetune_episodes_csv(base_dir, csv_prefix)
        if base_csv is None:
            print(
                f"⚠ [{model_label}] skip: not found {FINETUNE_EPISODES_CSV.format(prefix=csv_prefix)} in {base_dir}"
            )
            return []
    finetune_csv = resolve_finetune_episodes_csv(finetune_dir, csv_prefix)
    if finetune_csv is None:
        print(
            f"⚠ [{model_label}] skip: not found {FINETUNE_EPISODES_CSV.format(prefix=csv_prefix)} in {finetune_dir}"
        )
        return []

    stem = f"{csv_prefix}_without_vs_finetune_readable"
    out_paper = output_dir / model_label / f"{stem}.png"
    print(f"[{model_label}] {target_robot}: {base_csv.name} vs {finetune_csv.name}")
    saved_paths: list[Path] = []
    saved_paths.append(
        save_comparison_graph(
            base_csv=base_csv,
            finetune_csv=finetune_csv,
            out_path=out_paper,
            eval_robot=target_robot,
            show_legend=True,
            label_without=label_without,
            label_finetune=label_finetune,
        )
    )
    print(f"  Saved (paper): {saved_paths[-1]}")
    if write_poster:
        out_poster = output_dir / model_label / f"{stem}_poster.png"
        saved_paths.append(
            save_comparison_graph(
                base_csv=base_csv,
                finetune_csv=finetune_csv,
                out_path=out_poster,
                eval_robot=target_robot,
                show_legend=False,
                label_without=label_without,
                label_finetune=label_finetune,
            )
        )
        print(f"  Saved (poster): {saved_paths[-1]}")
    return saved_paths


def run_experiment_comparisons(
    exp_root: Path,
    output_dir: Path | None,
    selected: set[str] | None,
) -> int:
    """0% (no FT) vs 10% decoder FT from ``experience_decoder_ft_data_fraction_*`` eval CSVs."""
    exp_root = exp_root.resolve()
    eval_root = exp_root / "eval"
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Missing eval directory: {eval_root}")

    out_root = output_dir.resolve() if output_dir else exp_root / "readable_comparisons"
    saved = 0
    for model_key, _eval_robot, csv_prefix in EXP_MODEL_SPECS:
        if selected is not None and model_key not in selected:
            continue
        base_eval = eval_root / model_key / EXP_BASE_SUBDIR
        finetune_eval = eval_root / model_key / EXP_FINETUNE_SUBDIR
        paths = run_comparison(
            model_key,
            base_eval,
            finetune_eval,
            _eval_robot,
            csv_prefix,
            out_root,
            use_norm_from_training_base=False,
        )
        if paths:
            saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay without-finetune vs decoder-finetune readable episode graphs."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output root (default: vintix_go2/scripts/readable_eval_graphs/comparisons)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset: go1_without,go2_without,a1_without,minicheetah_without",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default=None,
        help=(
            "Use eval CSVs from a decoder FT data-fraction experiment "
            "(p00=no FT vs p10=10%% FT). Output defaults to <exp-root>/readable_comparisons."
        ),
    )
    args = parser.parse_args()

    vintix_root = Path(__file__).resolve().parent.parent
    selected = None
    if args.models:
        selected = {m.strip() for m in args.models.split(",") if m.strip()}

    if args.exp_root:
        exp_root = Path(args.exp_root)
        if not exp_root.is_absolute():
            exp_root = vintix_root / exp_root
        out_root = Path(args.output_dir) if args.output_dir else exp_root / "readable_comparisons"
        saved = run_experiment_comparisons(exp_root, out_root, selected)
        print(f"\nDone. Wrote {saved} comparison PNG(s) under {out_root}")
        return

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent / "readable_eval_graphs" / "comparisons"
    )

    saved = 0
    for model_label, base_rel, finetune_rel, target_robot, csv_prefix in DEFAULT_COMPARISONS:
        if selected is not None and model_label not in selected:
            continue
        base_dir = vintix_root / base_rel
        finetune_dir = vintix_root / finetune_rel
        paths = run_comparison(
            model_label, base_dir, finetune_dir, target_robot, csv_prefix, out_root
        )
        if paths:
            saved += 1

    print(f"\nDone. Wrote {saved} comparison PNG(s) under {out_root}")


if __name__ == "__main__":
    main()
