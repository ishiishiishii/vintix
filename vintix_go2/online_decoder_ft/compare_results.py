#!/usr/bin/env python3
"""ゼロショット / オフライン decoder FT / オンライン PPO FT の表・グラフ比較。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "scripts"))
from plot_style import FIGSIZE, LINE_WIDTH, STD_FILL_ALPHA, shade_mean_std

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
RESULT_DIR = EXP_DIR / "result"
REF_EXP = VINTIX_ROOT / "experience_decoder_ft_data_fraction_20260517"

FONT_SIZE_LABEL = 34
FONT_SIZE_TICK = 28
FONT_SIZE_LEGEND = 18

MODEL_META = {
    "go1_without": ("go1", "go1"),
    "go2_without": ("go2", "go2"),
    "a1_without": ("a1", "a1"),
    "minicheetah_without": ("minicheetah", "minicheetah"),
}

LABEL_ZS = "Zero-shot (in-context)"
LABEL_OFF = "Offline decoder FT (10%)"
LABEL_ON = "Online decoder PPO FT"

COLORS = {
    LABEL_ZS: "blue",
    LABEL_OFF: "darkorange",
    LABEL_ON: "green",
}


def load_episode_means(episode_csv: Path, max_episodes: int = 10):
    data = np.loadtxt(episode_csv, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    episode_numbers = data[:, 0].astype(int)
    cumulative_rewards = data[:, 3]
    ep_to_rews: dict[int, list[float]] = {}
    for ep, rew in zip(episode_numbers, cumulative_rewards):
        if ep < max_episodes:
            ep_to_rews.setdefault(int(ep), []).append(float(rew))
    eps = sorted(ep_to_rews)
    means = [float(np.mean(ep_to_rews[e])) for e in eps]
    stds = [float(np.std(ep_to_rews[e])) if len(ep_to_rews[e]) > 1 else 0.0 for e in eps]
    overall = float(np.mean(cumulative_rewards)) if len(cumulative_rewards) else float("nan")
    return eps, means, stds, overall


def find_online_episodes_csv(model_key: str, csv_prefix: str) -> Path | None:
    candidates: list[Path] = []
    eval_root = RESULT_DIR / model_key
    if eval_root.is_dir():
        candidates.extend(eval_root.rglob(f"{csv_prefix}_*_episodes.csv"))
        candidates.extend(eval_root.rglob("*_episodes.csv"))
    uniq = sorted(set(candidates), key=lambda p: p.stat().st_mtime)
    return uniq[-1] if uniq else None


def ref_csv(model_key: str, csv_prefix: str, split: str) -> Path:
    return REF_EXP / "eval" / model_key / split / f"{csv_prefix}_10envs_10episodes_episodes.csv"


def plot_comparison(
    series: list[tuple[str, list[int], list[float], list[float]]],
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_lim = (-5.0, 28.0)
    for label, eps, means, stds in series:
        x = [e + 1 for e in eps]
        color = COLORS.get(label, "black")
        ax.plot(x, means, label=label, color=color, linewidth=LINE_WIDTH)
        shade_mean_std(ax, x, means, stds, color=color, y_lim=y_lim, alpha=STD_FILL_ALPHA)
    ax.set_xlabel("Episode", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Average episode return", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.set_title(title, fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(*y_lim)
    ax.set_xlim(0, 11)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.92,
    )
    fig.canvas.draw()
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE_TICK)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        default="go2_without",
        help="Comma-separated model keys",
    )
    args = parser.parse_args()
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    table_rows = []
    for mk in model_keys:
        if mk not in MODEL_META:
            print(f"Skip unknown model key: {mk}")
            continue
        csv_prefix, _robot = MODEL_META[mk]
        zs_path = ref_csv(mk, csv_prefix, "p00")
        off_path = ref_csv(mk, csv_prefix, "p10")
        on_path = find_online_episodes_csv(mk, csv_prefix)

        series = []
        means_summary = {"model": mk}
        out_dir = RESULT_DIR / mk
        out_dir.mkdir(parents=True, exist_ok=True)

        for label, path in (
            (LABEL_ZS, zs_path),
            (LABEL_OFF, off_path),
            (LABEL_ON, on_path),
        ):
            if path is None or not Path(path).is_file():
                print(f"[warn] missing {label} for {mk}: {path}")
                means_summary[label] = None
                continue
            eps, means, stds, overall = load_episode_means(Path(path))
            series.append((label, eps, means, stds))
            means_summary[label] = overall
            print(f"{mk} | {label}: mean_return={overall:.4f} ({path})")

        table_rows.append(means_summary)
        if series:
            plot_comparison(
                series,
                out_dir / f"{mk}_zero_offline_online.png",
                title=f"{mk}: zero-shot vs offline FT vs online PPO FT",
            )

        train_png = RESULT_DIR / mk / "train_mean_reward.png"
        if train_png.is_file():
            print(f"Training reward curve: {train_png}")

    if len(model_keys) == 1:
        out_dir = RESULT_DIR / model_keys[0]
    else:
        out_dir = RESULT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    out_table = out_dir / "comparison_summary.csv"
    fieldnames = ["model", LABEL_ZS, LABEL_OFF, LABEL_ON]
    with open(out_table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"Wrote summary table: {out_table}")

    out_md = out_dir / "comparison_summary.md"
    lines = [
        "| model | Zero-shot | Offline decoder FT (10%) | Online decoder PPO FT |",
        "|---|---:|---:|---:|",
    ]
    for row in table_rows:
        def fmt(v):
            return f"{v:.2f}" if isinstance(v, (float, int)) and v is not None else "—"

        lines.append(
            f"| {row['model']} | {fmt(row.get(LABEL_ZS))} | {fmt(row.get(LABEL_OFF))} | {fmt(row.get(LABEL_ON))} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote summary markdown: {out_md}")


if __name__ == "__main__":
    main()
