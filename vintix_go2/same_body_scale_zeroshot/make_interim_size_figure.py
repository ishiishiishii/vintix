#!/usr/bin/env python3
"""Interim Go1 size figure: same-body Vintix vs leave-one-out (#18 vs #10).

  cd /workspace/vintix_go2
  python same_body_scale_zeroshot/make_interim_size_figure.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
SAME_ROOT = EXP_DIR / "results" / "go1" / "vintix" / "size"
LOO_CSV = EXP_DIR.parent / "robot_body_scale_zeroshot" / "results.csv"
FIG_DIR = EXP_DIR / "figures"
OUTPUT_STEM = "go1_10envs_10episodes"


def collect_same_body() -> dict[float, list[float]]:
    by_scale: dict[float, list[float]] = defaultdict(list)
    for meta in SAME_ROOT.glob("*_seed*/point.json"):
        parent = meta.parent.name
        scale_s, _seed = parent.rsplit("_seed", 1)
        scale = float(scale_s.replace("s", "").replace("p", "."))
        metrics = meta.parent / f"{OUTPUT_STEM}_metrics.json"
        if not metrics.exists():
            continue
        cr = float(json.load(open(metrics))["completion_rate"]) * 100.0
        by_scale[scale].append(cr)
    return dict(by_scale)


def collect_loo() -> dict[float, tuple[float, float, int]]:
    out: dict[float, tuple[float, float, int]] = {}
    with open(LOO_CSV) as f:
        for row in csv.DictReader(f):
            if row["robot"] != "go1" or row["policy"] != "vintix" or row["axis"] != "size":
                continue
            scale = float(row["scale"])
            out[scale] = (
                float(row["completion_mean"]),
                float(row["completion_std"]),
                int(row["n_seeds"]),
            )
    return out


def tick_label_indices(n: int, base: int, step: int = 3) -> list[int]:
    labeled = {0, n - 1, base}
    for i in range(base - step, -1, -step):
        labeled.add(i)
    for i in range(base + step, n, step):
        labeled.add(i)
    return sorted(labeled)


def main() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    same = collect_same_body()
    loo = collect_loo()
    # Nominal 1.0 lives under mass axis for same-body; inject LOO/same if present.
    scales = sorted(set(same) | set(loo) | {1.0})
    pos = {x: i for i, x in enumerate(scales)}
    base = scales.index(1.0) if 1.0 in scales else 0
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = FIG_DIR / f"go1_size_samebody_vs_loo_completion_mean_std_{day}.png"

    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "axes.linewidth": 1.4,
        }
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.2))

        xs = [x for x in scales if x in loo]
        xp = np.array([pos[x] for x in xs], dtype=float)
        mean = np.array([loo[x][0] for x in xs])
        std = np.array([loo[x][1] for x in xs])
        n0, n1 = min(loo[x][2] for x in xs), max(loo[x][2] for x in xs)
        seeds = f"{n0}-{n1} Seeds" if n0 != n1 else f"{n0} Seeds"
        ax.plot(xp, mean, "o-", color="#7f7f7f", label=f"LOO Vintix ({seeds})", lw=2.8, ms=6, zorder=3)
        ax.fill_between(xp, mean - std, mean + std, color="#7f7f7f", alpha=0.25, lw=0, zorder=2)

        xs = [x for x in scales if x in same]
        xp = np.array([pos[x] for x in xs], dtype=float)
        vals = [np.array(same[x]) for x in xs]
        mean = np.array([v.mean() for v in vals])
        std = np.array([v.std(ddof=0) for v in vals])
        n0, n1 = min(len(v) for v in vals), max(len(v) for v in vals)
        seeds = f"{n0}-{n1} Seeds" if n0 != n1 else f"{n0} Seeds"
        ax.plot(xp, mean, "o-", color="#1f77b4", label=f"Same-body Vintix ({seeds})", lw=2.8, ms=6, zorder=4)
        ax.fill_between(xp, mean - std, mean + std, color="#1f77b4", alpha=0.25, lw=0, zorder=2)

        if 1.0 in pos:
            ax.axvline(pos[1.0], color="black", lw=1.3, ls="--", alpha=0.5)
        ax.set_ylim(-5, 105)
        ax.set_ylabel("Success rate [%]\n(no fall in 1000 steps)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(width=1.3, length=6)
        tick_idx = set(tick_label_indices(len(scales), base, step=3))
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([f"{scales[i]:g}" if i in tick_idx else "" for i in range(len(scales))])
        ax.set_xlabel("Size scale (× Go1; equally spaced, not to scale)")
        fig.subplots_adjust(left=0.14, right=0.99, top=0.98, bottom=0.22)
        handles, labels = ax.get_legend_handles_labels()
        axes_mid = 0.5 * (0.14 + 0.99)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(axes_mid, 0.08),
            handlelength=2.0,
            columnspacing=1.6,
            handletextpad=0.5,
        )
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        plt.close(fig)

    table = FIG_DIR / f"go1_size_samebody_vs_loo_table_{day}.md"
    lines = [
        f"<!-- interim #18 size, {day} UTC -->",
        "",
        "| scale | same-body mean% (n) | same-body std% | LOO mean% (n) | LOO std% |",
        "|---:|---:|---:|---:|---:|",
    ]
    for x in scales:
        if x in same:
            v = np.array(same[x])
            sb = f"{v.mean():.1f} ({len(v)})"
            sb_std = f"{v.std(ddof=0):.1f}"
        else:
            sb, sb_std = "—", "—"
        if x in loo:
            m, s, n = loo[x]
            lo = f"{m:.1f} ({n})"
            lo_std = f"{s:.1f}"
        else:
            lo, lo_std = "—", "—"
        label = f"**{x:g}**" if x in (0.55, 0.6, 0.65, 1.0, 1.1, 1.15, 1.25) else f"{x:g}"
        lines.append(f"| {label} | {sb} | {sb_std} | {lo} | {lo_std} |")
    table.write_text("\n".join(lines) + "\n")
    print(out)
    print(table)
    return out


if __name__ == "__main__":
    main()
