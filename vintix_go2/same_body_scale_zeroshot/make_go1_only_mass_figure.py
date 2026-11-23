#!/usr/bin/env python3
"""Go1-only-trained Vintix mass limits (clear legend; no 'same-body' jargon).

  cd /workspace/vintix_go2
  python same_body_scale_zeroshot/make_go1_only_mass_figure.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
FIG = EXP / "figures"
METRICS = "go1_10envs_10episodes_metrics.json"
ROOT = EXP / "results" / "go1" / "vintix" / "mass"


def collect() -> dict[float, list[float]]:
    by: dict[float, list[float]] = defaultdict(list)
    for mp in ROOT.glob(f"*_seed*/{METRICS}"):
        tag = mp.parent.name.rsplit("_seed", 1)[0]
        scale = float(tag.replace("s", "").replace("p", "."))
        by[scale].append(float(json.loads(mp.read_text())["completion_rate"]) * 100.0)
    return dict(by)


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

    data = collect()
    scales = sorted(data)
    means = np.array([float(np.mean(data[s])) for s in scales])
    stds = np.array([float(np.std(data[s], ddof=0)) for s in scales])
    xs = np.arange(len(scales))
    base = scales.index(1.0) if 1.0 in scales else 0
    labels = tick_label_indices(len(scales), base)

    out = FIG / "go1_mass_scale_go1only_trained_vintix_completion_mean_std.png"
    FIG.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
        }
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
        color = "#1f77b4"
        ax.plot(xs, means, color=color, lw=2.4, marker="o", ms=5.5, label="Go1-only trained Vintix (8–10 Seeds)")
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.22, linewidth=0)
        ax.axvline(base, color="0.45", ls="--", lw=1.2)
        ax.set_ylim(-3, 105)
        ax.set_xlim(-0.5, len(scales) - 0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{scales[i]:g}" if i in labels else "" for i in range(len(scales))])
        ax.set_xlabel("Mass scale (× Go1; equally spaced, not to scale)")
        ax.set_ylabel("Success rate [%]\n(no fall in 1000 steps)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=1)
        ax.grid(True, axis="y", alpha=0.25)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.28)
        fig.savefig(out)
        plt.close(fig)
    print(out)
    return out


if __name__ == "__main__":
    main()
