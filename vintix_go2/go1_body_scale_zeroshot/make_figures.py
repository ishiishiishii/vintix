#!/usr/bin/env python3
"""Summary figures for Notion / README (completion rate and return vs body scale).

Separate from ``run_experiment.py`` so the running pipeline's workers are never affected.
Per point and seed, only the first ``max_episodes`` episodes of each env are used
(``save_expert.py`` keeps recording extra episodes on envs that fall early). Error bars are the
standard deviation of the per-seed values, i.e. how much one 10 env x 10 episode evaluation moves.

    cd /workspace/vintix_go2
    python go1_body_scale_zeroshot/make_figures.py            # -> figures/*_YYYYMMDD.png
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "figures"
OUTPUT_STEM = "go1_10envs_10episodes"
GO1_TOTAL_MASS_KG = 13.100528
GO1_THIGH_LENGTH_M = 0.213
LABELS = {"vintix": "Vintix zero-shot", "expert": "Go1 PPO expert (transferred)", "ppo": "PPO trained on the body"}
COLORS = {"vintix": "#1f77b4", "expert": "#7f7f7f", "ppo": "#2ca02c"}


def per_seed_values(point: Path, max_episodes: int) -> Dict[str, float]:
    rows = list(csv.DictReader(open(point / f"{OUTPUT_STEM}_episodes.csv")))
    by_env: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_env[int(r["env_index"])].append(r)
    kept = [r for env_rows in by_env.values() for r in sorted(env_rows, key=lambda r: int(r["episode_number"]))[:max_episodes]]
    reward = np.array([float(r["cumulative_reward"]) for r in kept])
    length = np.array([float(r["episode_length"]) for r in kept])
    return {"reward": float(reward.mean()), "completion": float((length >= 1000).mean() * 100)}


def collect(max_episodes: int) -> Dict[str, Dict[str, Dict[float, List[Dict[str, float]]]]]:
    data: Dict[str, Dict[str, Dict[float, List[Dict[str, float]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for meta in sorted(RESULTS_DIR.glob("*/*/*_seed*/point.json"), key=lambda p: (p.parent.parent, int(p.parent.name.rsplit("seed", 1)[1]))):
        point = meta.parent
        if not (point / f"{OUTPUT_STEM}_episodes.csv").exists():
            continue
        m = json.load(open(meta))
        data[m["policy"]][m["axis"]][float(m["scale"])].append(per_seed_values(point, max_episodes))
    return data


def plot_axis(data, axis: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Evaluated scales are placed at equal spacing so the dense steps near the drop stay readable.
    all_scales = sorted({x for policy in data for x in data[policy].get(axis, {})})
    pos = {x: i for i, x in enumerate(all_scales)}
    fig, (ax_c, ax_r) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for policy in ("vintix", "expert", "ppo"):
        pts = data.get(policy, {}).get(axis, {})
        if not pts:
            continue
        xs = sorted(pts)
        n = [len(pts[x]) for x in xs]
        seeds = f"{min(n)}-{max(n)} seeds/point" if min(n) != max(n) else f"{n[0]} seed{'s' if n[0] > 1 else ''}/point"
        for ax, key in ((ax_c, "completion"), (ax_r, "reward")):
            vals = [np.array([v[key] for v in pts[x]]) for x in xs]
            mean = np.array([v.mean() for v in vals])
            std = np.array([v.std() for v in vals])
            ax.errorbar([pos[x] for x in xs], mean, yerr=std, fmt="o-", color=COLORS[policy],
                        label=f"{LABELS[policy]} ({seeds})", lw=2, ms=4, capsize=3)
    for ax in (ax_c, ax_r):
        if 1.0 in pos:
            ax.axvline(pos[1.0], color="black", lw=1, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
    ax_c.set_ylabel("Completed episodes [%]\n(no fall in 1000 steps)")
    ax_c.set_ylim(-5, 105)
    ax_r.set_ylabel("Episode return")
    ax_r.set_xticks(range(len(all_scales)))
    if axis == "mass":
        ax_r.set_xticklabels([f"{t:g}\n{GO1_TOTAL_MASS_KG * t:.1f}kg" for t in all_scales], fontsize=8)
        ax_r.set_xlabel("Mass scale (Go1 = 1.0 = 13.1 kg; equally spaced, not to scale)")
    else:
        ax_r.set_xticklabels([f"{t:g}\n{GO1_THIGH_LENGTH_M * t * 100:.0f}cm" for t in all_scales], fontsize=8)
        ax_r.set_xlabel("Size scale (Go1 = 1.0 = thigh 21 cm, mass fixed; equally spaced, not to scale)")
    ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8, frameon=False)
    ax_c.set_title(f"Go1 {axis} scale - mean ± std over seeds ({time.strftime('%Y-%m-%d')})", fontsize=10, pad=24)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--tag", type=str, default=time.strftime("%Y%m%d"))
    args = p.parse_args()
    data = collect(args.max_episodes)
    for axis in ("mass", "size"):
        print(plot_axis(data, axis, FIG_DIR / f"{axis}_completion_return_{args.tag}.png"))


if __name__ == "__main__":
    main()
