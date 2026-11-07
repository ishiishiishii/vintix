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
COMMAND_SPEED_MPS = 0.5
# Per-step reward x dt x 1000 steps with zero velocity and no penalty:
# tracking_lin_vel 1.0 * exp(-0.5^2 / 0.25) + tracking_ang_vel 0.2 * exp(0), times 0.02 s.
STAND_STILL_RETURN = (1.0 * np.exp(-COMMAND_SPEED_MPS ** 2 / 0.25) + 0.2) * 0.02 * 1000
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
    done = length >= 1000
    # Only save_vintix.py writes locomotion metrics; they count each env until its first fall.
    speed = float("nan")
    summary = point / f"{OUTPUT_STEM}_mean_reward.txt"
    if summary.exists():
        for line in open(summary):
            if line.startswith("Avg Forward Speed (m/s):"):
                speed = float(line.split(":")[1])
    return {"reward": float(reward.mean()), "completion": float(done.mean() * 100),
            "finished_reward": float(reward[done].mean()) if done.any() else float("nan"), "speed": speed}


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
    ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=8, frameon=False)
    ax_c.set_title(f"Go1 {axis} scale - mean ± std over seeds ({time.strftime('%Y-%m-%d')})", fontsize=10, pad=34)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_walking_quality(data, axis: str, out: Path) -> Path:
    """How well the completed episodes walk: their mean return (all policies) and Vintix's forward speed.

    Completion alone counts an episode that survives 1000 steps without walking. Vintix speed is
    drawn hollow where completion < 80 %, since it only covers steps before each env's first fall.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_scales = sorted({x for policy in data for x in data[policy].get(axis, {})})
    pos = {x: i for i, x in enumerate(all_scales)}
    fig, (ax_q, ax_v) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for policy in ("vintix", "expert", "ppo"):
        pts = data.get(policy, {}).get(axis, {})
        if not pts:
            continue
        # NaN where no seed completed an episode, so the line breaks instead of bridging the gap.
        xs = sorted(pts)
        vals = [np.array([v["finished_reward"] for v in pts[x]]) for x in xs]
        mean = np.array([np.nanmean(v) if np.isfinite(v).any() else np.nan for v in vals])
        std = np.array([np.nanstd(v) if np.isfinite(v).any() else np.nan for v in vals])
        ax_q.errorbar([pos[x] for x in xs], mean, yerr=std, fmt="o-", color=COLORS[policy], label=LABELS[policy],
                      lw=2, ms=4, capsize=3)
        for x, m in zip(xs, mean):
            if m < -5:
                ax_q.annotate(f"{m:.1f}", (pos[x], -5), xytext=(pos[x], -2.5), color=COLORS[policy], fontsize=7,
                              ha="center", arrowprops={"arrowstyle": "->", "color": COLORS[policy]})
    ax_q.set_ylim(-5, 24)
    ax_q.axhline(STAND_STILL_RETURN, color="red", lw=1, ls=":")
    ax_q.text(0, STAND_STILL_RETURN + 0.4, f"standing still, no penalty ≈ {STAND_STILL_RETURN:.1f}", color="red", fontsize=7)
    ax_q.set_ylabel("Mean return of completed episodes\n(seeds with >= 1 completion)")
    pts = data.get("vintix", {}).get(axis, {})
    xs = sorted(pts)
    speed = np.array([np.nanmean([v["speed"] for v in pts[x]]) for x in xs])
    sd = np.array([np.nanstd([v["speed"] for v in pts[x]]) for x in xs])
    reliable = np.array([np.mean([v["completion"] for v in pts[x]]) >= 80 for x in xs])
    p = np.array([pos[x] for x in xs])
    ax_v.errorbar(p, speed, yerr=sd, fmt="-", color=COLORS["vintix"], lw=1.5, capsize=3, label=LABELS["vintix"])
    ax_v.plot(p[reliable], speed[reliable], "o", color=COLORS["vintix"], ms=5)
    ax_v.plot(p[~reliable], speed[~reliable], "o", mfc="white", color=COLORS["vintix"], ms=5,
              label="completion < 80 % (counts steps before the first fall only)")
    ax_v.axhline(COMMAND_SPEED_MPS, color="black", lw=1, ls=":", label=f"command {COMMAND_SPEED_MPS} m/s")
    ax_v.set_ylabel("Vintix forward speed [m/s]\n(mean over seeds)")
    ax_v.set_ylim(0, 0.7)
    ax_v.legend(fontsize=8, loc="lower right")
    for ax in (ax_q, ax_v):
        if 1.0 in pos:
            ax.axvline(pos[1.0], color="black", lw=1, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
    ax_v.set_xticks(range(len(all_scales)))
    if axis == "mass":
        ax_v.set_xticklabels([f"{t:g}\n{GO1_TOTAL_MASS_KG * t:.1f}kg" for t in all_scales], fontsize=8)
        ax_v.set_xlabel("Mass scale (Go1 = 1.0 = 13.1 kg; equally spaced, not to scale)")
    else:
        ax_v.set_xticklabels([f"{t:g}\n{GO1_THIGH_LENGTH_M * t * 100:.0f}cm" for t in all_scales], fontsize=8)
        ax_v.set_xlabel("Size scale (Go1 = 1.0 = thigh 21 cm, mass fixed; equally spaced, not to scale)")
    ax_q.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=8, frameon=False)
    ax_q.set_title(f"Go1 {axis} scale - how well the completed episodes walk ({time.strftime('%Y-%m-%d')})",
                   fontsize=10, pad=34)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_table(data, out: Path) -> Path:
    """Markdown table per axis for REPORT.md: completion of the three policies, Vintix speed, return vs PPO.

    Completion is the mean over seeds with the per-seed min-max. The return ratio divides Vintix's
    mean return of completed episodes by the PPO's mean return on the same body, and is left blank
    where the PPO itself did not learn to walk (completion < 50 %).
    """
    def stats(policy, axis, x, key):
        v = np.array([s[key] for s in data.get(policy, {}).get(axis, {}).get(x, [])], dtype=float)
        v = v[np.isfinite(v)]
        return (float(v.mean()), float(v.min()), float(v.max()), v.size) if v.size else None

    def completion(policy, axis, x):
        s = stats(policy, axis, x, "completion")
        if s is None:
            return "-"
        mean, lo, hi, n = s
        return f"{mean:.1f}" if lo == hi else f"{mean:.1f}（{lo:.0f}〜{hi:.0f}）"

    lines = [f"<!-- make_figures.py write_summary_table, {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} -->"]
    for axis in ("mass", "size"):
        unit = "質量" if axis == "mass" else "大きさ（質量は固定）"
        lines += ["", f"### {unit}", "",
                  "| 倍率 | シード数 | Vintix 完走率 % | 専門家 完走率 % | その体の PPO 完走率 % | Vintix 前進速度 m/s | Vintix 完走本の報酬 / PPO の報酬 |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for x in sorted({x for policy in data for x in data[policy].get(axis, {})}):
            n = max(len(data.get(p, {}).get(axis, {}).get(x, [])) for p in ("vintix", "expert"))
            speed = stats("vintix", axis, x, "speed")
            vintix_fin = stats("vintix", axis, x, "finished_reward")
            ppo_reward = stats("ppo", axis, x, "reward")
            ppo_completion = stats("ppo", axis, x, "completion")
            if ppo_completion is None:
                ratio = "-"
            elif ppo_completion[0] < 50:
                ratio = "（PPO 学習失敗）"
            elif vintix_fin is None:
                ratio = "（完走なし）"
            else:
                ratio = f"{vintix_fin[0]:.1f} / {ppo_reward[0]:.1f} = {vintix_fin[0] / ppo_reward[0]:.2f}"
            if speed is None:
                speed_text = "-"
            else:
                speed_text = f"{speed[0]:.2f}" + ("*" if stats("vintix", axis, x, "completion")[0] < 80 else "")
            lines.append(f"| {x:g} | {n} | {completion('vintix', axis, x)} | {completion('expert', axis, x)} | "
                         f"{completion('ppo', axis, x)} | {speed_text} | {ratio} |")
    lines += ["", "\\* Vintix の完走率 80% 未満。速度は各環境の最初の転倒までしか数えないので、転ぶ前の速さを表す。",
              "シード数は Vintix・専門家の評価シード数。PPO は各倍率 学習 1 本を 10 シードで評価。完走率は各環境の先頭 10 本。"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def episode_returns(point: Path, max_episodes: int, key: str = "cumulative_reward") -> np.ndarray:
    """(env, episode) matrix of ``key``, first ``max_episodes`` episodes per env (NaN if missing)."""
    rows = list(csv.DictReader(open(point / f"{OUTPUT_STEM}_episodes.csv")))
    by_env: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_env[int(r["env_index"])].append(r)
    out = np.full((max(by_env) + 1, max_episodes), np.nan)
    for env, env_rows in by_env.items():
        for i, r in enumerate(sorted(env_rows, key=lambda r: int(r["episode_number"]))[:max_episodes]):
            out[env, i] = float(r[key])
    return out


def plot_env_modes(policy: str, axis: str, scales: List[float], max_episodes: int, out: Path) -> Path:
    """Per-env, per-episode return heatmaps (rows = seeds, columns = scales)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({int(p.name.rsplit("seed", 1)[1]) for s in scales
                    for p in (RESULTS_DIR / policy / axis).glob(f"s{s:.3f}".replace(".", "p") + "_seed*")
                    if (p / f"{OUTPUT_STEM}_episodes.csv").exists()})
    fig, axes = plt.subplots(len(seeds), len(scales), figsize=(2.3 * len(scales) + 1, 2.3 * len(seeds) + 0.6),
                             squeeze=False, sharex=True, sharey=True)
    im = None
    for i, seed in enumerate(seeds):
        for j, s in enumerate(scales):
            ax = axes[i, j]
            point = RESULTS_DIR / policy / axis / (f"s{s:.3f}".replace(".", "p") + f"_seed{seed}")
            if (point / f"{OUTPUT_STEM}_episodes.csv").exists():
                im = ax.imshow(episode_returns(point, max_episodes), vmin=0, vmax=20, cmap="viridis", aspect="auto")
            else:
                ax.text(0.5, 0.5, "running", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            if i == 0:
                ax.set_title(f"{axis} x{s:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"seed {seed}\nenv index", fontsize=8)
            if i == len(seeds) - 1:
                ax.set_xlabel("episode", fontsize=8)
            ax.set_xticks(range(0, max_episodes, 3))
            ax.set_xticklabels(range(1, max_episodes + 1, 3), fontsize=7)
            ax.set_yticks(range(0, 10, 3))
            ax.tick_params(labelsize=7)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Episode return (falls early -> ~0-4)")
    fig.suptitle(f"{LABELS[policy]}: return of each env and episode ({time.strftime('%Y-%m-%d')})", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_first_episode_vs_heavy(axis: str, heavy_scales: List[float], max_episodes: int, out: Path) -> Path:
    """Per (seed, env): first-episode return on the normal body vs mean return on heavier bodies.

    The same seed gives each env the same initial offset on every body, so this shows whether the
    env's first episode on the normal body already tells which side it lands on near the boundary.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(heavy_scales), figsize=(3.2 * len(heavy_scales), 3.4), sharex=True, sharey=True,
                             squeeze=False)
    seeds = sorted(int(p.name.rsplit("seed", 1)[1]) for p in (RESULTS_DIR / "vintix" / "mass").glob("s1p000_seed*")
                   if (p / f"{OUTPUT_STEM}_episodes.csv").exists())
    for j, s in enumerate(heavy_scales):
        ax = axes[0, j]
        for k, seed in enumerate(seeds):
            point = RESULTS_DIR / "vintix" / axis / (f"s{s:.3f}".replace(".", "p") + f"_seed{seed}")
            if not (point / f"{OUTPUT_STEM}_episodes.csv").exists():
                continue
            first = episode_returns(RESULTS_DIR / "vintix" / "mass" / f"s1p000_seed{seed}", max_episodes)[:, 0]
            heavy = np.nanmean(episode_returns(point, max_episodes), axis=1)
            ax.scatter(first, heavy, s=28, color=plt.cm.tab10(k), label=f"seed {seed}", alpha=0.8)
        ax.set_title(f"{axis} x{s:g}", fontsize=9)
        ax.set_xlabel("1st-episode return, normal body", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(f"Mean return over {max_episodes} ep.\non the scaled body (1 dot = 1 env)", fontsize=8)
    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Vintix: does the normal-body 1st episode predict the heavy-side outcome? ({time.strftime('%Y-%m-%d')})",
                 fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_completion_by_episode(axis: str, scales: List[float], max_episodes: int, out: Path) -> Path:
    """Completion rate at each episode index (pooled over seeds x envs), Vintix vs the transferred expert.

    Vintix keeps earlier episodes in its context, so a rise after episode 1 is in-context adaptation;
    the expert has no memory, so its curve should stay flat.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scales), figsize=(3.0 * len(scales), 3.2), sharey=True, squeeze=False)
    for j, s in enumerate(scales):
        ax = axes[0, j]
        n_seeds = {}
        for policy in ("vintix", "expert"):
            done = []
            for point in sorted((RESULTS_DIR / policy / axis).glob(f"s{s:.3f}".replace(".", "p") + "_seed*")):
                f = point / f"{OUTPUT_STEM}_episodes.csv"
                if not f.exists():
                    continue
                by_env: Dict[int, List[dict]] = defaultdict(list)
                for r in csv.DictReader(open(f)):
                    by_env[int(r["env_index"])].append(r)
                for env_rows in by_env.values():
                    ep = sorted(env_rows, key=lambda r: int(r["episode_number"]))[:max_episodes]
                    if len(ep) == max_episodes:
                        done.append([float(r["episode_length"]) >= 1000 for r in ep])
            if done:
                n_seeds[policy] = len(done) // 10
                ax.plot(range(1, max_episodes + 1), np.array(done).mean(0) * 100, "o-", color=COLORS[policy], ms=3,
                        label=LABELS[policy].split(" (")[0])
        ax.set_title(f"{axis} x{s:g}  (seeds: Vintix {n_seeds.get('vintix', 0)}, expert {n_seeds.get('expert', 0)})",
                     fontsize=8)
        ax.set_xlabel("episode index", fontsize=8)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Completed [%] (seeds x envs pooled)", fontsize=8)
    axes[0, -1].legend(fontsize=7, loc="center right")
    fig.suptitle(f"Completion at each episode index near the heavy boundary ({time.strftime('%Y-%m-%d')})", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_completion_by_side(scales: List[float], max_episodes: int, out: Path, side_scale: float = 1.2) -> Path:
    """Completion of the Vintix walking / low side and of the expert on the same (seed, env) pairs.

    The side of each (seed, env) is fixed once on the ``side_scale`` body (episodes 2-10 mean return
    < 15 = low side). Within one policy a seed gives every env the same initial offset on every body;
    the expert script draws its offsets with the same code, but whether the GPU RNG state matches
    Vintix's at that point is not verified.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def point(policy: str, axis: str, s: float, seed: int) -> Path:
        return RESULTS_DIR / policy / axis / (f"s{s:.3f}".replace(".", "p") + f"_seed{seed}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, axis in zip(axes, ("mass", "size")):
        seeds = [k for k in range(10) if all((point(pol, axis, s, k) / f"{OUTPUT_STEM}_episodes.csv").exists()
                                             for pol in ("vintix", "expert") for s in scales + [side_scale])]
        low = np.concatenate([np.nanmean(episode_returns(point("vintix", axis, side_scale, k), max_episodes)[:, 1:],
                                         axis=1) < 15 for k in seeds])
        for policy, ls in (("vintix", "-"), ("expert", "--")):
            done = {s: np.concatenate([episode_returns(point(policy, axis, s, k), max_episodes, "episode_length")
                                       for k in seeds]) >= 1000 for s in scales}
            for side, mask, marker in (("walking side", ~low, "o"), ("low side", low, "x")):
                ax.plot(scales, [done[s][mask].mean() * 100 for s in scales], ls, marker=marker, ms=5,
                        color=COLORS[policy] if side == "walking side" else "#d62728" if policy == "vintix" else "#ff9896",
                        label=f"{LABELS[policy].split(' (')[0]} on Vintix {side} envs ({int(mask.sum())})")
        ax.set_title(f"{axis} axis (seeds x envs = {len(low)}; side fixed on x{side_scale:g})", fontsize=9)
        ax.set_xlabel(f"{axis} scale", fontsize=8)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Completed [%] (episodes 1-10)", fontsize=8)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Vintix low-side envs vs the expert on the same envs ({time.strftime('%Y-%m-%d')})", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ppo_learning_curves(out: Path) -> Path:
    """Mean training episode length of the from-scratch PPO runs, one line per body scale.

    Early in training every run first learns to end episodes quickly (the per-step reward is still
    negative and falling is not penalized); the dip gets deeper and longer for bigger bodies.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, axis in zip(axes, ("mass", "size")):
        runs = sorted((EXP_DIR / "ppo" / axis).glob("s*"), key=lambda d: float(d.name[1:].replace("p", ".")))
        cmap = plt.get_cmap("viridis")
        for k, d in enumerate(runs):
            events = sorted(d.glob("events.out.tfevents*"))
            if not events or not (d / "train_info.json").exists():
                continue
            ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
            ea.Reload()
            pts = ea.Scalars("Train/mean_episode_length")
            ax.plot([p.step for p in pts], [p.value for p in pts], color=cmap(k / max(len(runs) - 1, 1)), lw=1.2,
                    label=f"x{float(d.name[1:].replace('p', '.')):g}")
        ax.set_yscale("log")
        ax.set_xlim(0, 250)
        ax.set_title(f"{axis} axis: PPO trained on each body (1 run each)", fontsize=9)
        ax.set_xlabel("training iteration", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    axes[0].set_ylabel("mean episode length [steps] (log, 1000 = no fall)", fontsize=8)
    fig.suptitle(f"PPO learning: early dip to short episodes, size x1.5 and x2 never recover ({time.strftime('%Y-%m-%d')})",
                 fontsize=10)
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
        print(plot_walking_quality(data, axis, FIG_DIR / f"{axis}_walking_quality_{args.tag}.png"))
    print(plot_env_modes("vintix", "mass", [1.0, 1.1, 1.15, 1.2, 1.25, 1.3], args.max_episodes,
                         FIG_DIR / f"vintix_mass_env_modes_{args.tag}.png"))
    print(plot_env_modes("vintix", "size", [1.0, 1.1, 1.15, 1.2, 1.25, 1.3], args.max_episodes,
                         FIG_DIR / f"vintix_size_env_modes_{args.tag}.png"))
    for axis in ("mass", "size"):
        print(plot_first_episode_vs_heavy(axis, [1.1, 1.15, 1.2, 1.25], args.max_episodes,
                                          FIG_DIR / f"vintix_{axis}_first_episode_predictor_{args.tag}.png"))
        print(plot_completion_by_episode(axis, [1.15, 1.2, 1.25, 1.3], args.max_episodes,
                                         FIG_DIR / f"{axis}_completion_by_episode_{args.tag}.png"))
    print(plot_completion_by_side([1.0, 1.1, 1.15, 1.2, 1.25, 1.3], args.max_episodes,
                                  FIG_DIR / f"vintix_side_vs_expert_{args.tag}.png"))
    print(plot_ppo_learning_curves(FIG_DIR / f"ppo_learning_curves_{args.tag}.png"))
    print(write_summary_table(data, FIG_DIR / f"summary_table_{args.tag}.md"))


if __name__ == "__main__":
    main()
