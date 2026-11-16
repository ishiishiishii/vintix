#!/usr/bin/env python3
"""Multi-robot body-scale zero-shot study: workers, aggregation and figures.

Each job runs in its own subprocess (driven by ``pipeline.py``) and calls the shared evaluators
without modifying them; only inside the subprocess the robot's env class in ``save_vintix`` /
``save_expert`` is replaced by the scaled subclass.

    cd /workspace/vintix_go2
    python robot_body_scale_zeroshot/run_experiment.py --worker vintix --robot a1 --axis size --scale 1.2 --seed 0
    python robot_body_scale_zeroshot/run_experiment.py --worker train_ppo --robot go2 --axis mass --scale 2.0
    python robot_body_scale_zeroshot/run_experiment.py --stage report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
SCRIPTS_DIR = VINTIX_ROOT / "scripts"
GO1_EXP_DIR = VINTIX_ROOT / "go1_body_scale_zeroshot"
sys.path.insert(0, str(EXP_DIR))

from robots import (  # noqa: E402
    AXES,
    PPO_ITERATIONS,
    PPO_TRAIN_SEED,
    ROBOTS,
    canonical_axis,
    physical_summary,
    scale_tag,
    scaled_cfgs,
)

RESULTS_DIR = EXP_DIR / "results"
PPO_DIR = EXP_DIR / "ppo"
LOGS_DIR = EXP_DIR / "logs"
FIG_DIR = EXP_DIR / "figures"
POLICIES = ("vintix", "expert", "ppo")
LABELS = {"vintix": "Vintix zero-shot (<robot>_without)", "expert": "PPO expert of the nominal body (transferred)",
          "ppo": f"PPO trained on the body ({PPO_ITERATIONS} iters)"}
COLORS = {"vintix": "#1f77b4", "expert": "#7f7f7f", "ppo": "#2ca02c"}
AXIS_TITLES = {"mass": "mass x m (size fixed)", "size": "size x s (mass fixed)",
               "size_density": "size x s at constant density (mass x s^3)"}


def stem(robot: str) -> str:
    return f"{ROBOTS[robot]['robot_type']}_10envs_10episodes"


def point_dir(robot: str, policy: str, axis: str, scale: float, seed: int) -> Path:
    return RESULTS_DIR / robot / policy / canonical_axis(axis, scale) / f"{scale_tag(scale)}_seed{seed}"


def episodes_csv(robot: str, policy: str, axis: str, scale: float, seed: int) -> Path:
    return point_dir(robot, policy, axis, scale, seed) / f"{stem(robot)}_episodes.csv"


def probe_json(robot: str, axis: str, scale: float) -> Path:
    return RESULTS_DIR / robot / "probe" / canonical_axis(axis, scale) / f"{scale_tag(scale)}.json"


def ppo_dir(robot: str, axis: str, scale: float) -> Path:
    return PPO_DIR / robot / canonical_axis(axis, scale) / scale_tag(scale)


def ppo_final_ckpt(robot: str, axis: str, scale: float) -> Path:
    return ppo_dir(robot, axis, scale) / f"model_{PPO_ITERATIONS}.pt"


# --------------------------------------------------------------------------- workers


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scaled_class(robot: str, module):
    from scaled_env import scaled_env_class

    return scaled_env_class(getattr(module, ROBOTS[robot]["env_attr"]), ROBOTS[robot]["urdf"].name)


def worker_eval(robot: str, policy: str, axis: str, scale: float, seed: int, num_envs: int, max_episodes: int) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = ROBOTS[robot]
    out_dir = point_dir(robot, policy, axis, scale, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / stem(robot))
    _seed_everything(seed)
    if policy == "vintix":
        import save_vintix
        import genesis as gs

        setattr(save_vintix, spec["env_attr"], _scaled_class(robot, save_vintix))
        gs.init(seed=seed, performance_mode=True)
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = scaled_cfgs(robot, axis, scale)
        model = str(spec["vintix"])
        args = argparse.Namespace(robot_type=spec["robot_type"], vintix_path=model, num_envs=num_envs,
                                  max_episodes=max_episodes, output=output, base_mass=None,
                                  trajectory_stats_path=None, body_params_path=None)
        save_vintix._run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg)
    else:
        import save_expert
        import genesis as gs

        setattr(save_expert, spec["env_attr"], _scaled_class(robot, save_expert))
        gs.init(seed=seed, performance_mode=True)
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = scaled_cfgs(robot, axis, scale)
        model = str(spec["expert_ckpt"] if policy == "expert" else ppo_final_ckpt(robot, axis, scale))
        if not Path(model).exists():
            raise FileNotFoundError(model)
        args = argparse.Namespace(robot_type=spec["robot_type"], exp_name=spec["expert_exp"], expert_model_path=model,
                                  expert_model_type=None, num_envs=num_envs, max_steps=None,
                                  max_episodes=max_episodes, output=output)
        save_expert._run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    with open(out_dir / "point.json", "w") as f:
        json.dump({"robot": robot, "policy": policy, "axis": canonical_axis(axis, scale), "scale": scale,
                   "seed": seed, "model": model, "num_envs": num_envs, "max_episodes": max_episodes,
                   **physical_summary(robot, canonical_axis(axis, scale), scale)}, f, indent=2)


def worker_train_ppo(robot: str, axis: str, scale: float, num_envs: int, iterations: int, seed: int) -> None:
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner

    sys.path.insert(0, str(VINTIX_ROOT.parent / "Genesis" / "examples" / "locomotion"))
    import env as env_module

    _seed_everything(seed)
    gs.init(logging_level="warning", precision="64", seed=seed)  # as Genesis/examples/locomotion/train.py
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = scaled_cfgs(robot, axis, scale)
    train_cfg = copy.deepcopy(train_cfg)
    train_cfg["seed"] = seed
    train_cfg["runner"]["max_iterations"] = iterations
    train_cfg["runner"]["experiment_name"] = f"{robot}_{axis}_{scale_tag(scale)}"
    log_dir = ppo_dir(robot, axis, scale)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
    cls = _scaled_class(robot, env_module)
    env = cls(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    t0 = time.time()
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    final = log_dir / f"model_{iterations}.pt"
    if not final.exists():
        runner.save(str(final))
    with open(log_dir / "train_info.json", "w") as f:
        json.dump({"robot": robot, "axis": axis, "scale": scale, "iterations": iterations, "num_envs": num_envs,
                   "seed": seed, "minutes": (time.time() - t0) / 60, **physical_summary(robot, axis, scale)}, f, indent=2)


def worker_probe(robot: str, axis: str, scale: float, hold_steps: int = 150) -> None:
    import torch
    import genesis as gs

    sys.path.insert(0, str(VINTIX_ROOT.parent / "Genesis" / "examples" / "locomotion"))
    import env as env_module

    gs.init(performance_mode=True)
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = scaled_cfgs(robot, axis, scale)
    env = _scaled_class(robot, env_module)(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                                           command_cfg=command_cfg)
    env.reset()
    heights, falls = [], 0
    zero = torch.zeros(1, env.num_actions, device=gs.device)
    for _ in range(hold_steps):
        _, _, dones, _ = env.step(zero)
        falls += int(dones[0].item())
        heights.append(float(env.base_pos[0, 2].item()))
    out = {"robot": robot, "axis": axis, "scale": scale, "intended": physical_summary(robot, axis, scale),
           "sim_total_mass_kg": float(env.robot.get_mass()),
           "standing_height_m_last50_mean": float(np.mean(heights[-50:])),
           "base_height_target_m": reward_cfg.get("base_height_target"),
           "falls_while_holding_zero_action": falls}
    path = probe_json(robot, axis, scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- subprocess helpers (pipeline.py)


def worker_cmd(kind: str, robot: str, axis: str, scale: float, seed: Optional[int] = None) -> List[str]:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", kind, "--robot", robot,
           "--axis", axis, "--scale", str(scale)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    loco = str(VINTIX_ROOT.parent / "Genesis" / "examples" / "locomotion")
    env["PYTHONPATH"] = f"{VINTIX_ROOT}:{loco}:{env.get('PYTHONPATH', '')}"
    return env


# --------------------------------------------------------------------------- aggregation


def _per_seed(episodes: Path, max_episodes: int = 10) -> Dict[str, float]:
    rows = list(csv.DictReader(open(episodes)))
    by_env: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_env[int(r["env_index"])].append(r)
    kept = [r for env_rows in by_env.values() for r in sorted(env_rows, key=lambda r: int(r["episode_number"]))[:max_episodes]]
    reward = np.array([float(r["cumulative_reward"]) for r in kept])
    length = np.array([float(r["episode_length"]) for r in kept])
    return {"reward": float(reward.mean()), "completion": float((length >= 1000).mean() * 100),
            "length": float(length.mean()), "n": int(len(kept))}


def collect() -> Dict[tuple, List[Dict[str, float]]]:
    """(robot, policy, axis, scale) -> per-seed values. Nominal (1.0) is copied to every axis.

    Go1's mass / size axes come from go1_body_scale_zeroshot (same protocol and seeds).
    """
    data: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
    for meta in sorted(RESULTS_DIR.glob("*/*/*/*_seed*/point.json")):
        m = json.load(open(meta))
        ep = meta.parent / f"{stem(m['robot'])}_episodes.csv"
        if ep.exists():
            data[(m["robot"], m["policy"], m["axis"], float(m["scale"]))].append(_per_seed(ep))
    for meta in sorted(GO1_EXP_DIR.glob("results/*/*/*_seed*/point.json")):
        m = json.load(open(meta))
        ep = meta.parent / "go1_10envs_10episodes_episodes.csv"
        if ep.exists():
            data[("go1", m["policy"], m["axis"], float(m["scale"]))].append(_per_seed(ep))
    for (robot, policy, axis, scale), vals in list(data.items()):
        if axis == "mass" and abs(scale - 1.0) < 1e-9:
            for other in AXES[1:]:
                data.setdefault((robot, policy, other, scale), vals)
    return data


def write_results_csv(data) -> Path:
    path = EXP_DIR / "results.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["robot", "policy", "axis", "scale", "total_mass_kg", "thigh_length_m", "n_seeds",
                    "completion_mean", "completion_std", "reward_mean", "reward_std", "episode_length_mean"])
        for (robot, policy, axis, scale), vals in sorted(data.items()):
            c = np.array([v["completion"] for v in vals])
            r = np.array([v["reward"] for v in vals])
            l = np.array([v["length"] for v in vals])
            phys = physical_summary(robot, axis, scale)
            w.writerow([robot, policy, axis, scale, f"{phys['total_mass_kg']:.3f}", f"{phys['thigh_length_m']:.4f}",
                        len(vals), f"{c.mean():.2f}", f"{c.std():.2f}", f"{r.mean():.3f}", f"{r.std():.3f}", f"{l.mean():.1f}"])
    return path


def _tick_label_indices(scales: List[float], step: int = 3) -> List[int]:
    """Always label 1.0 (if present) and the ends; then every ``step`` ticks from 1.0."""
    n = len(scales)
    if n == 0:
        return []
    labeled = {0, n - 1}
    base = next((i for i, s in enumerate(scales) if abs(s - 1.0) < 1e-12), None)
    if base is None:
        labeled.update(range(0, n, step))
    else:
        labeled.add(base)
        for i in range(base - step, -1, -step):
            labeled.add(i)
        for i in range(base + step, n, step):
            labeled.add(i)
    return sorted(labeled)


def plot_robot_axis(data, robot: str, axis: str, tag: str) -> Optional[Path]:
    """Success rate + return vs scale for one robot×axis (mean line + std band).

    No in-figure title; filename carries meaning. Matches figure-style.md.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = sorted({k[3] for k in data if k[0] == robot and k[2] == axis})
    if not scales:
        return None
    # Align x-range to Body-PPO upper end only when that upper end exists (scale > 1.0).
    # Incomplete lower-only PPO (e.g. 0.5× still running) must not clip Vintix/Expert.
    ppo_scales = [k[3] for k in data if k[0] == robot and k[1] == "ppo" and k[2] == axis]
    if ppo_scales:
        xmax = max(ppo_scales)
        if xmax > 1.0 + 1e-12:
            scales = [x for x in scales if x <= xmax + 1e-12]
    pos = {x: i for i, x in enumerate(scales)}
    legend_names = {"vintix": "Vintix", "expert": "Expert", "ppo": "Body PPO"}
    axis_xlabel = {
        "mass": f"Mass scale (× {robot}; equally spaced, not to scale)",
        "size": f"Size scale (× {robot}; mass fixed; equally spaced, not to scale)",
        "size_density": f"Size scale (× {robot}; density fixed; equally spaced, not to scale)",
    }[axis]

    with plt.rc_context({
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "axes.linewidth": 1.4,
    }):
        fig, (ax_c, ax_r) = plt.subplots(2, 1, figsize=(10.5, 8.6), sharex=True)
        for policy in POLICIES:
            xs = [x for x in scales if (robot, policy, axis, x) in data]
            if not xs:
                continue
            n = [len(data[(robot, policy, axis, x)]) for x in xs]
            if min(n) != max(n):
                seeds = f"{min(n)}-{max(n)} Seeds"
            else:
                seeds = f"{n[0]} Seeds" if n[0] != 1 else "1 Seed"
            xp = np.array([pos[x] for x in xs], dtype=float)
            for ax, key in ((ax_c, "completion"), (ax_r, "reward")):
                vals = [np.array([v[key] for v in data[(robot, policy, axis, x)]]) for x in xs]
                mean = np.array([v.mean() for v in vals])
                std = np.array([v.std() for v in vals])
                ax.plot(xp, mean, "o-", color=COLORS[policy],
                        label=f"{legend_names[policy]} ({seeds})",
                        lw=2.8, ms=6, zorder=3)
                ax.fill_between(xp, mean - std, mean + std, color=COLORS[policy], alpha=0.25, lw=0, zorder=2)
        for ax in (ax_c, ax_r):
            if 1.0 in pos:
                ax.axvline(pos[1.0], color="black", lw=1.3, ls="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.tick_params(width=1.3, length=6)
        ax_c.set_ylabel("Success rate [%]\n(no fall in 1000 steps)")
        ax_c.set_ylim(-5, 105)
        ax_r.set_ylabel("Episode return")
        tick_idx = set(_tick_label_indices(scales, step=3))
        ax_r.set_xticks(range(len(scales)))
        ax_r.set_xticklabels([f"{scales[i]:g}" if i in tick_idx else "" for i in range(len(scales))])
        ax_r.set_xlabel(axis_xlabel)
        fig.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.14, hspace=0.12)
        handles, labels = ax_c.get_legend_handles_labels()
        axes_mid = 0.5 * (0.14 + 0.99)
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(axes_mid, 0.055),
                   handlelength=2.0, columnspacing=1.6, handletextpad=0.5)
        out = FIG_DIR / f"{robot}_{axis}_scale_completion_return_mean_std_{tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        plt.close(fig)
    return out


def stage_report(tag: Optional[str] = None) -> None:
    data = collect()
    path = write_results_csv(data)
    tag = tag or time.strftime("%Y%m%d")
    figs: List[Path] = []
    for robot in ROBOTS:
        for axis in AXES:
            p = plot_robot_axis(data, robot, axis, tag)
            if p:
                figs.append(p)
    with open(EXP_DIR / "summary.json", "w") as f:
        json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "n_points": len(data),
                   "results_csv": str(path.relative_to(EXP_DIR)),
                   "figures": [str(p.relative_to(EXP_DIR)) for p in figs]}, f, indent=2)
    print(f"[report] {len(data)} (robot, policy, axis, scale) points → {len(figs)} figures", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["report"], default=None)
    p.add_argument("--tag", type=str, default=None, help="figure file suffix for --stage report (default 'latest')")
    p.add_argument("--worker", choices=["vintix", "expert", "ppo", "probe", "train_ppo"], default=None)
    p.add_argument("--robot", choices=list(ROBOTS), default=None)
    p.add_argument("--axis", choices=list(AXES), default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--ppo_num_envs", type=int, default=4096)
    p.add_argument("--ppo_iterations", type=int, default=PPO_ITERATIONS)
    a = p.parse_args()
    if a.stage == "report":
        stage_report(a.tag)
    elif a.worker == "probe":
        worker_probe(a.robot, a.axis, a.scale)
    elif a.worker == "train_ppo":
        worker_train_ppo(a.robot, a.axis, a.scale, a.ppo_num_envs, a.ppo_iterations,
                         PPO_TRAIN_SEED if a.seed is None else a.seed)
    elif a.worker:
        worker_eval(a.robot, a.worker, a.axis, a.scale, a.seed, a.num_envs, a.max_episodes)
    else:
        p.error("give --stage report or --worker ...")


if __name__ == "__main__":
    main()
