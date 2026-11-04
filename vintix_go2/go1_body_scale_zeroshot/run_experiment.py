#!/usr/bin/env python3
"""Go1 mass / size sweep: how far does Vintix zero-shot keep walking?

Orchestration only. Each job runs in its own subprocess (fresh Genesis state) and calls the
shared code without modifying it:

- ``vintix``: ``scripts/save_vintix._run_parallel_evaluation`` with ``go1_without/0001_epoch``
  (Go1 never seen in training; unknown-task normalization = average of the known tasks,
  identical to the ISCIIA zero-shot protocol).
- ``expert``: ``scripts/save_expert._run_parallel_evaluation`` with the nominal Go1 PPO expert
  (``model_2000.pt``) transferred as-is to the modified body.
- ``ppo``: same evaluator with a PPO trained from scratch on that modified body
  (``--worker train_ppo``; rsl-rl ``OnPolicyRunner`` with the Go1 expert's train cfg).
- ``probe``: builds the variant, holds the default pose with zero actions and records the
  simulated total mass / standing height.

Evaluations take ``--seed``: python / numpy / torch / Genesis are seeded before the run, so each
seed is an independent, reproducible 10 envs x 10 episodes evaluation
(results/<policy>/<axis>/s<scale>_seed<k>/). The earlier unseeded single runs (``_r0``) are kept
for reference only and are not mixed into the seeded aggregation.

Usage (inside the container; normally driven by ``pipeline.py``)::

    cd /workspace/vintix_go2
    python go1_body_scale_zeroshot/run_experiment.py --worker vintix --axis mass --scale 1.25 --seed 0
    python go1_body_scale_zeroshot/run_experiment.py --worker train_ppo --axis mass --scale 1.3
    python go1_body_scale_zeroshot/run_experiment.py --stage report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
SCRIPTS_DIR = VINTIX_ROOT / "scripts"
sys.path.insert(0, str(EXP_DIR))

from body_scale import (  # noqa: E402
    AXES,
    GO1_EXPERT_CKPT,
    GO1_THIGH_LENGTH_M,
    GO1_TOTAL_MASS_KG,
    GO1_WITHOUT_CKPT,
    PPO_ITERATIONS,
    PPO_TRAIN_SEED,
    REFERENCE_ROBOTS,
    physical_summary,
    scale_tag,
    scaled_go1_cfgs,
)

RESULTS_DIR = EXP_DIR / "results"
PPO_DIR = EXP_DIR / "ppo"
LOGS_DIR = EXP_DIR / "logs"
GRAPHS_DIR = EXP_DIR / "graphs"
POLICIES = ("vintix", "expert", "ppo")
POLICY_LABELS = {
    "vintix": "Vintix zero-shot (go1_without)",
    "expert": "Go1 PPO expert (transferred)",
    "ppo": f"PPO trained on the body ({PPO_ITERATIONS} iters)",
}
POLICY_COLORS = {"vintix": "#1f77b4", "expert": "#7f7f7f", "ppo": "#2ca02c"}
OUTPUT_STEM = "go1_10envs_10episodes"


# --------------------------------------------------------------------------- paths


def run_name(seed: Optional[int]) -> str:
    return "r0" if seed is None else f"seed{seed}"


def point_dir(policy: str, axis: str, scale: float, seed: Optional[int]) -> Path:
    return RESULTS_DIR / policy / axis / f"{scale_tag(scale)}_{run_name(seed)}"


def episodes_csv(policy: str, axis: str, scale: float, seed: Optional[int]) -> Path:
    return point_dir(policy, axis, scale, seed) / f"{OUTPUT_STEM}_episodes.csv"


def probe_json(axis: str, scale: float) -> Path:
    return RESULTS_DIR / "probe" / axis / f"{scale_tag(scale)}.json"


def ppo_dir(axis: str, scale: float) -> Path:
    return PPO_DIR / axis / scale_tag(scale)


def ppo_final_ckpt(axis: str, scale: float) -> Path:
    return ppo_dir(axis, scale) / f"model_{PPO_ITERATIONS}.pt"


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


def _worker_eval(policy: str, axis: str, scale: float, seed: Optional[int], num_envs: int, max_episodes: int) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = scaled_go1_cfgs(axis, scale)
    out_dir = point_dir(policy, axis, scale, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / OUTPUT_STEM)

    from scaled_go1_env import ScaledGo1Env

    _seed_everything(seed)
    if policy == "vintix":
        import save_vintix
        import genesis as gs

        # Shared script untouched: only this process builds Go1 through the scaled subclass.
        save_vintix.Go1Env = ScaledGo1Env
        gs.init(seed=seed, performance_mode=True)
        args = argparse.Namespace(
            robot_type="go1",
            vintix_path=str(GO1_WITHOUT_CKPT),
            num_envs=num_envs,
            max_episodes=max_episodes,
            output=output,
            base_mass=None,
            trajectory_stats_path=None,
            body_params_path=None,
        )
        save_vintix._run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg)
        model_path = str(GO1_WITHOUT_CKPT)
    elif policy in ("expert", "ppo"):
        import save_expert
        import genesis as gs

        save_expert.Go1Env = ScaledGo1Env
        gs.init(seed=seed, performance_mode=True)
        if policy == "expert":
            model_path = str(GO1_EXPERT_CKPT)
        else:
            model_path = str(ppo_final_ckpt(axis, scale))
            if not Path(model_path).exists():
                raise FileNotFoundError(f"PPO checkpoint missing: {model_path}")
        args = argparse.Namespace(
            robot_type="go1",
            exp_name="go1-walking",
            expert_model_path=model_path,
            expert_model_type=None,
            num_envs=num_envs,
            max_steps=None,
            max_episodes=max_episodes,
            output=output,
        )
        save_expert._run_parallel_evaluation(args, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    else:
        raise ValueError(policy)

    with open(out_dir / "point.json", "w") as f:
        json.dump(
            {"policy": policy, "axis": axis, "scale": scale, "seed": seed, "model": model_path,
             "num_envs": num_envs, "max_episodes": max_episodes,
             "env_cfg_body": {k: env_cfg.get(k) for k in ("body_mass_scale", "body_size_scale", "base_init_pos")},
             "reward_heights": {k: reward_cfg[k] for k in ("base_height_target", "feet_height_target")},
             **physical_summary(axis, scale)},
            f, indent=2,
        )


def _worker_train_ppo(axis: str, scale: float, num_envs: int, iterations: int, seed: int) -> None:
    """PPO from scratch on one body, same algorithm/runner settings as the Go1 expert."""
    import torch
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner

    from scaled_go1_env import ScaledGo1Env

    _seed_everything(seed)
    gs.init(logging_level="warning", precision="64", seed=seed)  # as Genesis/examples/locomotion/train.py
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = scaled_go1_cfgs(axis, scale)
    train_cfg = copy.deepcopy(train_cfg)
    train_cfg["seed"] = seed
    train_cfg["runner"]["max_iterations"] = iterations
    train_cfg["runner"]["experiment_name"] = f"go1_{axis}_{scale_tag(scale)}"

    log_dir = ppo_dir(axis, scale)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    env = ScaledGo1Env(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                       command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    t0 = time.time()
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    final = log_dir / f"model_{iterations}.pt"
    if not final.exists():
        runner.save(str(final))
    with open(log_dir / "train_info.json", "w") as f:
        json.dump({"axis": axis, "scale": scale, "iterations": iterations, "num_envs": num_envs,
                   "seed": seed, "minutes": (time.time() - t0) / 60, "final_ckpt": str(final),
                   **physical_summary(axis, scale)}, f, indent=2)
    del torch


def _worker_probe(axis: str, scale: float, hold_steps: int = 150) -> None:
    import torch
    import genesis as gs

    from scaled_go1_env import ScaledGo1Env

    gs.init(performance_mode=True)
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = scaled_go1_cfgs(axis, scale)
    env = ScaledGo1Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
    env.reset()
    heights, terminations = [], 0
    zero = torch.zeros(1, env.num_actions, device=gs.device)
    for _ in range(hold_steps):
        _, _, dones, _ = env.step(zero)
        terminations += int(dones[0].item())
        heights.append(float(env.base_pos[0, 2].item()))

    image_path = None
    try:
        size = float(env_cfg.get("body_size_scale", 1.0))
        env.cam.set_pose(pos=(0.9 * size, -1.6 * size, 0.6 * size), lookat=(0.0, 0.0, 0.2 * size))
        rgb = env.cam.render()[0]
        import matplotlib.pyplot as plt

        image_path = probe_json(axis, scale).with_suffix(".png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(image_path, np.asarray(rgb))
    except Exception as e:  # rendering is optional (headless setups)
        print(f"[probe] render skipped: {e}")

    out = {
        "axis": axis,
        "scale": scale,
        "intended": physical_summary(axis, scale),
        "sim_total_mass_kg": float(env.robot.get_mass()),
        "standing_height_m_last50_mean": float(np.mean(heights[-50:])),
        "base_height_target_m": reward_cfg["base_height_target"],
        "base_init_z_m": env_cfg["base_init_pos"][2],
        "terminations_while_holding": terminations,
        "image": str(image_path) if image_path else None,
    }
    probe_json(axis, scale).parent.mkdir(parents=True, exist_ok=True)
    with open(probe_json(axis, scale), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- subprocess helper (used by pipeline.py)


def worker_cmd(kind: str, axis: str, scale: float, seed: Optional[int] = None,
               num_envs: int = 10, max_episodes: int = 10) -> List[str]:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", kind, "--axis", axis, "--scale", str(scale)]
    if kind in ("vintix", "expert", "ppo"):
        cmd += ["--num_envs", str(num_envs), "--max_episodes", str(max_episodes)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
    return cmd


def subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    env["PYTHONPATH"] = f"{VINTIX_ROOT}:{env.get('PYTHONPATH', '')}"
    return env


def run_subprocess(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(VINTIX_ROOT), env=subprocess_env())


# --------------------------------------------------------------------------- report


def _read_episodes(path: Path) -> Dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {
        "episode_number": np.array([int(r["episode_number"]) for r in rows]),
        "reward": np.array([float(r["cumulative_reward"]) for r in rows]),
        "length": np.array([float(r["episode_length"]) for r in rows]),
    }


_METRIC_PATTERNS = {
    "tracking_success_rate": r"Tracking Success Rate \(±15%\):\s*([-\d.eE+]+)",
    "avg_forward_speed_mps": r"Avg Forward Speed \(m/s\):\s*([-\d.eE+]+)",
    "cost_of_transport": r"Cost of Transport \(CoT\):\s*([-\d.eE+]+)",
}


def _read_loco_metrics(point: Path) -> Dict[str, float]:
    txt = point / f"{OUTPUT_STEM}_mean_reward.txt"
    if not txt.exists():
        return {}
    text = txt.read_text()
    out = {}
    for key, pat in _METRIC_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def collect_rows(seeded: bool = True) -> List[Dict[str, object]]:
    """One row per (policy, axis, scale), pooling every episode of every seed.

    ``reward_std`` is over all pooled episodes; ``reward_seed_std`` is the spread of the
    per-seed means (how much a single 10x10 evaluation can move).
    """
    rows = []
    for policy in POLICIES:
        for axis in AXES:
            base = RESULTS_DIR / policy / axis
            if not base.is_dir():
                continue
            by_scale: Dict[float, List[Path]] = {}
            for point in sorted(base.iterdir()):
                meta = point / "point.json"
                if not meta.exists() or not (point / f"{OUTPUT_STEM}_episodes.csv").exists():
                    continue
                is_seeded = point.name.split("_")[-1].startswith("seed")
                if is_seeded != seeded:
                    continue
                scale = float(json.load(open(meta))["scale"])
                by_scale.setdefault(scale, []).append(point)
            for scale, points in sorted(by_scale.items()):
                eps = [_read_episodes(p / f"{OUTPUT_STEM}_episodes.csv") for p in points]
                reward = np.concatenate([e["reward"] for e in eps])
                length = np.concatenate([e["length"] for e in eps])
                seed_means = np.array([e["reward"].mean() for e in eps])
                first = np.concatenate([e["reward"][e["episode_number"] == e["episode_number"].min()] for e in eps])
                last = np.concatenate([e["reward"][e["episode_number"] == e["episode_number"].max()] for e in eps])
                row = {
                    "policy": policy,
                    "axis": axis,
                    "scale": scale,
                    **physical_summary(axis, scale),
                    "n_seeds": len(points),
                    "n_episodes": int(reward.size),
                    "reward_mean": float(reward.mean()),
                    "reward_std": float(reward.std()),
                    "reward_seed_std": float(seed_means.std()),
                    "episode_length_mean": float(length.mean()),
                    "episode_length_std": float(length.std()),
                    "full_episode_rate": float((length >= 1000).mean()),
                    "reward_first_episode_mean": float(first.mean()),
                    "reward_last_episode_mean": float(last.mean()),
                }
                metrics = [_read_loco_metrics(p) for p in points]
                for key in _METRIC_PATTERNS:
                    vals = [m[key] for m in metrics if key in m]
                    row[key] = float(np.mean(vals)) if vals else ""
                rows.append(row)
    return rows


def write_results_csv(rows: List[Dict[str, object]], path: Path) -> Path:
    if not rows:
        return path
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def _axis_ticklabel(axis: str, scale: float) -> str:
    if axis == "mass":
        return f"{scale:g} ({GO1_TOTAL_MASS_KG * scale:.1f} kg)"
    return f"{scale:g} ({GO1_THIGH_LENGTH_M * scale * 100:.1f} cm)"


def plot(rows: List[Dict[str, object]], out_dir: Path, suffix: str = "") -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    xlabels = {"mass": "Mass scale (total mass)", "size": "Size scale (thigh length), mass fixed"}
    written = []
    for axis in AXES:
        axis_rows = [r for r in rows if r["axis"] == axis]
        if not axis_rows:
            continue
        fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for policy in POLICIES:
            pr = sorted((r for r in axis_rows if r["policy"] == policy), key=lambda r: r["scale"])
            if not pr:
                continue
            x = np.array([r["scale"] for r in pr])
            n_seeds = sorted({int(r.get("n_seeds", r.get("n_repeats", 1))) for r in pr})
            label = f"{POLICY_LABELS[policy]} [seeds/pt: {n_seeds[0]}" + (f"-{n_seeds[-1]}]" if len(n_seeds) > 1 else "]")
            for ax, key in ((ax_r, "reward"), (ax_l, "episode_length")):
                m = np.array([r[f"{key}_mean"] for r in pr])
                s = np.array([r[f"{key}_std"] for r in pr])
                ax.plot(x, m, "o-", color=POLICY_COLORS[policy], label=label, lw=2, ms=4)
                ax.fill_between(x, m - s, m + s, color=POLICY_COLORS[policy], alpha=0.15)
        for ax in (ax_r, ax_l):
            ax.set_xscale("log")
            ax.axvline(1.0, color="black", lw=1, ls="--", alpha=0.6)
            for name, ratio in REFERENCE_ROBOTS[axis].items():
                ax.axvline(ratio, color="tab:red", lw=0.8, ls=":", alpha=0.7)
            ax.grid(True, alpha=0.3, which="both")
        ymax = ax_r.get_ylim()[1]
        for name, ratio in REFERENCE_ROBOTS[axis].items():
            ax_r.text(ratio, ymax, name, rotation=90, va="top", ha="right", fontsize=9, color="tab:red")
        ticks = sorted({r["scale"] for r in axis_rows})
        ax_l.set_xticks(ticks)
        ax_l.set_xticklabels([_axis_ticklabel(axis, t) for t in ticks], fontsize=7, rotation=55, ha="right")
        ax_l.minorticks_off()
        ax_r.minorticks_off()
        ax_r.set_ylabel("Episode return (mean ± std over all episodes)")
        ax_l.set_ylabel("Episode length [steps] (mean ± std)")
        ax_l.set_xlabel(xlabels[axis])
        ax_r.legend(loc="center left", fontsize=8)
        fig.tight_layout()
        out = out_dir / f"{axis}_scale_zeroshot{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def stage_report() -> None:
    rows = collect_rows(seeded=True)
    path = write_results_csv(rows, EXP_DIR / "results.csv")
    graphs = plot(rows, GRAPHS_DIR)
    legacy = collect_rows(seeded=False)
    write_results_csv(legacy, EXP_DIR / "results_single_run.csv")
    plot(legacy, GRAPHS_DIR / "single_run")
    with open(EXP_DIR / "summary.json", "w") as f:
        json.dump({"n_points_seeded": len(rows), "n_points_single_run": len(legacy),
                   "results_csv": str(path.relative_to(EXP_DIR)),
                   "graphs": [str(g.relative_to(EXP_DIR)) for g in graphs]}, f, indent=2)
    print(f"[report] seeded {len(rows)} points, single-run {len(legacy)} points", flush=True)


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["report"], default=None)
    p.add_argument("--worker", choices=["vintix", "expert", "ppo", "probe", "train_ppo"], default=None)
    p.add_argument("--axis", choices=list(AXES), default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--ppo_num_envs", type=int, default=4096)
    p.add_argument("--ppo_iterations", type=int, default=PPO_ITERATIONS)
    args = p.parse_args(argv)

    if args.stage == "report":
        stage_report()
    elif args.worker == "probe":
        _worker_probe(args.axis, args.scale)
    elif args.worker == "train_ppo":
        _worker_train_ppo(args.axis, args.scale, args.ppo_num_envs, args.ppo_iterations,
                          PPO_TRAIN_SEED if args.seed is None else args.seed)
    elif args.worker is not None:
        _worker_eval(args.worker, args.axis, args.scale, args.seed, args.num_envs, args.max_episodes)
    else:
        p.error("give --stage report or --worker ... (the sweep itself is driven by pipeline.py)")


if __name__ == "__main__":
    main()
