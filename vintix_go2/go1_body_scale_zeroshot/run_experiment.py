#!/usr/bin/env python3
"""Go1 mass / size sweep: how far does Vintix zero-shot keep walking?

Orchestration only. Each (policy, axis, scale) point runs in its own subprocess
(fresh Genesis state) and calls the shared evaluators:

- ``vintix``: ``scripts/save_vintix._run_parallel_evaluation`` with ``go1_without/0001_epoch``
  (Go1 never seen in training; unknown-task normalization = average of the known tasks,
  identical to the ISCIIA zero-shot protocol).
- ``expert``: ``scripts/save_expert._run_parallel_evaluation`` with the nominal Go1 PPO expert
  (``model_2000.pt``) transferred as-is to the modified body.
- ``probe``: builds the variant, holds the default pose with zero actions and records the
  simulated total mass / standing height (sanity check that the body is what we intended).

Usage (inside the container)::

    cd /workspace/vintix_go2
    python go1_body_scale_zeroshot/run_experiment.py --stage probe
    python go1_body_scale_zeroshot/run_experiment.py --stage eval --policies expert,vintix
    python go1_body_scale_zeroshot/run_experiment.py --stage report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
    MASS_SCALES,
    REFERENCE_ROBOTS,
    SIZE_SCALES,
    physical_summary,
    scale_tag,
    scaled_go1_cfgs,
)

RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"
GRAPHS_DIR = EXP_DIR / "graphs"
POLICIES = ("vintix", "expert")
POLICY_LABELS = {"vintix": "Vintix zero-shot (go1_without)", "expert": "Go1 PPO expert (transferred)"}
POLICY_COLORS = {"vintix": "#1f77b4", "expert": "#7f7f7f"}
OUTPUT_STEM = "go1_10envs_10episodes"


# --------------------------------------------------------------------------- paths


def point_dir(policy: str, axis: str, scale: float, repeat: int) -> Path:
    return RESULTS_DIR / policy / axis / f"{scale_tag(scale)}_r{repeat}"


def episodes_csv(policy: str, axis: str, scale: float, repeat: int) -> Path:
    return point_dir(policy, axis, scale, repeat) / f"{OUTPUT_STEM}_episodes.csv"


def probe_json(axis: str, scale: float) -> Path:
    return RESULTS_DIR / "probe" / axis / f"{scale_tag(scale)}.json"


# --------------------------------------------------------------------------- workers


def _worker_eval(policy: str, axis: str, scale: float, repeat: int, num_envs: int, max_episodes: int) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = scaled_go1_cfgs(axis, scale)
    out_dir = point_dir(policy, axis, scale, repeat)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / OUTPUT_STEM)

    from scaled_go1_env import ScaledGo1Env

    if policy == "vintix":
        import save_vintix
        import genesis as gs

        # Shared script untouched: only this process builds Go1 through the scaled subclass.
        save_vintix.Go1Env = ScaledGo1Env
        gs.init(performance_mode=True)
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
    elif policy == "expert":
        import save_expert
        import genesis as gs

        save_expert.Go1Env = ScaledGo1Env
        gs.init(performance_mode=True)
        args = argparse.Namespace(
            robot_type="go1",
            exp_name="go1-walking",
            expert_model_path=str(GO1_EXPERT_CKPT),
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
            {"policy": policy, "axis": axis, "scale": scale, "repeat": repeat,
             "num_envs": num_envs, "max_episodes": max_episodes,
             "env_cfg_body": {k: env_cfg.get(k) for k in ("body_mass_scale", "body_size_scale", "base_init_pos")},
             "reward_heights": {k: reward_cfg[k] for k in ("base_height_target", "feet_height_target")},
             **physical_summary(axis, scale)},
            f, indent=2,
        )


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


# --------------------------------------------------------------------------- orchestration


def _run_subprocess(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    env["PYTHONPATH"] = f"{VINTIX_ROOT}:{env.get('PYTHONPATH', '')}"
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(VINTIX_ROOT), env=env)
    print(f"  -> rc={rc} ({(time.time() - t0) / 60:.1f} min) log={log_path.relative_to(EXP_DIR)}", flush=True)
    return rc


def _grid(args) -> Dict[str, Sequence[float]]:
    grids = {}
    for axis in args.axes:
        override = getattr(args, f"{axis}_scales")
        if override:
            grids[axis] = [float(x) for x in override.split(",")]
        else:
            grids[axis] = MASS_SCALES if axis == "mass" else SIZE_SCALES
    return grids


def _ordered(scales: Sequence[float]) -> List[float]:
    """Nominal first, then outward alternately, so a partial run still covers both sides."""
    return sorted(scales, key=lambda s: (abs(np.log(s)), s))


def stage_probe(args) -> None:
    for axis, scales in _grid(args).items():
        for scale in _ordered(scales):
            if probe_json(axis, scale).exists() and not args.force:
                continue
            print(f"[probe] {axis} x{scale}", flush=True)
            cmd = [sys.executable, "-u", __file__, "--worker", "probe", "--axis", axis, "--scale", str(scale)]
            _run_subprocess(cmd, LOGS_DIR / "probe" / f"{axis}_{scale_tag(scale)}.log")


def stage_eval(args) -> None:
    grids = _grid(args)
    for policy in args.policies:
        for axis, scales in grids.items():
            for scale in _ordered(scales):
                for repeat in range(args.repeats):
                    if episodes_csv(policy, axis, scale, repeat).exists() and not args.force:
                        continue
                    print(f"[eval] {policy} {axis} x{scale} r{repeat}", flush=True)
                    cmd = [sys.executable, "-u", __file__, "--worker", policy, "--axis", axis,
                           "--scale", str(scale), "--repeat", str(repeat),
                           "--num_envs", str(args.num_envs), "--max_episodes", str(args.max_episodes)]
                    log = LOGS_DIR / policy / f"{axis}_{scale_tag(scale)}_r{repeat}.log"
                    _run_subprocess(cmd, log)
                    if args.report_every_point:
                        stage_report(args)


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


def collect_rows() -> List[Dict[str, object]]:
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
                scale = float(json.load(open(meta))["scale"])
                by_scale.setdefault(scale, []).append(point)
            for scale, points in sorted(by_scale.items()):
                eps = [_read_episodes(p / f"{OUTPUT_STEM}_episodes.csv") for p in points]
                reward = np.concatenate([e["reward"] for e in eps])
                length = np.concatenate([e["length"] for e in eps])
                epn = np.concatenate([e["episode_number"] for e in eps])
                first = reward[epn == epn.min()]
                last = reward[epn == epn.max()]
                row = {
                    "policy": policy,
                    "axis": axis,
                    "scale": scale,
                    **physical_summary(axis, scale),
                    "n_repeats": len(points),
                    "n_episodes": int(reward.size),
                    "reward_mean": float(reward.mean()),
                    "reward_std": float(reward.std()),
                    "episode_length_mean": float(length.mean()),
                    "episode_length_std": float(length.std()),
                    "reward_first_episode_mean": float(first.mean()),
                    "reward_last_episode_mean": float(last.mean()),
                }
                metrics = [_read_loco_metrics(p) for p in points]
                for key in _METRIC_PATTERNS:
                    vals = [m[key] for m in metrics if key in m]
                    row[key] = float(np.mean(vals)) if vals else ""
                rows.append(row)
    return rows


def write_results_csv(rows: List[Dict[str, object]]) -> Path:
    path = EXP_DIR / "results.csv"
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
        return f"{scale:g}\n{GO1_TOTAL_MASS_KG * scale:.1f} kg"
    return f"{scale:g}\n{GO1_THIGH_LENGTH_M * scale * 100:.0f} cm"


def plot(rows: List[Dict[str, object]]) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    xlabels = {"mass": "Mass scale (total mass)", "size": "Size scale (thigh length), mass fixed"}
    written = []
    for axis in AXES:
        axis_rows = [r for r in rows if r["axis"] == axis]
        if not axis_rows:
            continue
        fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        for policy in POLICIES:
            pr = sorted((r for r in axis_rows if r["policy"] == policy), key=lambda r: r["scale"])
            if not pr:
                continue
            x = np.array([r["scale"] for r in pr])
            for ax, key in ((ax_r, "reward"), (ax_l, "episode_length")):
                m = np.array([r[f"{key}_mean"] for r in pr])
                s = np.array([r[f"{key}_std"] for r in pr])
                ax.plot(x, m, "o-", color=POLICY_COLORS[policy], label=POLICY_LABELS[policy], lw=2)
                ax.fill_between(x, m - s, m + s, color=POLICY_COLORS[policy], alpha=0.2)
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
        ax_l.set_xticklabels([_axis_ticklabel(axis, t) for t in ticks], fontsize=8)
        ax_l.minorticks_off()
        ax_r.minorticks_off()
        ax_r.set_ylabel("Episode return (mean ± std)")
        ax_l.set_ylabel("Episode length [steps] (mean ± std)")
        ax_l.set_xlabel(xlabels[axis])
        ax_r.legend(loc="lower center")
        fig.tight_layout()
        out = GRAPHS_DIR / f"{axis}_scale_zeroshot.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def stage_report(args=None) -> None:
    rows = collect_rows()
    path = write_results_csv(rows)
    graphs = plot(rows)
    with open(EXP_DIR / "summary.json", "w") as f:
        json.dump({"n_points": len(rows), "results_csv": str(path.relative_to(EXP_DIR)),
                   "graphs": [str(g.relative_to(EXP_DIR)) for g in graphs]}, f, indent=2)
    print(f"[report] {len(rows)} points -> {path.relative_to(EXP_DIR)}, graphs: {[g.name for g in graphs]}")


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["probe", "eval", "report", "all"], default="all")
    p.add_argument("--policies", type=str, default="expert,vintix")
    p.add_argument("--axes", type=str, default="mass,size")
    p.add_argument("--mass_scales", type=str, default=None, help="comma-separated override of the mass grid")
    p.add_argument("--size_scales", type=str, default=None, help="comma-separated override of the size grid")
    p.add_argument("--repeats", type=int, default=1, help="independent evaluations per point (r0, r1, ...)")
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--force", action="store_true", help="re-run points that already have results")
    p.add_argument("--report_every_point", action="store_true")
    # worker mode (internal)
    p.add_argument("--worker", choices=["vintix", "expert", "probe"], default=None)
    p.add_argument("--axis", choices=list(AXES), default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--repeat", type=int, default=0)
    args = p.parse_args(argv)

    if args.worker == "probe":
        _worker_probe(args.axis, args.scale)
        return
    if args.worker is not None:
        _worker_eval(args.worker, args.axis, args.scale, args.repeat, args.num_envs, args.max_episodes)
        return

    args.policies = [x for x in args.policies.split(",") if x]
    args.axes = [x for x in args.axes.split(",") if x]
    if args.stage in ("probe", "all"):
        stage_probe(args)
    if args.stage in ("eval", "all"):
        stage_eval(args)
    if args.stage in ("report", "all"):
        stage_report(args)


if __name__ == "__main__":
    main()
