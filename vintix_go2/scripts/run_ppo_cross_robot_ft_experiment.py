#!/usr/bin/env python3
"""
Cross-robot PPO fine-tuning experiment (4 sources × 3 targets × 10 seeds).

- Training: ``Genesis/examples/locomotion/train.py`` (no trajectory history)
- Init checkpoint per source: ``Genesis/logs/<robot>-walking/model_300.pt``
- Reward curve: TensorBoard ``Train/mean_reward`` (episode mean in training loop)
- Outputs: ``vintix_go2/ppo_cross_robot_ft_YYYYMMDD/`` (runs, curves, graphs, manifest)

Example::

    cd vintix_go2 && python scripts/run_ppo_cross_robot_ft_experiment.py
    python scripts/run_ppo_cross_robot_ft_experiment.py --plot_only
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = SCRIPT_DIR.parent
GENESIS_ROOT = VINTIX_ROOT.parent / "Genesis"
LOCOMOTION_ROOT = GENESIS_ROOT / "examples" / "locomotion"
TRAIN_SCRIPT = LOCOMOTION_ROOT / "train.py"

ROBOTS = ("go1", "go2", "a1", "minicheetah")
SOURCE_EXPERT_LOGDIRS = {
    "go1": "go1-walking",
    "go2": "go2-walking",
    "a1": "a1-walking",
    "minicheetah": "minicheetah-walking",
}
EXPERT_CKPT_NAME = "model_300.pt"
SEEDS = tuple(range(1, 11))
MAX_ITERATION = 300
MAX_ITERATIONS_TRAIN = 301
NUM_ENVS = 4096
NUM_STEPS_PER_ENV = 24
TB_REWARD_TAG = "Train/mean_reward"

# Readable style (generate_readable_finetune_comparison / save_vintix)
FONT_SIZE_LABEL = 34
FONT_SIZE_TICK = 28
FONT_SIZE_LEGEND = 18
Y_LIM = (-5.0, 28.0)

# Heuristic: 4096-env Genesis PPO typically needs ~10+ GiB free VRAM
MIN_GPU_FREE_MIB = 10_000
MAX_GPU_UTIL_PCT = 25

ROBOT_COLORS = {
    "go1": "#1f77b4",
    "go2": "#ff7f0e",
    "a1": "#2ca02c",
    "minicheetah": "#d62728",
}
ROBOT_LABELS = {
    "go1": "Go1",
    "go2": "Go2",
    "a1": "A1",
    "minicheetah": "MiniCheetah",
}


@dataclass(frozen=True)
class RunSpec:
    source_robot: str
    target_robot: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"ft_{self.source_robot}_to_{self.target_robot}_seed{self.seed:02d}"

    def log_dir(self, exp_root: Path) -> Path:
        return exp_root / "runs" / self.run_id

    def curve_csv(self, exp_root: Path) -> Path:
        return exp_root / "curves" / f"{self.run_id}.csv"


def default_experiment_dir() -> Path:
    stamp = date.today().strftime("%Y%m%d")
    return VINTIX_ROOT / f"ppo_cross_robot_ft_{stamp}"


def canon_robot_type(s: str) -> str:
    t = s.strip().lower()
    if t in ("a1", "unitreea1"):
        return "a1"
    return t


def expert_checkpoint(source_robot: str) -> Path:
    log_name = SOURCE_EXPERT_LOGDIRS[source_robot]
    return GENESIS_ROOT / "logs" / log_name / EXPERT_CKPT_NAME


def iter_cross_robot_pairs() -> Iterable[tuple[str, str]]:
    for src in ROBOTS:
        for tgt in ROBOTS:
            if src != tgt:
                yield src, tgt


def iter_all_runs() -> Iterable[RunSpec]:
    for src, tgt in iter_cross_robot_pairs():
        for seed in SEEDS:
            yield RunSpec(source_robot=src, target_robot=tgt, seed=seed)


def _subprocess_env() -> dict:
    env = {**os.environ}
    extra = f"{LOCOMOTION_ROOT}:{VINTIX_ROOT}"
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{extra}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = extra
    return env


def _ensure_genesis_importable() -> None:
    """Fail fast with a clear message if PPO training cannot import Genesis."""
    try:
        import genesis  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Genesis is not available in this Python. Run inside the Docker container, e.g.\n"
            "  docker compose exec genesis bash\n"
            "  cd /workspace/vintix_go2 && python scripts/run_ppo_cross_robot_ft_experiment.py\n"
            f"Original error: {exc}"
        ) from exc


def query_gpu_stats() -> Optional[dict[str, float]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if not out:
            return None
        total, used, free, util = [float(x.strip()) for x in out.split("\n")[0].split(",")]
        return {"total_mib": total, "used_mib": used, "free_mib": free, "util_pct": util}
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def query_ram_available_gib() -> Optional[float]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            lines = {k: int(v.split()[0]) for k, v in (ln.split(":", 1) for ln in f if ":" in ln)}
        avail_kib = lines.get("MemAvailable", lines.get("MemFree", 0))
        return avail_kib / (1024 * 1024)
    except OSError:
        return None


def resources_sufficient_for_training() -> tuple[bool, str]:
    gpu = query_gpu_stats()
    ram_gib = query_ram_available_gib()
    reasons: list[str] = []

    if gpu is None:
        reasons.append("GPU not detected (nvidia-smi unavailable); training requires CUDA.")
    else:
        if gpu["free_mib"] < MIN_GPU_FREE_MIB:
            reasons.append(
                f"GPU free memory {gpu['free_mib']:.0f} MiB < required {MIN_GPU_FREE_MIB} MiB "
                f"(used {gpu['used_mib']:.0f}/{gpu['total_mib']:.0f} MiB)."
            )
        if gpu["util_pct"] > MAX_GPU_UTIL_PCT:
            reasons.append(
                f"GPU utilization {gpu['util_pct']:.0f}% > allowed {MAX_GPU_UTIL_PCT}% "
                "(another job may be using the GPU)."
            )

    if ram_gib is not None and ram_gib < 8.0:
        reasons.append(f"System RAM available {ram_gib:.1f} GiB < 8 GiB.")

    if reasons:
        return False, " ".join(reasons)
    detail = []
    if gpu:
        detail.append(f"GPU free {gpu['free_mib']:.0f} MiB, util {gpu['util_pct']:.0f}%")
    if ram_gib is not None:
        detail.append(f"RAM available {ram_gib:.1f} GiB")
    return True, "; ".join(detail) if detail else "OK"


def read_train_mean_reward(log_dir: Path, max_iteration: int = MAX_ITERATION) -> tuple[np.ndarray, np.ndarray]:
    """Read ``Train/mean_reward`` vs iteration from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except ImportError as exc:
        raise ImportError("Install tensorboard to read training curves: pip install tensorboard") from exc

    event_files = glob.glob(str(log_dir / "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard events in {log_dir}")

    acc = ea.EventAccumulator(sorted(event_files)[-1])
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if TB_REWARD_TAG not in tags:
        raise KeyError(f"{TB_REWARD_TAG!r} not in {tags}")

    iters: list[int] = []
    rewards: list[float] = []
    for ev in acc.Scalars(TB_REWARD_TAG):
        it = int(ev.step)
        if it > max_iteration:
            continue
        iters.append(it)
        rewards.append(float(ev.value))

    if not iters:
        raise ValueError(f"No scalar points for {TB_REWARD_TAG} in {log_dir}")

    order = np.argsort(iters)
    it_arr = np.asarray(iters, dtype=np.int64)[order]
    rew_arr = np.asarray(rewards, dtype=np.float64)[order]
    return it_arr, rew_arr


def save_curve_csv(path: Path, iterations: np.ndarray, rewards: np.ndarray, num_envs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = iterations.astype(np.int64) * num_envs * NUM_STEPS_PER_ENV
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "env_steps", "train_mean_reward"])
        for it, st, rw in zip(iterations, steps, rewards):
            w.writerow([int(it), int(st), f"{float(rw):.8f}"])


def load_curve_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float64)


def run_train_job(spec: RunSpec, exp_root: Path, dry_run: bool) -> Path:
    log_dir = spec.log_dir(exp_root)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pretrained = expert_checkpoint(spec.source_robot)
    if not pretrained.is_file():
        raise FileNotFoundError(f"Missing expert checkpoint: {pretrained}")

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "-e",
        spec.run_id,
        "-r",
        spec.target_robot,
        "-B",
        str(NUM_ENVS),
        "--max_iterations",
        str(MAX_ITERATIONS_TRAIN),
        "--pretrained_path",
        str(pretrained),
        "--seed",
        str(spec.seed),
        "--log_dir",
        str(log_dir),
    ]
    print("\n[train]", " ".join(cmd), flush=True)
    if dry_run:
        return log_dir
    subprocess.run(cmd, check=True, cwd=str(LOCOMOTION_ROOT), env=_subprocess_env())
    return log_dir


def extract_and_save_curve(spec: RunSpec, exp_root: Path) -> Path:
    log_dir = spec.log_dir(exp_root)
    iters, rewards = read_train_mean_reward(log_dir, max_iteration=MAX_ITERATION)
    out = spec.curve_csv(exp_root)
    save_curve_csv(out, iters, rewards, NUM_ENVS)
    return out


def load_results_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_result_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    fields = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_already_ok(results: list[dict], spec: RunSpec) -> bool:
    for r in results:
        if r.get("run_id") == spec.run_id and r.get("status") == "ok":
            curve = r.get("curve_csv")
            if curve and Path(curve).is_file():
                return True
    return False


def aggregate_curves(
    exp_root: Path,
    source_robot: str,
    target_robot: str,
    x_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common x, mean reward, std reward across seeds."""
    xs_list: list[np.ndarray] = []
    ys_list: list[np.ndarray] = []

    for seed in SEEDS:
        spec = RunSpec(source_robot=source_robot, target_robot=target_robot, seed=seed)
        csv_path = spec.curve_csv(exp_root)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing curve CSV: {csv_path}")
        iters, steps, rewards = load_curve_csv(csv_path)
        if x_key == "iteration":
            xs = iters
        elif x_key == "env_steps":
            xs = steps
        else:
            raise ValueError(x_key)
        xs_list.append(xs)
        ys_list.append(rewards)

    max_len = min(len(x) for x in xs_list)
    common_x = xs_list[0][:max_len]
    for xs in xs_list[1:]:
        if not np.array_equal(xs[:max_len], common_x):
            # Align by interpolation on iteration index if needed
            common_x = np.arange(max_len, dtype=np.int64)
            break

    stacked = np.stack([y[:max_len] for y in ys_list], axis=0)
    return common_x, np.mean(stacked, axis=0), np.std(stacked, axis=0)


def _save_readable_multiline(
    *,
    out_path: Path,
    series: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]],
    xlabel: str,
    xlim: tuple[float, float],
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    for x, mean, std, label, color in series:
        ax.plot(x, mean, linewidth=2, label=label, color=color)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.25,
            color=color,
        )

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean Episode Reward", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(*Y_LIM)
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)
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


def plot_graphs(exp_root: Path) -> list[Path]:
    graphs_dir = exp_root / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    max_steps = MAX_ITERATION * NUM_ENVS * NUM_STEPS_PER_ENV

    for src in ROBOTS:
        targets = [t for t in ROBOTS if t != src]
        iter_series: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]] = []
        step_series: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]] = []

        for tgt in targets:
            x_it, mean_it, std_it = aggregate_curves(exp_root, src, tgt, "iteration")
            x_st, mean_st, std_st = aggregate_curves(exp_root, src, tgt, "env_steps")
            label = f"FT → {ROBOT_LABELS[tgt]}"
            color = ROBOT_COLORS[tgt]
            iter_series.append((x_it, mean_it, std_it, label, color))
            step_series.append((x_st, mean_st, std_st, label, color))

        p_it = _save_readable_multiline(
            out_path=graphs_dir / f"source_{src}_mean_reward_vs_iteration.png",
            series=iter_series,
            xlabel="PPO Iteration",
            xlim=(0, MAX_ITERATION),
        )
        p_st = _save_readable_multiline(
            out_path=graphs_dir / f"source_{src}_mean_reward_vs_env_steps.png",
            series=step_series,
            xlabel="Environment Steps",
            xlim=(0, max_steps),
        )
        saved.extend([p_it, p_st])

    return saved


def write_manifest(exp_root: Path) -> None:
    manifest = {
        "description": "Cross-robot PPO fine-tuning from model_300.pt experts (10 seeds per pair)",
        "robots": list(ROBOTS),
        "seeds": list(SEEDS),
        "max_iteration": MAX_ITERATION,
        "num_envs": NUM_ENVS,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "domain_randomization": False,
        "expert_checkpoints": {r: str(expert_checkpoint(r)) for r in ROBOTS},
        "reward_tag": TB_REWARD_TAG,
        "train_script": str(TRAIN_SCRIPT),
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run_experiment(
    exp_root: Path,
    *,
    dry_run: bool,
    skip_train: bool,
    plot_only: bool,
    force_run: bool,
) -> dict:
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "runs").mkdir(exist_ok=True)
    (exp_root / "curves").mkdir(exist_ok=True)
    (exp_root / "graphs").mkdir(exist_ok=True)
    write_manifest(exp_root)

    results_path = exp_root / "results.csv"
    existing = load_results_csv(results_path)

    if plot_only:
        paths = plot_graphs(exp_root)
        return {"status": "plot_only", "graphs": [str(p) for p in paths]}

    if not skip_train:
        ok, msg = resources_sufficient_for_training()
        if not ok and not force_run:
            report = {
                "status": "skipped_insufficient_resources",
                "reason": msg,
                "gpu": query_gpu_stats(),
                "ram_available_gib": query_ram_available_gib(),
            }
            (exp_root / "resource_check.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            print(f"[skip training] {msg}", flush=True)
            return report
        if ok:
            print(f"[resource check OK] {msg}", flush=True)
        else:
            print(f"[force run despite resources] {msg}", flush=True)

    for spec in iter_all_runs():
        if run_already_ok(existing, spec):
            print(f"[skip] done: {spec.run_id}", flush=True)
            continue

        row: dict = {
            "run_id": spec.run_id,
            "source_robot": spec.source_robot,
            "target_robot": spec.target_robot,
            "seed": str(spec.seed),
            "status": "pending",
        }
        try:
            if not skip_train:
                log_dir = run_train_job(spec, exp_root, dry_run=dry_run)
                row["log_dir"] = str(log_dir)
            curve_path = extract_and_save_curve(spec, exp_root)
            row["curve_csv"] = str(curve_path)
            row["status"] = "ok"
            print(f"[ok] {spec.run_id}", flush=True)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"[error] {spec.run_id}: {exc}", flush=True)

        append_result_row(results_path, row)
        existing.append(row)

    graphs = plot_graphs(exp_root)
    summary = {
        "status": "completed",
        "n_ok": sum(1 for r in load_results_csv(results_path) if r.get("status") == "ok"),
        "n_total": len(list(iter_all_runs())),
        "results_csv": str(results_path),
        "graphs": [str(p) for p in graphs],
    }
    (exp_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-robot PPO FT experiment (train + plot).")
    parser.add_argument(
        "--exp_root",
        type=Path,
        default=None,
        help="Experiment output directory (default: vintix_go2/ppo_cross_robot_ft_YYYYMMDD)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument("--skip_train", action="store_true", help="Only extract curves / plot from existing runs.")
    parser.add_argument("--plot_only", action="store_true", help="Regenerate graphs from curve CSVs.")
    parser.add_argument(
        "--force_run",
        action="store_true",
        help="Run training even if GPU/RAM check fails (not recommended).",
    )
    args = parser.parse_args()
    if not args.plot_only:
        _ensure_genesis_importable()
    exp_root = args.exp_root.resolve() if args.exp_root else default_experiment_dir().resolve()

    summary = run_experiment(
        exp_root,
        dry_run=args.dry_run,
        skip_train=args.skip_train,
        plot_only=args.plot_only,
        force_run=args.force_run,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
