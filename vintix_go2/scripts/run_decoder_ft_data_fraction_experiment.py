#!/usr/bin/env python3
"""
Decoder-only finetune with varying training data fraction (0%, 10%–100%),
then evaluation per checkpoint via ``save_vintix.py`` (10 envs × 10 episodes).

Outputs under ``vintix_go2/experience_decoder_ft_data_fraction_YYYYMMDD/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from plot_style import Y_LABEL, Y_LIM, finetune_legend_label  # noqa: E402

VINTIX_ROOT = SCRIPT_DIR.parent
GENESIS_ROOT = VINTIX_ROOT.parent / "Genesis"

NUM_EVAL_ENVS = 10
NUM_EVAL_EPISODES = 10

DATA_FRACTIONS_PCT = list(range(0, 101, 10))


def canon_robot_type(s: str) -> str:
    if not s:
        return s
    t = s.strip().lower()
    if t in ("a1", "unitreea1"):
        return "a1"
    return t


def mean_reward_all_cells(episode_returns_csv: Path) -> float:
    """Arithmetic mean of all reward cells in an episode-returns CSV."""
    vals: list[float] = []
    with open(episode_returns_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            for cell in row[2:]:
                cell = cell.strip()
                if cell:
                    vals.append(float(cell))
    if not vals:
        return float("nan")
    return float(np.mean(vals))

FONT_SIZE_LABEL = 22
FONT_SIZE_TICK = 18
FONT_SIZE_LEGEND = 14

TRAIN_HPARAMS = {
    "context_len": 2048,
    "trajectory_sparsity": 128,
    "lr": 0.0003,
    "betas": (0.9, 0.99),
    "weight_decay": 0.1,
    "precision": "bf16",
    "grad_accum_steps": 8,
    "warmup_ratio": 0.005,
    "batch_size": 8,
    "seed": 5,
    "epochs": 2,
    "save_every": 1,
    "data_dir": "data",
    "stats_path": "vintix/stats.json",
    "project": "Vintix_Go2",
    "group": "decoder_ft_data_fraction",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    base_ckpt: Path
    finetune_robot: str
    eval_robot: str
    dataset_config: str
    display_label: str
    plot_color: str


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="go1_without",
        base_ckpt=VINTIX_ROOT / "models/go1_without/go1_without/0001_epoch",
        finetune_robot="go1",
        eval_robot="go1",
        dataset_config="configs/go1_only_onegroup_config.yaml",
        display_label="Go1",
        plot_color="#1f77b4",
    ),
    ModelSpec(
        key="go2_without",
        base_ckpt=VINTIX_ROOT / "models/go2_without/go2_without/0001_epoch",
        finetune_robot="go2",
        eval_robot="go2",
        dataset_config="configs/go2_only_onegroup_config.yaml",
        display_label="Go2",
        plot_color="#ff7f0e",
    ),
    ModelSpec(
        key="a1_without",
        base_ckpt=VINTIX_ROOT / "models/a1_without/a1_without/0001_epoch",
        finetune_robot="a1",
        eval_robot="a1",
        dataset_config="configs/a1_finetune_config.yaml",
        display_label="A1",
        plot_color="#2ca02c",
    ),
    ModelSpec(
        key="minicheetah_without",
        base_ckpt=VINTIX_ROOT / "models/minicheetah_without/minicheetah_without/0001_epoch",
        finetune_robot="minicheetah",
        eval_robot="minicheetah",
        dataset_config="configs/minicheetah_only_onegroup_config.yaml",
        display_label="Minicheetah",
        plot_color="#d62728",
    ),
]


def default_experience_dir() -> Path:
    stamp = date.today().strftime("%Y%m%d")
    return VINTIX_ROOT / f"experience_decoder_ft_data_fraction_{stamp}"


def pct_tag(pct: int) -> str:
    return f"p{pct:02d}"


def ckpt_dir_for_fraction(exp_root: Path, spec: ModelSpec, pct: int) -> Path:
    if pct == 0:
        return spec.base_ckpt.resolve()
    return (exp_root / "checkpoints" / spec.key / pct_tag(pct)).resolve()


def latest_train_epoch_dir(train_parent: Path) -> Path:
    """After ``epochs=2``, the latest trained weights are in ``0001_epoch``."""
    candidates = sorted(train_parent.glob("*_epoch"), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No epoch dirs under {train_parent}")
    return candidates[-1].resolve()


def _episode_rewards_from_csv(episodes_csv: Path) -> list[float]:
    with open(episodes_csv, newline="", encoding="utf-8") as f:
        header = f.readline().strip()
    if "env_index" in header:
        data = np.loadtxt(episodes_csv, delimiter=",", skiprows=1)
        return [float(x) for x in data[:, 3]]
    data = np.loadtxt(episodes_csv, delimiter=",", skiprows=1)
    return [float(x) for x in data[:, 2]]


def extract_episode_reward_stats(
    eval_dir: Path,
    robot: str,
    num_envs: int = NUM_EVAL_ENVS,
    num_episodes: int = NUM_EVAL_EPISODES,
) -> tuple[float, float]:
    """Mean and std of per-episode cumulative reward (100 eval episodes)."""
    prefix = canon_robot_type(robot)
    mean_txt = eval_dir / f"{prefix}_{num_envs}envs_{num_episodes}episodes_mean_reward.txt"
    if mean_txt.is_file():
        mean_val = float("nan")
        std_val = float("nan")
        for line in mean_txt.read_text(encoding="utf-8").splitlines():
            if line.startswith("Mean Reward per Episode:"):
                mean_val = float(line.split(":", 1)[1].strip())
            elif line.startswith("Std Reward per Episode:"):
                std_val = float(line.split(":", 1)[1].strip())
        if np.isfinite(mean_val):
            return mean_val, std_val if np.isfinite(std_val) else 0.0

    episodes_csv = eval_dir / f"{prefix}_{num_envs}envs_{num_episodes}episodes_episodes.csv"
    if episodes_csv.is_file():
        vals = _episode_rewards_from_csv(episodes_csv)
        if vals:
            return float(np.mean(vals)), float(np.std(vals))

    csv_path = eval_dir / "episode_returns.csv"
    if csv_path.is_file():
        vals: list[float] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                for cell in row[2:]:
                    cell = cell.strip()
                    if cell:
                        vals.append(float(cell))
        if vals:
            return float(np.mean(vals)), float(np.std(vals))
    return float("nan"), float("nan")


def extract_mean_episode_reward(eval_dir: Path, robot: str, num_envs: int = NUM_EVAL_ENVS, num_episodes: int = NUM_EVAL_EPISODES) -> float:
    """Parse ``save_vintix``-style ``*_mean_reward.txt`` or fall back to ``episode_returns.csv``."""
    mean_val, _ = extract_episode_reward_stats(eval_dir, robot, num_envs, num_episodes)
    return mean_val


def _subprocess_env() -> dict:
    env = {**os.environ, "WANDB_MODE": "disabled"}
    py_path = str(VINTIX_ROOT)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{py_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = py_path
    return env


def run_train(
    spec: ModelSpec,
    pct: int,
    ckpt_parent: Path,
    dry_run: bool,
) -> Path:
    ckpt_parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "train_vintix.py"),
        "--data_dir",
        TRAIN_HPARAMS["data_dir"],
        "--dataset_config_paths",
        json.dumps([spec.dataset_config]),
        "--context_len",
        str(TRAIN_HPARAMS["context_len"]),
        "--trajectory_sparsity",
        str(TRAIN_HPARAMS["trajectory_sparsity"]),
        "--lr",
        str(TRAIN_HPARAMS["lr"]),
        "--weight_decay",
        str(TRAIN_HPARAMS["weight_decay"]),
        "--precision",
        TRAIN_HPARAMS["precision"],
        "--grad_accum_steps",
        str(TRAIN_HPARAMS["grad_accum_steps"]),
        "--warmup_ratio",
        str(TRAIN_HPARAMS["warmup_ratio"]),
        "--batch_size",
        str(TRAIN_HPARAMS["batch_size"]),
        "--seed",
        str(TRAIN_HPARAMS["seed"]),
        "--epochs",
        str(TRAIN_HPARAMS["epochs"]),
        "--save_every",
        str(TRAIN_HPARAMS["save_every"]),
        "--stats_path",
        TRAIN_HPARAMS["stats_path"],
        "--project",
        TRAIN_HPARAMS["project"],
        "--group",
        TRAIN_HPARAMS["group"],
        "--name",
        f"{spec.key}_{pct_tag(pct)}",
        "--load_ckpt",
        str(spec.base_ckpt.resolve()),
        "--finetune_decoder_only",
        "true",
        "--finetune_robot",
        spec.finetune_robot,
        "--finetune_output_subdir",
        str(ckpt_parent),
        "--random_sample_frac",
        str(pct / 100.0),
    ]
    env = _subprocess_env()
    print("\n[train]", " ".join(cmd), flush=True)
    if dry_run:
        return latest_train_epoch_dir(ckpt_parent) if list(ckpt_parent.glob("*_epoch")) else ckpt_parent / "0001_epoch"
    subprocess.run(cmd, check=True, cwd=str(VINTIX_ROOT), env=env)
    return latest_train_epoch_dir(ckpt_parent)


def _save_vintix_result_dir(ckpt: Path, eval_robot: str) -> Path:
    """``save_vintix.py`` writes under ``<ckpt>/../Result/<robot>/``."""
    return ckpt.resolve().parent / "Result" / canon_robot_type(eval_robot)


def run_eval(ckpt: Path, eval_robot: str, dry_run: bool) -> Path:
    """Run evaluation via ``save_vintix.py`` (10 envs × 10 episodes, max 1000 steps/ep)."""
    rt = canon_robot_type(eval_robot)
    result_dir = _save_vintix_result_dir(ckpt, eval_robot)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "save_vintix.py"),
        "--vintix_path",
        str(ckpt.resolve()),
        "-r",
        rt,
        "--num_envs",
        str(NUM_EVAL_ENVS),
        "--max_episodes",
        str(NUM_EVAL_EPISODES),
    ]

    print("\n[eval save_vintix]", " ".join(cmd), flush=True)
    env = _subprocess_env()
    if dry_run:
        return result_dir

    subprocess.run(cmd, check=True, cwd=str(VINTIX_ROOT), env=env)

    mean_txt = result_dir / f"{rt}_{NUM_EVAL_ENVS}envs_{NUM_EVAL_EPISODES}episodes_mean_reward.txt"
    if not mean_txt.is_file():
        raise FileNotFoundError(f"save_vintix output missing: {mean_txt}")
    return result_dir


def load_results_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_result_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    fields = list(row.keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def already_done(results: list[dict], model_key: str, pct: int) -> bool:
    tag = pct_tag(pct)
    for r in results:
        if r.get("model_key") == model_key and r.get("data_fraction_pct") == str(pct):
            if r.get("status") == "ok" and r.get("mean_cumulative_reward"):
                try:
                    float(r["mean_cumulative_reward"])
                    return True
                except ValueError:
                    pass
    return False


def plot_graphs(exp_root: Path, rows: list[dict]) -> None:
    graphs = exp_root / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)

    spec_by_key = {s.key: s for s in MODEL_SPECS}
    by_model: dict[str, list[tuple[int, float, float]]] = {s.key: [] for s in MODEL_SPECS}
    for r in rows:
        if r.get("status") != "ok":
            continue
        mk = r["model_key"]
        spec = spec_by_key.get(mk)
        if spec is None:
            continue
        try:
            pct = int(r["data_fraction_pct"])
        except (KeyError, ValueError):
            continue
        eval_dir = Path(r.get("eval_result_dir", ""))
        if eval_dir.is_dir():
            mean_val, std_val = extract_episode_reward_stats(eval_dir, spec.eval_robot)
        else:
            try:
                mean_val = float(r["mean_cumulative_reward"])
                std_val = float("nan")
            except (KeyError, ValueError):
                continue
        if np.isfinite(mean_val):
            if not np.isfinite(std_val):
                std_val = 0.0
            by_model[mk].append((pct, mean_val, std_val))

    def _plot_one(
        path: Path,
        series: Iterable[tuple[ModelSpec, list[tuple[int, float, float]]]],
        *,
        show_legend: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 10))
        for spec, pts in series:
            if not pts:
                continue
            pts = sorted(pts, key=lambda x: x[0])
            xs = np.array([p for p, _, _ in pts], dtype=float)
            means = np.array([m for _, m, _ in pts], dtype=float)
            stds = np.array([s for _, _, s in pts], dtype=float)
            label = finetune_legend_label(spec.eval_robot) if show_legend else "_nolegend_"
            ax.plot(
                xs,
                means,
                "o-",
                color=spec.plot_color,
                label=label,
                linewidth=2.6,
                markersize=8,
            )
            lower = np.maximum(means - stds, Y_LIM[0])
            upper = np.minimum(means + stds, Y_LIM[1])
            ax.fill_between(xs, lower, upper, alpha=0.15, color=spec.plot_color)
        ax.set_xlabel("Training data used (%)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(Y_LABEL, fontsize=FONT_SIZE_LABEL)
        ax.set_xticks(DATA_FRACTIONS_PCT)
        ax.set_ylim(*Y_LIM)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3)
        if show_legend:
            handles, _labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND, framealpha=0.95)
        plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.12)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    all_series = [(spec_by_key[k], by_model[k]) for k in by_model if by_model[k]]
    _plot_one(graphs / "all_models_data_fraction.png", all_series, show_legend=True)
    _plot_one(graphs / "all_models_data_fraction.pdf", all_series, show_legend=True)
    _plot_one(
        graphs / "all_models_data_fraction_poster.png",
        all_series,
        show_legend=False,
    )
    _plot_one(
        graphs / "all_models_data_fraction_poster.pdf",
        all_series,
        show_legend=False,
    )

    for spec in MODEL_SPECS:
        pts = by_model.get(spec.key, [])
        if not pts:
            continue
        one = [(spec, pts)]
        _plot_one(graphs / f"{spec.key}_data_fraction.png", one, show_legend=True)
        _plot_one(graphs / f"{spec.key}_data_fraction.pdf", one, show_legend=True)
        _plot_one(
            graphs / f"{spec.key}_data_fraction_poster.png",
            one,
            show_legend=False,
        )
        _plot_one(
            graphs / f"{spec.key}_data_fraction_poster.pdf",
            one,
            show_legend=False,
        )


def run_experiment(
    exp_root: Path,
    models: list[str],
    fractions: list[int],
    dry_run: bool,
    skip_train: bool,
    skip_eval: bool,
) -> None:
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "checkpoints").mkdir(exist_ok=True)
    (exp_root / "graphs").mkdir(exist_ok=True)

    manifest = {
        "train_hparams": TRAIN_HPARAMS,
        "data_fractions_pct": fractions,
        "eval": {
            "backend": "save_vintix.py",
            "num_envs": NUM_EVAL_ENVS,
            "num_episodes": NUM_EVAL_EPISODES,
            "max_steps_per_episode": 1000,
        },
        "models": [asdict(s) for s in MODEL_SPECS if s.key in models],
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    results_path = exp_root / "results.csv"
    existing = load_results_csv(results_path)
    specs = [s for s in MODEL_SPECS if s.key in models]

    for spec in specs:
        if not spec.base_ckpt.joinpath("model.pth").is_file():
            raise FileNotFoundError(f"Base checkpoint missing: {spec.base_ckpt}")

        for pct in fractions:
            if already_done(existing, spec.key, pct):
                print(f"[skip] done: {spec.key} {pct}%", flush=True)
                continue

            row: dict = {
                "model_key": spec.key,
                "data_fraction_pct": str(pct),
                "status": "pending",
            }
            try:
                if pct == 0:
                    ckpt = ckpt_dir_for_fraction(exp_root, spec, pct)
                else:
                    train_parent = ckpt_dir_for_fraction(exp_root, spec, pct)
                    if skip_train and latest_train_epoch_dir(train_parent).joinpath("model.pth").is_file():
                        ckpt = latest_train_epoch_dir(train_parent)
                    elif not skip_train:
                        ckpt = run_train(spec, pct, train_parent, dry_run=dry_run)
                    else:
                        raise FileNotFoundError(f"Training skipped but no checkpoint: {train_parent}")

                row["checkpoint_dir"] = str(ckpt)

                eval_result_dir = exp_root / "eval" / spec.key / pct_tag(pct)
                rt = canon_robot_type(spec.eval_robot)
                done_marker = eval_result_dir / f"{rt}_10envs_10episodes_mean_reward.txt"
                if skip_eval and done_marker.is_file():
                    result_subdir = eval_result_dir
                elif not skip_eval:
                    result_subdir = run_eval(ckpt, spec.eval_robot, dry_run=dry_run)
                    eval_result_dir.parent.mkdir(parents=True, exist_ok=True)
                    eval_result_dir.mkdir(parents=True, exist_ok=True)
                    if not dry_run:
                        for sp in result_subdir.iterdir():
                            if sp.is_file():
                                eval_result_dir.joinpath(sp.name).write_bytes(sp.read_bytes())
                    result_subdir = eval_result_dir
                else:
                    raise FileNotFoundError(f"Eval skipped but missing {done_marker}")

                mean_reward = extract_mean_episode_reward(result_subdir, spec.eval_robot)
                ep_csv = result_subdir / f"{rt}_10envs_10episodes_episodes.csv"
                alt_csv = result_subdir / "episode_returns.csv"
                if alt_csv.is_file():
                    alt = mean_reward_all_cells(alt_csv)
                elif ep_csv.is_file():
                    import numpy as np

                    d = np.loadtxt(ep_csv, delimiter=",", skiprows=1)
                    alt = float(d[:, 3].mean()) if d.size else mean_reward
                else:
                    alt = mean_reward
                row["mean_cumulative_reward"] = f"{mean_reward:.8f}"
                row["mean_cumulative_reward_all_cells"] = f"{alt:.8f}"
                row["eval_result_dir"] = str(result_subdir)
                row["status"] = "ok"
                print(f"[ok] {spec.key} {pct}% reward={mean_reward:.4f}", flush=True)
            except Exception as e:
                row["status"] = "error"
                row["error"] = str(e)
                print(f"[error] {spec.key} {pct}%: {e}", flush=True)

            append_result_row(results_path, row)
            existing.append(row)

    rows = load_results_csv(results_path)
    plot_graphs(exp_root, rows)
    summary = {
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_total": len(rows),
        "results_csv": str(results_path),
    }
    (exp_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp_root",
        type=Path,
        default=None,
        help="Experiment output root (default: experience_decoder_ft_data_fraction_<today>)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(s.key for s in MODEL_SPECS),
        help="Comma-separated model keys",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default=",".join(str(p) for p in DATA_FRACTIONS_PCT),
        help="Comma-separated data %% values (e.g. 0,10,...,100)",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    exp_root = (args.exp_root or default_experience_dir()).resolve()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    fractions = [int(x.strip()) for x in args.fractions.split(",") if x.strip()]

    run_experiment(
        exp_root=exp_root,
        models=models,
        fractions=fractions,
        dry_run=args.dry_run,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()
