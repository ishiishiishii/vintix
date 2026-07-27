#!/usr/bin/env python3
"""
Online decoder PPO FT のオーケストレーション。

  models/<model_key>/   … model_N.pt + model_N/ (Vintix), run_config.json
  result/<model_key>/   … 学習曲線, 評価, 比較表・グラフ
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
SCRIPTS = VINTIX_ROOT / "scripts"
GENESIS_LOCO = VINTIX_ROOT.parent / "Genesis" / "examples" / "locomotion"
MODELS_DIR = EXP_DIR / "models"
RESULT_DIR = EXP_DIR / "result"

sys.path.insert(0, str(SCRIPTS))
from vintix_actor_critic import load_vintix_ckpt_defaults  # noqa: E402

MODEL_SPECS = {
    "go1_without": {
        "robot": "go1",
        "ckpt": "models/go1_without/go1_without/0001_epoch",
    },
    "go2_without": {
        "robot": "go2",
        "ckpt": "models/go2_without/go2_without/0001_epoch",
    },
    "a1_without": {
        "robot": "a1",
        "ckpt": "models/a1_without/a1_without/0001_epoch",
    },
    "minicheetah_without": {
        "robot": "minicheetah",
        "ckpt": "models/minicheetah_without/minicheetah_without/0001_epoch",
    },
}


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    env = dict(**subprocess.os.environ)
    py_path = [
        str(VINTIX_ROOT),
        str(SCRIPTS),
        str(GENESIS_LOCO),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(p for p in py_path if p)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def model_dir(model_key: str) -> Path:
    return MODELS_DIR / model_key


def result_dir(model_key: str) -> Path:
    return RESULT_DIR / model_key


def final_vintix_dir(model_key: str, max_iterations: int) -> Path:
    return model_dir(model_key) / f"model_{max_iterations}"


def write_run_config(model_key: str, config: dict) -> None:
    d = model_dir(model_key)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def train_one(
    model_key: str,
    num_envs: int,
    max_iterations: int,
    seed: int,
    context_len: int | None,
    trajectory_sparsity: int | None,
    save_interval: int,
    skip_train: bool,
    init_noise_std: float = 0.2,
    saturate: bool = False,
    seed_retries: int = 0,
) -> Path:
    spec = MODEL_SPECS[model_key]
    robot = spec["robot"]
    ckpt = VINTIX_ROOT / spec["ckpt"]
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing Vintix ckpt: {ckpt}")

    ckpt_defaults = load_vintix_ckpt_defaults(str(ckpt))
    resolved_context_len = (
        context_len if context_len is not None else ckpt_defaults["context_len"]
    )
    resolved_traj_sparsity = (
        trajectory_sparsity
        if trajectory_sparsity is not None
        else ckpt_defaults["trajectory_sparsity"]
    )

    mdir = model_dir(model_key)
    rdir = result_dir(model_key)
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    # skip_train: prefer exact max_iterations export; if saturation stopped early, use latest
    final_export = final_vintix_dir(model_key, max_iterations)
    if skip_train:
        if (final_export / "model.pth").is_file():
            print(f"[skip_train] Using existing: {final_export}")
            return final_export
        candidates = sorted(mdir.glob("model_*/model.pth"), key=lambda p: p.parent.name)
        if candidates:
            print(f"[skip_train] Using latest: {candidates[-1].parent}")
            return candidates[-1].parent
        raise FileNotFoundError(f"No Vintix export under {mdir}")

    cmd = [
        sys.executable,
        str(SCRIPTS / "train_online_decoder_ppo.py"),
        "-e",
        f"online_ft_{model_key}",
        "-r",
        robot,
        "-B",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--seed",
        str(seed),
        "--vintix_path",
        str(ckpt),
        "--log_dir",
        str(mdir),
        "--result_dir",
        str(rdir),
        "--context_len",
        str(resolved_context_len),
        "--trajectory_sparsity",
        str(resolved_traj_sparsity),
        "--save_interval",
        str(save_interval),
        "--init_noise_std",
        str(init_noise_std),
        "--seed_retries",
        str(seed_retries),
    ]
    if saturate:
        cmd.append("--saturate")
    _run(cmd, cwd=VINTIX_ROOT)
    # 飽和早期終了時は model_{max_iterations} が無いので最新を返す
    if (final_export / "model.pth").is_file():
        return final_export
    candidates = sorted(mdir.glob("model_*/model.pth"), key=lambda p: p.parent.name)
    if not candidates:
        raise FileNotFoundError(f"No Vintix export under {mdir}")
    return candidates[-1].parent


def eval_one(
    model_key: str,
    vintix_dir: Path,
    num_envs: int,
    max_episodes: int,
    output_subdir: str | None = None,
) -> Path:
    spec = MODEL_SPECS[model_key]
    robot = spec["robot"]
    out_dir = result_dir(model_key).resolve()
    if output_subdir:
        out_dir = out_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "save_vintix.py"),
        "-r",
        robot,
        "--vintix_path",
        str(vintix_dir.resolve()),
        "--num_envs",
        str(num_envs),
        "--max_episodes",
        str(max_episodes),
        "--output_dir",
        str(out_dir),
    ]
    _run(cmd, cwd=VINTIX_ROOT)
    return out_dir


def compare_all(model_keys: list[str]) -> None:
    cmd = [
        sys.executable,
        str(EXP_DIR / "compare_results.py"),
        "--models",
        ",".join(model_keys),
    ]
    _run(cmd, cwd=VINTIX_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Online decoder PPO FT experiment runner")
    parser.add_argument(
        "--models",
        type=str,
        default="go2_without",
        help="Comma-separated: go1_without,go2_without,a1_without,minicheetah_without",
    )
    parser.add_argument("--num_envs", type=int, default=4096, help="PPO parallel envs")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Hard cap on PPO iterations (1-based). Use -1 with --saturate "
        "to stop only on reward saturation.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="SAR history length (default: from each ckpt config.json, typically 2048)",
    )
    parser.add_argument(
        "--trajectory_sparsity",
        type=int,
        default=None,
        help="Must match pretraining (default: from ckpt config.json, typically 128)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save checkpoint every N PPO iterations (1-based), plus final",
    )
    parser.add_argument(
        "--init_noise_std",
        type=float,
        default=0.2,
        help="Initial action noise std (default 0.2; smaller than train.py 1.0)",
    )
    parser.add_argument(
        "--saturate",
        action="store_true",
        help="Stop early when mean_reward saturates (built-in thresholds; see EXPERIMENT.md)",
    )
    parser.add_argument(
        "--seed_retries",
        type=int,
        default=0,
        help="On death-spiral, retry with seed+1 up to N times (recommended 3)",
    )
    parser.add_argument("--eval_num_envs", type=int, default=10)
    parser.add_argument("--eval_max_episodes", type=int, default=10)
    parser.add_argument(
        "--eval_checkpoints",
        type=str,
        default=None,
        help="Comma-separated 1-based iter tags to eval (e.g. 540,1000). "
        "Uses models/<key>/model_N/ for each N. Default: final max_iterations only.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_compare", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {mk}. Choose from {list(MODEL_SPECS)}")

    exports = {}
    for mk in model_keys:
        run_config = {
            "model_key": mk,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "num_steps_per_env": 24,
            "total_env_steps_per_iter": args.num_envs * 24,
            "seed": args.seed,
            "context_len": args.context_len,
            "trajectory_sparsity": args.trajectory_sparsity,
            "save_interval": args.save_interval,
            "init_noise_std": args.init_noise_std,
            "saturate": args.saturate,
            "seed_retries": args.seed_retries,
            "eval_num_envs": args.eval_num_envs,
            "eval_max_episodes": args.eval_max_episodes,
            "pretrained_ckpt": MODEL_SPECS[mk]["ckpt"],
            "checkpoint_layout": "model_N.pt (PPO) + model_N/ (Vintix export)",
            "paths": {
                "model_dir": f"online_decoder_ft/models/{mk}/",
                "result_dir": f"online_decoder_ft/result/{mk}/",
            },
        }
        write_run_config(mk, run_config)

        export_dir = train_one(
            mk,
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            seed=args.seed,
            context_len=args.context_len,
            trajectory_sparsity=args.trajectory_sparsity,
            save_interval=args.save_interval,
            skip_train=args.skip_train,
            init_noise_std=args.init_noise_std,
            saturate=args.saturate,
            seed_retries=args.seed_retries,
        )
        exports[mk] = str(export_dir)
        if args.train_only:
            continue
        if not args.skip_eval:
            if args.eval_checkpoints:
                eval_iters = []
                for part in args.eval_checkpoints.split(","):
                    part = part.strip()
                    if part:
                        eval_iters.append(int(part))
                for iter_tag in eval_iters:
                    ckpt_dir = model_dir(mk) / f"model_{iter_tag}"
                    if not (ckpt_dir / "model.pth").is_file():
                        raise FileNotFoundError(
                            f"Missing Vintix export for eval: {ckpt_dir}/model.pth"
                        )
                    print(f"[eval] {mk} model_{iter_tag} -> result/{mk}/eval_iter{iter_tag}/")
                    eval_one(
                        mk,
                        ckpt_dir,
                        args.eval_num_envs,
                        args.eval_max_episodes,
                        output_subdir=f"eval_iter{iter_tag}",
                    )
            else:
                eval_one(mk, export_dir, args.eval_num_envs, args.eval_max_episodes)

    if not args.train_only and not args.skip_compare:
        compare_all(model_keys)

    print("\nDone.")
    for mk in model_keys:
        print(f"  models/{mk}/  result/{mk}/")
    print("Exports:", json.dumps(exports, indent=2))


if __name__ == "__main__":
    main()
