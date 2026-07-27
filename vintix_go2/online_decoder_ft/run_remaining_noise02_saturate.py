#!/usr/bin/env python3
"""
安定化設定（init_noise_std=0.2 + --saturate + --seed_retries 3）で
MiniCheetah → Go1 → A1 を「学習 → 評価」の順に実行する。

Go2 (`go2_without_noise02_saturate`) は完了済みのため対象外。
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
SCRIPTS = VINTIX_ROOT / "scripts"
GENESIS_LOCO = VINTIX_ROOT.parent / "Genesis" / "examples" / "locomotion"
LOG_DIR = EXP_DIR / "logs"
MODELS = EXP_DIR / "models"
RESULT = EXP_DIR / "result"

ROBOTS = [
    {
        "key": "minicheetah_without_noise02_saturate",
        "robot": "minicheetah",
        "ckpt": "models/minicheetah_without/minicheetah_without/0001_epoch",
    },
    {
        "key": "go1_without_noise02_saturate",
        "robot": "go1",
        "ckpt": "models/go1_without/go1_without/0001_epoch",
    },
    {
        "key": "a1_without_noise02_saturate",
        "robot": "a1",
        "ckpt": "models/a1_without/a1_without/0001_epoch",
    },
]

NUM_ENVS = 64
MAX_ITERATIONS = 10000
SAVE_INTERVAL = 500
SEED = 1
SEED_RETRIES = 3
INIT_NOISE_STD = 0.2
CONTEXT_LEN = 2048
TRAJECTORY_SPARSITY = 128
EVAL_NUM_ENVS = 10
EVAL_MAX_EPISODES = 10


def _env() -> dict:
    import os

    e = dict(os.environ)
    e["PYTHONPATH"] = ":".join(
        [str(VINTIX_ROOT), str(SCRIPTS), str(GENESIS_LOCO), e.get("PYTHONPATH", "")]
    )
    e["PYTHONUNBUFFERED"] = "1"
    return e


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {' '.join(cmd)}\n    log={log_path}", flush=True)
    with open(log_path, "a") as lf:
        lf.write(f"\n===== START {datetime.now(timezone.utc).isoformat()} =====\n")
        lf.write("CMD: " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(VINTIX_ROOT), env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _latest_export(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("model_*/model.pth"), key=lambda p: p.parent.name)
    if not candidates:
        raise FileNotFoundError(f"No Vintix export under {model_dir}")
    return candidates[-1].parent


def _parse_eval_mean(summary_txt: Path) -> float | None:
    if not summary_txt.is_file():
        return None
    for line in summary_txt.read_text().splitlines():
        if line.startswith("Mean Reward per Episode:"):
            return float(line.split(":", 1)[1].strip())
    return None


def train_one(spec: dict) -> Path:
    key = spec["key"]
    mdir = MODELS / key
    rdir = RESULT / key
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    ckpt = VINTIX_ROOT / spec["ckpt"]
    if not ckpt.is_dir():
        raise FileNotFoundError(ckpt)

    cfg = {
        "model_key": key,
        "status": "running",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "num_envs": NUM_ENVS,
        "max_iterations": MAX_ITERATIONS,
        "saturate": True,
        "init_noise_std": INIT_NOISE_STD,
        "seed": SEED,
        "seed_retries": SEED_RETRIES,
        "context_len": CONTEXT_LEN,
        "trajectory_sparsity": TRAJECTORY_SPARSITY,
        "save_interval": SAVE_INTERVAL,
        "pretrained_ckpt": spec["ckpt"],
        "robot": spec["robot"],
    }
    (mdir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    cmd = [
        sys.executable,
        str(SCRIPTS / "train_online_decoder_ppo.py"),
        "-e",
        f"online_ft_{key}",
        "-r",
        spec["robot"],
        "-B",
        str(NUM_ENVS),
        "--max_iterations",
        str(MAX_ITERATIONS),
        "--saturate",
        "--init_noise_std",
        str(INIT_NOISE_STD),
        "--seed",
        str(SEED),
        "--seed_retries",
        str(SEED_RETRIES),
        "--vintix_path",
        str(ckpt),
        "--log_dir",
        str(mdir),
        "--result_dir",
        str(rdir),
        "--context_len",
        str(CONTEXT_LEN),
        "--trajectory_sparsity",
        str(TRAJECTORY_SPARSITY),
        "--save_interval",
        str(SAVE_INTERVAL),
    ]
    _run(cmd, LOG_DIR / f"{key}.log")
    export = _latest_export(mdir)
    cfg["status"] = "train_done"
    cfg["final_export"] = str(export.relative_to(VINTIX_ROOT))
    cfg["finished_train_at"] = datetime.now(timezone.utc).isoformat()
    (mdir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return export


def eval_one(spec: dict, export_dir: Path) -> dict:
    key = spec["key"]
    # export dir name is model_N
    tag = export_dir.name  # e.g. model_4006
    out = RESULT / key / f"eval_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "save_vintix.py"),
        "-r",
        spec["robot"],
        "--vintix_path",
        str(export_dir.resolve()),
        "--num_envs",
        str(EVAL_NUM_ENVS),
        "--max_episodes",
        str(EVAL_MAX_EPISODES),
        "--output_dir",
        str(out.resolve()),
    ]
    _run(cmd, LOG_DIR / f"eval_{key}_{tag}.log")
    # find mean reward summary
    summaries = list(out.glob("*_mean_reward.txt"))
    mean_ep = _parse_eval_mean(summaries[0]) if summaries else None
    return {
        "eval_dir": str(out.relative_to(VINTIX_ROOT)),
        "mean_episode_reward": mean_ep,
        "export": str(export_dir.relative_to(VINTIX_ROOT)),
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    batch_log = LOG_DIR / "remaining_robots_noise02_saturate_batch.json"
    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "num_envs": NUM_ENVS,
            "max_iterations": MAX_ITERATIONS,
            "init_noise_std": INIT_NOISE_STD,
            "saturate": True,
            "seed": SEED,
            "seed_retries": SEED_RETRIES,
            "order": [r["key"] for r in ROBOTS],
        },
        "robots": {},
    }
    batch_log.write_text(json.dumps(results, indent=2) + "\n")

    for spec in ROBOTS:
        key = spec["key"]
        print(f"\n======== TRAIN {key} ========", flush=True)
        export = train_one(spec)
        print(f"\n======== EVAL {key} ({export.name}) ========", flush=True)
        ev = eval_one(spec, export)
        results["robots"][key] = ev
        results["updated_at"] = datetime.now(timezone.utc).isoformat()
        batch_log.write_text(json.dumps(results, indent=2) + "\n")
        # mark run_config
        cfg_path = MODELS / key / "run_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["status"] = "completed"
        cfg["eval"] = ev
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"DONE {key}: eval_mean={ev.get('mean_episode_reward')}", flush=True)

    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    results["status"] = "all_completed"
    batch_log.write_text(json.dumps(results, indent=2) + "\n")
    print("\nAll remaining robots train+eval completed.", flush=True)
    print(f"Summary: {batch_log}", flush=True)


if __name__ == "__main__":
    main()
