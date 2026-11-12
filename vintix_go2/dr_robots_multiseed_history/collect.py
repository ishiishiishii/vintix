#!/usr/bin/env python3
"""Train every DR robot with PPO under several seeds and keep each run's learning history (AD data).

Orchestration only: per (robot, seed) it calls the same shared steps as
``dr_robots_ppo_history/run_dr_ppo_history.py`` writes into each robot's ``run_experiment.sh``:
``Genesis/examples/locomotion/auto_reward_design.py`` (once per robot) and
``vintix_go2/scripts/train_with_history.py`` (PPO 301 iters, 4096 envs, history of the first 256 envs).

Outputs stay inside each robot package:
- seed 1: ``model/`` and ``ppo_history/`` (the layout already used by dr_robots_ppo_history)
- seed k>=2: ``model_seed<k>/`` and ``ppo_history_seed<k>/``

Order is seed-major (every robot gets seed k before any robot gets seed k+1) so stopping at any
time leaves a balanced dataset. A run is done when ``<history>/<robot_id>.json`` and
``<model>/model_300.pt`` exist; a crashed partial run is moved aside (``*_incomplete_<time>``),
never deleted. Stops when free disk space falls below --min_free_gb or when ``STOP`` exists here.

    cd /workspace/vintix_go2
    setsid nohup python -u dr_robots_multiseed_history/collect.py > dr_robots_multiseed_history/logs/collect.log 2>&1 < /dev/null &
    python dr_robots_multiseed_history/collect.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
WORKSPACE = VINTIX_ROOT.parent
LOCO_ROOT = WORKSPACE / "Genesis" / "examples" / "locomotion"
DR_DIR = VINTIX_ROOT / "dr_robots_ppo_history"
SETS = {
    "narrow": DR_DIR / "dr_narrow_genloco" / "generated",
    "wide": DR_DIR / "dr_wide_extended" / "generated",
}
STATUS_PATH = EXP_DIR / "status.json"
STOP_PATH = EXP_DIR / "STOP"
LOG_DIR = EXP_DIR / "logs"
ITERATIONS = 301
FINAL_CKPT = f"model_{ITERATIONS - 1}.pt"


def out_dirs(pkg: Path, seed: int) -> Tuple[Path, Path]:
    if seed == 1:
        return pkg / "model", pkg / "ppo_history"
    return pkg / f"model_seed{seed}", pkg / f"ppo_history_seed{seed}"


def done(pkg: Path, seed: int) -> bool:
    model, hist = out_dirs(pkg, seed)
    return (hist / f"{pkg.name}.json").exists() and (model / FINAL_CKPT).exists()


def robots() -> List[Tuple[str, Path]]:
    out = []
    for name, root in SETS.items():
        out += [(name, p) for p in sorted(root.iterdir()) if (p / "manifest.json").exists()]
    return out


def jobs(seeds: List[int]) -> List[Tuple[str, Path, int]]:
    return [(s, p, seed) for seed in seeds for s, p in robots()]


def env() -> dict:
    e = dict(os.environ)
    e.setdefault("WANDB_MODE", "disabled")
    e["PYTHONPATH"] = f"{LOCO_ROOT}:{VINTIX_ROOT}:{VINTIX_ROOT / 'scripts'}:{e.get('PYTHONPATH', '')}"
    return e


def free_gb() -> float:
    return shutil.disk_usage(WORKSPACE).free / 1e9


def needs_reward_design(pkg: Path) -> bool:
    cfg = json.loads((pkg / "config" / "reward_cfg.json").read_text())
    return "auto_reward_design" not in cfg


def run(cmd: List[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(VINTIX_ROOT), env=env())


def write_status(all_jobs, current=None, note="") -> None:
    by_seed = {}
    for s, p, seed in all_jobs:
        d = by_seed.setdefault(seed, {"done": 0, "total": 0})
        d["total"] += 1
        d["done"] += done(p, seed)
    STATUS_PATH.write_text(json.dumps({
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "free_gb": round(free_gb(), 1),
        "current": current, "note": note, "by_seed": by_seed,
        "done_total": sum(v["done"] for v in by_seed.values()), "total": len(all_jobs)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="1-10")
    ap.add_argument("--min_free_gb", type=float, default=150.0)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--record_num_envs", type=int, default=256)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    all_jobs = jobs(seeds)
    if a.dry_run:
        for seed in seeds:
            sj = [j for j in all_jobs if j[2] == seed]
            print(f"seed {seed}: {sum(done(p, seed) for _, p, _ in sj)}/{len(sj)} done")
        print(f"free disk {free_gb():.0f} GB")
        return

    print(f"[{time.strftime('%F %T')}] collect start: {len(all_jobs)} runs, "
          f"{sum(done(p, s) for _, p, s in all_jobs)} done, free {free_gb():.0f} GB", flush=True)
    for set_name, pkg, seed in all_jobs:
        if done(pkg, seed):
            continue
        if STOP_PATH.exists():
            write_status(all_jobs, note="STOP file present")
            print("STOP file present -> exit", flush=True)
            return
        if free_gb() < a.min_free_gb:
            write_status(all_jobs, note=f"free disk {free_gb():.0f} GB < {a.min_free_gb:.0f} GB")
            print(f"free disk below {a.min_free_gb} GB -> exit", flush=True)
            return
        robot_id = pkg.name
        model, hist = out_dirs(pkg, seed)
        log = LOG_DIR / set_name / f"{robot_id}_seed{seed}.log"
        current = f"{robot_id} seed {seed}"
        write_status(all_jobs, current=current)
        t0 = time.time()
        for d in (model, hist):
            if d.exists():
                d.rename(d.with_name(f"{d.name}_incomplete_{time.strftime('%Y%m%d_%H%M%S')}"))
        if needs_reward_design(pkg):
            run(["python", str(LOCO_ROOT / "auto_reward_design.py"), "--package", str(pkg)], log)
        rc = run(["python", "scripts/train_with_history.py", "--robot_package", str(pkg),
                  "-B", str(a.num_envs), "--max_iterations", str(ITERATIONS), "--seed", str(seed),
                  "--log_dir", str(model), "--record_output", str(hist), "--task_name", robot_id,
                  "--record_num_envs", str(a.record_num_envs), "--relax_output_permissions"], log)
        status = "ok" if done(pkg, seed) else f"FAILED rc={rc}"
        print(f"[{time.strftime('%F %T')}] {status} {current} {(time.time() - t0) / 60:.1f} min free {free_gb():.0f} GB",
              flush=True)
        write_status(all_jobs)
    write_status(all_jobs, note="all done")
    print("collect finished", flush=True)


if __name__ == "__main__":
    main()
