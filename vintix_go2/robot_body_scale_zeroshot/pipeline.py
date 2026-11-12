#!/usr/bin/env python3
"""Detached, resumable scheduler for the multi-robot body-scale study.

Stages (a later stage's jobs may start when slots are free, except PPO which waits for ``main``):
1. probe : body sanity check (sim mass, standing height) for every point
2. main  : Vintix zero-shot + transferred expert, seed-major (all points get seed 0 before seed 1)
3. ppo   : PPO from scratch at PPO_GRID, then each trained PPO evaluated with EVAL_SEEDS

A job is done when its output exists, so restarting skips finished work. ``touch STOP`` in this
directory stops launching new jobs.

    cd /workspace/vintix_go2
    setsid nohup python -u robot_body_scale_zeroshot/pipeline.py > robot_body_scale_zeroshot/logs/pipeline.log 2>&1 < /dev/null &
    python robot_body_scale_zeroshot/pipeline.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import run_experiment as rx  # noqa: E402
from robots import EVAL_SEEDS, GRIDS, PPO_GRID, ROBOT_AXES, canonical_axis, nominal_point, scale_tag  # noqa: E402

STATUS_PATH = EXP_DIR / "pipeline_status.json"
STOP_PATH = EXP_DIR / "STOP"
STAGES = ("probe", "main", "ppo")
ROBOT_ORDER = ("a1", "go2", "minicheetah", "go1")


@dataclass
class Job:
    stage: str
    kind: str  # probe | vintix | expert | ppo | train_ppo
    robot: str
    axis: str
    scale: float
    seed: Optional[int] = None
    deps: List[str] = field(default_factory=list)
    attempts: int = 0

    @property
    def id(self) -> str:
        s = "" if self.seed is None else f":seed{self.seed}"
        return f"{self.kind}:{self.robot}:{canonical_axis(self.axis, self.scale)}:{scale_tag(self.scale)}{s}"

    @property
    def weight(self) -> int:
        return 2 if self.kind == "train_ppo" else 1

    def done(self) -> bool:
        if self.kind == "probe":
            return rx.probe_json(self.robot, self.axis, self.scale).exists()
        if self.kind == "train_ppo":
            return rx.ppo_final_ckpt(self.robot, self.axis, self.scale).exists()
        return rx.episodes_csv(self.robot, self.kind, self.axis, self.scale, self.seed).exists()

    def log_path(self) -> Path:
        s = "" if self.seed is None else f"_seed{self.seed}"
        return rx.LOGS_DIR / self.robot / self.kind / f"{canonical_axis(self.axis, self.scale)}_{scale_tag(self.scale)}{s}.log"

    def cmd(self) -> List[str]:
        return rx.worker_cmd(self.kind, self.robot, self.axis, self.scale, self.seed)


def points(robot: str):
    """(axis, scale) pairs to evaluate; nominal once per robot (Go1's nominal already exists)."""
    out = []
    for axis in ROBOT_AXES[robot]:
        for s in sorted(GRIDS[axis], key=lambda s: (abs(math.log(s)), s)):
            if nominal_point(axis, s) and (axis != "mass" or robot == "go1"):
                continue
            out.append((axis, s))
    return out


def build_jobs() -> List[Job]:
    jobs: List[Job] = []
    for robot in ROBOT_ORDER:
        for axis, s in points(robot):
            jobs.append(Job("probe", "probe", robot, axis, s))
    for seed in EVAL_SEEDS:
        for robot in ROBOT_ORDER:
            for axis, s in points(robot):
                jobs.append(Job("main", "vintix", robot, axis, s, seed))
                jobs.append(Job("main", "expert", robot, axis, s, seed))
    for robot in ROBOT_ORDER:
        for axis in ROBOT_AXES[robot]:
            for s in PPO_GRID[axis]:
                train = Job("ppo", "train_ppo", robot, axis, s)
                jobs.append(train)
                for seed in EVAL_SEEDS:
                    jobs.append(Job("ppo", "ppo", robot, axis, s, seed, deps=[train.id]))
    seen, unique = set(), []
    for j in jobs:
        if j.id not in seen:
            seen.add(j.id)
            unique.append(j)
    return unique


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _write_status(jobs: List[Job], running: Dict[str, tuple], failed: Dict[str, int]) -> None:
    stages = {}
    for st in STAGES:
        sj = [j for j in jobs if j.stage == st]
        stages[st] = {"total": len(sj), "done": sum(j.done() for j in sj), "running": sum(j.id in running for j in sj)}
    STATUS_PATH.write_text(json.dumps({
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": stages,
        "running": [{"job": jid, "minutes": round((time.time() - t0) / 60, 1)} for jid, (_, t0, _) in running.items()],
        "failed": failed, "stop_file": STOP_PATH.exists()}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_evals", type=int, default=4)
    p.add_argument("--capacity", type=int, default=5, help="total weight (eval=1, PPO training=2)")
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--poll", type=float, default=15.0)
    p.add_argument("--report_every_min", type=float, default=20.0)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs = build_jobs()
    by_id = {j.id: j for j in jobs}
    if args.dry_run:
        for st in STAGES:
            sj = [j for j in jobs if j.stage == st]
            print(st, "total", len(sj), "done", sum(j.done() for j in sj))
        return

    running: Dict[str, tuple] = {}
    failed: Dict[str, int] = {}
    last_report = 0.0
    _log(f"pipeline start: {len(jobs)} jobs, {sum(j.done() for j in jobs)} already done")
    while True:
        for jid in list(running):
            proc, t0, job = running[jid]
            rc = proc.poll()
            if rc is None:
                continue
            del running[jid]
            mins = (time.time() - t0) / 60
            if job.done():
                _log(f"done   {jid} rc={rc} {mins:.1f} min")
            else:
                failed[jid] = job.attempts
                _log(f"FAILED {jid} rc={rc} {mins:.1f} min (attempt {job.attempts}) log={job.log_path().relative_to(EXP_DIR)}")
        if time.time() - last_report > args.report_every_min * 60:
            try:
                rx.stage_report()
            except Exception as e:  # reporting must never kill the scheduler
                _log(f"report error: {e}")
            last_report = time.time()

        main_done = all(j.done() for j in jobs if j.stage in ("probe", "main"))
        pending = [j for j in jobs if not j.done() and j.id not in running and j.attempts < args.max_attempts]
        if not pending and not running:
            rx.stage_report()
            _write_status(jobs, running, failed)
            _log("pipeline finished")
            break
        if not STOP_PATH.exists():
            used = sum(running[j][2].weight for j in running)
            n_eval = sum(1 for j in running if running[j][2].kind != "train_ppo")
            n_train = len(running) - n_eval
            for job in pending:
                if job.kind == "train_ppo" and (not main_done or n_train >= 1):
                    continue
                if any(not by_id[d].done() for d in job.deps):
                    continue
                if used + job.weight > args.capacity or (job.kind != "train_ppo" and n_eval >= args.max_evals):
                    continue
                job.attempts += 1
                log = job.log_path()
                log.parent.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(job.cmd(), stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                        cwd=str(rx.VINTIX_ROOT), env=rx.subprocess_env())
                running[job.id] = (proc, time.time(), job)
                used += job.weight
                n_train += job.kind == "train_ppo"
                n_eval += job.kind != "train_ppo"
                _log(f"start  {job.id} (attempt {job.attempts})")
        elif not running:
            _write_status(jobs, running, failed)
            _log("STOP file present and nothing running -> exit")
            break
        _write_status(jobs, running, failed)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
