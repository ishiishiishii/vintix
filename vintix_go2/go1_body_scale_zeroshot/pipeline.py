#!/usr/bin/env python3
"""Detached scheduler for the whole Go1 body-scale study (resumable, runs jobs in parallel).

Order of work (higher stage first; a stage's jobs may overlap with the next when slots are free):

1. probe   : body sanity check for every point of FULL_GRID (cheap)
2. boundary: Vintix + transferred expert at BOUNDARY_GRID x EVAL_SEEDS
             (seed-major, so all boundary points get seed 0 before seed 1, ...)
3. ppo     : PPO from scratch at PPO_GRID (starts only after stage 2 is complete),
             then each trained PPO evaluated with EVAL_SEEDS
4. rest    : Vintix + expert at the remaining FULL_GRID points x EVAL_SEEDS

A job is "done" when its output exists, so restarting the pipeline skips finished work.
Touch ``STOP`` in this directory to stop launching new jobs (running ones finish).

    cd /workspace/vintix_go2
    nohup python -u go1_body_scale_zeroshot/pipeline.py > go1_body_scale_zeroshot/logs/pipeline.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import run_experiment as rx  # noqa: E402
from body_scale import BOUNDARY_GRID, EVAL_SEEDS, FULL_GRID, PPO_GRID, scale_tag  # noqa: E402

STATUS_PATH = EXP_DIR / "pipeline_status.json"
STOP_PATH = EXP_DIR / "STOP"
STAGES = ("probe", "boundary", "ppo", "rest")


@dataclass
class Job:
    stage: str
    kind: str  # probe | vintix | expert | ppo | train_ppo
    axis: str
    scale: float
    seed: Optional[int] = None
    deps: List[str] = field(default_factory=list)
    attempts: int = 0

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.axis}:{scale_tag(self.scale)}:{rx.run_name(self.seed)}"

    @property
    def weight(self) -> int:
        return 2 if self.kind == "train_ppo" else 1

    def done(self) -> bool:
        if self.kind == "probe":
            return rx.probe_json(self.axis, self.scale).exists()
        if self.kind == "train_ppo":
            return rx.ppo_final_ckpt(self.axis, self.scale).exists()
        return rx.episodes_csv(self.kind, self.axis, self.scale, self.seed).exists()

    def log_path(self) -> Path:
        name = f"{self.axis}_{scale_tag(self.scale)}" + ("" if self.seed is None else f"_seed{self.seed}")
        return rx.LOGS_DIR / self.kind / f"{name}.log"

    def cmd(self) -> List[str]:
        return rx.worker_cmd(self.kind, self.axis, self.scale, self.seed)


def _ordered(scales) -> List[float]:
    import math

    return sorted(scales, key=lambda s: (abs(math.log(s)), s))


def build_jobs() -> List[Job]:
    jobs: List[Job] = []
    for axis, scales in FULL_GRID.items():
        for s in _ordered(scales):
            jobs.append(Job("probe", "probe", axis, s))
    for seed in EVAL_SEEDS:
        for axis, scales in BOUNDARY_GRID.items():
            for s in _ordered(scales):
                jobs.append(Job("boundary", "vintix", axis, s, seed))
                jobs.append(Job("boundary", "expert", axis, s, seed))
    for axis, scales in PPO_GRID.items():
        for s in _ordered(scales):
            train = Job("ppo", "train_ppo", axis, s)
            jobs.append(train)
            for seed in EVAL_SEEDS:
                jobs.append(Job("ppo", "ppo", axis, s, seed, deps=[train.id]))
    for seed in EVAL_SEEDS:
        for axis, scales in FULL_GRID.items():
            for s in _ordered(scales):
                if s in BOUNDARY_GRID[axis]:
                    continue
                jobs.append(Job("rest", "vintix", axis, s, seed))
                jobs.append(Job("rest", "expert", axis, s, seed))
    return jobs


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _write_status(jobs: List[Job], running: Dict[str, tuple], failed: Dict[str, int]) -> None:
    by_stage = {}
    for st in STAGES:
        sj = [j for j in jobs if j.stage == st]
        by_stage[st] = {"total": len(sj), "done": sum(j.done() for j in sj),
                        "running": sum(j.id in running for j in sj)}
    with open(STATUS_PATH, "w") as f:
        json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": by_stage,
                   "running": [{"job": jid, "minutes": round((time.time() - t0) / 60, 1)}
                               for jid, (_, t0, _) in running.items()],
                   "failed": failed, "stop_file": STOP_PATH.exists()}, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_evals", type=int, default=4, help="concurrent evaluation jobs")
    p.add_argument("--capacity", type=int, default=5, help="total GPU weight (eval=1, PPO training=2)")
    p.add_argument("--max_trains", type=int, default=1)
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--poll", type=float, default=15.0)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs = build_jobs()
    by_id = {j.id: j for j in jobs}
    if args.dry_run:
        for st in STAGES:
            sj = [j for j in jobs if j.stage == st]
            print(st, "total", len(sj), "done", sum(j.done() for j in sj))
        return

    running: Dict[str, tuple] = {}  # id -> (Popen, t0, job)
    failed: Dict[str, int] = {}
    _log(f"pipeline start: {len(jobs)} jobs, {sum(j.done() for j in jobs)} already done")
    while True:
        # reap finished
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
            try:
                rx.stage_report()
            except Exception as e:  # report must never kill the scheduler
                _log(f"report error: {e}")

        boundary_done = all(j.done() for j in jobs if j.stage in ("probe", "boundary"))
        pending = [j for j in jobs if not j.done() and j.id not in running and j.attempts < args.max_attempts]
        if not pending and not running:
            _write_status(jobs, running, failed)
            _log("pipeline finished")
            break

        if not STOP_PATH.exists():
            used = sum(running[j][2].weight for j in running)
            n_eval = sum(1 for j in running if running[j][2].kind != "train_ppo")
            n_train = sum(1 for j in running if running[j][2].kind == "train_ppo")
            for job in pending:  # already in priority order
                if job.kind == "train_ppo" and not boundary_done:
                    continue
                if any(not by_id[d].done() for d in job.deps):
                    continue
                if used + job.weight > args.capacity:
                    continue
                if job.kind == "train_ppo" and n_train >= args.max_trains:
                    continue
                if job.kind != "train_ppo" and n_eval >= args.max_evals:
                    continue
                job.attempts += 1
                log = job.log_path()
                log.parent.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(job.cmd(), stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                        cwd=str(rx.VINTIX_ROOT), env=rx.subprocess_env())
                running[job.id] = (proc, time.time(), job)
                used += job.weight
                if job.kind == "train_ppo":
                    n_train += 1
                else:
                    n_eval += 1
                _log(f"start  {job.id} (attempt {job.attempts})")
        elif not running:
            _write_status(jobs, running, failed)
            _log("STOP file present and nothing running -> exit")
            break

        _write_status(jobs, running, failed)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
