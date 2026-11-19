#!/usr/bin/env python3
"""Go1 same-body Vintix body-scale eval (#18), 3 axes, Vintix only.

Compares against leave-one-out results already on disk (go1_body_scale_zeroshot mass/size,
robot_body_scale_zeroshot size_density). Does not re-run PPO/expert.

    cd /workspace/vintix_go2
    setsid nohup python -u same_body_scale_zeroshot/eval_go1_pipeline.py \
      > same_body_scale_zeroshot/logs/eval_go1.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
RB_DIR = VINTIX_ROOT / "robot_body_scale_zeroshot"
GO1_DIR = VINTIX_ROOT / "go1_body_scale_zeroshot"
# go1_body_scale also has run_experiment.py — keep robot_body_scale first on sys.path.
sys.path.insert(0, str(GO1_DIR))
sys.path.insert(0, str(RB_DIR))

import run_experiment as rx  # noqa: E402
from body_scale import FULL_GRID as GO1_FULL_GRID  # noqa: E402
from robots import EVAL_SEEDS, GRIDS, ROBOTS, canonical_axis, scale_tag  # noqa: E402

# Same-body Go1 checkpoint (5 epochs finished).
GO1_SAME = (
    EXP_DIR / "models" / "go1_data1M_5epochs" / "go1_data1M_5epochs" / "0004_epoch"
)
assert GO1_SAME.is_dir(), GO1_SAME

# Write under this experiment so LOO results are not overwritten.
rx.RESULTS_DIR = EXP_DIR / "results"
rx.LOGS_DIR = EXP_DIR / "logs" / "eval"
rx.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
rx.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Point Vintix path at same-body model; enable all 3 axes for Go1.
ROBOTS["go1"]["vintix"] = GO1_SAME

STATUS_PATH = EXP_DIR / "eval_go1_status.json"
STOP_PATH = EXP_DIR / "STOP"
MAX_PARALLEL = 4

# mass/size: same grid as Go1 LOO study; size_density: same as #10 four-robot grid.
AXIS_GRIDS: Dict[str, Tuple[float, ...]] = {
    "mass": GO1_FULL_GRID["mass"],
    "size": GO1_FULL_GRID["size"],
    "size_density": GRIDS["size_density"],
}


@dataclass
class Job:
    axis: str
    scale: float
    seed: int

    @property
    def id(self) -> str:
        return f"vintix:go1:{canonical_axis(self.axis, self.scale)}:{scale_tag(self.scale)}:seed{self.seed}"

    def done(self) -> bool:
        return rx.episodes_csv("go1", "vintix", self.axis, self.scale, self.seed).exists()

    def log_path(self) -> Path:
        d = rx.LOGS_DIR / "go1" / "vintix"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{canonical_axis(self.axis, self.scale)}_{scale_tag(self.scale)}_seed{self.seed}.log"

    def cmd(self) -> List[str]:
        # Inline worker: patch model path via robots, call run_experiment CLI with env.
        return [
            sys.executable, "-u", str(EXP_DIR / "eval_go1_worker.py"),
            "--axis", self.axis, "--scale", str(self.scale), "--seed", str(self.seed),
        ]


def build_jobs() -> List[Job]:
    jobs: List[Job] = []
    for seed in EVAL_SEEDS:
        for axis, scales in AXIS_GRIDS.items():
            for s in sorted(scales, key=lambda x: (abs(math.log(x)), x)):
                # Share nominal 1.0 under mass (same convention as #10).
                if abs(s - 1.0) < 1e-9 and axis != "mass":
                    continue
                jobs.append(Job(axis, s, seed))
    return jobs


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _write_status(jobs: List[Job], running: Dict[str, subprocess.Popen], failed: Dict[str, int]) -> None:
    STATUS_PATH.write_text(json.dumps({
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(GO1_SAME),
        "total": len(jobs),
        "done": sum(j.done() for j in jobs),
        "running": list(running),
        "failed": failed,
        "stop_file": STOP_PATH.exists(),
    }, indent=2))


def main(dry_run: bool = False) -> None:
    jobs = build_jobs()
    _log(f"Go1 same-body eval: {len(jobs)} vintix jobs, model={GO1_SAME}")
    _log(f"grids: mass={len(AXIS_GRIDS['mass'])} size={len(AXIS_GRIDS['size'])} "
         f"size_density={len(AXIS_GRIDS['size_density'])} seeds={len(EVAL_SEEDS)}")
    if dry_run:
        done = sum(j.done() for j in jobs)
        print(json.dumps({"total": len(jobs), "done": done, "remaining": len(jobs) - done}, indent=2))
        return

    running: Dict[str, tuple] = {}
    failed: Dict[str, int] = {}
    while True:
        if STOP_PATH.exists():
            _log("STOP present; not launching new jobs")
        # reap
        for jid in list(running):
            proc, job = running[jid]
            rc = proc.poll()
            if rc is None:
                continue
            del running[jid]
            if rc == 0 and job.done():
                _log(f"done   {jid}")
            else:
                failed[jid] = failed.get(jid, 0) + 1
                _log(f"fail   {jid} rc={rc} attempts={failed[jid]}")
        # launch
        if not STOP_PATH.exists():
            for job in jobs:
                if len(running) >= MAX_PARALLEL:
                    break
                if job.done() or job.id in running:
                    continue
                if failed.get(job.id, 0) >= 2:
                    continue
                logf = job.log_path().open("w")
                proc = subprocess.Popen(
                    job.cmd(), cwd=str(VINTIX_ROOT),
                    stdout=logf, stderr=subprocess.STDOUT,
                )
                running[job.id] = (proc, job)
                _log(f"start  {job.id} pid={proc.pid}")
        _write_status(jobs, {k: v[0] for k, v in running.items()}, failed)
        if all(j.done() or failed.get(j.id, 0) >= 2 for j in jobs) and not running:
            _log("eval_go1 finished")
            break
        time.sleep(15)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)
