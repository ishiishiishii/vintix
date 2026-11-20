#!/usr/bin/env python3
"""Go1 3-axis body-scale eval for Vintix trained on narrow-DR histories (#21).

Same grids / seeds as ``same_body_scale_zeroshot/eval_go1_pipeline.py`` (#18).
Results go under this experiment's ``results/`` so they do not overwrite #18.

Early-stop (user 2026-11-19): within each axis and side of 1.0, once a scale is already
~0% completion with enough seeds, do not launch farther-from-1.0 scales. Order is
axis → |log scale| → seed so work moves to the next axis after a cliff.

    cd /workspace/vintix_go2
    setsid nohup python -u dr_narrow_vintix_bodyscale/eval_go1_pipeline.py \
      > dr_narrow_vintix_bodyscale/logs/eval_go1.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

EXP_DIR = Path(__file__).resolve().parent
VINTIX_ROOT = EXP_DIR.parent
RB_DIR = VINTIX_ROOT / "robot_body_scale_zeroshot"
GO1_DIR = VINTIX_ROOT / "go1_body_scale_zeroshot"
sys.path.insert(0, str(GO1_DIR))
sys.path.insert(0, str(RB_DIR))

import run_experiment as rx  # noqa: E402
from body_scale import FULL_GRID as GO1_FULL_GRID  # noqa: E402
from robots import EVAL_SEEDS, GRIDS, ROBOTS, canonical_axis, scale_tag  # noqa: E402

MODEL = (
    EXP_DIR / "models" / "dr_narrow_success_5epochs" / "dr_narrow_success_5epochs" / "0004_epoch"
)

rx.RESULTS_DIR = EXP_DIR / "results"
rx.LOGS_DIR = EXP_DIR / "logs" / "eval"
rx.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
rx.LOGS_DIR.mkdir(parents=True, exist_ok=True)
ROBOTS["go1"]["vintix"] = MODEL

STATUS_PATH = EXP_DIR / "eval_go1_status.json"
STOP_PATH = EXP_DIR / "STOP"
MAX_PARALLEL = 4

AXIS_GRIDS: Dict[str, Tuple[float, ...]] = {
    "mass": GO1_FULL_GRID["mass"],
    "size": GO1_FULL_GRID["size"],
    "size_density": GRIDS["size_density"],
}

# Early-stop: mean completion_rate <= this with >= MIN_SEEDS → cliff.
DEAD_COMPLETION = 0.01
MIN_SEEDS_FOR_CLIFF = 5
METRICS_NAME = "go1_10envs_10episodes_metrics.json"


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
        return [
            sys.executable, "-u", str(EXP_DIR / "eval_go1_worker.py"),
            "--axis", self.axis, "--scale", str(self.scale), "--seed", str(self.seed),
        ]


def build_jobs() -> List[Job]:
    """Axis → closest-to-1.0 scale → seed (so a dead mass side yields to size sooner)."""
    jobs: List[Job] = []
    for axis, scales in AXIS_GRIDS.items():
        for s in sorted(scales, key=lambda x: (abs(math.log(x)), x)):
            if abs(s - 1.0) < 1e-9 and axis != "mass":
                continue
            for seed in EVAL_SEEDS:
                jobs.append(Job(axis, s, seed))
    return jobs


def _completion_rates(axis: str, scale: float) -> List[float]:
    rates: List[float] = []
    cax = canonical_axis(axis, scale)
    tag = scale_tag(scale)
    root = rx.RESULTS_DIR / "go1" / "vintix" / cax
    if not root.is_dir():
        return rates
    for seed in EVAL_SEEDS:
        mp = root / f"{tag}_seed{seed}" / METRICS_NAME
        if not mp.exists():
            continue
        try:
            rates.append(float(json.loads(mp.read_text())["completion_rate"]))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return rates


def cliff_bounds() -> Dict[str, Dict[str, Optional[float]]]:
    """Per axis, farthest-from-1.0 scale still allowed on each side (inclusive)."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for axis, scales in AXIS_GRIDS.items():
        lo_cliff: Optional[float] = None
        hi_cliff: Optional[float] = None
        for s in sorted(scales):
            if abs(s - 1.0) < 1e-9:
                continue
            rates = _completion_rates(axis, s)
            if len(rates) < MIN_SEEDS_FOR_CLIFF:
                continue
            if sum(rates) / len(rates) > DEAD_COMPLETION:
                continue
            if s < 1.0:
                lo_cliff = s if lo_cliff is None else max(lo_cliff, s)
            else:
                hi_cliff = s if hi_cliff is None else min(hi_cliff, s)
        out[axis] = {"lo": lo_cliff, "hi": hi_cliff}
    return out


def should_skip(job: Job, cliffs: Dict[str, Dict[str, Optional[float]]]) -> bool:
    if job.done():
        return False
    bounds = cliffs.get(job.axis, {})
    if job.scale < 1.0:
        lo = bounds.get("lo")
        return lo is not None and job.scale < lo - 1e-12
    if job.scale > 1.0:
        hi = bounds.get("hi")
        return hi is not None and job.scale > hi + 1e-12
    return False


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _write_status(
    jobs: List[Job],
    running: Dict[str, subprocess.Popen],
    failed: Dict[str, int],
    skipped: Set[str],
    cliffs: Dict[str, Dict[str, Optional[float]]],
) -> None:
    STATUS_PATH.write_text(json.dumps({
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(MODEL),
        "total": len(jobs),
        "done": sum(j.done() for j in jobs),
        "skipped": sorted(skipped),
        "n_skipped": len(skipped),
        "cliffs": cliffs,
        "early_stop": {
            "dead_completion": DEAD_COMPLETION,
            "min_seeds": MIN_SEEDS_FOR_CLIFF,
        },
        "running": list(running),
        "failed": failed,
        "stop_file": STOP_PATH.exists(),
    }, indent=2))


def main(dry_run: bool = False) -> None:
    if not MODEL.is_dir():
        raise SystemExit(f"missing model checkpoint: {MODEL}")
    jobs = build_jobs()
    cliffs = cliff_bounds()
    skipped = {j.id for j in jobs if should_skip(j, cliffs)}
    _log(f"DR-narrow Vintix Go1 eval: {len(jobs)} jobs, model={MODEL}")
    _log(f"grids: mass={len(AXIS_GRIDS['mass'])} size={len(AXIS_GRIDS['size'])} "
         f"size_density={len(AXIS_GRIDS['size_density'])} seeds={len(EVAL_SEEDS)}")
    _log(f"early-stop cliffs={cliffs} skip_now={len(skipped)}")
    if dry_run:
        done = sum(j.done() for j in jobs)
        print(json.dumps({
            "total": len(jobs),
            "done": done,
            "skipped": len(skipped),
            "remaining": len(jobs) - done - len(skipped),
            "cliffs": cliffs,
        }, indent=2))
        return

    running: Dict[str, tuple] = {}
    failed: Dict[str, int] = {}
    logged_skip: Set[str] = set()
    while True:
        cliffs = cliff_bounds()
        skipped = {j.id for j in jobs if should_skip(j, cliffs)}
        for jid in sorted(skipped - logged_skip):
            _log(f"skip   {jid} (beyond cliff)")
            logged_skip.add(jid)

        if STOP_PATH.exists():
            _log("STOP present; not launching new jobs")
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
        if not STOP_PATH.exists():
            for job in jobs:
                if len(running) >= MAX_PARALLEL:
                    break
                if job.done() or job.id in running or job.id in skipped:
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
        _write_status(jobs, {k: v[0] for k, v in running.items()}, failed, skipped, cliffs)
        finished = all(
            j.done() or j.id in skipped or failed.get(j.id, 0) >= 2 for j in jobs
        )
        if finished and not running:
            _log(f"eval_go1 finished (done={sum(j.done() for j in jobs)} skipped={len(skipped)})")
            break
        time.sleep(15)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
