#!/usr/bin/env python3
"""Decide the next autopilot sleep interval.

Rules (2026-11-18):
- If there is actionable *non-experiment* open work (survey / write-up / impl / discussion),
  use INTERVAL_MINUTES_ACTIVE (default 30).
- Else if only waiting on experiments:
  - If ``state/experiment_eta_epoch`` is a future unix time, sleep until then
    (clamped to [1 min, INTERVAL_HOURS_DEFAULT]).
  - Otherwise use INTERVAL_HOURS_DEFAULT (default 8h).
- When the ETA has passed, clear it and fall back to the default (or active if
  non-experiment work appeared).

Prints one line: ``<minutes> <mode> <reason>``
  mode: active | eta | waiting
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

AP = Path(__file__).resolve().parent
STATE = AP / "state"
ETA_FILE = STATE / "experiment_eta_epoch"
ETA_NOTE = STATE / "experiment_eta_note"

# Defaults; config.env may export these before invoking.
ACTIVE_MIN = int(os.environ.get("INTERVAL_MINUTES_ACTIVE", "30"))
DEFAULT_HOURS = float(os.environ.get("INTERVAL_HOURS_DEFAULT", os.environ.get("INTERVAL_HOURS", "8")))
DEFAULT_MIN = max(1, int(round(DEFAULT_HOURS * 60)))


def gh_open_issues() -> list[dict]:
    raw = subprocess.check_output(
        [
            "gh", "issue", "list", "-R", "ishiishiishii/vintix",
            "--state", "open", "--limit", "50",
            "--json", "number,title,labels",
        ],
        text=True,
    )
    return json.loads(raw)


def labels(it: dict) -> set[str]:
    return {l["name"] for l in (it.get("labels") or [])}


def has_non_experiment_work(issues: list[dict]) -> tuple[bool, str]:
    """True when open issues include survey / write-up / light impl (not later, not GPU-wait-only).

    - ``discussion`` (with or without ``experiment``) → investigation → active (e.g. #20)
    - ``task`` without ``experiment`` → docs/impl/orchestration → active
    - pure ``experiment`` / ``experiment+autopilot`` → GPU wait path (not active by itself)
    - ``needs-decision`` without other actionable labels → user wait (not active)
    - agent may force active via ``state/schedule_mode=active``
    """
    mode_file = STATE / "schedule_mode"
    if mode_file.exists():
        mode = mode_file.read_text().strip().split()[0].lower()
        if mode == "active":
            return True, "state/schedule_mode=active"
        if mode == "waiting":
            return False, "state/schedule_mode=waiting"

    hits = []
    for it in issues:
        labs = labels(it)
        if "later" in labs:
            continue
        if "needs-decision" in labs and "discussion" not in labs and "task" not in labs:
            continue
        if "discussion" in labs:
            hits.append(f"#{it['number']}")
            continue
        if "task" in labs and "experiment" not in labs:
            hits.append(f"#{it['number']}")
            continue
        # #16 / #22: task+experiment but non-GPU work continues during mainline (user 2026-11-19).
        if "task" in labs and "experiment" in labs and it["number"] in (16, 22):
            hits.append(f"#{it['number']}")
            continue
    if hits:
        return True, "non-exp: " + ",".join(hits[:8])
    return False, "no actionable non-experiment open issues"


def read_eta() -> tuple[int | None, str]:
    if not ETA_FILE.exists():
        return None, ""
    try:
        epoch = int(ETA_FILE.read_text().strip().split()[0])
    except (ValueError, OSError):
        return None, ""
    note = ETA_NOTE.read_text().strip() if ETA_NOTE.exists() else ""
    return epoch, note


def clear_eta() -> None:
    ETA_FILE.unlink(missing_ok=True)
    ETA_NOTE.unlink(missing_ok=True)


def main() -> None:
    issues = gh_open_issues()
    active, why = has_non_experiment_work(issues)
    if active:
        print(f"{ACTIVE_MIN} active {why}")
        return

    epoch, note = read_eta()
    now = int(time.time())
    if epoch is not None:
        if epoch <= now:
            clear_eta()
            print(f"{DEFAULT_MIN} waiting eta_passed_cleared -> default {DEFAULT_HOURS:g}h")
            return
        mins = max(1, (epoch - now + 59) // 60)
        # Cap: do not sleep longer than the default wait interval.
        mins = min(mins, DEFAULT_MIN)
        reason = f"eta_in_{mins}m"
        if note:
            reason += f" ({note[:80]})"
        print(f"{mins} eta {reason}")
        return

    print(f"{DEFAULT_MIN} waiting default {DEFAULT_HOURS:g}h (experiment wait / idle)")


if __name__ == "__main__":
    main()
