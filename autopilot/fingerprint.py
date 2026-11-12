#!/usr/bin/env python3
"""Cheap "did anything change?" check for autopilot (no AI call, no Cursor usage).

Prints one line: a hash of everything the agent would react to, followed by a short summary.
autopilot.sh calls the agent only when this hash differs from the one saved after the
previous AI tick.

Inputs:
- GitHub: open/closed issues and PRs of ishiishiishii/vintix (number, state, labels, updated_at,
  comment count) -> any new comment, label change or new issue by the user changes the hash
- running experiment processes (command lines of pipeline / run_experiment / train / collect / eval)
- experiment progress files ``vintix_go2/*/pipeline_status.json`` (stage done counts and failures,
  not the timestamps)
- git HEADs of ``main`` and the current branch (new commits by the user)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
REPO = "ishiishiishii/vintix"
PROC_PATTERNS = ("pipeline.py", "run_experiment.py", "train", "collect_ad_data", "save_vintix", "save_expert")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=WORKSPACE).stdout
    except Exception as e:  # a failed probe must change nothing silently: record the error text
        return f"ERR:{e}"


def github_state() -> list:
    gh = os.path.expanduser("~/.local/bin/gh")
    out = _run([gh, "api", f"repos/{REPO}/issues?state=all&per_page=100",
                "--jq", "[.[] | {n: .number, s: .state, l: [.labels[].name], u: .updated_at, c: .comments}]"])
    try:
        return sorted(json.loads(out), key=lambda x: x["n"])
    except Exception:
        return [{"error": out.strip()[:200]}]


def processes() -> list:
    out = _run(["ps", "-eo", "args"])
    lines = [l.strip() for l in out.splitlines()]
    return sorted({l for l in lines if "python" in l and any(p in l for p in PROC_PATTERNS)})


def pipelines() -> dict:
    res = {}
    for f in glob.glob(str(WORKSPACE / "vintix_go2" / "*" / "status.json")):
        try:
            s = json.load(open(f))
            res[Path(f).parent.name] = {"done_total": s.get("done_total"), "note": s.get("note")}
        except Exception as e:
            res[Path(f).parent.name] = {"error": str(e)}
    for f in glob.glob(str(WORKSPACE / "vintix_go2" / "*" / "pipeline_status.json")):
        try:
            s = json.load(open(f))
            res[Path(f).parent.name] = {"stages": {k: v.get("done") for k, v in s.get("stages", {}).items()},
                                        "failed": sorted(s.get("failed", {}))}
        except Exception as e:
            res[Path(f).parent.name] = {"error": str(e)}
    return res


def git_heads() -> dict:
    return {"main": _run(["git", "rev-parse", "main"]).strip(), "HEAD": _run(["git", "rev-parse", "HEAD"]).strip()}


def main() -> None:
    state = {"github": github_state(), "processes": processes(), "pipelines": pipelines(), "git": git_heads()}
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]
    open_issues = [i for i in state["github"] if i.get("s") == "open"]
    waiting = [i["n"] for i in open_issues if "needs-decision" in i.get("l", [])]
    print(f"{digest} issues_open={len(open_issues)} needs_decision={waiting} "
          f"experiment_procs={len(state['processes'])} pipelines={json.dumps(state['pipelines'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
