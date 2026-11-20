#!/usr/bin/env python3
"""Prepare / optionally run standing renders for #16 newly introduced robots.

CPU dry-run validates URDF joint aliases (from JOINT_NAMING_QA) and writes a
manifest. ``--run`` launches Genesis renders and must only be used when GPU is free.

  cd /workspace
  python vintix_go2/robot_eval_expansion/prepare_standing_render.py          # dry-run
  python vintix_go2/robot_eval_expansion/prepare_standing_render.py --run    # GPU
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Reuse QA tables
from check_joint_names_cpu import ALIAS_HINTS, GO2_JOINTS, URDFS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "figures" / "standing"
MANIFEST = Path(__file__).resolve().parent / "standing_render_manifest.json"

# Unitree-family robots can use Go2 joint template; others need ALIAS_HINTS.
UNITREE_LIKE = {"go1", "go2", "aliengo", "b1"}
# Task-1 standing set (new / comparison robots).
RENDER_SET = ["aliengo", "b1", "anymalb", "anymalc", "spot", "laikago"]


def _joint_list(robot: str) -> list[str]:
    if robot in UNITREE_LIKE:
        return list(GO2_JOINTS)
    hint = ALIAS_HINTS.get(robot)
    if not hint:
        raise KeyError(f"no joint alias for {robot}")
    return list(hint["joints"])


def _validate(robot: str) -> dict:
    path = URDFS[robot]
    if not path.is_file():
        return {"robot": robot, "ok": False, "error": f"missing {path}"}
    root = ET.parse(path).getroot()
    urdf_joints = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    names = _joint_list(robot)
    missing = [n for n in names if n not in urdf_joints]
    return {
        "robot": robot,
        "ok": not missing,
        "urdf": str(path),
        "joint_names": names,
        "n_joints": len(names),
        "missing_in_urdf": missing,
        "needs_explicit_joint_map": robot not in UNITREE_LIKE,
        "out_png": str(OUT_DIR / f"{robot}_standing.png"),
    }


def dry_run() -> int:
    rows = [_validate(r) for r in RENDER_SET]
    MANIFEST.write_text(json.dumps({"robots": rows, "mode": "dry_run"}, indent=2) + "\n")
    ok = all(r["ok"] for r in rows)
    print(json.dumps({"ok": ok, "manifest": str(MANIFEST), "n": len(rows)}, indent=2))
    for r in rows:
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  [{flag}] {r['robot']:8s} joints={r.get('n_joints')} "
              f"explicit_map={r.get('needs_explicit_joint_map')} "
              f"missing={r.get('missing_in_urdf')}")
    return 0 if ok else 1


def run_renders() -> int:
    """GPU path: standing PNG for Unitree-like robots (Go2 default angles).

    Non-Unitree robots need per-Env default_joint_angles; those are left for a
    follow-up once angles are extracted from Env cfgs (joint aliases are ready).
    """
    rows = [_validate(r) for r in RENDER_SET]
    if not all(r["ok"] for r in rows):
        print("validation failed; refusing --run", file=sys.stderr)
        return 1

    runnable = [r for r in rows if r["robot"] in UNITREE_LIKE]
    deferred = [r["robot"] for r in rows if r["robot"] not in UNITREE_LIKE]
    if deferred:
        print(f"[defer] need Env default angles: {deferred}", file=sys.stderr)

    sys.path.insert(0, str(ROOT / "Genesis" / "examples" / "locomotion"))
    from render_robot_comparison import RobotRenderSpec, render_robots  # noqa: WPS433

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in runnable:
        print(f"[render] {r['robot']} → {r['out_png']}")
        spec = RobotRenderSpec(label=r["robot"], urdf_path=r["urdf"])
        render_robots(
            [spec],
            Path(r["out_png"]),
            joint_names=r["joint_names"],
        )
    MANIFEST.write_text(json.dumps({
        "robots": rows,
        "mode": "run",
        "rendered": [r["robot"] for r in runnable],
        "deferred_need_default_angles": deferred,
    }, indent=2) + "\n")
    print(f"done → {OUT_DIR} (rendered {len(runnable)}, deferred {len(deferred)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Launch Genesis standing renders (GPU)")
    args = ap.parse_args()
    if args.run:
        return run_renders()
    return dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
