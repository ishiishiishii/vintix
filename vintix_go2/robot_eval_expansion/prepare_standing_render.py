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

# Standing-render URDF overrides (Env may still use a different file).
# Laikago: stock URDF is Y-up; use ``render_laikago_standing.py`` (Env quat)
# instead of toes_zup (camera/pose empty in practice). Keep overrides empty.
URDF_OVERRIDES: dict[str, Path] = {}

# Standing angles from Genesis/examples/locomotion/train.py get_*_cfgs()
# (snapshot; keep in sync if train defaults change). Used for non-Unitree --run.
STANDING_ANGLES: dict[str, dict[str, float]] = {
    "anymalb": {
        "LF_HAA": -0.3, "LF_HFE": 0.8, "LF_KFE": -1.3,
        "RF_HAA": 0.3, "RF_HFE": 0.8, "RF_KFE": -1.3,
        "LH_HAA": -0.3, "LH_HFE": -0.8, "LH_KFE": 1.3,
        "RH_HAA": 0.3, "RH_HFE": -0.8, "RH_KFE": 1.3,
    },
    "anymalc": {
        "LF_HAA": -0.3, "LF_HFE": 0.8, "LF_KFE": -1.3,
        "RF_HAA": 0.3, "RF_HFE": 0.8, "RF_KFE": -1.3,
        "LH_HAA": -0.3, "LH_HFE": -0.8, "LH_KFE": 1.3,
        "RH_HAA": 0.3, "RH_HFE": -0.8, "RH_KFE": 1.3,
    },
    "spot": {
        "fl_hx": 0.0, "fr_hx": 0.0, "hl_hx": 0.0, "hr_hx": 0.0,
        "fl_hy": 0.9, "fr_hy": 0.9, "hl_hy": 0.9, "hr_hy": 0.9,
        "fl_kn": -1.5, "fr_kn": -1.5, "hl_kn": -1.5, "hr_kn": -1.5,
    },
    "laikago": {
        "FR_hip_motor_2_chassis_joint": 0.0,
        "FR_upper_leg_2_hip_motor_joint": 0.0,
        "FR_lower_leg_2_upper_leg_joint": -0.75,
        "FL_hip_motor_2_chassis_joint": 0.0,
        "FL_upper_leg_2_hip_motor_joint": 0.0,
        "FL_lower_leg_2_upper_leg_joint": -0.75,
        "RR_hip_motor_2_chassis_joint": 0.0,
        "RR_upper_leg_2_hip_motor_joint": 0.0,
        "RR_lower_leg_2_upper_leg_joint": -0.75,
        "RL_hip_motor_2_chassis_joint": 0.0,
        "RL_upper_leg_2_hip_motor_joint": 0.0,
        "RL_lower_leg_2_upper_leg_joint": -0.75,
    },
}


def _joint_list(robot: str) -> list[str]:
    if robot in UNITREE_LIKE:
        return list(GO2_JOINTS)
    hint = ALIAS_HINTS.get(robot)
    if not hint:
        raise KeyError(f"no joint alias for {robot}")
    return list(hint["joints"])


def _urdf_path(robot: str) -> Path:
    return URDF_OVERRIDES.get(robot, URDFS[robot])


def _validate(robot: str) -> dict:
    path = _urdf_path(robot)
    if not path.is_file():
        return {"robot": robot, "ok": False, "error": f"missing {path}"}
    root = ET.parse(path).getroot()
    urdf_joints = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    names = _joint_list(robot)
    missing = [n for n in names if n not in urdf_joints]
    angles = STANDING_ANGLES.get(robot)
    angle_missing = []
    if angles is not None:
        angle_missing = [n for n in names if n not in angles]
    return {
        "robot": robot,
        "ok": not missing and not angle_missing,
        "urdf": str(path),
        "urdf_override": robot in URDF_OVERRIDES,
        "joint_names": names,
        "n_joints": len(names),
        "missing_in_urdf": missing,
        "missing_standing_angles": angle_missing,
        "has_standing_angles": angles is not None and not angle_missing,
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


def run_renders(*, only: list[str] | None = None, force: bool = False) -> int:
    """GPU path: standing PNG for all RENDER_SET robots with known angles.

    Unitree-like robots use Go2 default angles in render_robot_comparison.
    Others pass STANDING_ANGLES from train.py get_*_cfgs().
    """
    targets = list(RENDER_SET) if not only else list(only)
    unknown = [r for r in targets if r not in RENDER_SET]
    if unknown:
        print(f"unknown robots (not in RENDER_SET): {unknown}", file=sys.stderr)
        return 1

    rows = [_validate(r) for r in targets]
    if not all(r["ok"] for r in rows):
        print("validation failed; refusing --run", file=sys.stderr)
        return 1

    runnable = [r for r in rows if r.get("has_standing_angles") or r["robot"] in UNITREE_LIKE]
    deferred = [r["robot"] for r in rows if r["robot"] not in {x["robot"] for x in runnable}]
    if deferred:
        print(f"[defer] need standing angles: {deferred}", file=sys.stderr)

    sys.path.insert(0, str(ROOT / "Genesis" / "examples" / "locomotion"))
    from render_robot_comparison import RobotRenderSpec, render_robots  # noqa: WPS433

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    skipped_existing: list[str] = []
    failed: dict[str, str] = {}
    for r in runnable:
        out = Path(r["out_png"])
        if (not force) and out.is_file() and out.stat().st_size > 0:
            print(f"[skip] {r['robot']} already has {out}")
            skipped_existing.append(r["robot"])
            rendered.append(r["robot"])
            continue
        if force and out.is_file():
            out.unlink()
            print(f"[force] removed {out}")
        print(f"[render] {r['robot']} → {out}")
        spec = RobotRenderSpec(label=r["robot"], urdf_path=r["urdf"])
        kwargs = {"joint_names": r["joint_names"]}
        if r["robot"] in STANDING_ANGLES:
            kwargs["default_joint_angles"] = STANDING_ANGLES[r["robot"]]
        try:
            render_robots([spec], out, **kwargs)
            rendered.append(r["robot"])
        except Exception as exc:  # noqa: BLE001 — keep remaining robots going
            failed[r["robot"]] = str(exc)
            print(f"[fail] {r['robot']}: {exc}", file=sys.stderr)
    MANIFEST.write_text(json.dumps({
        "robots": rows,
        "mode": "run",
        "force": force,
        "only": targets,
        "rendered": rendered,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "deferred_need_default_angles": deferred,
        "standing_angles_source": "train.py get_*_cfgs (snapshot in STANDING_ANGLES)",
        "urdf_overrides": {k: str(v) for k, v in URDF_OVERRIDES.items()},
    }, indent=2) + "\n")
    print(
        f"done → {OUT_DIR} (rendered {len(rendered)}, "
        f"failed {len(failed)}, deferred {len(deferred)})"
    )
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Launch Genesis standing renders (GPU)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if PNG already exists",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Subset of RENDER_SET (e.g. aliengo laikago)",
    )
    args = ap.parse_args()
    if args.run:
        return run_renders(only=args.only, force=args.force)
    return dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
