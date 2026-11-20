#!/usr/bin/env python3
"""CPU: fill URMA description slots that can be read from Go2 URDF (no MuJoCo/RL-X).

Official URMA builds per-joint description (23) and per-foot (10) from MuJoCo.
This script extracts the subset available in the Genesis Go2 URDF:
  - link masses
  - revolute joint lower/upper limits
and writes a JSON map for later wiring into ``obs_adapter``.

  cd /workspace/vintix_go2/general_locomotion_baselines
  python wrappers/urma/extract_go2_urdf_fields.py
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from joint_description import FOOT_DESC_LAYOUT, JOINT_DESC_LAYOUT

ROOT = Path(__file__).resolve().parents[4]
URDF = ROOT / "Genesis" / "genesis" / "assets" / "urdf" / "go2" / "urdf" / "go2.urdf"
OUT = Path(__file__).resolve().parent / "go2_urdf_description_fields.json"

# Genesis Go2 joint order (matches loco env / GO2_JOINT_NAMES).
GO2_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# Child link of each joint (URDF child) — used for per-joint mass proxy.
GO2_FEET = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


def _masses(root: ET.Element) -> dict[str, float]:
    out: dict[str, float] = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        if mass_el is None:
            continue
        out[name] = float(mass_el.get("value", "0"))
    return out


def _joint_limits(root: ET.Element) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for j in root.findall("joint"):
        name = j.get("name")
        lim = j.find("limit")
        if lim is None or name is None:
            continue
        lo = lim.get("lower")
        hi = lim.get("upper")
        if lo is None or hi is None:
            continue
        out[name] = (float(lo), float(hi))
    return out


def _child_link(root: ET.Element) -> dict[str, str]:
    return {
        j.get("name"): (j.find("child").get("link") if j.find("child") is not None else "")
        for j in root.findall("joint")
        if j.get("name")
    }


def main() -> int:
    if not URDF.is_file():
        raise SystemExit(f"missing {URDF}")
    root = ET.parse(URDF).getroot()
    masses = _masses(root)
    limits = _joint_limits(root)
    child = _child_link(root)
    total_mass = sum(masses.values())

    fillable_joint = {"mass", "joint_range"}
    fillable_foot = {"mass"}
    joint_slots = [n for n, _ in JOINT_DESC_LAYOUT]
    foot_slots = [n for n, _ in FOOT_DESC_LAYOUT]

    joints = []
    for jn in GO2_JOINTS:
        cl = child.get(jn, "")
        lo_hi = limits.get(jn)
        joints.append({
            "joint": jn,
            "child_link": cl,
            "mass_kg": masses.get(cl),
            "joint_range": list(lo_hi) if lo_hi else None,
            "urdf_fillable_slots": sorted(fillable_joint),
            "still_need_mujoco_slots": [s for s in joint_slots if s not in fillable_joint],
        })

    feet = []
    for fn in GO2_FEET:
        feet.append({
            "foot_link": fn,
            "mass_kg": masses.get(fn),
            "urdf_fillable_slots": sorted(fillable_foot),
            "still_need_mujoco_slots": [s for s in foot_slots if s not in fillable_foot],
        })

    payload = {
        "urdf": str(URDF),
        "total_mass_kg": total_mass,
        "n_links_with_mass": len(masses),
        "note": (
            "Subset of URMA description; rel_pos/axis/PD/torque etc. still need "
            "MuJoCo or Env-side values. Use to replace placeholder mass/range in obs_adapter."
        ),
        "joints": joints,
        "feet": feet,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    missing_j = [j["joint"] for j in joints if j["mass_kg"] is None or j["joint_range"] is None]
    missing_f = [f["foot_link"] for f in feet if f["mass_kg"] is None]
    print(json.dumps({
        "out": str(OUT),
        "total_mass_kg": round(total_mass, 3),
        "joints_ok": len(GO2_JOINTS) - len(missing_j),
        "feet_ok": len(GO2_FEET) - len(missing_f),
        "missing_joints": missing_j,
        "missing_feet": missing_f,
    }, indent=2))
    return 0 if not missing_j and not missing_f else 1


if __name__ == "__main__":
    raise SystemExit(main())
