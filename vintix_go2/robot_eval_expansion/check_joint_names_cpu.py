#!/usr/bin/env python3
"""CPU-only joint / foot naming QA for #16 standing-render prep.

Does not import Genesis or touch the GPU. Compares each introduced URDF against
the Go2-style names used by ``robot_standing_pose.py`` / ``render_robot_comparison.py``.

  cd /workspace
  python vintix_go2/robot_eval_expansion/check_joint_names_cpu.py
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URDF_ROOT = ROOT / "Genesis" / "genesis" / "assets" / "urdf"

GO2_JOINTS = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
GO2_FEET = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

URDFS: dict[str, Path] = {
    "go1": URDF_ROOT / "go1/urdf/go1.urdf",
    "go2": URDF_ROOT / "go2/urdf/go2.urdf",
    "aliengo": URDF_ROOT / "aliengo/urdf/aliengo.urdf",
    "b1": URDF_ROOT / "b1/xacro/b1.urdf",
    "anymalb": URDF_ROOT / "anymal_b/urdf/anymal.urdf",
    "anymalc": URDF_ROOT / "anymal_c/urdf/anymal_c.urdf",
    "spot": URDF_ROOT / "spot/urdf/spot_simple.urdf",
    "laikago": URDF_ROOT / "laikago/urdf/laikago.urdf",
}

# Known aliases for non-Unitree naming (standing render must pass these explicitly).
ALIAS_HINTS: dict[str, dict[str, list[str]]] = {
    "anymalb": {
        "joints": ["LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
                   "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"],
        "feet": ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    },
    "anymalc": {
        "joints": ["LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
                   "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"],
        "feet": ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    },
    "spot": {
        "joints": ["fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
                   "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn"],
        "feet": ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"],
    },
    "laikago": {
        "joints": [
            "FR_hip_motor_2_chassis_joint",
            "FR_upper_leg_2_hip_motor_joint",
            "FR_lower_leg_2_upper_leg_joint",
            "FL_hip_motor_2_chassis_joint",
            "FL_upper_leg_2_hip_motor_joint",
            "FL_lower_leg_2_upper_leg_joint",
            "RR_hip_motor_2_chassis_joint",
            "RR_upper_leg_2_hip_motor_joint",
            "RR_lower_leg_2_upper_leg_joint",
            "RL_hip_motor_2_chassis_joint",
            "RL_upper_leg_2_hip_motor_joint",
            "RL_lower_leg_2_upper_leg_joint",
        ],
        "feet": ["FR_lower_leg", "FL_lower_leg", "RR_lower_leg", "RL_lower_leg"],
    },
}


def _names(urdf: Path) -> tuple[list[str], list[str]]:
    root = ET.parse(urdf).getroot()
    joints = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    links = [l.get("name") for l in root.findall("link")]
    return joints, links


def main() -> int:
    rows = []
    for name, path in URDFS.items():
        if not path.is_file():
            rows.append({"robot": name, "ok": False, "error": f"missing {path}"})
            continue
        joints, links = _names(path)
        go2_joint_ok = all(j in joints for j in GO2_JOINTS)
        go2_foot_ok = all(f in links for f in GO2_FEET)
        alias = ALIAS_HINTS.get(name)
        alias_ok = None
        if alias:
            alias_ok = all(j in joints for j in alias["joints"]) and all(
                f in links for f in alias["feet"]
            )
        rows.append(
            {
                "robot": name,
                "path": str(path.relative_to(ROOT)),
                "n_revolute_joints": len(joints),
                "go2_joint_names_ok": go2_joint_ok,
                "go2_foot_links_ok": go2_foot_ok,
                "render_default_ok": go2_joint_ok and go2_foot_ok,
                "alias_scheme": None if alias is None else ("ok" if alias_ok else "mismatch"),
                "needs_explicit_joint_map": not (go2_joint_ok and go2_foot_ok),
            }
        )

    out = {
        "ok": True,
        "go2_joint_template": GO2_JOINTS,
        "robots": rows,
        "standing_render_note": (
            "Robots with needs_explicit_joint_map=true must pass joint_names / "
            "default_joint_angles into render_robot_comparison (Env already has them)."
        ),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    md = Path(__file__).resolve().parent / "JOINT_NAMING_QA.md"
    lines = [
        "# 関節・足リンク命名 QA（#16・CPU）",
        "",
        "立位レンダ前の準備。Genesis / GPU は使わない。",
        "",
        "| ロボット | Go2 名でそのまま描ける | 別名スキーム | 備考 |",
        "|----------|------------------------|--------------|------|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['robot']} | — | — | {r['error']} |")
            continue
        go2 = "yes" if r["render_default_ok"] else "**no**"
        alias = r["alias_scheme"] or "（Unitree 系）"
        note = "Env の joint_names を render に渡す" if r["needs_explicit_joint_map"] else "GO2 既定で可"
        lines.append(f"| {r['robot']} | {go2} | {alias} | {note} |")
    lines += [
        "",
        "## レンダ時の注意",
        "",
        "- **Unitree 系（go1/go2/aliengo/b1）:** `GO2_JOINT_NAMES` で立位可",
        "- **ANYmal-B/C:** `LF/RF/LH/RH` × `HAA/HFE/KFE`（Env 既存）",
        "- **Spot:** `fl/fr/hl/hr` × `hx/hy/kn`（足リンクは `*_lleg`）",
        "- **Laikago:** `*_hip_motor_2_chassis_joint` 等の長い名前",
        "",
        "再現: `python vintix_go2/robot_eval_expansion/check_joint_names_cpu.py`",
        "",
    ]
    md.write_text("\n".join(lines))
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
