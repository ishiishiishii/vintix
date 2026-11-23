#!/usr/bin/env python3
"""Laikago standing PNG matching ``LaikagoEnv.reset_idx`` pose (Y-up URDF).

Stock ``laikago.urdf`` is Y-up. Env applies ``base_init_quat = [0.5, 0.5, 0.5, 0.5]``
(Y90→X90) with ``base_init_pos z=0.50``. Do **not** use ``apply_standing_pose``
(assumes Z-up feet) — that leaves the robot inverted after the quat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import genesis as gs
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LOCO = ROOT / "Genesis" / "examples" / "locomotion"
sys.path.insert(0, str(LOCO))

URDF = ROOT / "Genesis/genesis/assets/urdf/laikago/urdf/laikago.urdf"
OUT = Path(__file__).resolve().parent / "figures" / "standing" / "laikago_standing.png"

JOINT_NAMES = [
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
]
DEFAULT_ANGLES = {n: 0.0 for n in JOINT_NAMES}
for prefix in ("FR", "FL", "RR", "RL"):
    DEFAULT_ANGLES[f"{prefix}_lower_leg_2_upper_leg_joint"] = -0.75

BASE_POS = [0.0, 0.0, 0.50]
BASE_QUAT_WXYZ = [0.5, 0.5, 0.5, 0.5]  # train.get_laikago_cfgs()


def main() -> int:
    if not gs._initialized:
        gs.init()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    # Match Env: spawn identity quat, then set_quat after build.
    ent = scene.add_entity(
        gs.morphs.URDF(
            file=str(URDF.resolve()),
            pos=BASE_POS,
            quat=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    cam = scene.add_camera(
        res=(1600, 900),
        pos=(2.8, -2.8, 1.6),
        lookat=(0.0, 0.0, 0.35),
        fov=42.0,
        GUI=False,
    )
    scene.build(n_envs=1)

    dof_idx = [ent.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES]
    dof_pos = torch.tensor(
        [DEFAULT_ANGLES[n] for n in JOINT_NAMES],
        device=gs.device,
        dtype=gs.tc_float,
    )
    ent.set_dofs_position(
        position=dof_pos.unsqueeze(0),
        dofs_idx_local=dof_idx,
        zero_velocity=True,
    )
    ent.set_pos(
        torch.tensor([BASE_POS], device=gs.device, dtype=gs.tc_float),
        zero_velocity=True,
        envs_idx=[0],
    )
    ent.set_quat(
        torch.tensor([BASE_QUAT_WXYZ], device=gs.device, dtype=gs.tc_float),
        envs_idx=[0],
        zero_velocity=True,
    )

    rgb = cam.render()[0]
    if hasattr(rgb, "cpu"):
        arr = rgb.cpu().numpy()
    else:
        arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, "laikago", (720, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), bgr)
    print(f"wrote {OUT} bytes={OUT.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
