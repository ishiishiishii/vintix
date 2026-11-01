"""Go1 body-scale variants: sweep grids and scaled Genesis cfgs.

The physical scaling itself is done by ``scaled_go1_env.ScaledGo1Env`` from two env_cfg keys:

- ``body_mass_scale``: every link mass/inertia x m (geometry unchanged).
- ``body_size_scale``: URDF loaded with Genesis ``scale=s`` (all lengths x s) while the
  total mass stays nominal (inertia x s^2).

This module only builds the cfg dicts for one variant from the Go1 expert's ``cfgs.pkl``.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

WORKSPACE = Path(__file__).resolve().parents[2]
GO1_LOG_DIR = WORKSPACE / "Genesis" / "logs" / "go1-walking"
GO1_EXPERT_CKPT = GO1_LOG_DIR / "model_2000.pt"
GO1_WITHOUT_CKPT = WORKSPACE / "vintix_go2" / "models" / "go1_without" / "go1_without" / "0001_epoch"

GO1_TOTAL_MASS_KG = 13.100528
GO1_THIGH_LENGTH_M = 0.213

AXES = ("mass", "size")

MASS_SCALES = (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
SIZE_SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0)

# Existing robots expressed as Go1 ratios (robot_specifications_table.md), for plot markers.
REFERENCE_ROBOTS = {
    "mass": {"SpotMicro": 4.5 / 13.1, "Mini Cheetah": 8.3 / 13.1, "A1": 13.7 / 13.1,
             "Go2": 15.0 / 13.1, "ANYmal C": 52.1 / 13.1},
    "size": {"SpotMicro": 0.12 / 0.21, "ANYmal C": 0.28 / 0.21},
}


def scale_tag(scale: float) -> str:
    return f"s{scale:.3f}".replace(".", "p")


def load_go1_cfgs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    with open(GO1_LOG_DIR / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def scaled_go1_cfgs(axis: str, scale: float):
    """Return (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg) for one Go1 variant.

    Controller, torque limits, reward weights and the velocity command stay at the Go1 values.
    On the size axis the initial height and the height targets of the reward scale with s so a
    larger body standing normally is not penalized for its height.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = (copy.deepcopy(c) for c in load_go1_cfgs())

    if axis == "mass":
        env_cfg["body_mass_scale"] = float(scale)
    else:
        env_cfg["body_size_scale"] = float(scale)
        pos = list(env_cfg["base_init_pos"])
        pos[2] = pos[2] * scale
        env_cfg["base_init_pos"] = pos
        reward_cfg["base_height_target"] = reward_cfg["base_height_target"] * scale
        reward_cfg["feet_height_target"] = reward_cfg["feet_height_target"] * scale
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def physical_summary(axis: str, scale: float) -> Dict[str, float]:
    """Nominal (intended) body numbers for one variant."""
    if axis == "mass":
        return {"total_mass_kg": GO1_TOTAL_MASS_KG * scale, "thigh_length_m": GO1_THIGH_LENGTH_M}
    return {"total_mass_kg": GO1_TOTAL_MASS_KG, "thigh_length_m": GO1_THIGH_LENGTH_M * scale}
