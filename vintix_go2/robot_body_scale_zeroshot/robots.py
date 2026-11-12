"""Per-robot specs, sweep grids and scaled cfgs for the multi-robot body-scale study.

Targets: Go2, A1, Mini Cheetah (all three axes) and Go1 (density-preserving size axis only;
Go1's mass and fixed-mass size axes are in ``vintix_go2/go1_body_scale_zeroshot``).

Axes (scale x relative to the robot's own nominal body):
- ``mass``         every link mass/inertia x m, geometry unchanged
- ``size``         all lengths x s, total mass kept at nominal (inertia x s^2)
- ``size_density`` all lengths x s at constant density -> mass x s^3, inertia x s^5
On both size axes the initial height and the reward height targets scale with s.
"""
from __future__ import annotations

import copy
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Tuple

WORKSPACE = Path(__file__).resolve().parents[2]
LOGS = WORKSPACE / "Genesis" / "logs"
ASSETS = WORKSPACE / "Genesis" / "genesis" / "assets" / "urdf"
MODELS = WORKSPACE / "vintix_go2" / "models"

# cfg_exp mirrors save_vintix.get_exp_name_for_robot (cfgs.pkl if present, else train.get_*_cfgs),
# expert_* mirrors save_expert.get_default_expert_model_path, so the nominal body reproduces the
# ISCIIA zero-shot / expert numbers.
ROBOTS: Dict[str, Dict[str, Any]] = {
    "go2": {
        "robot_type": "go2", "env_attr": "Go2Env", "urdf": ASSETS / "go2" / "urdf" / "go2.urdf",
        "cfg_exp": "go2-walking", "cfg_fn": "get_go2_cfgs",
        "expert_exp": "go2-walking", "expert_ckpt": LOGS / "go2-walking" / "model_300.pt",
        "vintix": MODELS / "go2_without" / "go2_without" / "0001_epoch", "thigh_m": 0.213,
    },
    "a1": {
        "robot_type": "a1", "env_attr": "A1Env", "urdf": ASSETS / "a1" / "urdf" / "a1.urdf",
        "cfg_exp": "a1-walking", "cfg_fn": "get_unitreea1_cfgs",
        "expert_exp": "a1-walking", "expert_ckpt": LOGS / "a1-walking" / "model_350.pt",
        "vintix": MODELS / "a1_without" / "a1_without" / "0001_epoch", "thigh_m": 0.20,
    },
    "minicheetah": {
        "robot_type": "minicheetah", "env_attr": "MiniCheetahEnv",
        "urdf": ASSETS / "mini_cheetah" / "urdf" / "mini_cheetah.urdf",
        "cfg_exp": "minicheetah-walking2", "cfg_fn": "get_minicheetah_cfgs",
        "expert_exp": "minicheetah-walking", "expert_ckpt": LOGS / "minicheetah-walking" / "model_2990.pt",
        "vintix": MODELS / "minicheetah_without" / "minicheetah_without" / "0001_epoch", "thigh_m": 0.21,
    },
    "go1": {
        "robot_type": "go1", "env_attr": "Go1Env", "urdf": ASSETS / "go1" / "urdf" / "go1.urdf",
        "cfg_exp": "go1-walking", "cfg_fn": "get_go1_cfgs",
        "expert_exp": "go1-walking", "expert_ckpt": LOGS / "go1-walking" / "model_2000.pt",
        "vintix": MODELS / "go1_without" / "go1_without" / "0001_epoch", "thigh_m": 0.213,
    },
}

AXES = ("mass", "size", "size_density")
GRIDS = {
    "mass": (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0),
    "size": (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0),
    "size_density": (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5),
}
# Go1 mass / size are already done in go1_body_scale_zeroshot; only the density axis is new.
ROBOT_AXES = {"go2": AXES, "a1": AXES, "minicheetah": AXES, "go1": ("size_density",)}
EVAL_SEEDS = tuple(range(10))

# PPO from scratch at the extremes, to tell "the policy cannot" from "the body cannot".
PPO_GRID = {"mass": (0.5, 2.0), "size": (0.6, 1.3), "size_density": (0.6, 1.3)}
PPO_ITERATIONS = 500
PPO_TRAIN_SEED = 1


def scale_tag(scale: float) -> str:
    return f"s{scale:.3f}".replace(".", "p")


def nominal_point(axis: str, scale: float) -> bool:
    """The unmodified body is evaluated once (under ``mass``) and shared by all axes."""
    return abs(scale - 1.0) < 1e-9


def canonical_axis(axis: str, scale: float) -> str:
    return "mass" if nominal_point(axis, scale) else axis


def urdf_total_mass(robot: str) -> float:
    root = ET.parse(ROBOTS[robot]["urdf"]).getroot()
    return sum(float(m.get("value", 0)) for m in root.iter("mass"))


def body_scales(axis: str, scale: float) -> Tuple[float, float]:
    """(body_size_scale, body_mass_scale) for ScaledEnv."""
    if axis == "mass":
        return 1.0, scale
    if axis == "size":
        return scale, 1.0
    if axis == "size_density":
        return scale, scale**3
    raise ValueError(axis)


def physical_summary(robot: str, axis: str, scale: float) -> Dict[str, float]:
    size, mass = body_scales(axis, scale)
    return {"total_mass_kg": urdf_total_mass(robot) * mass, "thigh_length_m": ROBOTS[robot]["thigh_m"] * size,
            "size_scale": size, "mass_scale": mass}


def load_cfgs(robot: str):
    spec = ROBOTS[robot]
    path = LOGS / spec["cfg_exp"] / "cfgs.pkl"
    if path.exists():
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(path, "rb"))
    else:
        import train  # Genesis/examples/locomotion/train.py (same fallback as save_vintix)

        env_cfg, obs_cfg, reward_cfg, command_cfg = getattr(train, spec["cfg_fn"])()
    _, _, _, _, train_cfg = pickle.load(open(LOGS / spec["expert_exp"] / "cfgs.pkl", "rb"))
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def scaled_cfgs(robot: str, axis: str, scale: float):
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = (copy.deepcopy(c) for c in load_cfgs(robot))
    size, mass = body_scales(axis, scale)
    env_cfg["body_size_scale"] = size
    env_cfg["body_mass_scale"] = mass
    if not math.isclose(size, 1.0):
        pos = list(env_cfg["base_init_pos"])
        pos[2] *= size
        env_cfg["base_init_pos"] = pos
        for key in ("base_height_target", "feet_height_target"):
            if key in reward_cfg:
                reward_cfg[key] *= size
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg
