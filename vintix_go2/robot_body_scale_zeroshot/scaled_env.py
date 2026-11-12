"""Robot envs with a scaled body, built on the unmodified shared env classes.

env_cfg keys (absent -> nominal body):
- ``body_size_scale`` s: the robot URDF is loaded with Genesis ``scale=s`` (all lengths x s)
- ``body_mass_scale`` m: total mass = m x nominal URDF mass, independent of s

Genesis' URDF scale already multiplies link masses by s^3 and inertia by s^5, so after the
build every link mass is multiplied by m / s^3 (``set_mass`` rescales inertia by the same ratio).
m = s^3 therefore keeps density constant, m = 1 keeps the nominal mass.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

LOCOMOTION_DIR = Path(__file__).resolve().parents[2] / "Genesis" / "examples" / "locomotion"
if str(LOCOMOTION_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMOTION_DIR))

import genesis as gs  # noqa: E402


@contextlib.contextmanager
def _urdf_scale(urdf_name: str, scale: float):
    original = gs.morphs.URDF

    # A subclass (not a function) so Genesis' isinstance(morph, gs.morphs.URDF) checks keep working.
    class ScaledURDF(original):
        def __init__(self, **data):
            if str(data.get("file", "")).endswith(urdf_name):
                data["scale"] = scale
            super().__init__(**data)

    gs.morphs.URDF = ScaledURDF
    try:
        yield
    finally:
        gs.morphs.URDF = original


def scaled_env_class(base_cls, urdf_name: str):
    class ScaledEnv(base_cls):
        def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, **kwargs):
            size = float(env_cfg.get("body_size_scale", 1.0))
            mass = float(env_cfg.get("body_mass_scale", 1.0))
            with _urdf_scale(urdf_name, size):
                super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=show_viewer, **kwargs)
            factor = mass / size**3
            if abs(factor - 1.0) > 1e-12:
                for link in self.robot.links:
                    m = link.get_mass()
                    if m > 0:
                        link.set_mass(m * factor)

    ScaledEnv.__name__ = f"Scaled{base_cls.__name__}"
    return ScaledEnv
