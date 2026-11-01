"""Go1 env with a scaled body, built on the unmodified shared ``go1_env.Go1Env``.

env_cfg keys read here (absent -> nominal Go1):

- ``body_size_scale`` s: the Go1 URDF is loaded with Genesis ``scale=s`` (all lengths x s).
- ``body_mass_scale`` m: total mass = m x nominal URDF mass, independent of s.

Genesis' URDF scale also multiplies every link mass by s^3 and inertia by s^5, so after the
build every link mass is multiplied by m / s^3 (``set_mass`` rescales inertia by the same
ratio). Size-only scaling therefore keeps the nominal mass with inertia x s^2, and mass-only
scaling keeps geometry and the centre of mass.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

LOCOMOTION_DIR = Path(__file__).resolve().parents[2] / "Genesis" / "examples" / "locomotion"
if str(LOCOMOTION_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMOTION_DIR))

import genesis as gs  # noqa: E402
from go1_env import Go1Env  # noqa: E402


@contextlib.contextmanager
def _go1_urdf_scale(scale: float):
    """Inject ``scale`` into the Go1 robot URDF morph created inside ``Go1Env.__init__``."""
    original = gs.morphs.URDF

    # A subclass (not a function) so Genesis' isinstance(morph, gs.morphs.URDF) checks keep working.
    class ScaledURDF(original):
        def __init__(self, **data):
            if str(data.get("file", "")).endswith("go1.urdf"):
                data["scale"] = scale
            super().__init__(**data)

    gs.morphs.URDF = ScaledURDF
    try:
        yield
    finally:
        gs.morphs.URDF = original


class ScaledGo1Env(Go1Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, **kwargs):
        size_scale = float(env_cfg.get("body_size_scale", 1.0))
        mass_scale = float(env_cfg.get("body_mass_scale", 1.0))
        with _go1_urdf_scale(size_scale):
            super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=show_viewer, **kwargs)
        factor = mass_scale / size_scale**3
        if abs(factor - 1.0) > 1e-12:
            for link in self.robot.links:
                mass = link.get_mass()
                if mass > 0:
                    link.set_mass(mass * factor)
