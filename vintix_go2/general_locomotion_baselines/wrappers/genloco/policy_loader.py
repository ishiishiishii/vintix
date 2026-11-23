"""Inspect / smoke-check GenLoco Stable-Baselines PPO zip without PyBullet.

Loads `morphology_generator_*.zip` NPZ weights and runs a minimal actor MLP
(451 → 1024 → 512 → 12, ReLU) that matches `policy_kwargs.net_arch`.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .obs_adapter import GENLOCO_OBS_DIM, GenesisToGenlocoObs

DEFAULT_CKPT = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "GenLoco"
    / "motion_imitation"
    / "data"
    / "policies"
    / "morphology_generator_pace_model.zip"
)

POLICY_ZIPS = (
    "morphology_generator_pace_model.zip",
    "morphology_generator_spin_model.zip",
    "morphology_generator_inverse.zip",
)


@dataclass(frozen=True)
class GenlocoActorDims:
    obs_dim: int = GENLOCO_OBS_DIM  # 451
    hidden0: int = 1024
    hidden1: int = 512
    action_dim: int = 12


def _open_policy_zip(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"GenLoco policy zip not found: {path}\n"
            "Run: bash vintix_go2/general_locomotion_baselines/clone_third_party.sh"
        )
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("data"))
        params = dict(np.load(io.BytesIO(zf.read("parameters"))))
    return params, meta


def load_weight_shapes(ckpt_path: str | Path | None = None) -> dict[str, tuple[int, ...]]:
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    params, _ = _open_policy_zip(path)
    keys = (
        "model/pi_fc0/w:0",
        "model/pi_fc0/b:0",
        "model/pi_fc1/w:0",
        "model/pi_fc1/b:0",
        "model/pi/w:0",
        "model/pi/b:0",
    )
    out = {}
    for key in keys:
        if key not in params:
            raise KeyError(f"missing {key} in {path}")
        out[key] = tuple(params[key].shape)
    return out


def assert_dims_match_checkpoint(ckpt_path: str | Path | None = None) -> GenlocoActorDims:
    shapes = load_weight_shapes(ckpt_path)
    dims = GenlocoActorDims()
    w0 = shapes["model/pi_fc0/w:0"]
    if w0 != (dims.obs_dim, dims.hidden0):
        raise AssertionError(f"pi_fc0 {w0} != {(dims.obs_dim, dims.hidden0)}")
    if shapes["model/pi_fc1/w:0"] != (dims.hidden0, dims.hidden1):
        raise AssertionError(shapes["model/pi_fc1/w:0"])
    if shapes["model/pi/w:0"] != (dims.hidden1, dims.action_dim):
        raise AssertionError(shapes["model/pi/w:0"])
    return dims


class GenlocoActor(nn.Module):
    """Minimal mean-actor path (ignores value / log-std heads)."""

    def __init__(self, params: dict[str, np.ndarray]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GENLOCO_OBS_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 12),
        )
        self.net[0].weight.data.copy_(torch.from_numpy(params["model/pi_fc0/w:0"].T))
        self.net[0].bias.data.copy_(torch.from_numpy(params["model/pi_fc0/b:0"]))
        self.net[2].weight.data.copy_(torch.from_numpy(params["model/pi_fc1/w:0"].T))
        self.net[2].bias.data.copy_(torch.from_numpy(params["model/pi_fc1/b:0"]))
        self.net[4].weight.data.copy_(torch.from_numpy(params["model/pi/w:0"].T))
        self.net[4].bias.data.copy_(torch.from_numpy(params["model/pi/b:0"]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GenlocoActorPolicy(nn.Module):
    """Mean-actor wrapper for Genesis eval (loads zip → GenlocoActor)."""

    def __init__(self, ckpt_path: str | Path | None = None, device: str | torch.device = "cpu"):
        super().__init__()
        path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
        self.dims = assert_dims_match_checkpoint(path)
        params, _ = _open_policy_zip(path)
        self.actor = GenlocoActor(params)
        self.device = torch.device(device)
        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def act(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """(B, 451) → (B, 12) actions."""
        return self.actor(obs_flat.to(self.device))


def smoke_forward_shapes(
    ckpt_path: str | Path | None = None,
    batch_size: int = 2,
    steps: int = 5,
) -> dict[str, tuple[int, ...]]:
    dims = assert_dims_match_checkpoint(ckpt_path)
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    params, _ = _open_policy_zip(path)
    adapter = GenesisToGenlocoObs(batch_size=batch_size, device="cpu")
    actor = GenlocoActor(params).eval()

    with torch.no_grad():
        last_action = None
        for _ in range(steps):
            fake = torch.randn(batch_size, 45)
            fake[:, 3:6] = torch.nn.functional.normalize(fake[:, 3:6], dim=-1)
            fake[:, 6:9] = 0.1 * torch.randn(batch_size, 3)
            packed = adapter.step(fake)
            if packed["obs_flat"].shape[-1] != dims.obs_dim:
                raise AssertionError(packed["obs_flat"].shape)
            last_action = actor(packed["obs_flat"])
            if last_action.shape != (batch_size, dims.action_dim):
                raise AssertionError(last_action.shape)
            if not torch.isfinite(last_action).all():
                raise AssertionError("non-finite actions")

    return {
        "obs_flat": tuple(packed["obs_flat"].shape),
        "action": tuple(last_action.shape),
    }


def assert_all_policy_zips_match() -> list[str]:
    """Sanity: pace / spin / inverse share the same actor dims."""
    root = DEFAULT_CKPT.parent
    checked = []
    for name in POLICY_ZIPS:
        assert_dims_match_checkpoint(root / name)
        checked.append(name)
    return checked
