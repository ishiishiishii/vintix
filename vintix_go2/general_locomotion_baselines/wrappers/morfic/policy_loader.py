"""Inspect / smoke-check MorFiC `ac_weights_last.pt` without Isaac Gym.

Does not reconstruct the full AC (FiLM/hypernet critic). Only verifies that
history / morphology input dims match the pretrained tensors, and that a
minimal actor-body path accepts concatenated features of the right size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .morphology import go2_nominal_morphology, normalize_morphology
from .obs_adapter import (
    MORFIC_HISTORY_FLAT,
    MORFIC_HISTORY_LEN,
    MORFIC_OBS_DIM,
    GenesisToMorficObs,
)

DEFAULT_CKPT = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "MorFiC"
    / "runs"
    / "gait-conditioned-agility"
    / "2026-05-15"
    / "train"
    / "003739.836224"
    / "checkpoints"
    / "ac_weights_last.pt"
)


@dataclass(frozen=True)
class MorficActorDims:
    obs_dim: int = MORFIC_OBS_DIM
    history_len: int = MORFIC_HISTORY_LEN
    history_flat: int = MORFIC_HISTORY_FLAT  # 2100
    morph_raw: int = 11
    morph_latent: int = 64
    adapt_latent: int = 2
    actor_in: int = 2166  # 2100 + 64 + 2
    action_dim: int = 12


def load_ac_weight_shapes(ckpt_path: str | Path | None = None) -> dict[str, tuple[int, ...]]:
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not path.is_file():
        raise FileNotFoundError(
            f"MorFiC checkpoint not found: {path}\n"
            "Run: bash vintix_go2/general_locomotion_baselines/clone_third_party.sh"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key in (
        "morph_encoder.net.0.weight",
        "morph_encoder.net.2.weight",
        "adaptation_module.0.weight",
        "adaptation_module.4.weight",
        "actor_body.0.weight",
        "actor_body.6.weight",
    ):
        if key not in state:
            raise KeyError(f"missing {key} in {path}")
        out[key] = tuple(state[key].shape)
    return out


def assert_dims_match_checkpoint(ckpt_path: str | Path | None = None) -> MorficActorDims:
    shapes = load_ac_weight_shapes(ckpt_path)
    dims = MorficActorDims()
    if shapes["morph_encoder.net.0.weight"][1] != dims.morph_raw:
        raise AssertionError(shapes["morph_encoder.net.0.weight"])
    if shapes["morph_encoder.net.2.weight"][0] != dims.morph_latent:
        raise AssertionError(shapes["morph_encoder.net.2.weight"])
    if shapes["adaptation_module.0.weight"][1] != dims.history_flat:
        raise AssertionError(
            f"adaptation in {shapes['adaptation_module.0.weight'][1]} != {dims.history_flat}"
        )
    if shapes["adaptation_module.4.weight"][0] != dims.adapt_latent:
        raise AssertionError(shapes["adaptation_module.4.weight"])
    if shapes["actor_body.0.weight"][1] != dims.actor_in:
        raise AssertionError(
            f"actor in {shapes['actor_body.0.weight'][1]} != {dims.actor_in}"
        )
    if shapes["actor_body.6.weight"][0] != dims.action_dim:
        raise AssertionError(shapes["actor_body.6.weight"])
    return dims


class _MorphEncoder(nn.Module):
    def __init__(self, state: dict):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128),
            nn.ELU(),
            nn.Linear(128, 64),
        )
        self.net[0].weight.data.copy_(state["morph_encoder.net.0.weight"])
        self.net[0].bias.data.copy_(state["morph_encoder.net.0.bias"])
        self.net[2].weight.data.copy_(state["morph_encoder.net.2.weight"])
        self.net[2].bias.data.copy_(state["morph_encoder.net.2.bias"])

    def forward(self, morph_norm: torch.Tensor) -> torch.Tensor:
        return self.net(morph_norm)


class _Adaptation(nn.Module):
    def __init__(self, state: dict):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2100, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 2),
        )
        self.net[0].weight.data.copy_(state["adaptation_module.0.weight"])
        self.net[0].bias.data.copy_(state["adaptation_module.0.bias"])
        self.net[2].weight.data.copy_(state["adaptation_module.2.weight"])
        self.net[2].bias.data.copy_(state["adaptation_module.2.bias"])
        self.net[4].weight.data.copy_(state["adaptation_module.4.weight"])
        self.net[4].bias.data.copy_(state["adaptation_module.4.bias"])

    def forward(self, hist_flat: torch.Tensor) -> torch.Tensor:
        return self.net(hist_flat)


class _ActorBody(nn.Module):
    def __init__(self, state: dict):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2166, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12),
        )
        self.net[0].weight.data.copy_(state["actor_body.0.weight"])
        self.net[0].bias.data.copy_(state["actor_body.0.bias"])
        self.net[2].weight.data.copy_(state["actor_body.2.weight"])
        self.net[2].bias.data.copy_(state["actor_body.2.bias"])
        self.net[4].weight.data.copy_(state["actor_body.4.weight"])
        self.net[4].bias.data.copy_(state["actor_body.4.bias"])
        self.net[6].weight.data.copy_(state["actor_body.6.weight"])
        self.net[6].bias.data.copy_(state["actor_body.6.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def smoke_forward_shapes(
    ckpt_path: str | Path | None = None,
    batch_size: int = 2,
    steps: int = 5,
) -> dict[str, tuple[int, ...]]:
    """CPU smoke: synthetic Genesis obs → adapter → morph/adapt/actor tensors."""
    dims = assert_dims_match_checkpoint(ckpt_path)
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    state = torch.load(path, map_location="cpu", weights_only=False)

    adapter = GenesisToMorficObs(batch_size=batch_size, device="cpu")
    morph_enc = _MorphEncoder(state).eval()
    adapt = _Adaptation(state).eval()
    actor = _ActorBody(state).eval()

    morph = normalize_morphology(go2_nominal_morphology()).unsqueeze(0).expand(batch_size, -1)
    with torch.no_grad():
        z_m = morph_enc(morph)
        last_action = None
        for _ in range(steps):
            fake = torch.randn(batch_size, 45)
            # Keep gravity unit-ish and commands small so values are finite.
            fake[:, 3:6] = torch.nn.functional.normalize(fake[:, 3:6], dim=-1)
            fake[:, 6:9] = 0.1 * torch.randn(batch_size, 3)
            packed = adapter.step(fake)
            z_a = adapt(packed["obs_history_flat"])
            actor_in = torch.cat([packed["obs_history_flat"], z_m, z_a], dim=-1)
            if actor_in.shape[-1] != dims.actor_in:
                raise AssertionError(actor_in.shape)
            last_action = actor(actor_in)
            if last_action.shape != (batch_size, dims.action_dim):
                raise AssertionError(last_action.shape)
            if not torch.isfinite(last_action).all():
                raise AssertionError("non-finite actions")

    return {
        "obs": tuple(packed["obs"].shape),
        "obs_history_flat": tuple(packed["obs_history_flat"].shape),
        "z_morph": tuple(z_m.shape),
        "z_adapt": tuple(z_a.shape),
        "action": tuple(last_action.shape),
    }
