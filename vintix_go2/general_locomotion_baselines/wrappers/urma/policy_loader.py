"""URMA pretrained inspection without RL-X / MuJoCo.

The official `experiments/pre_trained_model` is an RL-X (Flax/msgpack) dump.
Loading the actor needs the RL-X stack, which is intentionally deferred
(GPU + heavy deps). This module only verifies Go2 observation geometry and
that the checkpoint file exists / has the expected magic header.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .joint_description import UrmaGo2Dims
from .obs_adapter import GenesisToUrmaGo2Obs

DEFAULT_CKPT = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "URMA_one_policy_to_run_them_all"
    / "experiments"
    / "pre_trained_model"
)

# msgpack map that starts with key "config_algorit..." in the official dump
_URMA_MAGIC = b"\x83\xb0config_algorit"


@dataclass(frozen=True)
class UrmaCheckpointInfo:
    path: str
    size_bytes: int
    magic_ok: bool
    format_note: str = (
        "RL-X Flax/msgpack dump; actor forward requires RL-X (deferred). "
        "Use official experiments/test.sh on GPU vacancy."
    )


def inspect_checkpoint(ckpt_path: str | Path | None = None) -> UrmaCheckpointInfo:
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not path.is_file():
        raise FileNotFoundError(
            f"URMA pretrained not found: {path}\n"
            "Run: bash vintix_go2/general_locomotion_baselines/clone_third_party.sh"
        )
    raw = path.read_bytes()[:64]
    magic_ok = raw.startswith(_URMA_MAGIC) or b"config_algorit" in raw
    return UrmaCheckpointInfo(
        path=str(path),
        size_bytes=path.stat().st_size,
        magic_ok=magic_ok,
    )


def assert_go2_dims() -> UrmaGo2Dims:
    dims = UrmaGo2Dims()
    if dims.obs_dim != 380:
        raise AssertionError(dims.obs_dim)
    if dims.joint_description != 23:
        raise AssertionError(dims.joint_description)
    if dims.foot_description != 10:
        raise AssertionError(dims.foot_description)
    if dims.action_dim != 12:
        raise AssertionError(dims.action_dim)
    return dims


def smoke_obs_shapes(batch_size: int = 2, steps: int = 5) -> dict[str, tuple[int, ...]]:
    dims = assert_go2_dims()
    adapter = GenesisToUrmaGo2Obs(batch_size=batch_size, device="cpu")
    last = None
    for _ in range(steps):
        fake = torch.randn(batch_size, 45)
        fake[:, 3:6] = torch.nn.functional.normalize(fake[:, 3:6], dim=-1)
        fake[:, 6:9] = 0.1 * torch.randn(batch_size, 3)
        last = adapter.step(fake)
        if last["obs_flat"].shape != (batch_size, dims.obs_dim):
            raise AssertionError(last["obs_flat"].shape)
        if not torch.isfinite(last["obs_flat"]).all():
            raise AssertionError("non-finite obs")
    assert last is not None
    return {
        "obs_flat": tuple(last["obs_flat"].shape),
        "joint_block": tuple(last["joint_block"].shape),
        "foot_block": tuple(last["foot_block"].shape),
        "general": tuple(last["general"].shape),
    }
